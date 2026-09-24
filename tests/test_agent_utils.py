import flashbax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from roxie.exploration.noisy import GaussianNoise
from roxie.models.actors import (
    TanhNormal,
    deterministic_step_fn,
    stochastic_step_fn,
)
from roxie.agents.utils import (
    Transition,
    fused_grad_steps,
    make_optimizer,
    serialize_bound,
    soft_update,
    transition_prototype,
)
from roxie.utils.math import (
    finite_or_zero,
    init_obs_stats,
    normalize_obs,
    obs_mean_std,
    scale_to_env,
    update_obs_stats,
)
from roxie.utils.memory import ReplayManager


class TestScaleToEnv:
    def test_maps_the_unit_interval_onto_the_env_bounds(self):
        """An affine map from the actor's [-1, 1] onto `[low, high]`: the two
        ends and the midpoint pin it completely, symmetric or not."""
        for x, low, high, expected in (
            (0.0, -1.0, 1.0, 0.0),
            (-1.0, -2.0, 2.0, -2.0),
            (1.0, -2.0, 2.0, 2.0),
            (0.0, 0.0, 10.0, 5.0),
        ):
            result = scale_to_env(
                jnp.array([x], dtype=jnp.float32),
                jnp.array([low], dtype=jnp.float32),
                jnp.array([high], dtype=jnp.float32),
            )
            assert jnp.isclose(result, expected), (x, low, high)

    def test_applies_per_actuator_bounds_across_a_batch(self):
        """A heterogeneous ctrlrange is the real case: each column takes its
        own bound, and the leading batch axis broadcasts."""
        result = scale_to_env(
            jnp.array([[-1.0, 0.0, 1.0]], dtype=jnp.float32),
            jnp.zeros(3, dtype=jnp.float32),
            jnp.ones(3, dtype=jnp.float32),
        )
        assert jnp.allclose(result, jnp.array([[0.0, 0.5, 1.0]], dtype=jnp.float32))


class _NaNDeterministicActor(nnx.Module):
    """Stands in for an actor whose weights have blown up."""

    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def __call__(self, obs):
        return jnp.full((obs.shape[0], self.action_dim), jnp.nan)


class _NaNStochasticActor(nnx.Module):
    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def __call__(self, obs):
        nan = jnp.full((obs.shape[0], self.action_dim), jnp.nan)
        return TanhNormal(nan, jnp.ones_like(nan))


class TestFiniteOrZero:
    def test_replaces_non_finite(self):
        x = jnp.array([jnp.nan, jnp.inf, -jnp.inf, 0.5, -1.0])
        result = finite_or_zero(x)
        assert jnp.array_equal(result, jnp.array(
            [0.0, 0.0, 0.0, 0.5, -1.0], dtype=jnp.float32
        ))

    def test_passes_finite_through(self):
        x = jnp.array([[-1.0, 0.0, 1.0]], dtype=jnp.float32)
        assert jnp.array_equal(finite_or_zero(x), x)

    def test_deterministic_step_is_finite(self):
        actor = _NaNDeterministicActor(3)
        noise = GaussianNoise(action_shape=(3,), initial_noise_scale=0.1)
        action, applied_noise = deterministic_step_fn(
            actor, jnp.zeros((4, 8), dtype=jnp.float32), jax.random.PRNGKey(0), noise
        )
        assert jnp.isfinite(action).all()
        assert jnp.isfinite(applied_noise).all()
        assert jnp.all(jnp.abs(action) <= 1.0)

    def test_stochastic_step_is_finite(self):
        actor = _NaNStochasticActor(3)
        action, deviation, _, _, _ = stochastic_step_fn(
            actor, jnp.zeros((4, 8), dtype=jnp.float32), False, jax.random.PRNGKey(0)
        )
        assert jnp.array_equal(action, jnp.zeros((4, 3), dtype=jnp.float32))
        assert jnp.isfinite(deviation).all()

    def test_stochastic_eval_step_is_finite(self):
        actor = _NaNStochasticActor(3)
        action, deviation, _, _, _ = stochastic_step_fn(
            actor, jnp.zeros((4, 8), dtype=jnp.float32), True, jax.random.PRNGKey(0)
        )
        assert jnp.array_equal(action, jnp.zeros((4, 3), dtype=jnp.float32))
        assert jnp.isfinite(deviation).all()


def _manager(prototype):
    """A `ReplayManager` over an unallocated buffer: `prepare` reads the
    prototype alone, and `make_flat_buffer` only builds a struct of functions."""
    return ReplayManager(
        flashbax.buffers.make_flat_buffer(
            max_length=8, min_length=1, sample_batch_size=1, add_batch_size=1,
        ),
        prototype,
    )


class TestNonFiniteNeverPersists:
    """The two stores a NaN would survive in for the rest of the run.

    Both are cumulative: the statistics are running sums that never recover, and
    a buffer item is resampled into batches until the run ends. One NaN write is
    not one bad gradient step, it is a permanently poisoned source.
    """

    def test_obs_stats_survive_a_nan_observation(self):
        stats = init_obs_stats((3,))
        obs = jnp.array([[1.0, 2.0, 3.0], [jnp.nan, jnp.inf, 1.0]])

        stats = update_obs_stats(stats, obs)
        mean, std = obs_mean_std(stats, 1e-8)

        assert jnp.isfinite(stats.sum).all() and jnp.isfinite(stats.sumsq).all()
        assert jnp.isfinite(mean).all() and jnp.isfinite(std).all()

    def test_a_later_clean_batch_still_normalizes(self):
        """The failure this prevents: once the sums are NaN, EVERY agent sees
        NaN for EVERY env forever, whether or not anything is still diverging."""
        stats = init_obs_stats((3,))
        stats = update_obs_stats(stats, jnp.full((2, 3), jnp.nan))
        stats = update_obs_stats(stats, jnp.ones((2, 3), dtype=jnp.float32))

        mean, std = obs_mean_std(stats, 1e-8)
        clean = normalize_obs(jnp.ones((1, 3), dtype=jnp.float32), mean, std, 10.0)
        assert jnp.isfinite(clean).all()

    def test_buffer_write_is_scrubbed(self):
        written = _manager(transition_prototype(3, 2)).prepare(
            Transition(
                observation=jnp.full((1, 3), jnp.nan),
                action=jnp.zeros((1, 2), dtype=jnp.float32),
                reward=jnp.array([jnp.nan]),
                terminal=jnp.array([False]),
                truncation=jnp.array([False]),
            ),
        )

        assert jnp.array_equal(written.observation, jnp.zeros(
            (1, 3), dtype=jnp.float32
        ))
        assert jnp.array_equal(written.reward, jnp.zeros((1,), dtype=jnp.float32))
        # Flags are not floats and must pass through untouched.
        assert written.terminal.dtype == jnp.bool_

    def test_buffer_pruning_still_drops_unused_fields(self):
        """The scrub must not resurrect a field this buffer never allocated."""
        written = _manager(transition_prototype(3, 2, truncation=False)).prepare(
            Transition(
                observation=jnp.zeros((1, 3), dtype=jnp.float32),
                action=jnp.zeros((1, 2), dtype=jnp.float32),
                reward=jnp.zeros((1,), dtype=jnp.float32),
                terminal=jnp.array([False]),
                truncation=jnp.array([False]),
            ),
        )
        assert written.truncation is None


class TestObsStats:
    def test_accumulates_moments_and_recovers_mean_and_std(self):
        """Sums in, mean/std out. The variance is taken as E[x^2] - E[x]^2, so
        this pins the recovery rather than only the accumulation."""
        stats = init_obs_stats(2)
        assert stats.sum.shape == stats.sumsq.shape == (2,)
        # Empty is mean 0 / std 1, so normalization before any data is inert.
        assert jnp.allclose(obs_mean_std(stats, eps=1e-8)[0], 0.0)

        batch = jnp.array([[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]], dtype=jnp.float32)
        stats = update_obs_stats(stats, batch)
        assert jnp.isclose(stats.count, 3.0)
        assert jnp.allclose(stats.sum, jnp.array([12.0, 18.0], dtype=jnp.float32))

        mean, std = obs_mean_std(stats, eps=1e-8)
        assert jnp.allclose(mean, jnp.array([4.0, 6.0], dtype=jnp.float32))
        expected_var = jnp.array([(4 + 16 + 36) / 3 - 16, (16 + 36 + 64) / 3 - 36])
        assert jnp.allclose(std, jnp.sqrt(expected_var + 1e-8), atol=1e-5)

    def test_normalize_obs_is_a_z_score_inside_the_clip(self):
        mean = jnp.array([5.0, 10.0], dtype=jnp.float32)
        std = jnp.array([5.0, 5.0], dtype=jnp.float32)
        result = normalize_obs(
            jnp.array([10.0, 20.0], dtype=jnp.float32), mean, std, clip=5.0
        )
        assert jnp.allclose(result, jnp.array([1.0, 2.0], dtype=jnp.float32))

        # Past the clip it saturates — an outlier must not reach the network
        # as a hundred standard deviations.
        clipped = normalize_obs(
            jnp.array([100.0], dtype=jnp.float32),
            jnp.zeros(1, dtype=jnp.float32),
            jnp.ones(1, dtype=jnp.float32),
            clip=3.0,
        )
        assert jnp.isclose(clipped, 3.0)

    def test_clip_none_leaves_the_z_score_unbounded(self):
        """`obs_norm_clip: null` is what HumanoidWalk needs.

        Its running moments are set by the 92% of every episode a fallen
        humanoid spends motionless, so the states carrying all of the reward sit
        at z ~ 8.5 and any bound of 5 maps upright and half-fallen onto the same
        number. Unbounded, they stay distinguishable.
        """
        far = normalize_obs(
            jnp.array([8.5, 6.5], dtype=jnp.float32),
            jnp.zeros(2, dtype=jnp.float32),
            jnp.ones(2, dtype=jnp.float32),
            clip=None,
        )
        assert jnp.allclose(far, jnp.array([8.5, 6.5], dtype=jnp.float32))
        # The property that matters is that they did not collapse onto each other.
        assert float(far[0] - far[1]) == pytest.approx(2.0)


class TestSerializeBound:
    def test_every_bound_a_checkpoint_can_carry_becomes_json(self):
        """A bound reaches the agent as a Python float, a JAX scalar, a numpy
        scalar or a per-actuator vector. Recording only the first entry of the
        last one would report the wrong bound for every other actuator.
        """
        assert serialize_bound(1.5) == 1.5
        for scalar in (jnp.array(2.0, dtype=jnp.float32), np.float32(2.0)):
            assert serialize_bound(scalar) == 2.0
            assert isinstance(serialize_bound(scalar), float)
        # Heterogeneous ctrlrange, kept entry for entry.
        assert serialize_bound(
            jnp.array([-1.0, -0.5, -2.0], dtype=jnp.float32)
        ) == [-1.0, -0.5, -2.0]


class TestTransitionPrototype:
    """The buffer schema factory shared by every agent."""

    def test_off_policy_stores_the_transition_and_its_truncation(self):
        p = transition_prototype(4, 2)
        assert p.observation.shape == (4,) and p.observation.dtype == jnp.float32
        assert p.action.shape == (2,) and p.action.dtype == jnp.float32
        assert p.reward.shape == () and p.reward.dtype == jnp.float32
        assert p.terminal.dtype == jnp.bool_
        # n-step windows must be able to stop at a truncation, which the
        # terminal flag does not mark. No behaviour stats: nothing rescores.
        assert p.truncation is not None
        assert p.log_probs is None and p.value is None

    def test_the_two_opt_outs_are_honoured(self):
        # MPO's 1-step target reads only `terminal`; the field is then an empty
        # pytree node and costs no buffer memory.
        assert transition_prototype(4, 2, truncation=False).truncation is None
        # PPO rescores its own rollout, so it stores what it acted under —
        # and its GAE needs the truncation flag too.
        on_policy = transition_prototype(4, 2, on_policy=True)
        assert on_policy.log_probs is not None and on_policy.value is not None
        assert on_policy.truncation is not None


class _Tiny(nnx.Module):
    def __init__(self, value: float):
        self.w = nnx.Param(jnp.full((3,), value, dtype=jnp.float32))

    def __call__(self, x):
        return x * self.w


class TestSoftUpdate:
    def test_interpolates_the_target_toward_the_source(self):
        """Polyak averaging across its whole range, and the source left alone —
        a soft update that wrote back would collapse the two networks."""
        for tau, expected in ((0.0, 2.0), (0.25, 3.75), (1.0, 9.0)):
            target, source = _Tiny(2.0), _Tiny(9.0)
            soft_update(target, source, tau)
            assert jnp.allclose(target.w[...], expected), tau
            assert jnp.allclose(source.w[...], 9.0)

    def test_matches_optax_incremental_update(self):
        target, source = _Tiny(-1.0), _Tiny(3.0)
        expected = optax.incremental_update(
            new_tensors=nnx.state(source, nnx.Param),
            old_tensors=nnx.state(target, nnx.Param),
            step_size=0.005,
        )
        soft_update(target, source, 0.005)
        assert jnp.allclose(target.w[...], expected["w"][...])


class TestFusedGradSteps:
    """The `lax.scan` sequence every agent's `_<agent>_grad_steps` is built on."""

    def test_carries_updates_across_steps(self):
        module = _Tiny(0.0)

        def step(m, _key):
            m.w[...] = m.w[...] + 1.0
            return m.w[...].sum()

        module, sums = fused_grad_steps(module, jax.random.PRNGKey(0), 4, step)
        # Accumulating (not restarting from the entry value) every step.
        assert jnp.allclose(module.w[...], 4.0)
        assert jnp.allclose(sums, jnp.array([3.0, 6.0, 9.0, 12.0], dtype=jnp.float32))

    def test_each_step_gets_a_distinct_key(self):
        module = _Tiny(0.0)

        def step(m, key):
            return jax.random.uniform(key, dtype=jnp.float32)

        _, draws = fused_grad_steps(module, jax.random.PRNGKey(0), 5, step)
        assert len(jnp.unique(draws)) == 5

    def test_optimizer_slots_stay_in_the_carry(self):
        """A side module in the tuple keeps its Adam state across the pass."""
        module = _Tiny(1.0)
        optimizer = make_optimizer(module, None, learning_rate=0.1)

        def step(nodes, _key):
            m, opt = nodes
            loss, grads = nnx.value_and_grad(lambda mm: jnp.sum(mm.w[...] ** 2))(m)
            opt.update(m, grads)
            return loss

        (module, optimizer), losses = fused_grad_steps(
            (module, optimizer), jax.random.PRNGKey(0), 3, step
        )
        # Descending, so the slots carried forward rather than resetting.
        assert losses[0] > losses[-1]
        assert jnp.all(module.w[...] < 1.0)

    def test_extras_are_scanned_alongside_the_keys(self):
        module = _Tiny(0.0)
        mask = jnp.array([True, False, True, False])

        def step(m, _key, do_update):
            m.w[...] = m.w[...] + jnp.where(do_update, 1.0, 0.0)
            return m.w[...][0]

        module, trace = fused_grad_steps(
            module, jax.random.PRNGKey(0), 4, step, extras=(mask,)
        )
        assert jnp.allclose(module.w[...], 2.0)
        assert jnp.allclose(trace, jnp.array([1.0, 1.0, 2.0, 2.0], dtype=jnp.float32))

    def test_none_extra_rides_along_as_an_empty_node(self):
        """How `policy_delay == 1` skips the branch without a mask."""
        module = _Tiny(0.0)

        def step(m, _key, do_update):
            assert do_update is None  # static at trace time
            m.w[...] = m.w[...] + 1.0
            return m.w[...][0]

        module, _ = fused_grad_steps(
            module, jax.random.PRNGKey(0), 3, step, extras=(None,)
        )
        assert jnp.allclose(module.w[...], 3.0)
