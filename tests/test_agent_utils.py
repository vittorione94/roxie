import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from roxie.agents.agent import Agent, ObsStats
from roxie.exploration.noisy import GaussianNoise
from roxie.models.actors import TanhNormal
from roxie.agents.utils import (
    Transition,
    fused_grad_steps,
    make_optimizer,
    serialize_bound,
    soft_update,
    transition_prototype,
)


class TestScaleToEnv:
    def test_center(self):
        x = jnp.array([0.0])
        result = Agent.scale_to_env(x, jnp.array([-1.0]), jnp.array([1.0]))
        assert jnp.isclose(result, 0.0)

    def test_low_bound(self):
        x = jnp.array([-1.0])
        result = Agent.scale_to_env(x, jnp.array([-2.0]), jnp.array([2.0]))
        assert jnp.isclose(result, -2.0)

    def test_high_bound(self):
        x = jnp.array([1.0])
        result = Agent.scale_to_env(x, jnp.array([-2.0]), jnp.array([2.0]))
        assert jnp.isclose(result, 2.0)

    def test_asymmetric_bounds(self):
        x = jnp.array([0.0])
        result = Agent.scale_to_env(x, jnp.array([0.0]), jnp.array([10.0]))
        assert jnp.isclose(result, 5.0)

    def test_batch(self):
        x = jnp.array([[-1.0, 0.0, 1.0]])
        low = jnp.array([0.0, 0.0, 0.0])
        high = jnp.array([1.0, 1.0, 1.0])
        result = Agent.scale_to_env(x, low, high)
        expected = jnp.array([[0.0, 0.5, 1.0]])
        assert jnp.allclose(result, expected)


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
        result = Agent.finite_or_zero(x)
        assert jnp.array_equal(result, jnp.array([0.0, 0.0, 0.0, 0.5, -1.0]))

    def test_passes_finite_through(self):
        x = jnp.array([[-1.0, 0.0, 1.0]])
        assert jnp.array_equal(Agent.finite_or_zero(x), x)

    def test_deterministic_step_is_finite(self):
        actor = _NaNDeterministicActor(3)
        noise = GaussianNoise(action_shape=(3,), initial_noise_scale=0.1)
        action, applied_noise = Agent.deterministic_step_fn(
            actor, jnp.zeros((4, 8)), jax.random.PRNGKey(0), noise
        )
        assert jnp.isfinite(action).all()
        assert jnp.isfinite(applied_noise).all()
        assert jnp.all(jnp.abs(action) <= 1.0)

    def test_stochastic_step_is_finite(self):
        actor = _NaNStochasticActor(3)
        action, deviation, _, _, _ = Agent.stochastic_step_fn(
            actor, jnp.zeros((4, 8)), False, jax.random.PRNGKey(0)
        )
        assert jnp.array_equal(action, jnp.zeros((4, 3)))
        assert jnp.isfinite(deviation).all()

    def test_stochastic_eval_step_is_finite(self):
        actor = _NaNStochasticActor(3)
        action, deviation, _, _, _ = Agent.stochastic_step_fn(
            actor, jnp.zeros((4, 8)), True, jax.random.PRNGKey(0)
        )
        assert jnp.array_equal(action, jnp.zeros((4, 3)))
        assert jnp.isfinite(deviation).all()


class _FakeBufferState:
    """A buffer_state stand-in: `_pruned_transition` reads only `.experience`."""

    def __init__(self, experience):
        self.experience = experience


class TestNonFiniteNeverPersists:
    """The two stores a NaN would survive in for the rest of the run.

    Both are cumulative: the statistics are running sums that never recover, and
    a buffer item is resampled into batches until the run ends. One NaN write is
    not one bad gradient step, it is a permanently poisoned source.
    """

    def test_obs_stats_survive_a_nan_observation(self):
        stats = Agent.init_obs_stats((3,))
        obs = jnp.array([[1.0, 2.0, 3.0], [jnp.nan, jnp.inf, 1.0]])

        stats = Agent.update_obs_stats(stats, obs)
        mean, std = Agent.obs_mean_std(stats, 1e-8)

        assert jnp.isfinite(stats.sum).all() and jnp.isfinite(stats.sumsq).all()
        assert jnp.isfinite(mean).all() and jnp.isfinite(std).all()

    def test_a_later_clean_batch_still_normalizes(self):
        """The failure this prevents: once the sums are NaN, EVERY agent sees
        NaN for EVERY env forever, whether or not anything is still diverging."""
        stats = Agent.init_obs_stats((3,))
        stats = Agent.update_obs_stats(stats, jnp.full((2, 3), jnp.nan))
        stats = Agent.update_obs_stats(stats, jnp.ones((2, 3)))

        mean, std = Agent.obs_mean_std(stats, 1e-8)
        clean = Agent.normalize_obs(jnp.ones((1, 3)), mean, std, 10.0)
        assert jnp.isfinite(clean).all()

    def test_buffer_write_is_scrubbed(self):
        state = _FakeBufferState(transition_prototype(3, 2))
        written = Agent._pruned_transition(
            state,
            observation=jnp.full((1, 3), jnp.nan),
            action=jnp.zeros((1, 2)),
            reward=jnp.array([jnp.nan]),
            terminal=jnp.array([False]),
            truncation=jnp.array([False]),
        )

        assert jnp.array_equal(written.observation, jnp.zeros((1, 3)))
        assert jnp.array_equal(written.reward, jnp.zeros((1,)))
        # Flags are not floats and must pass through untouched.
        assert written.terminal.dtype == jnp.bool_

    def test_buffer_pruning_still_drops_unused_fields(self):
        """The scrub must not resurrect a field this buffer never allocated."""
        state = _FakeBufferState(transition_prototype(3, 2, truncation=False))
        written = Agent._pruned_transition(
            state,
            observation=jnp.zeros((1, 3)),
            action=jnp.zeros((1, 2)),
            reward=jnp.zeros((1,)),
            terminal=jnp.array([False]),
            truncation=jnp.array([False]),
        )
        assert written.truncation is None


class TestObsStats:
    def test_init_obs_stats(self):
        stats = Agent.init_obs_stats(4)
        assert jnp.isclose(stats.count, 0.0)
        assert stats.sum.shape == (4,)
        assert stats.sumsq.shape == (4,)
        assert jnp.allclose(stats.sum, 0.0)
        assert jnp.allclose(stats.sumsq, 0.0)

    def test_update_obs_stats(self):
        stats = Agent.init_obs_stats(3)
        batch = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        updated = Agent.update_obs_stats(stats, batch)
        assert jnp.isclose(updated.count, 2.0)
        assert jnp.allclose(updated.sum, jnp.array([5.0, 7.0, 9.0]))

    def test_obs_mean_std(self):
        stats = Agent.init_obs_stats(2)
        batch = jnp.array([[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]])
        stats = Agent.update_obs_stats(stats, batch)
        mean, std = Agent.obs_mean_std(stats, eps=1e-8)
        assert jnp.allclose(mean, jnp.array([4.0, 6.0]))
        expected_var = jnp.array([(4+16+36)/3 - 16, (16+36+64)/3 - 36])
        expected_std = jnp.sqrt(expected_var + 1e-8)
        assert jnp.allclose(std, expected_std, atol=1e-5)

    def test_obs_mean_std_empty(self):
        stats = Agent.init_obs_stats(3)
        mean, std = Agent.obs_mean_std(stats, eps=1e-8)
        assert jnp.allclose(mean, 0.0)

    def test_normalize_obs(self):
        x = jnp.array([10.0, 20.0])
        mean = jnp.array([5.0, 10.0])
        std = jnp.array([5.0, 5.0])
        result = Agent.normalize_obs(x, mean, std, clip=5.0)
        expected = jnp.array([1.0, 2.0])
        assert jnp.allclose(result, expected)

    def test_normalize_obs_clipping(self):
        x = jnp.array([100.0])
        mean = jnp.array([0.0])
        std = jnp.array([1.0])
        result = Agent.normalize_obs(x, mean, std, clip=3.0)
        assert jnp.isclose(result, 3.0)

    def test_full_pipeline(self):
        stats = Agent.init_obs_stats(2)
        for i in range(100):
            batch = jnp.array([[float(i), float(i) * 2]])
            stats = Agent.update_obs_stats(stats, batch)
        mean, std = Agent.obs_mean_std(stats, eps=1e-8)
        normalized = Agent.normalize_obs(jnp.array([50.0, 100.0]), mean, std, clip=5.0)
        assert jnp.all(jnp.abs(normalized) <= 5.0)


class TestTransition:
    def test_creation(self):
        t = Transition(
            observation=jnp.zeros(4),
            action=jnp.zeros(2),
            reward=jnp.array(1.0),
            terminal=jnp.array(False),
        )
        assert t.observation.shape == (4,)
        assert t.action.shape == (2,)
        assert t.log_probs is None
        assert t.value is None

    def test_with_optional_fields(self):
        t = Transition(
            observation=jnp.zeros(4),
            action=jnp.zeros(2),
            reward=jnp.array(1.0),
            terminal=jnp.array(False),
            log_probs=jnp.array(-0.5),
            value=jnp.array(3.0),
        )
        assert t.log_probs is not None
        assert t.value is not None


class TestSerializeBound:
    def test_scalar_float(self):
        result = serialize_bound(1.5)
        assert result == 1.5

    def test_jax_scalar(self):
        result = serialize_bound(jnp.array(2.0))
        assert result == 2.0
        assert isinstance(result, float)

    def test_numpy_scalar(self):
        result = serialize_bound(np.float32(3.0))
        assert result == 3.0
        assert isinstance(result, float)

    def test_per_actuator_vector_keeps_every_entry(self):
        # Heterogeneous ctrlrange: recording only the first entry would report
        # the wrong bound for every other actuator.
        result = serialize_bound(jnp.array([-1.0, -0.5, -2.0]))
        assert result == [-1.0, -0.5, -2.0]


class TestTransitionPrototype:
    """The buffer schema factory shared by every agent."""

    def test_shapes_and_dtypes(self):
        p = transition_prototype(4, 2)
        assert p.observation.shape == (4,) and p.observation.dtype == jnp.float32
        assert p.action.shape == (2,) and p.action.dtype == jnp.float32
        assert p.reward.shape == () and p.reward.dtype == jnp.float32
        assert p.terminal.dtype == jnp.bool_

    def test_truncation_on_by_default(self):
        # Off-policy n-step windows must be able to stop at a truncation, which
        # the terminal flag does not mark.
        assert transition_prototype(4, 2).truncation is not None

    def test_truncation_can_be_dropped(self):
        # MPO's 1-step target reads only `terminal`; the field is then an empty
        # pytree node and costs no buffer memory.
        assert transition_prototype(4, 2, truncation=False).truncation is None

    def test_on_policy_adds_behaviour_stats(self):
        p = transition_prototype(4, 2, on_policy=True)
        assert p.log_probs is not None and p.value is not None
        # PPO's GAE needs the truncation flag too.
        assert p.truncation is not None

    def test_off_policy_has_no_behaviour_stats(self):
        p = transition_prototype(4, 2)
        assert p.log_probs is None and p.value is None


class _Tiny(nnx.Module):
    def __init__(self, value: float):
        self.w = nnx.Param(jnp.full((3,), value, dtype=jnp.float32))

    def __call__(self, x):
        return x * self.w


class TestSoftUpdate:
    def test_interpolates_towards_the_source(self):
        target, source = _Tiny(0.0), _Tiny(1.0)
        soft_update(target, source, 0.25)
        assert jnp.allclose(target.w.value, 0.25)
        # The source is untouched — only the target moves.
        assert jnp.allclose(source.w.value, 1.0)

    def test_tau_zero_freezes_the_target(self):
        target, source = _Tiny(2.0), _Tiny(9.0)
        soft_update(target, source, 0.0)
        assert jnp.allclose(target.w.value, 2.0)

    def test_tau_one_copies(self):
        target, source = _Tiny(2.0), _Tiny(9.0)
        soft_update(target, source, 1.0)
        assert jnp.allclose(target.w.value, 9.0)

    def test_matches_optax_incremental_update(self):
        target, source = _Tiny(-1.0), _Tiny(3.0)
        expected = optax.incremental_update(
            new_tensors=nnx.state(source, nnx.Param),
            old_tensors=nnx.state(target, nnx.Param),
            step_size=0.005,
        )
        soft_update(target, source, 0.005)
        assert jnp.allclose(target.w.value, expected["w"].value)


class TestFusedGradSteps:
    """The `lax.scan` burst every agent's `_grad_steps` is built on."""

    def test_carries_updates_across_steps(self):
        module = _Tiny(0.0)

        def step(m, _key):
            m.w.value = m.w.value + 1.0
            return m.w.value.sum()

        module, sums = fused_grad_steps(module, jax.random.PRNGKey(0), 4, step)
        # Accumulating (not restarting from the entry value) every step.
        assert jnp.allclose(module.w.value, 4.0)
        assert jnp.allclose(sums, jnp.array([3.0, 6.0, 9.0, 12.0]))

    def test_each_step_gets_a_distinct_key(self):
        module = _Tiny(0.0)

        def step(m, key):
            return jax.random.uniform(key)

        _, draws = fused_grad_steps(module, jax.random.PRNGKey(0), 5, step)
        assert len(jnp.unique(draws)) == 5

    def test_optimizer_slots_stay_in_the_carry(self):
        """A side module in the tuple keeps its Adam state across the burst."""
        module = _Tiny(1.0)
        optimizer = make_optimizer(module, None, learning_rate=0.1)

        def step(nodes, _key):
            m, opt = nodes
            loss, grads = nnx.value_and_grad(lambda mm: jnp.sum(mm.w.value**2))(m)
            opt.update(m, grads)
            return loss

        (module, optimizer), losses = fused_grad_steps(
            (module, optimizer), jax.random.PRNGKey(0), 3, step
        )
        # Descending, so the slots carried forward rather than resetting.
        assert losses[0] > losses[-1]
        assert jnp.all(module.w.value < 1.0)

    def test_extras_are_scanned_alongside_the_keys(self):
        module = _Tiny(0.0)
        mask = jnp.array([True, False, True, False])

        def step(m, _key, do_update):
            m.w.value = m.w.value + jnp.where(do_update, 1.0, 0.0)
            return m.w.value[0]

        module, trace = fused_grad_steps(
            module, jax.random.PRNGKey(0), 4, step, extras=(mask,)
        )
        assert jnp.allclose(module.w.value, 2.0)
        assert jnp.allclose(trace, jnp.array([1.0, 1.0, 2.0, 2.0]))

    def test_none_extra_rides_along_as_an_empty_node(self):
        """How `policy_delay == 1` skips the branch without a mask."""
        module = _Tiny(0.0)

        def step(m, _key, do_update):
            assert do_update is None  # static at trace time
            m.w.value = m.w.value + 1.0
            return m.w.value[0]

        module, _ = fused_grad_steps(
            module, jax.random.PRNGKey(0), 3, step, extras=(None,)
        )
        assert jnp.allclose(module.w.value, 3.0)
