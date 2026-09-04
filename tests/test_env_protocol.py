"""The env boundary, pinned.

These tests are the executable statement of what `roxie.environment.functional`
and `roxie.environment.vector` promise, written against a toy `FuncEnv` rather
than a MuJoCo one so they run in milliseconds on CPU and fail for exactly one
reason. The semantics they pin — the termination/truncation three-way split and
the pre-vs-post-auto-reset observation — are the two things a refactor of this
env boundary is most likely to move silently, and moving either invalidates
every number in the release grid.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from roxie.environment import functional
from roxie.environment.vector import JaxVectorEnv, Timestep, VecState


@struct.dataclass
class _CounterState:
    """Position on a line, plus the start it was reset to."""

    x: jnp.ndarray
    origin: jnp.ndarray


class CounterEnv(functional.FuncEnv):
    """A one-dimensional walker with an env-internal, non-failure cutoff.

    `terminal` (failure) fires at x <= -5; `truncal` (the env ran out of road,
    NOT a failure) fires at x >= +5. Having both lets one env exercise every
    branch of the driver's done logic.
    """

    def __init__(self, start: float = 0.0):
        self.start = start
        self.observation_space = functional.unbounded_box(2)
        self.action_space = functional.box(-1.0, 1.0, shape=(1,))
        self.metadata = {"jax": True, "impl": "counter"}

    def initial(self, rng, params=None):
        # `params`, when given, shifts the start — the stand-in for an env that
        # adapts its own params, and what proves params stays traced.
        shift = 0.0 if params is None else params
        x = jnp.asarray(self.start + shift, dtype=jnp.float32)
        return _CounterState(x=x, origin=x)

    def transition(self, state, action, rng, params=None):
        return state.replace(x=state.x + jnp.squeeze(action))

    def observation(self, state, rng, params=None):
        return jnp.stack([state.x, state.origin])

    def reward(self, state, action, next_state, rng, params=None):
        return next_state.x - state.x

    def terminal(self, state, rng, params=None):
        return state.x <= -5.0

    def truncal(self, state, rng, params=None):
        return state.x >= 5.0

    def transition_info(self, state, action, next_state, params=None):
        return {"metrics": {"x": next_state.x}}


@pytest.fixture
def vec_env():
    return JaxVectorEnv(CounterEnv(), num_envs=4, max_episode_steps=3)


def _pool(env, key, n=4, params=None):
    state, _ = env.reset(key, params=params, num_envs=n)
    return state


def test_reset_returns_state_and_timestep(vec_env, rng_key):
    state, ts = vec_env.reset(rng_key)

    assert isinstance(state, VecState) and isinstance(ts, Timestep)
    assert state.obs.shape == (4, 2)
    assert np.all(np.asarray(state.steps) == 0)
    # A fresh episode is neither terminated nor truncated, and scores nothing.
    assert not np.any(np.asarray(ts.terminated))
    assert not np.any(np.asarray(ts.truncated))
    assert np.all(np.asarray(ts.reward) == 0.0)


def test_spaces_are_single_env(vec_env):
    """The driver reports SINGLE-env spaces; the agent derives its layer widths
    from them, so a batched space here would size every network by num_envs."""
    assert vec_env.single_observation_space.shape == (2,)
    assert vec_env.single_action_space.shape == (1,)
    assert vec_env.num_envs == 4


def test_params_stay_traced_and_reach_initial(vec_env, rng_key):
    """`params` must be an ARGUMENT, not a closure constant: an env that adapts
    its params refreshes them every epoch, and a recompile there would cost more
    than the adaptation buys."""
    jitted = jax.jit(lambda k, p: vec_env.reset(k, params=p)[0].obs)
    at_zero = np.asarray(jitted(rng_key, 0.0))
    at_two = np.asarray(jitted(rng_key, 2.0))

    assert np.allclose(at_two - at_zero, 2.0)
    # One compile for both values is the property under test.
    assert jitted._cache_size() == 1


def test_failure_is_terminated_not_truncated(vec_env, rng_key):
    state = _pool(vec_env, rng_key)
    pool = _pool(vec_env, rng_key)
    # -5 in one step: x goes 0 -> -5, tripping `terminal`.
    actions = jnp.full((4, 1), -5.0)

    _, ts = vec_env.step(state, actions, rng_key, pool)

    assert np.all(np.asarray(ts.terminated))
    assert not np.any(np.asarray(ts.truncated))


def test_env_internal_cutoff_is_truncated_and_clears_termination(vec_env, rng_key):
    """The load-bearing rule: a non-failure cutoff must NOT read as terminal.

    A value-based agent zeroes its Bellman bootstrap on `terminated`, so
    reporting the clip-end cutoff as a termination collapses Q at the cutoff.
    """
    state = _pool(vec_env, rng_key)
    pool = _pool(vec_env, rng_key)
    actions = jnp.full((4, 1), 5.0)  # x: 0 -> +5, tripping `truncal`

    _, ts = vec_env.step(state, actions, rng_key, pool)

    assert np.all(np.asarray(ts.truncated))
    assert not np.any(np.asarray(ts.terminated))


def test_step_limit_truncates_without_terminating(vec_env, rng_key):
    """max_episode_steps=3, and a step limit is the driver's own truncation."""
    state = _pool(vec_env, rng_key)
    pool = _pool(vec_env, rng_key)
    actions = jnp.zeros((4, 1))  # never trips either env rule

    for i in range(3):
        state, ts = vec_env.step(state, actions, rng_key, pool)
        expected = i == 2
        assert bool(np.all(np.asarray(ts.truncated))) is expected
        assert not np.any(np.asarray(ts.terminated))


def test_step_limit_does_not_mask_a_genuine_failure(vec_env, rng_key):
    """Unlike an env-internal cutoff, the step limit does NOT clear termination:
    a fall on the very last step is still a fall."""
    state = _pool(vec_env, rng_key)
    pool = _pool(vec_env, rng_key)
    zero, fall = jnp.zeros((4, 1)), jnp.full((4, 1), -5.0)

    state, _ = vec_env.step(state, zero, rng_key, pool)
    state, _ = vec_env.step(state, zero, rng_key, pool)
    _, ts = vec_env.step(state, fall, rng_key, pool)  # step 3 == the limit

    assert np.all(np.asarray(ts.terminated))
    assert np.all(np.asarray(ts.truncated))


def test_timestep_obs_is_pre_reset_and_state_obs_is_post_reset(rng_key):
    """The distinction the replay buffer depends on.

    `Timestep.obs` is the true next observation of the transition that just
    happened; `VecState.obs` is what the next action is selected from. Conflating
    them stores the reset observation as the terminal transition's next_obs.
    """
    env = JaxVectorEnv(CounterEnv(start=0.0), num_envs=4, max_episode_steps=0)
    state = _pool(env, rng_key)
    # A pool whose starts are all at +100, so a reset is unmistakable.
    pool = _pool(env, rng_key, params=100.0)
    actions = jnp.full((4, 1), -5.0)  # terminates immediately

    next_state, ts = env.step(state, actions, rng_key, pool)

    assert np.allclose(np.asarray(ts.obs)[:, 0], -5.0)          # pre-reset
    assert np.allclose(np.asarray(next_state.obs)[:, 0], 100.0)  # post-reset


def test_autoreset_leaves_live_envs_untouched(rng_key):
    env = JaxVectorEnv(CounterEnv(), num_envs=4, max_episode_steps=0)
    state = _pool(env, rng_key)
    pool = _pool(env, rng_key, params=100.0)
    # Env 0 terminates; the rest step normally.
    actions = jnp.array([[-5.0], [1.0], [1.0], [1.0]])

    next_state, _ = env.step(state, actions, rng_key, pool)

    x = np.asarray(next_state.obs)[:, 0]
    assert x[0] == pytest.approx(100.0)
    assert np.allclose(x[1:], 1.0)


def test_autoreset_zeroes_the_step_counter(rng_key):
    """A reset env starts a fresh episode, so its time limit restarts too —
    otherwise a long-lived worker truncates ever sooner."""
    env = JaxVectorEnv(CounterEnv(), num_envs=4, max_episode_steps=0)
    state = _pool(env, rng_key)
    pool = _pool(env, rng_key)
    actions = jnp.array([[-5.0], [0.0], [0.0], [0.0]])

    state, _ = env.step(state, actions, rng_key, pool)
    state, _ = env.step(state, jnp.zeros((4, 1)), rng_key, pool)

    steps = np.asarray(state.steps)
    assert steps[0] == 1      # reset on the first step, then stepped once
    assert np.all(steps[1:] == 2)


def test_whole_step_is_jittable_and_scannable(rng_key):
    """The property the trainer is built on: warmup fills the replay buffer with
    one scanned dispatch, and gymnasium's own vector env cannot do this."""
    env = JaxVectorEnv(CounterEnv(), num_envs=4, max_episode_steps=0)
    state = _pool(env, rng_key)
    pool = _pool(env, rng_key)

    def body(carry, _):
        state, key = carry
        key, step_key = jax.random.split(key)
        state, ts = env.step(state, jnp.full((4, 1), 0.5), step_key, pool)
        return (state, key), ts.reward

    (final, _), rewards = jax.jit(
        lambda s, k: jax.lax.scan(body, (s, k), None, length=8)
    )(state, rng_key)

    assert rewards.shape == (8, 4)
    assert np.allclose(np.asarray(rewards), 0.5)
    assert np.allclose(np.asarray(final.obs)[:, 0], 4.0)


def test_transition_info_metrics_are_batched(vec_env, rng_key):
    """Per-step env metrics ride in `info["metrics"]` on BOTH backends; the
    trainer means them over an epoch and logs them as `train/<key>`."""
    state = _pool(vec_env, rng_key)
    pool = _pool(vec_env, rng_key)

    _, ts = vec_env.step(state, jnp.ones((4, 1)), rng_key, pool)

    assert set(ts.info["metrics"]) == {"x"}
    assert np.asarray(ts.info["metrics"]["x"]).shape == (4,)


def test_timestep_unpacks_as_the_gymnasium_five_tuple(vec_env, rng_key):
    state = _pool(vec_env, rng_key)
    pool = _pool(vec_env, rng_key)

    _, ts = vec_env.step(state, jnp.zeros((4, 1)), rng_key, pool)
    obs, reward, terminated, truncated, info = ts

    assert obs.shape == (4, 2) and reward.shape == (4,)
    assert terminated.shape == (4,) and truncated.shape == (4,)
    assert isinstance(info, dict)
