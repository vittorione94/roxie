"""n-step return machinery (roxie.agents.utils.repack_samples + TD3 wiring).

Covers the target math against hand-computed values — clean windows, terminals,
truncations, done-on-first-step — plus n=1 flat-vs-trajectory parity and an
end-to-end TD3 update() smoke test on a real flashbax trajectory buffer.
Everything runs on CPU.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flashbax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from roxie.agents.utils import Transition, repack_samples

GAMMA = 0.9
N = 3
OBS_DIM = 4


@struct.dataclass
class _FakeSample:
    experience: Transition


def _traj_sample(rewards, terminals, truncations):
    """Build a (B=1, T=N+1) trajectory sample; obs_j = j+1 everywhere so the
    selected bootstrap obs is identifiable by value."""
    T = N + 1
    obs = jnp.tile(jnp.arange(1, T + 1, dtype=jnp.float32)[None, :, None], (1, 1, OBS_DIM))
    pad = lambda xs: jnp.array([list(xs) + [0.0] * (T - len(xs))], dtype=jnp.float32)
    return _FakeSample(experience=Transition(
        observation=obs,
        action=jnp.zeros((1, T, 2)),
        reward=pad(rewards),
        terminal=pad(terminals).astype(bool),
        truncation=pad(truncations).astype(bool),
    ))


class TestNStepRepack:
    def test_clean_window(self):
        out = repack_samples(_traj_sample([1.0, 2.0, 3.0], [0, 0, 0], [0, 0, 0]), GAMMA, N)
        expected = 1.0 + GAMMA * 2.0 + GAMMA**2 * 3.0
        np.testing.assert_allclose(out["rewards"][0], expected, rtol=1e-6)
        np.testing.assert_allclose(out["bootstrap"][0], GAMMA**N, rtol=1e-6)
        # bootstrap obs = o_n (value N+1); first obs = o_0 (value 1)
        np.testing.assert_allclose(out["next_observations"][0], (N + 1.0) * np.ones(OBS_DIM))
        np.testing.assert_allclose(out["observations"][0], np.ones(OBS_DIM))

    def test_terminal_mid_window(self):
        # terminal at j=1: r0 + gamma*r1, no bootstrap, later rewards masked
        out = repack_samples(_traj_sample([1.0, 2.0, 99.0], [0, 1, 0], [0, 0, 0]), GAMMA, N)
        np.testing.assert_allclose(out["rewards"][0], 1.0 + GAMMA * 2.0, rtol=1e-6)
        assert out["bootstrap"][0] == 0.0

    def test_truncation_mid_window(self):
        # truncation at j=1: r0 only (truncated step's reward folded into the
        # bootstrap), bootstrap gamma^1 at o_1 (value 2)
        out = repack_samples(_traj_sample([1.0, 2.0, 99.0], [0, 0, 0], [0, 1, 0]), GAMMA, N)
        np.testing.assert_allclose(out["rewards"][0], 1.0, rtol=1e-6)
        np.testing.assert_allclose(out["bootstrap"][0], GAMMA, rtol=1e-6)
        np.testing.assert_allclose(out["next_observations"][0], 2.0 * np.ones(OBS_DIM))

    def test_terminal_first_step(self):
        out = repack_samples(_traj_sample([5.0, 99.0, 99.0], [1, 0, 0], [0, 0, 0]), GAMMA, N)
        np.testing.assert_allclose(out["rewards"][0], 5.0, rtol=1e-6)
        assert out["bootstrap"][0] == 0.0

    def test_truncation_first_step(self):
        # window is pure bootstrap-at-o_0: no rewards, coeff gamma^0 = 1
        out = repack_samples(_traj_sample([5.0, 99.0, 99.0], [0, 0, 0], [1, 0, 0]), GAMMA, N)
        np.testing.assert_allclose(out["rewards"][0], 0.0, atol=1e-7)
        np.testing.assert_allclose(out["bootstrap"][0], 1.0, rtol=1e-6)
        np.testing.assert_allclose(out["next_observations"][0], 1.0 * np.ones(OBS_DIM))

    def test_terminal_wins_over_truncation(self):
        # both flags on one step -> treated as terminal (reward counts, no boot)
        out = repack_samples(_traj_sample([1.0, 2.0, 99.0], [0, 1, 0], [0, 1, 0]), GAMMA, N)
        np.testing.assert_allclose(out["rewards"][0], 1.0 + GAMMA * 2.0, rtol=1e-6)
        assert out["bootstrap"][0] == 0.0

    def test_n1_matches_flat_pair_path(self):
        # same underlying transition through both layouts -> identical dicts
        obs0 = jnp.full((1, OBS_DIM), 1.0)
        obs1 = jnp.full((1, OBS_DIM), 2.0)
        for term in (False, True):
            traj = _FakeSample(experience=Transition(
                observation=jnp.stack([obs0, obs1], axis=1),
                action=jnp.zeros((1, 2, 2)),
                reward=jnp.array([[3.0, 0.0]]),
                terminal=jnp.array([[term, False]]),
                truncation=jnp.array([[False, False]]),
            ))
            first = Transition(observation=obs0, action=jnp.zeros((1, 2)),
                               reward=jnp.array([3.0]), terminal=jnp.array([term]))
            second = Transition(observation=obs1, action=jnp.zeros((1, 2)),
                                reward=jnp.array([0.0]), terminal=jnp.array([False]))

            @struct.dataclass
            class _Pair:
                first: Transition
                second: Transition

            @struct.dataclass
            class _PairSample:
                experience: _Pair

            a = repack_samples(traj, GAMMA, 1)
            b = repack_samples(_PairSample(experience=_Pair(first, second)), GAMMA, 1)
            for k in ("observations", "rewards", "bootstrap"):
                np.testing.assert_allclose(a[k], b[k], rtol=1e-6, err_msg=f"{k} term={term}")
            # next_observations only matters where the bootstrap coefficient is
            # nonzero (at terminals the traj path zeroes it, the pair path
            # carries the unused next obs — both multiplied by 0 in the target).
            if not term:
                np.testing.assert_allclose(
                    a["next_observations"], b["next_observations"], rtol=1e-6,
                )


class TestTrajectoryBufferEndToEnd:
    def test_buffer_windows_respect_layout(self):
        """Push a known stream through a real flashbax trajectory buffer and
        verify sampled windows are consecutive items from one env row."""
        buf = flashbax.make_trajectory_buffer(
            add_batch_size=2, sample_batch_size=32, sample_sequence_length=N + 1,
            period=1, min_length_time_axis=N + 1, max_length_time_axis=64,
        )
        proto = Transition(
            observation=jnp.zeros(1, dtype=jnp.float32),
            action=jnp.zeros(1, dtype=jnp.float32),
            reward=jnp.zeros((), dtype=jnp.float32),
            terminal=jnp.zeros((), dtype=jnp.bool_),
            truncation=jnp.zeros((), dtype=jnp.bool_),
        )
        state = buf.init(proto)
        # obs value encodes (env_row * 1000 + t) so windows are checkable
        for t in range(20):
            item = Transition(
                observation=jnp.array([[1000.0 + t], [2000.0 + t]])[:, None, :],
                action=jnp.zeros((2, 1, 1)),
                reward=jnp.full((2, 1), float(t)),
                terminal=jnp.zeros((2, 1), dtype=bool),
                truncation=jnp.zeros((2, 1), dtype=bool),
            )
            state = buf.add(state, item)
        sample = buf.sample(state, jax.random.PRNGKey(0))
        seq = np.asarray(sample.experience.observation[..., 0])  # (B, N+1)
        diffs = np.diff(seq, axis=1)
        assert np.all(diffs == 1.0), "windows must be consecutive within one env row"

    def test_td3_update_smoke(self):
        """Full TD3 update() with n_step=3 on a trajectory buffer: shapes, jit,
        finiteness. Tiny nets, CPU."""
        from omegaconf import OmegaConf

        from roxie.agents.td3 import TD3

        obs_dim, act_dim, n_envs = 6, 3, 4
        agent = TD3(
            env_obs_size=obs_dim,
            env_action_size=act_dim,
            action_low=-jnp.ones(act_dim),
            action_high=jnp.ones(act_dim),
            actor_config=OmegaConf.create({
                "_target_": "roxie.models.actors.DeterministicActor",
                "features": [16], "use_layer_norm": True, "dropout_rate": 0.0,
            }),
            critic_config=OmegaConf.create({
                "_target_": "roxie.models.critics.QCritic",
                "features": [16], "use_layer_norm": True, "dropout_rate": 0.0,
            }),
            memory_config=OmegaConf.create({
                "_target_": "flashbax.buffers.make_flat_buffer",
                "max_length": 512, "min_length": 8, "sample_batch_size": 16,
                "add_sequences": False, "add_batch_size": n_envs,
            }),
            noise_config=OmegaConf.create({
                "_target_": "roxie.exploration.noisy.OrnsteinUhlenbeckNoise",
                "initial_noise_scale": 0.1, "theta": 2.0, "dt": 0.025,
                "clip": 2.0, "mu": 0.0,
            }),
            steps_before_learning=0,
            steps_between_updates=1,
            learning_steps=2,
            memory_warmup=0,
            n_step=N,
        )
        assert agent.n_step == N

        key = jax.random.PRNGKey(0)
        for t in range(40):
            key, k1, k2 = jax.random.split(key, 3)
            tr = Transition(
                observation=jax.random.normal(k1, (n_envs, obs_dim)),
                action=jax.random.normal(k2, (n_envs, act_dim)),
                reward=jnp.ones(n_envs),
                terminal=jnp.zeros(n_envs, dtype=bool).at[0].set(t % 13 == 0),
                truncation=jnp.zeros(n_envs, dtype=bool).at[1].set(t % 17 == 0),
            )
            agent.state.buffer_state = agent.replay_add(agent.state.buffer_state, tr)

        grad_steps, actor_loss, critic_loss = agent.update(
            steps=0, agent_rng=jax.random.PRNGKey(1),
        )
        assert grad_steps == 2
        assert jnp.isfinite(actor_loss) and jnp.isfinite(critic_loss)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
