import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from omegaconf import OmegaConf

import roxie.agents  # noqa: F401  (avoid circular import)
from roxie.agents.ppo import PPO, _grad_step, _grad_steps, _prepare_rollout
from roxie.agents.utils import Transition
from roxie.environment.vector import Timestep


def _buffer(agent, prev_obs, timestep):
    """What `SyncLearner.buffer` does: assemble the batch the agent stores."""
    agent.buffer_transitions(
        Transition(
            observation=prev_obs,
            action=agent.last_action,
            reward=timestep.reward,
            terminal=timestep.terminated,
            truncation=timestep.truncated,
            **(agent.last_extras or {}),
        ),
        timestep.obs,
    )


NUM_ENVS = 4
OBS_DIM = 6
ACT_DIM = 2
ROLLOUT = 8


def _make_agent(**kwargs):
    actor_config = OmegaConf.create(
        {"_target_": "roxie.models.actors.StochasticActor", "features": [16, 16]}
    )
    critic_config = OmegaConf.create(
        {"_target_": "roxie.models.critics.VCritic", "features": [16, 16]}
    )
    memory_config = OmegaConf.create(
        {
            "_target_": "flashbax.buffers.make_trajectory_queue",
            "max_length_time_axis": 64,
            "add_batch_size": NUM_ENVS,
            "add_sequence_length": 1,
            "sample_sequence_length": ROLLOUT,
        }
    )
    params = dict(
        env_obs_size=OBS_DIM,
        env_action_size=ACT_DIM,
        action_low=jnp.full((ACT_DIM,), -1.0),
        action_high=jnp.full((ACT_DIM,), 1.0),
        actor_config=actor_config,
        critic_config=critic_config,
        memory_config=memory_config,
        # One pass over one minibatch, so the single reported approx_kl IS the
        # first pass's -- measured at the behaviour parameters themselves.
        learning_steps=1,
        num_minibatches=1,
        target_kl=None,
    )
    params.update(kwargs)
    return PPO(**params)


def _timestep(key, scale):
    """One batched env step, as `JaxVectorEnv.step` would report it."""
    false = jnp.zeros((NUM_ENVS,), jnp.bool_)
    return Timestep(
        obs=jax.random.normal(key, (NUM_ENVS, OBS_DIM)) * scale + scale,
        reward=jax.random.normal(key, (NUM_ENVS,)),
        terminated=false,
        truncated=false,
        info={},
    )


def _collect_and_update(agent, drift=2.0, unfreeze_norm=False):
    """Roll out under a deliberately drifting observation distribution."""
    key = jax.random.PRNGKey(0)
    key, k0 = jax.random.split(key)
    obs = _timestep(k0, 1.0).obs

    for t in range(ROLLOUT):
        key, act_key, step_key = jax.random.split(key, 3)
        agent.step(obs, evaluate=False, key=act_key)
        timestep = _timestep(step_key, 1.0 + drift * t)
        _buffer(agent, obs, timestep)
        obs = timestep.obs

    if unfreeze_norm:
        # Throw away BOTH caches so the update re-derives mean/std from the
        # drifted running stats — the failure mode the freeze prevents.
        # `_obs_norm` alone would re-derive the same numbers.
        agent._obs_stats_snapshot = None
        agent._obs_norm = None

    key, update_key = jax.random.split(key)
    agent.update(steps=ROLLOUT * NUM_ENVS, agent_rng=update_key)
    return agent.pop_diagnostics()


class TestPPOObsNormFreeze:
    def test_first_pass_ratio_is_one_despite_stat_drift(self):
        """The behaviour log-probs must be reproducible at the same parameters.

        `norm_obs` is rebuilt at update time while the running obs statistics
        moved throughout the rollout. Unless the normalizer is pinned to the
        snapshot that `step()` acted under, re-evaluating the rollout gives a
        ratio != 1 before any gradient step, so the clip and the KL early stop
        trip on normalization drift rather than policy drift.
        """
        diag = _collect_and_update(_make_agent(normalize_observations=True))
        assert diag["ppo/approx_kl"] == pytest.approx(0.0, abs=1e-5)
        assert diag["ppo/clip_frac"] == pytest.approx(0.0, abs=1e-6)

    def test_drifting_stats_would_break_the_ratio(self):
        """Guards the test above against silently passing for the wrong reason."""
        diag = _collect_and_update(
            _make_agent(normalize_observations=True), unfreeze_norm=True
        )
        assert diag["ppo/approx_kl"] > 1e-3

    def test_snapshot_refreshes_between_rollouts(self):
        agent = _make_agent(normalize_observations=True)
        _collect_and_update(agent)
        first = agent._frozen_obs_norm()
        # More data at a different scale, then another rollout boundary.
        _collect_and_update(agent, drift=5.0)
        second = agent._frozen_obs_norm()
        assert not jnp.allclose(first[0], second[0])

    def test_unnormalized_agent_leaves_obs_untouched(self):
        """With normalization off the stats stay empty; obs must not be rescaled.

        Normalizing unconditionally divides by a zero-variance std and pins
        every feature to the clip bound.
        """
        agent = _make_agent(normalize_observations=False)
        diag = _collect_and_update(agent)
        assert float(agent.state.obs_stats.count) == 0.0
        assert diag["ppo/approx_kl"] == pytest.approx(0.0, abs=1e-5)


# The fused update burst must be the Python loop, only faster.


def _prepared_rollout(agent, key):
    """Drive one rollout into the queue, then dequeue the tensors that both
    update paths get fed. Going through the real `_prepare_rollout` rather than
    synthesizing arrays is what keeps the time axes (T for the policy tensors,
    T-1 for the GAE ones) honest."""
    key, k0 = jax.random.split(key)
    obs = _timestep(k0, 1.0).obs
    for _ in range(ROLLOUT):
        key, act_key, step_key = jax.random.split(key, 3)
        agent.step(obs, evaluate=False, key=act_key)
        timestep = _timestep(step_key, 1.0)
        _buffer(agent, obs, timestep)
        obs = timestep.obs

    obs_mean, obs_std = agent._frozen_obs_norm()
    # `_prepare_rollout` returns the state it was handed, so the agent re-adopts
    # it — the same contract every burst has.
    agent.state, *tensors = _prepare_rollout(
        agent.state,
        gamma=agent.gamma,
        gae_lambda=agent.gae_lambda,
        replay_get_fn=agent.replay.sample,
        obs_clip=agent.obs_clip,
        normalize=agent.normalize_observations,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )
    return tuple(tensors)


def _looped_update(state, key, agent, tensors):
    """The reference `_grad_steps` is held against: nested loops, a host `float`
    on `approx_kl`, and a `break` that abandons the rest of the rollout."""
    mb = agent.minibatch_size
    steps = stops = 0
    sums = dict(actor=0.0, critic=0.0, kl=0.0, clip=0.0)
    stop_epochs = False
    for _ in range(agent.learning_steps):
        key, perm_key = jax.random.split(key)
        perm = jax.random.permutation(perm_key, agent.batch_size)
        shuffled = tuple(leaf[perm] for leaf in tensors)

        for m in range(agent.num_minibatches):
            key, step_key = jax.random.split(key)
            sl = slice(m * mb, (m + 1) * mb)
            actor_loss, critic_loss, approx_kl, clip_frac = _grad_step(
                state, step_key, *[leaf[sl] for leaf in shuffled],
                clip_eps=agent.clip_eps,
                entropy_coef=agent.entropy_coef,
            )
            steps += 1
            sums["actor"] += float(actor_loss)
            sums["critic"] += float(critic_loss)
            sums["kl"] += float(approx_kl)
            sums["clip"] += float(clip_frac)

            if agent.target_kl is not None and float(approx_kl) > agent.target_kl:
                stops += 1
                stop_epochs = True
                break
        if stop_epochs:
            break
    return steps, stops, sums


def _params(module):
    return jax.tree.leaves(nnx.state(module, nnx.Param))


class TestFusedUpdateMatchesTheLoop:
    """`_grad_steps` moved the `target_kl` early stop from a host `break` into a
    sticky flag in the scan carry. That is only a throughput change if it lands
    exactly where the loop landed, so these run one rollout through both."""

    # `None` exercises the no-stop path; the tiny bound trips MID-rollout, on
    # the second of six steps (`approx_kl` is identically 0 on the first). A
    # bound that tripped immediately, or never, would not cover the sticky flag.
    @pytest.mark.parametrize("target_kl", [None, 1e-9])
    def test_fused_update_matches_the_per_minibatch_loop(self, target_kl):
        agent = _make_agent(
            learning_steps=3, num_minibatches=2, target_kl=target_kl,
        )
        tensors = _prepared_rollout(agent, jax.random.PRNGKey(0))

        burst_key = jax.random.PRNGKey(7)
        fused_state = copy.deepcopy(agent.state)
        looped_state = copy.deepcopy(agent.state)

        (
            new_state, _key, steps, actor_sum, critic_sum,
            kl_sum, clip_sum, stops, _last_kl, _last_clip,
        ) = _grad_steps(
            fused_state, burst_key, *tensors,
            agent.learning_steps, agent.num_minibatches, agent.minibatch_size,
            agent.clip_eps, agent.entropy_coef, agent.target_kl,
        )
        ref_steps, ref_stops, ref_sums = _looped_update(
            looped_state, burst_key, agent, tensors,
        )

        assert int(steps) == ref_steps, "different number of passes ran"
        assert int(stops) == ref_stops
        for name, got in (("actor", actor_sum), ("critic", critic_sum),
                          ("kl", kl_sum), ("clip", clip_sum)):
            np.testing.assert_allclose(
                float(got), ref_sums[name], rtol=1e-5, atol=1e-5,
                err_msg=f"{name} sum diverged between the two paths",
            )

        # The parameters are the real assertion: the diagnostics could agree
        # while the updates landed differently.
        for network in ("actor", "critic"):
            for fused, looped in zip(_params(getattr(new_state, network)),
                                     _params(getattr(looped_state, network))):
                np.testing.assert_allclose(
                    fused, looped, rtol=1e-5, atol=1e-5,
                    err_msg=f"{network} params diverged",
                )

    def test_the_tiny_bound_really_stops_mid_rollout(self):
        """Guards the parametrization above: if the bound never tripped, the
        `target_kl` case would just be re-testing the no-stop path."""
        agent = _make_agent(
            learning_steps=3, num_minibatches=2, target_kl=1e-9,
        )
        tensors = _prepared_rollout(agent, jax.random.PRNGKey(0))
        _state, _key, steps, *_rest = _grad_steps(
            copy.deepcopy(agent.state),
            jax.random.PRNGKey(7), *tensors,
            agent.learning_steps, agent.num_minibatches, agent.minibatch_size,
            agent.clip_eps, agent.entropy_coef, agent.target_kl,
        )
        assert 0 < int(steps) < agent.learning_steps * agent.num_minibatches, (
            f"{int(steps)} passes ran: the bound did not stop mid-rollout"
        )


class TestSaturatedRolloutStillTrains:
    """`tests/test_losses.py::TestRatioSurvivesSaturation` pins the loss-level
    property; this pins the PLUMBING that feeds it.

    The pre-tanh draw only reaches the loss if acting returns it, `extras`
    carries it, the queue has a slot allocated for it, and `_prepare_rollout`
    hands it back in the right position. Any of those silently reverting puts
    `approx_kl` back at the rail, and the visible symptom is not a crash but a
    trust region that abandons every rollout after its first minibatch -- which
    is what the release runs did for their last 493M steps.
    """

    @staticmethod
    def _saturate(agent, mean=20.0):
        """Force the policy past the arctanh rail, where the old path lost `u`."""
        layer = agent.state.actor.output_layer
        layer.kernel[...] = jnp.zeros_like(layer.kernel[...])
        layer.bias[...] = jnp.full_like(layer.bias[...], mean)

    def test_saturated_policy_does_not_early_stop_every_rollout(self):
        agent = _make_agent(learning_steps=4, num_minibatches=2, target_kl=0.03)
        self._saturate(agent)
        diag = _collect_and_update(agent, drift=0.0)

        assert jnp.isfinite(diag["ppo/approx_kl"])
        # The whole budget, not the one step a tripped trust region allows.
        assert diag["ppo/steps_per_rollout"] == 8
        assert diag["ppo/kl_early_stops"] == 0

    def test_stored_pre_actions_reproduce_the_behaviour_log_probs(self):
        """The ratio is 1 on the first pass only if the stored `u` is the `u`
        the behaviour policy actually drew from."""
        agent = _make_agent()
        self._saturate(agent)
        _norm_obs, pre_actions, old_log_probs, _returns, _adv = _prepared_rollout(
            agent, jax.random.PRNGKey(0)
        )
        # Never round-tripped through tanh: past the rail that is lossy.
        assert float(jnp.min(jnp.abs(pre_actions))) > 8.0
        assert jnp.isfinite(old_log_probs).all()
