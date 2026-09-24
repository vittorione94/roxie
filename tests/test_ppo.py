"""What PPO adds that no other agent has.

The shared contract — that a learning pass trains, that the diagnostics arrive
namespaced and finite, that acting is bounded — is `test_agents.py`'s, and PPO
goes through it with the other six. What is here needs machinery no other agent
needs: a reference implementation of the update to hold the fused one against, a
deliberately drifting observation distribution, and an actor forced past the
tanh rail.

The agent itself still comes from the shipped `ppo.yaml` through
`tests/harness.py`. That matters more here than anywhere else: PPO's log-ratio
divides by sigma, so `std_min` sets how fast `approx_kl` can move, and the
config states it (1e-2) rather than taking `StochasticActor`'s default (1e-4).
A hand-written actor block would run every trust-region test below in a regime
a hundred times more sensitive than the one that ships.
"""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from roxie.agents.ppo import _ppo_grad_step, _ppo_grad_steps, _prepare_rollout
from tests import harness
from tests.harness import ENVS


def _rollout_len(agent) -> int:
    """One rollout, in env steps per env: PPO's update window IS its queue's
    width, so a rollout that over- or under-filled would leave the queue in a
    state the next one inherits."""
    return int(agent.sample_sequence_length)


def _collect_and_update(agent, key, drift=2.0, unfreeze_norm=False):
    """One rollout into the queue, then the learning pass that drains it."""
    harness.drive(agent, _rollout_len(agent), key, drift=drift)

    if unfreeze_norm:
        # Throw away BOTH caches so the update re-derives mean/std from the
        # drifted running stats — the failure mode the freeze prevents.
        # `_obs_norm` alone would re-derive the same numbers.
        agent._obs_stats_snapshot = None
        agent._obs_norm = None

    # The trainer's `steps` is cumulative, and the gate CONSUMES the boundary it
    # fires on, so a second rollout on the same agent must be asked at its own
    # step count rather than at the first one's again.
    steps = max(agent._last_update_boundary, 0) + _rollout_len(agent) * ENVS
    assert agent.due_for_update(steps), "the queue never filled"
    agent.learn(jax.random.fold_in(key, 99))
    return agent.pop_diagnostics()


def _agent(**hyperparams):
    """A PPO whose single reported `approx_kl` IS the first pass's — measured at
    the behaviour parameters themselves — unless a test asks otherwise."""
    return harness.build(
        "ppo", **{"learning_steps": 1, "num_minibatches": 1, "target_kl": None,
                  **hyperparams}
    )


class TestObsNormFreeze:
    def test_first_pass_ratio_is_one_despite_stat_drift(self):
        """The behaviour log-probs must be reproducible at the same parameters.

        `norm_obs` is rebuilt at update time while the running obs statistics
        moved throughout the rollout. Unless the normalizer is pinned to the
        snapshot that `step()` acted under, re-evaluating the rollout gives a
        ratio != 1 before any gradient step, so the clip and the KL early stop
        trip on normalization drift rather than policy drift.
        """
        diag = _collect_and_update(_agent(), jax.random.PRNGKey(0))
        assert diag["ppo/approx_kl"] == pytest.approx(0.0, abs=1e-5)
        assert diag["ppo/clip_frac"] == pytest.approx(0.0, abs=1e-6)

    def test_drifting_stats_would_break_the_ratio(self):
        """Guards the test above against silently passing for the wrong reason:
        with the snapshot thrown away, the same rollout must NOT score 0."""
        diag = _collect_and_update(
            _agent(), jax.random.PRNGKey(0), unfreeze_norm=True
        )
        assert diag["ppo/approx_kl"] > 1e-3

    def test_snapshot_refreshes_between_rollouts(self):
        """Pinned per rollout, not once for the run — the next rollout acts
        under the statistics it actually accumulated."""
        agent = _agent()
        _collect_and_update(agent, jax.random.PRNGKey(0))
        first = agent._frozen_obs_norm()
        # More data at a different scale, then another rollout boundary.
        _collect_and_update(agent, jax.random.PRNGKey(1), drift=5.0)
        second = agent._frozen_obs_norm()
        assert not jnp.allclose(first[0], second[0])

    def test_unnormalized_agent_leaves_obs_untouched(self):
        """With normalization off the stats stay empty; obs must not be rescaled.

        Normalizing unconditionally divides by a zero-variance std and pins
        every feature to the clip bound.
        """
        agent = _agent(normalize_observations=False)
        diag = _collect_and_update(agent, jax.random.PRNGKey(0))
        assert float(agent.state.obs_stats.count) == 0.0
        assert diag["ppo/approx_kl"] == pytest.approx(0.0, abs=1e-5)


# The fused learning pass must be the Python loop, only faster.


def _prepared_rollout(agent, key):
    """Drive one rollout into the queue, then dequeue the tensors that both
    update paths get fed. Going through the real `_prepare_rollout` rather than
    synthesizing arrays is what keeps the flattening honest: it slices the
    policy tensors to the GAE horizon and folds `(env, time)` into one
    transition axis, and a hand-built `(B, T)` block would not catch that
    reverting."""
    harness.drive(agent, _rollout_len(agent), key)

    obs_mean, obs_std = agent._frozen_obs_norm()
    # `_prepare_rollout` returns the state it was handed, so the agent re-adopts
    # it — the same contract every pass has.
    agent.state, agent.adv_scale, *tensors, _rollout_stats = _prepare_rollout(
        agent.state,
        agent.adv_scale,
        hp=agent.hp,
        replay_get_fn=agent.replay.sample,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )
    return tuple(tensors)


class TestRolloutIsFlattenedToTransitions:
    """A minibatch has to be a draw from the rollout's TRANSITIONS.

    Permuting the env axis alone hands every minibatch whole trajectories, so a
    gradient step sees `minibatch_size / T` distinct start states instead of
    `minibatch_size`, and every step inside one of them shares a policy and a
    reset. The flattening happens in `_prepare_rollout` and its only visible
    trace is the shape, which is why that is what these assert on.
    """

    def test_prepared_tensors_are_flat_over_env_and_time(self):
        agent = _agent()
        norm_obs, pre_actions, old_log_probs, returns_t, adv_t = _prepared_rollout(
            agent, jax.random.PRNGKey(0)
        )
        # T - 1: GAE spends the last step of every trajectory on the bootstrap.
        n = ENVS * (agent.sample_sequence_length - 1)
        assert agent.rollout_transitions == n
        assert norm_obs.shape == (n, norm_obs.shape[-1])
        assert pre_actions.shape == (n, pre_actions.shape[-1])
        for leaf in (old_log_probs, returns_t, adv_t):
            assert leaf.shape == (n,)

    def test_advantages_are_standardized_when_the_ema_is_off(self):
        """`adv_norm_decay: 0.0` is the reference's `(adv - adv.mean()) /
        (adv.std() + 1e-8)`, which it can only be if the running accumulator
        squares the CENTERED advantage: uncentered it holds the RMS, and the two
        part company by exactly the batch mean that GAE does not pin to zero.
        """
        agent = _agent(adv_norm_decay=0.0)
        *_, adv_t = _prepared_rollout(agent, jax.random.PRNGKey(0))
        assert float(jnp.mean(adv_t)) == pytest.approx(0.0, abs=1e-4)
        assert float(jnp.std(adv_t)) == pytest.approx(1.0, abs=1e-3)


def _looped_update(state, key, agent, tensors):
    """The reference `_ppo_grad_steps` is held against: nested loops, a host `float`
    on `approx_kl`, and a `break` that abandons the rest of the rollout."""
    mb = agent.minibatch_size
    steps = stops = 0
    sums = dict(actor=0.0, critic=0.0, kl=0.0, clip=0.0)
    stop_epochs = False
    for _ in range(agent.hp.learning_steps):
        key, perm_key = jax.random.split(key)
        perm = jax.random.permutation(perm_key, agent.rollout_transitions)
        shuffled = tuple(leaf[perm] for leaf in tensors)

        for m in range(agent.hp.num_minibatches):
            key, step_key = jax.random.split(key)
            sl = slice(m * mb, (m + 1) * mb)
            actor_loss, critic_loss, aux = _ppo_grad_step(
                state, step_key, *[leaf[sl] for leaf in shuffled], hp=agent.hp,
            )
            approx_kl, clip_frac = aux["approx_kl"], aux["clip_frac"]
            steps += 1
            sums["actor"] += float(actor_loss)
            sums["critic"] += float(critic_loss)
            sums["kl"] += float(approx_kl)
            sums["clip"] += float(clip_frac)

            if (
                agent.hp.target_kl is not None
                and float(approx_kl) > agent.hp.target_kl
            ):
                stops += 1
                stop_epochs = True
                break
        if stop_epochs:
            break
    return steps, stops, sums


def _params(module):
    return jax.tree.leaves(nnx.state(module, nnx.Param))


class TestFusedUpdateMatchesTheLoop:
    """`_ppo_grad_steps` moved the `target_kl` early stop from a host `break` into a
    sticky flag in the scan carry. That is only a throughput change if it lands
    exactly where the loop landed, so these run one rollout through both."""

    # `None` exercises the no-stop path; the tiny bound trips MID-rollout
    # (`approx_kl` is identically 0 on the first pass). A bound that tripped
    # immediately, or never, would not cover the sticky flag — which is why the
    # tripping case asserts where it stopped, not just that it agreed.
    @pytest.mark.parametrize("target_kl", [None, 1e-9])
    def test_fused_update_matches_the_per_minibatch_loop(self, target_kl):
        agent = _agent(learning_steps=3, num_minibatches=2, target_kl=target_kl)
        tensors = _prepared_rollout(agent, jax.random.PRNGKey(0))

        learn_key = jax.random.PRNGKey(7)
        fused_state = copy.deepcopy(agent.state)
        looped_state = copy.deepcopy(agent.state)

        steps, output = _ppo_grad_steps(
            fused_state, learn_key, *tensors, agent.hp, agent.minibatch_size,
        )
        ref_steps, ref_stops, ref_sums = _looped_update(
            looped_state, learn_key, agent, tensors,
        )

        steps = int(steps)
        assert steps == ref_steps, "different number of passes ran"
        assert int(output.diagnostics["kl_early_stops"]) == ref_stops
        if target_kl is not None:
            budget = agent.hp.learning_steps * agent.hp.num_minibatches
            assert 0 < steps < budget, (
                f"{steps} passes ran: the bound did not stop mid-rollout"
            )
        # The pass reports per-minibatch averages; the loop accumulates sums.
        for name, got in (("actor", output.actor_loss),
                          ("critic", output.critic_loss),
                          ("kl", output.diagnostics["approx_kl"]),
                          ("clip", output.diagnostics["clip_frac"])):
            np.testing.assert_allclose(
                float(got) * steps, ref_sums[name], rtol=1e-5, atol=1e-5,
                err_msg=f"{name} sum diverged between the two paths",
            )

        # The parameters are the real assertion: the diagnostics could agree
        # while the updates landed differently.
        for network in ("actor", "critic"):
            for fused, looped in zip(_params(getattr(output.state, network)),
                                     _params(getattr(looped_state, network))):
                np.testing.assert_allclose(
                    fused, looped, rtol=1e-5, atol=1e-5,
                    err_msg=f"{network} params diverged",
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
        """Force the policy past the arctanh rail, where the old path lost `u`.

        Only the mean: PPO's sigma is a free `log_std` parameter, so this leaves
        the spread the shipped `init_std` / `std_min` / `std_max` produce.
        """
        layer = agent.state.actor.output_layer
        layer.kernel[...] = jnp.zeros_like(layer.kernel[...])
        layer.bias[...] = jnp.full_like(layer.bias[...], mean)

    def test_saturated_policy_does_not_early_stop_every_rollout(self):
        agent = _agent(learning_steps=4, num_minibatches=2, target_kl=0.03)
        self._saturate(agent)
        diag = _collect_and_update(agent, jax.random.PRNGKey(0), drift=0.0)

        assert jnp.isfinite(diag["ppo/approx_kl"])
        # The whole budget, not the one step a tripped trust region allows.
        assert diag["ppo/steps_per_rollout"] == 8
        assert diag["ppo/kl_early_stops"] == 0

    def test_stored_pre_actions_reproduce_the_behaviour_log_probs(self):
        """The ratio is 1 on the first pass only if the stored `u` is the `u`
        the behaviour policy actually drew from."""
        agent = _agent()
        self._saturate(agent)
        _norm_obs, pre_actions, old_log_probs, _returns, _adv = _prepared_rollout(
            agent, jax.random.PRNGKey(0)
        )
        # Never round-tripped through tanh: past the rail that is lossy.
        assert float(jnp.min(jnp.abs(pre_actions))) > 8.0
        assert jnp.isfinite(old_log_probs).all()
