"""Twin Delayed Deep Deterministic Policy Gradient (TD3) agent."""

import functools

import jax
import jax.numpy as jnp
from flax import nnx

from roxie.agents.ddpg import DDPG
from roxie.agents.agent import LearningOutput, TrainState
from roxie.agents.hyperparams import TD3Hyperparams
from roxie.agents.utils import (
    fused_grad_steps,
    reduce_diagnostics,
    repack_samples,
    soft_update,
)
from roxie.losses.actor_losses import skipped_actor_aux, td3_actor_loss_fn
from roxie.losses.critic_losses import td3_critic_loss_fn
from roxie.utils.math import normalize_samples, obs_mean_std


def _td3_grad_step(
    state: TrainState,
    key: jax.Array,
    update_actor,
    *,
    hp: TD3Hyperparams,
    replay_sample_fn,
    action_low: float,
    action_high: float,
    obs_mean: jnp.ndarray,
    obs_std: jnp.ndarray,
):
    """Executes a single TD3 gradient step, conditionally updating policy and target networks."""
    key, noise_key = jax.random.split(key)
    samples = replay_sample_fn(state.buffer_state, key)
    re_packed_samples = repack_samples(samples, hp.gamma, hp.n_step)
    re_packed_samples = normalize_samples(
        re_packed_samples,
        obs_mean,
        obs_std,
        hp.obs_norm_clip,
        hp.normalize_observations,
    )

    (critic_loss, critic_aux), critic_grads = nnx.value_and_grad(
        td3_critic_loss_fn, has_aux=True
    )(
        state.critic,
        state.target_actor,
        state.target_critic,
        re_packed_samples,
        noise_key,
        hp.target_policy_noise,
        hp.target_noise_clip,
        action_low,
        action_high,
        hp.twin_q_weight,
    )
    state.critic_optimizer.update(state.critic, critic_grads)

    def _actor_update(state):
        (actor_loss, actor_aux), actor_grads = nnx.value_and_grad(
            td3_actor_loss_fn, has_aux=True
        )(
            state.actor,
            state.critic,
            re_packed_samples,
            action_low,
            action_high,
            hp.pre_activation_coef,
        )
        state.actor_optimizer.update(state.actor, actor_grads)

        soft_update(state.target_actor, state.actor, hp.tau)
        soft_update(state.target_critic, state.critic, hp.tau)
        return actor_loss, actor_aux

    def _skip_actor_update(state):
        zero = jnp.array(0.0, dtype=critic_loss.dtype)
        return zero, skipped_actor_aux(critic_loss.dtype, "pre_act_penalty")

    actor_loss, actor_aux = nnx.cond(
        update_actor, _actor_update, _skip_actor_update, state
    )

    return actor_loss, critic_loss, actor_aux, critic_aux


@functools.partial(
    jax.jit,
    static_argnames=("hp", "replay_sample_fn", "n_steps"),
    donate_argnums=(0,),
)
def _td3_grad_steps(
    state: TrainState,
    key: jax.Array,
    n_steps: int,
    hp: TD3Hyperparams,
    replay_sample_fn,
    action_low: float,
    action_high: float,
):
    """Runs one learning pass of TD3 gradient steps via `jax.lax.scan`."""
    obs_mean, obs_std = obs_mean_std(state.obs_stats, hp.obs_norm_eps)

    update_mask = (jnp.arange(n_steps, dtype=jnp.int32) % hp.policy_delay) == 0

    state, (actor_losses, critic_losses, actor_aux, critic_aux) = fused_grad_steps(
        state,
        key,
        n_steps,
        functools.partial(
            _td3_grad_step,
            hp=hp,
            replay_sample_fn=replay_sample_fn,
            action_low=action_low,
            action_high=action_high,
            obs_mean=obs_mean,
            obs_std=obs_std,
        ),
        extras=(update_mask,),
    )

    n_actor_updates = jnp.maximum(jnp.sum(update_mask), 1)
    return LearningOutput(
        state=state,
        actor_loss=jnp.sum(actor_losses) / n_actor_updates,
        critic_loss=jnp.mean(critic_losses),
        diagnostics={
            **reduce_diagnostics(actor_aux, n_actor_updates),
            **reduce_diagnostics(critic_aux, n_steps),
        },
    )


class TD3(DDPG):
    """Twin Delayed Deep Deterministic Policy Gradient agent.

    Reference:
        Fujimoto et al., 2018: https://arxiv.org/abs/1802.09477
    """

    hyperparams_cls = TD3Hyperparams

    def _compile_and_run(self, agent_rng: jax.Array, target_steps: int):
        """Runs one fused learning pass to update the agent."""
        return target_steps, _td3_grad_steps(
            self.state,
            agent_rng,
            target_steps,
            self.hp,
            self.replay.sample,
            self.action_low,
            self.action_high,
        )