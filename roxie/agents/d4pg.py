"""Distributed Distributional Deep Deterministic Policy Gradient (D4PG) agent."""

import functools

import jax
import jax.numpy as jnp
from flax import nnx

from roxie.agents.ddpg import DDPG
from roxie.agents.agent import LearningOutput, TrainState
from roxie.agents.hyperparams import DDPGHyperparams
from roxie.agents.utils import (
    fused_grad_steps,
    reduce_diagnostics,
    repack_samples,
    soft_update,
)
from roxie.losses.actor_losses import d4pg_actor_loss_fn
from roxie.losses.critic_losses import d4pg_critic_loss_fn
from roxie.utils.math import normalize_samples, obs_mean_std


def _d4pg_grad_step(
    state: TrainState,
    key: jax.Array,
    *,
    hp: DDPGHyperparams,
    replay_sample_fn,
    action_low: float,
    action_high: float,
    obs_mean: jnp.ndarray,
    obs_std: jnp.ndarray,
):
    """Executes a single D4PG gradient step updating actor and categorical critic networks."""
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
        d4pg_critic_loss_fn, has_aux=True
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
    )
    state.critic_optimizer.update(state.critic, critic_grads)

    (actor_loss, actor_aux), actor_grads = nnx.value_and_grad(
        d4pg_actor_loss_fn, has_aux=True
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

    return actor_loss, critic_loss, actor_aux, critic_aux


@functools.partial(
    jax.jit,
    static_argnames=("hp", "replay_sample_fn", "n_steps"),
    donate_argnums=(0,),
)
def _d4pg_grad_steps(
    state: TrainState,
    key: jax.Array,
    n_steps: int,
    hp: DDPGHyperparams,
    replay_sample_fn,
    action_low: float,
    action_high: float,
):
    """Runs one learning pass of D4PG gradient steps via `jax.lax.scan`."""
    obs_mean, obs_std = obs_mean_std(state.obs_stats, hp.obs_norm_eps)

    state, (actor_losses, critic_losses, actor_aux, critic_aux) = fused_grad_steps(
        state,
        key,
        n_steps,
        functools.partial(
            _d4pg_grad_step,
            hp=hp,
            replay_sample_fn=replay_sample_fn,
            action_low=action_low,
            action_high=action_high,
            obs_mean=obs_mean,
            obs_std=obs_std,
        ),
    )

    return LearningOutput(
        state=state,
        actor_loss=jnp.mean(actor_losses),
        critic_loss=jnp.mean(critic_losses),
        diagnostics={
            **reduce_diagnostics(actor_aux, n_steps),
            **reduce_diagnostics(critic_aux, n_steps),
        },
    )


class D4PG(DDPG):
    """Distributed Distributional Deep Deterministic Policy Gradient agent.

    Reference:
        Barth-Maron et al., 2018: https://arxiv.org/abs/1804.08617
    """

    def _compile_and_run(self, agent_rng: jax.Array, target_steps: int):
        """Runs one fused learning pass to update the agent."""
        return target_steps, _d4pg_grad_steps(
            self.state,
            agent_rng,
            target_steps,
            self.hp,
            self.replay.sample,
            self.action_low,
            self.action_high,
        )