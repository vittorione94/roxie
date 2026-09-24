"""Soft Actor-Critic (SAC) agent."""

import functools
from typing import Any

import hydra
import jax
import jax.numpy as jnp
from flax import nnx

from roxie.agents.agent import Agent, LearningOutput, TrainState
from roxie.agents.hyperparams import SACHyperparams, build_hyperparams
from roxie.agents.utils import (
    build_replay,
    fused_grad_steps,
    make_optimizer,
    reduce_diagnostics,
    repack_samples,
    soft_update,
    transition_prototype,
)
from roxie.losses.actor_losses import (
    sac_actor_loss_fn,
    sac_alpha_loss_fn,
    skipped_actor_aux,
)
from roxie.losses.critic_losses import sac_critic_loss_fn
from roxie.models.actors import stochastic_step_fn
from roxie.utils.math import (
    normalize_obs,
    normalize_samples,
    obs_mean_std,
    scale_to_env,
)
from roxie.utils.memory import ReplayManager


class LogAlpha(nnx.Module):
    """Wraps the log entropy temperature parameter as an NNX module."""

    def __init__(self, init_value: float = 0.0):
        self.log_alpha = nnx.Param(jnp.array(init_value, dtype=jnp.float32))


def _sac_grad_step(
    nodes,
    key: jax.Array,
    update_actor,
    *,
    hp: SACHyperparams,
    replay_sample_fn,
    target_entropy: float,
    action_low: float,
    action_high: float,
    obs_mean: jnp.ndarray,
    obs_std: jnp.ndarray,
):
    """Executes a single SAC gradient step updating critic, actor, and entropy temperature."""
    state, log_alpha_module, alpha_optimizer = nodes

    key, sample_key, actor_key, critic_key = jax.random.split(key, 4)
    samples = replay_sample_fn(state.buffer_state, sample_key)
    re_packed_samples = repack_samples(samples, hp.gamma, hp.n_step)
    re_packed_samples = normalize_samples(
        re_packed_samples,
        obs_mean,
        obs_std,
        hp.obs_norm_clip,
        hp.normalize_observations,
    )

    alpha = jnp.exp(log_alpha_module.log_alpha[...])

    (critic_loss, critic_aux), critic_grads = nnx.value_and_grad(
        sac_critic_loss_fn, has_aux=True
    )(
        state.critic,
        state.actor,
        state.target_critic,
        re_packed_samples,
        alpha,
        critic_key,
        action_low,
        action_high,
        hp.twin_q_weight,
    )
    state.critic_optimizer.update(state.critic, critic_grads)

    def _actor_update(operand):
        st, la, aopt = operand
        (actor_loss, (log_probs, actor_aux)), actor_grads = nnx.value_and_grad(
            sac_actor_loss_fn, has_aux=True
        )(
            st.actor,
            st.critic,
            alpha,
            re_packed_samples,
            actor_key,
            action_low,
            action_high,
            hp.twin_q_weight,
        )
        st.actor_optimizer.update(st.actor, actor_grads)

        if hp.auto_alpha:
            _, alpha_grads = nnx.value_and_grad(sac_alpha_loss_fn)(
                la,
                jax.lax.stop_gradient(log_probs),
                target_entropy,
            )
            aopt.update(la, alpha_grads)
        return actor_loss, actor_aux

    def _skip_actor_update(operand):
        zero = jnp.array(0.0, dtype=critic_loss.dtype)
        return zero, skipped_actor_aux(critic_loss.dtype, "alpha", "entropy")

    operand = (state, log_alpha_module, alpha_optimizer)
    if update_actor is None:
        actor_loss, actor_aux = _actor_update(operand)
    else:
        actor_loss, actor_aux = nnx.cond(
            update_actor, _actor_update, _skip_actor_update, operand
        )

    soft_update(state.target_critic, state.critic, hp.tau)

    return actor_loss, critic_loss, actor_aux, critic_aux


@functools.partial(
    jax.jit,
    static_argnames=("hp", "replay_sample_fn", "n_steps", "target_entropy"),
    donate_argnums=(0, 1, 2),
)
def _sac_grad_steps(
    state: TrainState,
    log_alpha_module: LogAlpha,
    alpha_optimizer: nnx.Optimizer,
    key: jax.Array,
    n_steps: int,
    hp: SACHyperparams,
    replay_sample_fn,
    target_entropy: float,
    action_low: float,
    action_high: float,
) -> LearningOutput:
    """Runs one learning pass of SAC gradient steps via `jax.lax.scan`."""
    obs_mean, obs_std = obs_mean_std(state.obs_stats, hp.obs_norm_eps)

    update_mask = (
        None
        if hp.policy_delay <= 1
        else (jnp.arange(n_steps, dtype=jnp.int32) % hp.policy_delay) == 0
    )

    (state, log_alpha_module, alpha_optimizer), (
        actor_losses,
        critic_losses,
        actor_aux,
        critic_aux,
    ) = fused_grad_steps(
        (state, log_alpha_module, alpha_optimizer),
        key,
        n_steps,
        functools.partial(
            _sac_grad_step,
            hp=hp,
            replay_sample_fn=replay_sample_fn,
            target_entropy=target_entropy,
            action_low=action_low,
            action_high=action_high,
            obs_mean=obs_mean,
            obs_std=obs_std,
        ),
        extras=(update_mask,),
    )

    n_actor_updates = (
        n_steps if update_mask is None else jnp.maximum(jnp.sum(update_mask), 1)
    )
    return LearningOutput(
        state=state,
        actor_loss=jnp.sum(actor_losses) / n_actor_updates,
        critic_loss=jnp.mean(critic_losses),
        diagnostics={
            **reduce_diagnostics(actor_aux, n_actor_updates),
            **reduce_diagnostics(critic_aux, n_steps),
        },
        extra_state=(log_alpha_module, alpha_optimizer),
    )


class SAC(Agent):
    """Soft Actor-Critic agent.

    Reference:
        Haarnoja et al., 2018: https://arxiv.org/abs/1801.01290
    """

    hyperparams_cls = SACHyperparams

    def __init__(
        self,
        env_obs_size: int,
        env_action_size: int,
        action_low: jnp.ndarray,
        action_high: jnp.ndarray,
        actor_config: dict,
        critic_config: dict,
        memory_config: dict,
        *,
        hyperparams: SACHyperparams = None,
        actor_optimizer_config: dict = None,
        critic_optimizer_config: dict = None,
        alpha_optimizer_config: dict = None,
    ):
        self.hp = build_hyperparams(type(self).hyperparams_cls, hyperparams)

        actor = hydra.utils.instantiate(actor_config)
        twin_critic = hydra.utils.instantiate(critic_config)

        self.replay = ReplayManager(
            build_replay(memory_config, self.hp.n_step),
            transition_prototype(env_obs_size, env_action_size),
            time_axis=self.hp.n_step > 1,
        )
        self.batch_size = memory_config.sample_batch_size
        self.buffer_size = memory_config.max_length
        buffer_state = self.replay.init()

        self.log_alpha_module = LogAlpha(self.hp.init_log_alpha)

        self.target_entropy = (
            self.hp.target_entropy
            if self.hp.target_entropy is not None
            else -self.hp.target_entropy_scale * float(env_action_size)
        )

        self.alpha_optimizer = make_optimizer(
            self.log_alpha_module,
            alpha_optimizer_config,
            learning_rate=self.hp.alpha_learning_rate,
        )

        self._init_train_state(
            actor,
            twin_critic,
            buffer_state,
            actor_optimizer_config=actor_optimizer_config,
            critic_optimizer_config=critic_optimizer_config,
            target_actor=False,
        )

        self.action_low = action_low
        self.action_high = action_high

        print("SAC agent initialized.")
        print("Hyper Params:", self._export_hyperparams())

    def _checkpoint_modules(self) -> dict:
        """Returns extra stateful modules outside `TrainState` to checkpoint."""
        return {
            "log_alpha_module": self.log_alpha_module,
            "alpha_optimizer": self.alpha_optimizer,
        }

    def select_action(
        self,
        observation: jnp.ndarray,
        key: jax.Array = None,
        evaluate: bool = False,
        *,
        actor: nnx.Module = None,
        obs_stats: Any = None,
        noise_module: nnx.Module = None,
        critic: nnx.Module = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
        """Selects an action given an observation using the stochastic policy."""
        del noise_module, critic
        if self.hp.normalize_observations:
            stats = self.state.obs_stats if obs_stats is None else obs_stats
            mean, std = obs_mean_std(stats, self.hp.obs_norm_eps)
            observation = normalize_obs(
                observation, mean, std, self.hp.obs_norm_clip
            )

        action, noise, _, _, _ = stochastic_step_fn(
            self.state.actor if actor is None else actor,
            observation,
            evaluate,
            key,
        )
        return (
            scale_to_env(action, self.action_low, self.action_high),
            noise,
            {},
        )

    def _compile_and_run(self, agent_rng: jax.Array, target_steps: int):
        """Runs one fused learning pass to update the agent."""
        return target_steps, _sac_grad_steps(
            self.state,
            self.log_alpha_module,
            self.alpha_optimizer,
            agent_rng,
            target_steps,
            self.hp,
            self.replay.sample,
            self.target_entropy,
            self.action_low,
            self.action_high,
        )

    def _apply_extra_state(self, extra_state) -> None:
        self.log_alpha_module, self.alpha_optimizer = extra_state

    def _export_hyperparams(self) -> dict:
        params = super()._export_hyperparams()
        params.update(self._replay_hyperparams())
        params.update(
            {
                "init_log_alpha": float(self.log_alpha_module.log_alpha[...]),
                "target_entropy": float(self.target_entropy),
            }
        )
        return params