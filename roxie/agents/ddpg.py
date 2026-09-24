"""Deep Deterministic Policy Gradient (DDPG) agent."""

import functools
from typing import Any

import hydra
import jax
import jax.numpy as jnp
from flax import nnx

from roxie.agents.agent import Agent, LearningOutput, TrainState
from roxie.agents.hyperparams import DDPGHyperparams, build_hyperparams
from roxie.agents.utils import (
    build_replay,
    fused_grad_steps,
    reduce_diagnostics,
    repack_samples,
    soft_update,
    transition_prototype,
)
from roxie.losses.actor_losses import ddpg_actor_loss_fn
from roxie.losses.critic_losses import ddpg_critic_loss_fn
from roxie.models.actors import deterministic_step_fn
from roxie.utils.math import (
    normalize_obs,
    normalize_samples,
    obs_mean_std,
    scale_to_env,
)
from roxie.utils.memory import ReplayManager


def _ddpg_grad_step(
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
    """Executes a single DDPG gradient step updating actor and critic networks."""
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
        ddpg_critic_loss_fn, has_aux=True
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
        ddpg_actor_loss_fn, has_aux=True
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
def _ddpg_grad_steps(
    state: TrainState,
    key: jax.Array,
    n_steps: int,
    hp: DDPGHyperparams,
    replay_sample_fn,
    action_low: float,
    action_high: float,
):
    """Runs one learning pass of DDPG gradient steps via `jax.lax.scan`."""
    obs_mean, obs_std = obs_mean_std(state.obs_stats, hp.obs_norm_eps)

    state, (actor_losses, critic_losses, actor_aux, critic_aux) = fused_grad_steps(
        state,
        key,
        n_steps,
        functools.partial(
            _ddpg_grad_step,
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


class DDPG(Agent):
    """Deep Deterministic Policy Gradient agent.

    Reference:
        Lillicrap et al., 2015: https://arxiv.org/abs/1509.02971
    """

    hyperparams_cls = DDPGHyperparams

    def __init__(
        self,
        env_obs_size: int,
        env_action_size: int,
        action_low: jnp.ndarray,
        action_high: jnp.ndarray,
        actor_config: dict,
        critic_config: dict,
        memory_config: dict,
        noise_config: dict,
        *,
        hyperparams: DDPGHyperparams = None,
        actor_optimizer_config: dict = None,
        critic_optimizer_config: dict = None,
    ):
        self.hp = build_hyperparams(type(self).hyperparams_cls, hyperparams)

        actor = hydra.utils.instantiate(actor_config)
        critic = hydra.utils.instantiate(critic_config)

        self.replay = ReplayManager(
            build_replay(memory_config, self.hp.n_step),
            transition_prototype(env_obs_size, env_action_size),
            time_axis=self.hp.n_step > 1,
        )
        self.batch_size = memory_config.sample_batch_size
        self.buffer_size = memory_config.max_length

        buffer_state = self.replay.init()

        self.noise_module = hydra.utils.instantiate(
            noise_config, action_shape=(env_action_size,)
        )

        self._init_train_state(
            actor,
            critic,
            buffer_state,
            actor_optimizer_config=actor_optimizer_config,
            critic_optimizer_config=critic_optimizer_config,
        )

        self.action_low = action_low
        self.action_high = action_high

        print(f"{type(self).__name__} agent initialized.")
        print("Noise module hyperparameters:", self.noise_module.hyperparameters())
        print("Hyper Params:", self._export_hyperparams())

    def _checkpoint_modules(self) -> dict:
        """Returns extra stateful modules outside `TrainState` to checkpoint."""
        return {"noise_module": self.noise_module}

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
        """Selects an action given an observation and applies exploration noise."""
        del critic
        if self.hp.normalize_observations:
            stats = self.state.obs_stats if obs_stats is None else obs_stats
            mean, std = obs_mean_std(stats, self.hp.obs_norm_eps)
            observation = normalize_obs(observation, mean, std, self.hp.obs_norm_clip)

        action, noise = deterministic_step_fn(
            self.state.actor if actor is None else actor,
            observation,
            key,
            self.noise_module if noise_module is None else noise_module,
            evaluate,
        )
        return (
            scale_to_env(action, self.action_low, self.action_high),
            noise,
            {},
        )

    def _compile_and_run(self, agent_rng: jax.Array, target_steps: int):
        """Runs one fused learning pass to update the agent."""
        return target_steps, _ddpg_grad_steps(
            self.state,
            agent_rng,
            target_steps,
            self.hp,
            self.replay.sample,
            self.action_low,
            self.action_high,
        )

    def _export_hyperparams(self) -> dict:
        params = super()._export_hyperparams()
        params.update(self._replay_hyperparams())
        return params