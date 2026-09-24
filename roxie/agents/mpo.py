"""Maximum a Posteriori Policy Optimization (MPO) agent."""

import functools
import math
from typing import Any

import hydra
import jax
import jax.numpy as jnp
from flax import nnx

from roxie.agents.agent import Agent, LearningOutput, TrainState
from roxie.agents.hyperparams import MPOHyperparams, build_hyperparams
from roxie.agents.utils import (
    build_replay,
    fused_grad_steps,
    make_optimizer,
    reduce_diagnostics,
    repack_samples,
    soft_update,
    transition_prototype,
)
from roxie.losses.actor_losses import mpo_actor_loss_fn
from roxie.losses.critic_losses import mpo_critic_loss_fn
from roxie.models.actors import stochastic_step_fn
from roxie.utils.math import (
    normalize_obs,
    normalize_samples,
    obs_mean_std,
    scale_to_env,
    inv_softplus,
)
from roxie.utils.memory import ReplayManager


class MPODualParams(nnx.Module):
    """Lagrange dual variables for MPO stored in pre-softplus space.

    Attributes:
        log_temperature: E-step temperature parameter for the KL bound constraint.
        log_alpha_mean: Per-dimension M-step KL multiplier for the mean trust region.
        log_alpha_stddev: Per-dimension M-step KL multiplier for the covariance trust region.
    """

    def __init__(
        self,
        action_dim: int,
        init_temperature: float = 1.0,
        init_alpha_mean: float = 1.0,
        init_alpha_stddev: float = 1.0,
    ):
        self.log_temperature = nnx.Param(
            jnp.asarray(inv_softplus(init_temperature), dtype=jnp.float32)
        )
        self.log_alpha_mean = nnx.Param(
            jnp.full((action_dim,), inv_softplus(init_alpha_mean), dtype=jnp.float32)
        )
        self.log_alpha_stddev = nnx.Param(
            jnp.full((action_dim,), inv_softplus(init_alpha_stddev), dtype=jnp.float32)
        )


def _mpo_grad_step(
    nodes,
    key: jax.Array,
    *,
    hp: MPOHyperparams,
    replay_sample_fn,
    action_low: float,
    action_high: float,
    obs_mean: jnp.ndarray,
    obs_std: jnp.ndarray,
):
    """Executes a single MPO gradient step updating policy, critic, and dual parameters."""
    state, dual_params, dual_optimizer = nodes

    key, sample_key, critic_key, actor_key = jax.random.split(key, 4)

    samples = replay_sample_fn(state.buffer_state, sample_key)
    re_packed_samples = repack_samples(samples, hp.gamma, hp.n_step)
    re_packed_samples = normalize_samples(
        re_packed_samples,
        obs_mean,
        obs_std,
        hp.obs_norm_clip,
        hp.normalize_observations,
    )

    (critic_loss, critic_aux), critic_grads = nnx.value_and_grad(
        mpo_critic_loss_fn, has_aux=True
    )(
        state.critic,
        state.target_actor,
        state.target_critic,
        re_packed_samples,
        critic_key,
        hp.num_action_samples,
        action_low,
        action_high,
    )
    state.critic_optimizer.update(state.critic, critic_grads)

    (actor_loss, actor_aux), (actor_grads, dual_grads) = nnx.value_and_grad(
        mpo_actor_loss_fn, argnums=(0, 1), has_aux=True
    )(
        state.actor,
        dual_params,
        state.target_actor,
        state.critic,
        re_packed_samples,
        actor_key,
        hp.num_action_samples,
        hp.epsilon,
        hp.epsilon_mean,
        hp.epsilon_stddev,
        action_low,
        action_high,
    )
    state.actor_optimizer.update(state.actor, actor_grads)
    dual_optimizer.update(dual_params, dual_grads)

    soft_update(state.target_actor, state.actor, hp.tau)
    soft_update(state.target_critic, state.critic, hp.tau)

    return actor_loss, critic_loss, actor_aux, critic_aux


@functools.partial(
    jax.jit,
    static_argnames=("hp", "replay_sample_fn", "n_steps"),
    donate_argnums=(0, 1, 2),
)
def _mpo_grad_steps(
    state: TrainState,
    dual_params: MPODualParams,
    dual_optimizer: nnx.Optimizer,
    key: jax.Array,
    n_steps: int,
    hp: MPOHyperparams,
    replay_sample_fn,
    action_low: float,
    action_high: float,
):
    """Runs one learning pass of MPO gradient steps via `jax.lax.scan`."""
    obs_mean, obs_std = obs_mean_std(state.obs_stats, hp.obs_norm_eps)

    (state, dual_params, dual_optimizer), (
        actor_losses,
        critic_losses,
        actor_aux,
        critic_aux,
    ) = fused_grad_steps(
        (state, dual_params, dual_optimizer),
        key,
        n_steps,
        functools.partial(
            _mpo_grad_step,
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
        extra_state=(dual_params, dual_optimizer),
    )


class MPO(Agent):
    """Maximum a Posteriori Policy Optimization agent.
    
    Reference:
        Abdolmaleki et al., 2018: https://arxiv.org/abs/1806.06920
    """

    hyperparams_cls = MPOHyperparams

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
        hyperparams: MPOHyperparams = None,
        actor_optimizer_config: dict = None,
        critic_optimizer_config: dict = None,
        dual_optimizer_config: dict = None,
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

        self.dual_params = MPODualParams(
            action_dim=env_action_size,
            init_temperature=self.hp.init_temperature,
            init_alpha_mean=self.hp.init_alpha_mean,
            init_alpha_stddev=self.hp.init_alpha_stddev,
        )

        self.dual_optimizer = make_optimizer(
            self.dual_params,
            dual_optimizer_config,
            learning_rate=self.hp.dual_learning_rate,
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

        print("MPO agent initialized.")
        print("Hyper Params:", self._export_hyperparams())

    def _checkpoint_modules(self) -> dict:
        """Returns extra stateful modules outside `TrainState` to checkpoint."""
        return {
            "dual_params": self.dual_params,
            "dual_optimizer": self.dual_optimizer,
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
        """Selects an action given an observation."""
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
        return target_steps, _mpo_grad_steps(
            self.state,
            self.dual_params,
            self.dual_optimizer,
            agent_rng,
            target_steps,
            self.hp,
            self.replay.sample,
            self.action_low,
            self.action_high,
        )

    def _apply_extra_state(self, extra_state) -> None:
        self.dual_params, self.dual_optimizer = extra_state

    def _export_hyperparams(self) -> dict:
        params = super()._export_hyperparams()
        params.update(self._replay_hyperparams())
        return params