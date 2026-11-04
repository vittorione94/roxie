"""Short-Horizon Actor-Critic (SHAC), trained through a learned world model."""

import functools

import hydra
import jax
import jax.numpy as jnp
from flax import nnx

from roxie.agents.agent import Agent, LearningOutput, TrainState
from roxie.agents.ddpg import NoiseModuleActing
from roxie.agents.hyperparams import SHACHyperparams, build_hyperparams
from roxie.agents.utils import fused_grad_steps, reduce_diagnostics
from roxie.losses.actor_losses import bptt_actor_loss_fn
from roxie.losses.critic_losses import (
    lambda_return_targets,
    shac_critic_loss_fn,
)
from roxie.utils.math import normalize_obs, obs_mean_std, scale_to_env


def _fit_critic(critic, critic_optimizer, key, observations, targets, *, hp):
    """Fits the critic to fixed targets over shuffled epochs of minibatches.

    Returns:
        `(critic, critic_optimizer, losses)`, the losses stacked per minibatch.
    """
    n = targets.shape[0]
    per_batch = n // hp.critic_minibatches

    def epoch(carry, epoch_key):
        perm = jax.random.permutation(epoch_key, n)
        batches = perm[: per_batch * hp.critic_minibatches].reshape(
            hp.critic_minibatches, per_batch
        )

        def minibatch(carry, idx):
            critic, critic_optimizer = carry
            (loss, _aux), grads = nnx.value_and_grad(
                shac_critic_loss_fn, has_aux=True
            )(critic, observations[idx], targets[idx])
            critic_optimizer.update(critic, grads)
            return (critic, critic_optimizer), loss

        return jax.lax.scan(minibatch, carry, batches)

    (critic, critic_optimizer), losses = jax.lax.scan(
        epoch, (critic, critic_optimizer),
        jax.random.split(key, hp.critic_epochs),
    )
    return critic, critic_optimizer, losses


def _shac_grad_step(
    carry,
    key: jax.Array,
    *,
    hp: SHACHyperparams,
    world,
    buffer_state,
    start_sample_fn,
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
    obs_mean: jnp.ndarray,
    obs_std: jnp.ndarray,
    world_mean: jnp.ndarray,
    world_std: jnp.ndarray,
):
    """One imagined rollout: an actor step through it, then a critic fit to it.

    `carry` is `(train_state, dynamics)`: the world's network rides in the scan
    carry, unchanged, because nnx refuses a module closed over from outside
    the scan's trace. The buffer is plain arrays and is closed over.
    """
    state, dynamics = carry
    sample_key, critic_key = jax.random.split(key)
    starts = start_sample_fn(buffer_state, sample_key).experience.observation[:, 0]

    # Clipped exactly as acting clips, so the actor trains on the inputs it
    # will act on; an imagined state past the clip passes no gradient back
    # through it. The world's own normalization is the unclipped one.
    def normalize(obs):
        if not hp.normalize_observations:
            return obs
        return normalize_obs(obs, obs_mean, obs_std, hp.obs_norm_clip)

    def policy_fn(actor, obs):
        return scale_to_env(actor(normalize(obs)), action_low, action_high)

    def value_fn(critic, obs):
        return critic(normalize(obs))[..., 0]

    def model_fn(dynamics, obs, action):
        return world.step(dynamics, world_mean, world_std, obs, action)

    (actor_loss, actor_aux), actor_grads = nnx.value_and_grad(
        functools.partial(
            bptt_actor_loss_fn,
            policy_fn=policy_fn,
            model_fn=model_fn,
            value_fn=value_fn,
            gamma=hp.gamma,
            horizon=hp.horizon,
        ),
        has_aux=True,
    )(state.actor, state.critic, dynamics, starts)
    state.actor_optimizer.update(state.actor, actor_grads)

    # Targets from the rollout the actor was scored on, under the critic as it
    # stood when the actor bootstrapped off it, and held fixed for the fit.
    states, rewards, term_probs = actor_aux.pop("trajectory")   # time-major
    next_values = value_fn(state.critic, states[1:])            # (H, B)
    targets = lambda_return_targets(
        rewards.T, term_probs.T, next_values.T, hp.gamma, hp.gae_lambda,
    )                                                           # (B, H)
    width = states.shape[-1]
    fit_obs = normalize(jnp.swapaxes(states[:-1], 0, 1).reshape(-1, width))
    fit_targets = targets.reshape(-1)

    value_before = state.critic(fit_obs)[..., 0]
    residual_var = jnp.var(fit_targets - value_before)

    state.critic, state.critic_optimizer, critic_losses = _fit_critic(
        state.critic, state.critic_optimizer, critic_key, fit_obs, fit_targets,
        hp=hp,
    )
    critic_aux = {
        "value_target": jnp.mean(fit_targets),
        "value_ev": 1.0 - residual_var / (jnp.var(fit_targets) + 1e-8),
    }
    return actor_loss, jnp.mean(critic_losses), {**actor_aux, **critic_aux}


@functools.partial(
    jax.jit,
    static_argnames=("hp", "world", "start_sample_fn", "n_steps"),
    # Only the agent's own state is donated. `world_state` is read here but
    # still owned by the world, and donating it would free its replay buffer
    # out from under the next collect chunk.
    donate_argnums=(0,),
)
def _shac_grad_steps(
    state: TrainState,
    world_state,
    key: jax.Array,
    n_steps: int,
    hp: SHACHyperparams,
    world,
    start_sample_fn,
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
):
    """Runs one learning pass of SHAC gradient steps via `jax.lax.scan`."""
    obs_mean, obs_std = obs_mean_std(state.obs_stats, hp.obs_norm_eps)
    world_mean, world_std = world.normalizer(world_state.obs_stats)

    (state, _), (actor_losses, critic_losses, aux) = fused_grad_steps(
        (state, world_state.dynamics),
        key,
        n_steps,
        functools.partial(
            _shac_grad_step,
            hp=hp,
            world=world,
            buffer_state=world_state.buffer_state,
            start_sample_fn=start_sample_fn,
            action_low=action_low,
            action_high=action_high,
            obs_mean=obs_mean,
            obs_std=obs_std,
            world_mean=world_mean,
            world_std=world_std,
        ),
    )
    return LearningOutput(
        state=state,
        actor_loss=jnp.mean(actor_losses),
        critic_loss=jnp.mean(critic_losses),
        diagnostics=reduce_diagnostics(aux, n_steps),
    )


class SHAC(NoiseModuleActing, Agent):
    """Short-horizon actor-critic, backpropagating through a learned model.

    A deterministic actor is trained on the analytic gradient of an imagined
    `horizon`-step return, rolled through a `roxie.worlds.WorldModel` from real
    observations and bootstrapped by a state-value critic; the critic is fit to
    lambda-returns on the same imagined rollout. Real data only ever reaches the
    world: this agent holds no replay buffer.

    References:
        Xu et al., 2022: https://arxiv.org/abs/2204.07137
        Liao et al., 2026: https://arxiv.org/abs/2610.28480
    """

    hyperparams_cls = SHACHyperparams
    uses_world = True

    def __init__(
        self,
        env_obs_size: int,
        env_action_size: int,
        action_low: jnp.ndarray,
        action_high: jnp.ndarray,
        actor_config: dict,
        critic_config: dict,
        noise_config: dict,
        *,
        hyperparams: SHACHyperparams = None,
        actor_optimizer_config: dict = None,
        critic_optimizer_config: dict = None,
    ):
        self.hp = build_hyperparams(type(self).hyperparams_cls, hyperparams)
        self.env_obs_size = int(env_obs_size)
        self.env_action_size = int(env_action_size)

        actor = hydra.utils.instantiate(actor_config)
        critic = hydra.utils.instantiate(critic_config)

        self.noise_module = hydra.utils.instantiate(
            noise_config, action_shape=(env_action_size,)
        )

        # No targets: the critic's targets are recomputed from scratch on every
        # imagined rollout, which is what a target network would stabilize.
        self._init_train_state(
            actor,
            critic,
            None,
            actor_optimizer_config=actor_optimizer_config,
            critic_optimizer_config=critic_optimizer_config,
            target_actor=False,
            target_critic=False,
            obs_size=self.env_obs_size,
            actor_clip_value=self.hp.actor_clip_value,
        )

        self.action_low = action_low
        self.action_high = action_high

        print(f"{type(self).__name__} agent initialized.")
        print("Noise module hyperparameters:", self.noise_module.hyperparameters())
        print("Hyper Params:", self._export_hyperparams())

    def _compile_and_run(self, agent_rng: jax.Array, target_steps: int, world):
        """Runs one fused learning pass through `world`."""
        return target_steps, _shac_grad_steps(
            self.state,
            world.state,
            agent_rng,
            target_steps,
            self.hp,
            world,
            world.start_sampler(self.hp.imagined_batch_size),
            self.action_low,
            self.action_high,
        )

    def _export_hyperparams(self) -> dict:
        params = super()._export_hyperparams()
        params.update(
            env_obs_size=self.env_obs_size,
            env_action_size=self.env_action_size,
        )
        return params
