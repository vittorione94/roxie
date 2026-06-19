import copy
import functools

import hydra
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from roxie.agents.agent import Agent, TrainState
from roxie.agents.utils import Transition, serialize_bound
from roxie.losses.actor_losses import sac_actor_loss_fn, sac_alpha_loss_fn
from roxie.losses.critic_losses import sac_critic_loss_fn
from roxie.models.critics import TwinCritic


class LogAlpha(nnx.Module):
    def __init__(self, init_value=0.0):
        self.log_alpha = nnx.Param(jnp.array(init_value, dtype=jnp.float32))


@functools.partial(nnx.jit, static_argnames=("evaluate",))
def _sac_step_fn(actor_model, observation, evaluate, key):
    distribution = actor_model(observation)
    if evaluate:
        try:
            mean = distribution.mean()
        except TypeError:
            mean = distribution.mean
        return jnp.tanh(mean)
    u = distribution.sample(seed=key)
    return jnp.tanh(u)


@functools.partial(
    nnx.jit,
    static_argnames=("gamma", "tau", "replay_sample_fn", "target_entropy", "auto_alpha"),
)
def _sac_grad_step(
    state: TrainState,
    log_alpha_module: LogAlpha,
    alpha_optimizer: nnx.Optimizer,
    key: jax.random.PRNGKey,
    gamma: float,
    tau: float,
    replay_sample_fn,
    target_entropy: float,
    auto_alpha: bool,
    action_low: float,
    action_high: float,
    obs_eps: float,
    obs_clip: float,
):
    key, sample_key, actor_key, critic_key = jax.random.split(key, 4)

    samples = replay_sample_fn(state.buffer_state, sample_key)
    re_packed_samples = {
        "observations": samples.experience.first.observation,
        "actions": samples.experience.first.action,
        "rewards": samples.experience.first.reward,
        "next_observations": samples.experience.second.observation,
        "terminals": samples.experience.first.terminal,
    }

    obs_mean, obs_std = Agent.obs_mean_std(state.obs_stats, obs_eps)
    alpha = jnp.exp(log_alpha_module.log_alpha.value)

    # 1. Critic update
    critic_loss, critic_grads = nnx.value_and_grad(sac_critic_loss_fn)(
        state.critic,
        state.actor,
        state.target_critic,
        re_packed_samples,
        gamma,
        alpha,
        critic_key,
        action_low,
        action_high,
        obs_mean,
        obs_std,
        obs_clip,
    )
    state.critic_optimizer.update(critic_grads)

    # 2. Actor update
    (actor_loss, log_probs), actor_grads = nnx.value_and_grad(
        sac_actor_loss_fn, has_aux=True
    )(
        state.actor,
        state.critic,
        alpha,
        re_packed_samples,
        actor_key,
        obs_mean,
        obs_std,
        obs_clip,
        action_low,
        action_high,
    )
    state.actor_optimizer.update(actor_grads)

    # 3. Alpha update
    if auto_alpha:
        _, alpha_grads = nnx.value_and_grad(sac_alpha_loss_fn)(
            log_alpha_module,
            jax.lax.stop_gradient(log_probs),
            target_entropy,
        )
        alpha_optimizer.update(alpha_grads)

    # 4. Soft update target critics
    new_critic_tensors = nnx.state(state.critic, nnx.Param)
    old_critic_tensors = nnx.state(state.target_critic, nnx.Param)
    new_target_critic_tensors = optax.incremental_update(
        new_tensors=new_critic_tensors,
        old_tensors=old_critic_tensors,
        step_size=tau,
    )
    nnx.update(state.target_critic, new_target_critic_tensors)

    new_state = TrainState(
        actor=state.actor,
        critic=state.critic,
        actor_optimizer=state.actor_optimizer,
        target_actor=state.target_actor,
        target_critic=state.target_critic,
        critic_optimizer=state.critic_optimizer,
        buffer_state=state.buffer_state,
        obs_stats=state.obs_stats,
    )

    return new_state, log_alpha_module, alpha_optimizer, actor_loss, critic_loss


class SAC(Agent):
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
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        alpha_learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        init_log_alpha: float = 0.0,
        auto_alpha: bool = True,
        target_entropy: float = None,
        steps_before_learning: int = 100,
        steps_between_updates: int = 10,
        learning_steps: int = 5,
        memory_warmup: int = 100,
        max_grad_norm: float = 1.0,
        normalize_observations: bool = True,
        obs_norm_clip: float = 5.0,
        obs_norm_eps: float = 1e-8,
    ):
        actor_rngs = nnx.Rngs(params=0, dropout=1)
        critic_rngs_1 = nnx.Rngs(params=2, dropout=3)
        critic_rngs_2 = nnx.Rngs(params=4, dropout=5)

        # Stochastic actor (outputs distribution for reparameterized sampling)
        actor = hydra.utils.instantiate(
            actor_config,
            in_features=env_obs_size,
            action_dim=env_action_size,
            rngs=actor_rngs,
        )

        # Twin Q-networks to reduce overestimation bias
        critic1 = hydra.utils.instantiate(
            critic_config,
            in_features=env_obs_size + env_action_size,
            rngs=critic_rngs_1,
        )
        critic2 = hydra.utils.instantiate(
            critic_config,
            in_features=env_obs_size + env_action_size,
            rngs=critic_rngs_2,
        )
        twin_critic = TwinCritic(critic1, critic2)

        # Replay buffer
        prototype = Transition(
            observation=jnp.zeros(env_obs_size, dtype=jnp.float32),
            action=jnp.zeros(env_action_size, dtype=jnp.float32),
            reward=jnp.zeros((), dtype=jnp.float32),
            terminal=jnp.zeros((), dtype=jnp.bool_),
        )
        replay = hydra.utils.instantiate(memory_config)
        self.batch_size = memory_config.sample_batch_size
        self.buffer_size = memory_config.max_length
        buffer_state = replay.init(prototype)

        # Target critic (no target actor in SAC)
        target_twin_critic = copy.deepcopy(twin_critic)

        # Learnable entropy temperature
        self.log_alpha_module = LogAlpha(init_log_alpha)
        self.auto_alpha = auto_alpha
        self.target_entropy = (
            target_entropy if target_entropy is not None else -float(env_action_size)
        )

        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.alpha_learning_rate = alpha_learning_rate
        self.max_grad_norm = max_grad_norm

        actor_optimizer = nnx.Optimizer(
            actor,
            optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.adam(self.actor_learning_rate),
            ),
            wrt=nnx.Param,
        )

        critic_optimizer = nnx.Optimizer(
            twin_critic,
            optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.adam(self.critic_learning_rate),
            ),
            wrt=nnx.Param,
        )

        self.alpha_optimizer = nnx.Optimizer(
            self.log_alpha_module,
            optax.adam(self.alpha_learning_rate),
            wrt=nnx.Param,
        )

        obs_shape = buffer_state.experience.observation.shape[-1]
        obs_stats = Agent.init_obs_stats(obs_shape)

        self.state = TrainState(
            actor=actor,
            critic=twin_critic,
            target_actor=None,
            target_critic=target_twin_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            buffer_state=buffer_state,
            obs_stats=obs_stats,
        )

        self.gamma = gamma
        self.tau = tau
        self.action_low = action_low
        self.action_high = action_high
        self.replay = replay
        self.steps_before_learning = steps_before_learning
        self.steps_between_updates = steps_between_updates
        self.learning_steps = learning_steps
        self.memory_warmup = memory_warmup
        self.normalize_observations = normalize_observations
        self.obs_clip = float(obs_norm_clip)
        self.obs_eps = float(obs_norm_eps)

        print("SAC agent initialized.")
        print("Hyper Params:", self._export_hyperparams())

    def step(
        self,
        observation: jnp.ndarray,
        evaluate: bool = False,
        key: jax.random.PRNGKey = None,
    ) -> jnp.ndarray:
        if self.normalize_observations:
            mean, std = Agent.obs_mean_std(self.state.obs_stats, self.obs_eps)
            observation = Agent.normalize_obs(observation, mean, std, self.obs_clip)

        action = _sac_step_fn(self.state.actor, observation, evaluate, key)
        self.last_action = Agent.scale_to_env(action, self.action_low, self.action_high)
        return self.last_action

    def add(self, prev_states, states):
        experiences = Transition(
            observation=prev_states.obs,
            action=self.last_action,
            reward=states.reward,
            terminal=states.done,
        )

        self.state.buffer_state = self.replay.add(self.state.buffer_state, experiences)

        if self.normalize_observations:
            obs_batch = jnp.concatenate([prev_states.obs, states.obs], axis=0)
            self.state.obs_stats = Agent.update_obs_stats(
                self.state.obs_stats, obs_batch
            )

    def update(self, steps, agent_rng):
        gradient_steps, actor_loss, critic_loss = 0, 0, 0

        if (
            steps >= self.steps_before_learning
            and (steps - self.steps_before_learning) % self.steps_between_updates == 0
        ):
            for _ in range(self.learning_steps):
                agent_rng, key = jax.random.split(agent_rng, 2)
                (
                    self.state,
                    self.log_alpha_module,
                    self.alpha_optimizer,
                    actor_loss,
                    critic_loss,
                ) = _sac_grad_step(
                    self.state,
                    self.log_alpha_module,
                    self.alpha_optimizer,
                    key,
                    self.gamma,
                    self.tau,
                    self.replay.sample,
                    self.target_entropy,
                    self.auto_alpha,
                    self.action_low,
                    self.action_high,
                    self.obs_eps,
                    self.obs_clip,
                )
            gradient_steps += self.learning_steps

        return gradient_steps, actor_loss, critic_loss

    def _export_hyperparams(self) -> dict:
        return {
            "gamma": float(self.gamma),
            "tau": float(self.tau),
            "actor_learning_rate": float(self.actor_learning_rate),
            "critic_learning_rate": float(self.critic_learning_rate),
            "alpha_learning_rate": float(self.alpha_learning_rate),
            "max_grad_norm": float(self.max_grad_norm),
            "env_obs_size": self.state.buffer_state.experience.observation.shape[2],
            "env_action_size": self.state.buffer_state.experience.action.shape[2],
            "init_log_alpha": float(self.log_alpha_module.log_alpha.value),
            "auto_alpha": bool(self.auto_alpha),
            "target_entropy": float(self.target_entropy),
            "steps_before_learning": int(self.steps_before_learning),
            "steps_between_updates": int(self.steps_between_updates),
            "learning_steps": int(self.learning_steps),
            "memory_warmup": int(self.memory_warmup),
            "memory_capacity": int(self.buffer_size),
            "memory_batch_size": int(self.batch_size),
            "normalize_observations": bool(self.normalize_observations),
            "obs_norm_clip": float(self.obs_clip),
            "obs_norm_eps": float(self.obs_eps),
            "action_low": serialize_bound(self.action_low),
            "action_high": serialize_bound(self.action_high),
        }
