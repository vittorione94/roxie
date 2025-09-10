from roxie.agents.agent import Agent, TrainState 
import jax
import jax.numpy as jnp
import functools
from roxie.agents.utils import Transition
from flax import nnx
import optax
import copy
from roxie.agents.utils import serialize_bound
from typing import Any
import hydra
from roxie.losses.actor_losses import ddpg_actor_loss_fn
from roxie.losses.critic_losses import ddpg_critic_loss_fn

# This is the core computational kernel that will be JIT-compiled.
@functools.partial(nnx.jit, static_argnames=('gamma', 'tau', 'replay_sample_fn'))
def _grad_step(state: TrainState, key: jax.random.PRNGKey, gamma: float, tau: float, replay_sample_fn, \
               target_policy_noise: float, target_noise_clip: float, action_low: float, action_high: float,
               obs_eps: float, obs_clip: float):
    """Performs one full gradient update step and returns the new state."""
    # 1. Sample from the replay buffer
    key, noise_key = jax.random.split(key)
    samples = replay_sample_fn(state.buffer_state, key)

    # type(samples)  flashbax.buffers.flat_buffer.TransitionSample
    # type(samples.experience) flashbax.buffers.flat_buffer.ExperiencePair
    # type(samples.experience.first) roxie.replays.buffer.Transition
    re_packed_samples =  {
        "observations": samples.experience.first.observation,
        "actions": samples.experience.first.action,
        "rewards": samples.experience.first.reward,
        "next_observations": samples.experience.second.observation,
        "terminals": samples.experience.first.terminal,
    }

    # 1.1 Compute current normalization parameters
    obs_mean, obs_std = Agent.obs_mean_std(state.obs_stats, obs_eps)

    # 2. Critic update
    critic_loss, critic_grads = nnx.value_and_grad(ddpg_critic_loss_fn)(
        state.critic, state.target_actor, state.target_critic, re_packed_samples, gamma, noise_key, \
            target_policy_noise, target_noise_clip, action_low, action_high, \
            obs_mean, obs_std, obs_clip
    )
    state.critic_optimizer.update(critic_grads)

    # 3. Actor update (use scaled actions)
    actor_loss, actor_grads = nnx.value_and_grad(ddpg_actor_loss_fn)(
        state.actor, state.critic, re_packed_samples, obs_mean, obs_std, obs_clip, action_low, action_high
    )
    state.actor_optimizer.update(actor_grads)
    
    # 4. Update target networks using soft updates
    new_actor_tensors = nnx.state(state.actor, nnx.Param)  # Use updated_actor
    old_actor_tensors = nnx.state(state.target_actor, nnx.Param)

    new_target_actor_tensors = optax.incremental_update(
          new_tensors=new_actor_tensors,
          old_tensors=old_actor_tensors,
          step_size=tau)
    
    new_critic_tensors = nnx.state(state.critic, nnx.Param)
    old_critic_tensors = nnx.state(state.target_critic, nnx.Param)
    new_target_critic_tensors = optax.incremental_update(
          new_tensors=new_critic_tensors,
          old_tensors=old_critic_tensors,
          step_size=tau)
    

    nnx.update(state.target_actor, new_target_actor_tensors)
    nnx.update(state.target_critic, new_target_critic_tensors)

    # 5. Return the new, updated state object
    return TrainState(
            actor=state.actor,
            critic=state.critic,
            actor_optimizer=state.actor_optimizer,
            target_actor=state.target_actor,
            target_critic=state.target_critic,
            critic_optimizer=state.critic_optimizer,
            buffer_state=state.buffer_state,
            obs_stats=state.obs_stats
        ), actor_loss, critic_loss


class DDPG(Agent):
    def __init__(self, 
                 env_obs_size: int,
                 env_action_size: int,
                 action_low: jnp.ndarray,
                 action_high: jnp.ndarray,
                 actor_config: dict,
                 critic_config: dict,
                 memory_config: dict,
                 noise_config: dict,
                 *,
                 actor_learning_rate: float = 3e-4,
                 critic_learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 steps_before_learning: int = 100,
                 steps_between_updates: int = 10,
                 learning_steps: int = 5,
                 memory_warmup: int = 100,
                 target_noise_clip: float = 0.1,
                 target_policy_noise: float = 0.1,
                 max_grad_norm: float = 1.0,
                 normalize_observations: bool = True,
                 obs_norm_clip: float = 5.0,
                 obs_norm_eps: float = 1e-8,
                 ):

        actor_rngs = nnx.Rngs(params=0, dropout=1)
        critic_rngs = nnx.Rngs(params=0, dropout=1)        

        # Instantiate actor
        actor = hydra.utils.instantiate(
            actor_config,
            in_features=env_obs_size,
            action_dim=env_action_size,
            rngs=actor_rngs
        )

        # Instantiate critic
        critic = hydra.utils.instantiate(
            critic_config,
            in_features=env_obs_size + env_action_size,
            rngs=critic_rngs
        )

        # Instantiate replay buffer
        prototype = Transition(
            observation=jnp.zeros(env_obs_size, dtype=jnp.float32),
            action=jnp.zeros(env_action_size, dtype=jnp.float32),
            reward=jnp.zeros((), dtype=jnp.float32),
            # next_observation=jnp.zeros(env_obs_size, dtype=jnp.float32),
            terminal=jnp.zeros((), dtype=jnp.bool_)
        )
        replay = hydra.utils.instantiate(memory_config)
        self.batch_size = memory_config.sample_batch_size
        self.buffer_size = memory_config.max_length

        buffer_state = replay.init(prototype)

        # Instantiate noise module
        noise_module = hydra.utils.instantiate(
            noise_config,
            action_shape=(env_action_size,)
        )

        # Create targets
        target_actor = copy.deepcopy(actor)
        target_critic = copy.deepcopy(critic)

        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate = actor_learning_rate
        self.max_grad_norm = max_grad_norm
        self.noise_module = noise_module

        # Add gradient clipping to optimizers
        actor_optimizer = nnx.Optimizer(
            actor,
            optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.adam(self.actor_learning_rate)
            ),
            wrt=nnx.Param
        )

        critic_optimizer = nnx.Optimizer(
            critic,
            optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.adam(self.critic_learning_rate)
            ),
            wrt=nnx.Param
        )

        # Init observation stats from buffer state's observation shape
        obs_shape = buffer_state.experience.observation.shape[-1]  # Exclude batch dimension
        obs_stats = Agent.init_obs_stats(obs_shape)

        self.state = TrainState(
            actor=actor,
            critic=critic,
            target_actor=target_actor,
            target_critic=target_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            buffer_state=buffer_state,
            obs_stats=obs_stats
        )

        # Store hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.action_low = action_low
        self.action_high = action_high
        self.replay = replay
        self.steps_before_learning = steps_before_learning
        self.steps_between_updates = steps_between_updates
        self.learning_steps = learning_steps
        self.memory_warmup = memory_warmup
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise
        self.normalize_observations = normalize_observations
        self.obs_clip = float(obs_norm_clip)
        self.obs_eps = float(obs_norm_eps)

        print("DDPG agent initialized.")
        print("Noise module hyperparameters:", self.noise_module.hyperparameters())
        print("Hyper Params:", self._export_hyperparams())

    def step(self, observation: jnp.ndarray, evaluate: bool = False, key: jax.random.PRNGKey = None) -> jnp.ndarray:
        """
        Selects an action by calling the pure, JIT-compiled step function.
        """
        # Call the standalone function, passing in the required parts from the agent's state.
        if self.normalize_observations:
            mean, std = Agent.obs_mean_std(self.state.obs_stats, self.obs_eps)
            observation = Agent.normalize_obs(observation, mean, std, self.obs_clip)

        action, noise = Agent.deterministic_step_fn(
            self.state.actor,
            observation,
            key,
            self.noise_module,
            evaluate,
        )

        self.last_action = Agent.scale_to_env(action, self.action_low, self.action_high)

        return self.last_action

    def update(self, prev_states, states, steps, agent_rng):
        # prev_states.obs.shape   (num_envs, obs_dim)
        # states.reward.shape     (num_envs,)
        # states.done.shape       (num_envs,)
        # self.last_action.shape  (num_envs, action_dim)
        # states.obs.shape        (num_envs, obs_dim)

        experiences = Transition(
            observation=prev_states.obs,
            action=self.last_action,
            reward=states.reward,
            # next_observation=states.obs,
            terminal=states.done,
        )

        # store in memory
        self.state.buffer_state = self.replay.add(self.state.buffer_state, experiences)

        # Update observation normalization stats with both current and next observations
        if self.normalize_observations:
            obs_batch = jnp.concatenate([prev_states.obs, states.obs], axis=0)
            # obs_batch.shape --> (2 * num_envs, obs_dim)
            self.state.obs_stats = Agent.update_obs_stats(self.state.obs_stats, obs_batch)

        gradient_steps, actor_loss, critic_loss = 0, 0, 0

        # Conditionally call the JIT-compiled gradient step
        if steps >= self.steps_before_learning and \
           (steps - self.steps_before_learning) % self.steps_between_updates == 0:
            for _ in range(self.learning_steps):
                agent_rng, key = jax.random.split(agent_rng, 2)
                self.state, actor_loss, critic_loss = _grad_step(
                    self.state, 
                    key, 
                    self.gamma, 
                    self.tau,
                    self.replay.sample, # Pass the sample method itself,
                    self.target_policy_noise,  # Use target_policy_noise
                    self.target_noise_clip,
                    self.action_low,
                    self.action_high,
                    self.obs_eps,
                    self.obs_clip
                )
            gradient_steps += self.learning_steps

        return gradient_steps, actor_loss, critic_loss

    def _export_hyperparams(self) -> dict:
        # self.state.buffer_state.experience.observation.shape  (num_envs, steps, obs_dim)
        # self.state.buffer_state.experience.action.shape       (num_envs, steps, action_dim)
        # Keep this minimal and JSON-serializable
        return {
            "gamma": float(self.gamma),
            "tau": float(self.tau),
            "actor_learning_rate": float(self.actor_learning_rate),
            "critic_learning_rate": float(self.critic_learning_rate),
            "max_grad_norm": float(self.max_grad_norm),

            "env_obs_size": self.state.buffer_state.experience.observation.shape[2],
            "env_action_size": self.state.buffer_state.experience.action.shape[2],
            "target_policy_noise": float(self.target_policy_noise),
            "target_noise_clip": float(self.target_noise_clip),
            "steps_before_learning": int(self.steps_before_learning),
            "steps_between_updates": int(self.steps_between_updates),
            "learning_steps": int(self.learning_steps),
            "memory_warmup": int(self.memory_warmup),
            "memory_capacity": int(self.buffer_size),
            "memory_batch_size": int(self.batch_size),
            "normalize_observations": bool(self.normalize_observations),
            "obs_norm_clip": float(self.obs_clip),
            "obs_norm_eps": float(self.obs_eps),
            # bounds can be scalar or arrays → use your helpers
            "action_low": serialize_bound(self.action_low),
            "action_high": serialize_bound(self.action_high),
        }