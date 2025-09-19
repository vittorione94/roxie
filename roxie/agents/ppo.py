import functools

import hydra
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from roxie.agents.agent import Agent, TrainState
from roxie.agents.utils import Transition, serialize_bound
from roxie.losses.actor_losses import ppo_loss_fn
from roxie.losses.critic_losses import ppo_critic_loss_fn


# This is the core computational kernel that will be JIT-compiled.
@functools.partial(
    nnx.jit,
    static_argnames=(
        "gamma",
        "gae_lambda",
        "clip_eps",
        "entropy_coef",
        "value_coef",
        "replay_get_fn",
    ),
)
def _grad_step(
    state: TrainState,
    key: jax.random.PRNGKey,
    gamma: float,
    gae_lambda: float,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    replay_get_fn,  # Function to get the on-policy data
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
    obs_eps: float,
    obs_clip: float,
):
    """
    Performs a single gradient update step for the PPO agent.
    This function is JIT-compiled for performance.
    """
    # 1. Get the most recent on-policy data from the buffer
    # This function is expected to be `replay.get_recent_window`
    data = replay_get_fn(state.buffer_state)

    # Unpack data for convenience
    observations = data["observations"]
    actions = data["actions"]
    rewards = data["rewards"]
    terminals = data["terminals"]
    old_log_probs = data["log_probs"]
    bootstrap_obs = data["bootstrap_observation"]

    # 2. Normalize observations
    mean, std = Agent.obs_mean_std(state.obs_stats, obs_eps)
    norm_obs = Agent.normalize_obs(observations, mean, std, obs_clip)
    norm_bootstrap_obs = Agent.normalize_obs(bootstrap_obs, mean, std, obs_clip)

    # 3. Calculate GAE (Generalized Advantage Estimation)
    # Get value estimates for all observations in the rollout
    values = state.critic(norm_obs)
    # Get the value estimate for the bootstrap state (the state after the last action)
    bootstrap_value = state.critic(norm_bootstrap_obs)

    # GAE requires values for s_t and s_{t+1}. We concatenate the rollout values
    # with the bootstrap value to easily compute deltas.
    all_values = jnp.concatenate([values, bootstrap_value[jnp.newaxis, ...]], axis=0)

    # Calculate temporal difference errors (deltas)
    deltas = rewards + gamma * (1.0 - terminals) * all_values[1:] - all_values[:-1]

    # Define the GAE scan function
    def gae_scan_fn(carry, delta_t):
        advantage = delta_t + gamma * gae_lambda * carry
        return advantage, advantage

    # Compute advantages by scanning backwards over the deltas
    # The initial carry is zero.
    _, advantages = jax.lax.scan(
        gae_scan_fn, jnp.zeros_like(bootstrap_value), deltas, reverse=True
    )

    # Calculate returns (targets for the value function)
    returns = advantages + values

    # Normalize advantages for stability
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 4. Compute Actor and Critic Gradients

    # Actor update
    actor_grad_fn = nnx.value_and_grad(
        lambda actor: ppo_loss_fn(
            actor,
            observations=norm_obs,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            clip_eps=clip_eps,
            entropy_coef=entropy_coef,
        ),
        wrt=nnx.Param,
    )
    actor_loss, actor_grads = actor_grad_fn(state.actor)
    state.actor_optimizer.apply_gradient(actor_grads)

    # Critic update
    critic_grad_fn = nnx.value_and_grad(
        lambda critic: ppo_critic_loss_fn(
            critic, observations=norm_obs, returns=returns, value_coef=value_coef
        ),
        wrt=nnx.Param,
    )
    critic_loss, critic_grads = critic_grad_fn(state.critic)
    state.critic_optimizer.apply_gradient(critic_grads)

    # 5. Return the new, updated state object
    return (
        TrainState(
            actor=state.actor,
            critic=state.critic,
            actor_optimizer=state.actor_optimizer,
            target_actor=state.target_actor,
            target_critic=state.target_critic,
            critic_optimizer=state.critic_optimizer,
            buffer_state=state.buffer_state,
            obs_stats=state.obs_stats,
        ),
        actor_loss,
        critic_loss,
    )


class PPO(Agent):
    """Proximal Policy Optimization agent."""

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
        gamma: float = 0.99,
        learning_steps: int = 5,
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
            rngs=actor_rngs,
        )

        # Instantiate critic
        critic = hydra.utils.instantiate(
            critic_config, in_features=env_obs_size + env_action_size, rngs=critic_rngs
        )

        # Instantiate replay buffer
        prototype = Transition(
            observation=jnp.zeros(env_obs_size, dtype=jnp.float32),
            action=jnp.zeros(env_action_size, dtype=jnp.float32),
            reward=jnp.zeros((), dtype=jnp.float32),
            terminal=jnp.zeros((), dtype=jnp.bool_),
            log_probs=jnp.zeros((), dtype=jnp.float32),
        )
        replay = hydra.utils.instantiate(memory_config)
        self.add_sequence_length = memory_config.add_sequence_length

        buffer_state = replay.init(prototype)

        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate = actor_learning_rate
        self.max_grad_norm = max_grad_norm

        # Add gradient clipping to optimizers
        actor_optimizer = nnx.Optimizer(
            actor,
            optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.adam(self.actor_learning_rate),
            ),
            wrt=nnx.Param,
        )

        critic_optimizer = nnx.Optimizer(
            critic,
            optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.adam(self.critic_learning_rate),
            ),
            wrt=nnx.Param,
        )

        # Init observation stats from buffer state's observation shape
        obs_shape = buffer_state.experience.observation.shape[
            -1
        ]  # Exclude batch dimension
        obs_stats = Agent.init_obs_stats(obs_shape)

        self.state = TrainState(
            actor=actor,
            critic=critic,
            target_actor=None,
            target_critic=None,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            buffer_state=buffer_state,
            obs_stats=obs_stats,
        )

        # Store hyperparameters
        self.gamma = gamma
        self.action_low = action_low
        self.action_high = action_high
        self.replay = replay
        self.learning_steps = learning_steps
        self.normalize_observations = normalize_observations
        self.obs_clip = float(obs_norm_clip)
        self.obs_eps = float(obs_norm_eps)

        print("PPO agent initialized.")

    def step(
        self,
        observation: jnp.ndarray,
        evaluate: bool = False,
        key: jax.random.PRNGKey = None,
    ):
        """
        Selects an action by calling the pure, JIT-compiled step function.
        """
        # Call the standalone function, passing in the required parts from the agent's state.
        if self.normalize_observations:
            mean, std = Agent.obs_mean_std(self.state.obs_stats, self.obs_eps)
            observation = Agent.normalize_obs(observation, mean, std, self.obs_clip)

        action, self.last_log_prob = Agent.stochastic_step_fn(
            self.state.actor,
            observation,
            key,
        )

        self.last_action = Agent.scale_to_env(action, self.action_low, self.action_high)

        return self.last_action

    def update(self, prev_states, states, steps, agent_rng):
        experiences = Transition(
            observation=prev_states.obs,
            action=self.last_action,
            reward=states.reward,
            terminal=states.done,
            log_probs=self.last_log_prob,
        )
        # store in memory
        self.state.buffer_state = self.replay.add(
            self.state.buffer_state, experiences
        )

        # Update observation normalization stats with both current and next observations
        if self.normalize_observations:
            obs_batch = jnp.concatenate([prev_states.obs, states.obs], axis=0)
            self.state.obs_stats = Agent.update_obs_stats(
                self.state.obs_stats, obs_batch
            )

        gradient_steps, actor_loss, critic_loss = 0, 0, 0

        # The buffer is full, so we can start learning
        if self.state.buffer_state == self.replay.capacity:
            for _ in range(self.learning_steps):
                agent_rng, key = jax.random.split(agent_rng, 2)

                self.state, actor_loss, critic_loss = _grad_step(
                    self.state,
                    key,
                    self.gamma,
                    self.replay.get_recent_window,  # Pass the sample method itself,
                    self.action_low,
                    self.action_high,
                    self.obs_eps,
                    self.obs_clip,
                )
            gradient_steps += self.learning_steps

            # empty the buffer after learning
            self.state.buffer_state = self.replay.flush(self.state.buffer_state)

    def _export_hyperparams(self) -> dict:
        # Keep this minimal and JSON-serializable
        return {
            "gamma": float(self.gamma),
            "actor_learning_rate": float(self.actor_learning_rate),
            "critic_learning_rate": float(self.critic_learning_rate),
            "max_grad_norm": float(self.max_grad_norm),
            "learning_steps": int(self.learning_steps),
            "memory_capacity": int(self.replay.capacity),
            "memory_batch_size": int(self.replay.batch_size),
            "normalize_observations": bool(self.normalize_observations),
            "obs_norm_clip": float(self.obs_clip),
            "obs_norm_eps": float(self.obs_eps),
            # bounds can be scalar or arrays → use your helpers
            "action_low": serialize_bound(self.action_low),
            "action_high": serialize_bound(self.action_high),
        }
