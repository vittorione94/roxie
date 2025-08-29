from myojit.agents.agent import Agent
import jax.numpy as jnp
from flax import nnx
import hydra
from myojit.replays.buffer import Transition
import optax
from myojit.agents.agent import Agent, TrainState 
from myojit.agents.utils import serialize_bound
import jax
from myojit.agents.ppo import ppo_loss_fn, ppo_critic_loss_fn


class PPO(Agent):
    '''Proximal Policy Optimization agent.'''

    def __init__(self, 
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
                 obs_norm_eps: float = 1e-8,):
        
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
            next_observation=jnp.zeros(env_obs_size, dtype=jnp.float32),
            terminal=jnp.zeros((), dtype=jnp.bool_)
        )
        replay = hydra.utils.instantiate(memory_config)
        buffer_state = replay.init(prototype)

        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate = actor_learning_rate
        self.max_grad_norm = max_grad_norm

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
        data = None
        if buffer_state is not None:
            if hasattr(buffer_state, "data"):
                data = buffer_state.data
            elif isinstance(buffer_state, dict):
                data = buffer_state.get("data", buffer_state)

        if data is None:
            # Fallback: infer from the replay instance (adjust to your replay API)
            obs_src = getattr(replay, "data", {}).get("observation", None) if isinstance(getattr(replay, "data", {}), dict) else None
            if obs_src is None:
                raise ValueError("Could not infer observation shape from buffer_state or replay.")
            obs_shape = tuple(obs_src.shape[1:])
        else:
            obs = data["observation"] if isinstance(data, dict) else data.observation
            obs_shape = tuple(obs.shape[1:])

        obs_stats = Agent.init_obs_stats(obs_shape)

        self.state = TrainState(
            actor=actor,
            critic=critic,
            target_actor=None,
            target_critic=None,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            buffer_state=buffer_state,
            obs_stats=obs_stats
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


    def step(self, observation: jnp.ndarray, evaluate: bool = False, key: jax.random.PRNGKey = None):
        """
        Selects an action by calling the pure, JIT-compiled step function.
        """
        # Call the standalone function, passing in the required parts from the agent's state.
        if self.normalize_observations:
            mean, std = Agent.obs_mean_std(self.state.obs_stats, self.obs_eps)
            observation = Agent.normalize_obs(observation, mean, std, self.obs_clip)
        
        action, log_prob = Agent.stochastic_step_fn(
            self.state.actor,
            observation,
            key,
        )
        
        return Agent.scale_to_env(action, self.action_low, self.action_high)

    def update(self, prev_states, states, steps, agent_rng, actions):
        experiences = Transition(
            observation=prev_states.obs,
            action=actions,
            reward=states.reward,
            next_observation=states.obs,
            terminal=states.done,
        )
        # store in memory
        self.state.buffer_state = self.replay.add_batch(self.state.buffer_state, experiences)

        # Update observation normalization stats with both current and next observations
        if self.normalize_observations:
            obs_batch = jnp.concatenate([prev_states.obs, states.obs], axis=0)
            self.state.obs_stats = Agent.update_obs_stats(self.state.obs_stats, obs_batch)


        gradient_steps, actor_loss, critic_loss = 0, 0, 0

        # The buffer is full, so we can start learning
        if self.state.buffer_state == self.replay.capacity :
            for _ in range(self.learning_steps):
                agent_rng, key = jax.random.split(agent_rng, 2)

                self.state, actor_loss, critic_loss = Agent._grad_step(
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
                    self.obs_clip,
                    actor_loss_fn=ppo_loss_fn,
                    critic_loss_fn=ppo_critic_loss_fn
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
