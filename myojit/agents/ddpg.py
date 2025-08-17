from myojit.agents import agent
import jax
import jax.numpy as jnp
import functools
from myojit.replays.buffer import Transition, JaxReplayBuffer
from flax import nnx
import orbax.checkpoint as ocp
from pathlib import Path
from typing import Union
import optax
import copy
from typing import Any
import flax.struct as struct


class TrainState(nnx.Module):
  def __init__(
      self,
      *,
      actor: nnx.Module,
      critic: nnx.Module,
      target_actor: nnx.Module,
      target_critic: nnx.Module,
      actor_optimizer: nnx.Optimizer,
      critic_optimizer: nnx.Optimizer,
      buffer_state: Any,
      obs_stats: Any
  ):
    """Initializes the training state.

    The state components are defined as attributes of the module.
    `nnx.Module` will automatically know how to handle them for
    JAX transformations.
    """
    self.actor = actor
    self.critic = critic
    self.target_actor = target_actor
    self.target_critic = target_critic

    self.actor_optimizer = actor_optimizer
    self.critic_optimizer = critic_optimizer

    self.buffer_state = buffer_state
    self.obs_stats = obs_stats  # Stores observation statistics for normalization

@struct.dataclass
class ObsStats:
    count: jnp.ndarray      # shape: ()
    sum: jnp.ndarray        # shape: obs_shape
    sumsq: jnp.ndarray      # shape: obs_shape


def _init_obs_stats(obs_shape) -> ObsStats:
    return ObsStats(
        count=jnp.array(0.0, dtype=jnp.float32),
        sum=jnp.zeros(obs_shape, dtype=jnp.float32),
        sumsq=jnp.zeros(obs_shape, dtype=jnp.float32),
    )

def _update_obs_stats(stats: ObsStats, batch_obs: jnp.ndarray) -> ObsStats:
    # batch_obs: (B, *obs_shape)
    b = batch_obs.shape[0]
    batch_sum = jnp.sum(batch_obs, axis=0)
    batch_sumsq = jnp.sum(jnp.square(batch_obs), axis=0)
    return stats.replace(
        count=stats.count + b,
        sum=stats.sum + batch_sum,
        sumsq=stats.sumsq + batch_sumsq,
    )

def _obs_mean_std(stats: ObsStats, eps: float):
    # If no data yet, return zeros and ones to effectively skip normalization.
    def compute():
        mean = stats.sum / stats.count
        var = stats.sumsq / stats.count - jnp.square(mean)
        std = jnp.sqrt(jnp.maximum(var, 0.0))  # avoid negatives due to numerics
        std = jnp.maximum(std, eps)
        return mean, std
    def skip():
        return jnp.zeros_like(stats.sum), jnp.ones_like(stats.sum)
    return jax.lax.cond(stats.count > 0.0, compute, skip)

def _normalize_obs(x: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray, clip: float):
    return jnp.clip((x - mean) / std, -clip, clip)

def _scale_to_env(x: jnp.ndarray, low: jnp.ndarray, high: jnp.ndarray):
    # x in [-1, 1] -> [low, high]
    return low + 0.5 * (x + 1.0) * (high - low)


def _critic_loss_fn(critic_model, target_actor_model, target_critic_model, samples, \
                    gamma, noise_key, target_policy_noise, target_noise_clip,  action_low, action_high, \
                    obs_mean, obs_std, obs_clip):
    """Calculates the MSE loss for the critic."""
    # Normalize observations
    obs = _normalize_obs(samples['observations'], obs_mean, obs_std, obs_clip)
    next_obs = _normalize_obs(samples['next_observations'], obs_mean, obs_std, obs_clip)

    # Target actions in env scale
    next_actions = target_actor_model(next_obs)                 # [-1, 1]
    next_actions = _scale_to_env(next_actions, action_low, action_high)

    # Target smoothing noise in env units
    act_span = (action_high - action_low)
    noise = jax.random.normal(noise_key, next_actions.shape) * (target_policy_noise * act_span)
    noise_clip = target_noise_clip * act_span
    noise = jnp.clip(noise, -noise_clip, noise_clip)

    next_actions = jnp.clip(next_actions + noise, action_low, action_high)

    next_q = target_critic_model(next_obs, next_actions)
    
    term = samples['terminals'].astype(jnp.float32)
    reward = jnp.squeeze(samples['rewards'])
    next_q = jnp.squeeze(next_q)
    target_q = reward + gamma * (1.0 - term) * next_q
    target_q = jax.lax.stop_gradient(target_q)

    current_q = critic_model(obs, samples['actions'])
    
    critic_loss = jnp.mean((jnp.squeeze(current_q) - target_q)**2)
    return critic_loss

def _actor_loss_fn(actor_model, critic_model, samples, obs_mean, obs_std, obs_clip, action_low, action_high):
    """Calculates the loss for the actor (aims to maximize Q-value)."""
    obs = _normalize_obs(samples['observations'], obs_mean, obs_std, obs_clip)
    actions = actor_model(obs)                                  # [-1, 1]
    actions = _scale_to_env(actions, action_low, action_high)   # [low, high]
    q_values = critic_model(obs, actions)
    actor_loss = -jnp.mean(q_values)
    return actor_loss


# This is the core computational kernel that will be JIT-compiled.
@functools.partial(nnx.jit, static_argnames=('gamma', 'tau', 'replay_sample_fn'))
def _grad_step(state: TrainState, key: jax.random.PRNGKey, gamma: float, tau: float, replay_sample_fn, \
               exploration_noise: float, target_policy_noise: float, target_noise_clip: float, action_low: float, action_high: float,
               obs_eps: float, obs_clip: float):
    """Performs one full gradient update step and returns the new state."""
    # 1. Sample from the replay buffer
    key, noise_key = jax.random.split(key)
    samples = replay_sample_fn(state.buffer_state, key)

    # 1.1 Compute current normalization parameters
    obs_mean, obs_std = _obs_mean_std(state.obs_stats, obs_eps)

    # 2. Critic update
    critic_loss, critic_grads = nnx.value_and_grad(_critic_loss_fn)(
        state.critic, state.target_actor, state.target_critic, samples, gamma, noise_key, \
            target_policy_noise, target_noise_clip, action_low, action_high, \
            obs_mean, obs_std, obs_clip
    )
    state.critic_optimizer.update(critic_grads)

    # 3. Actor update (use scaled actions)
    actor_loss, actor_grads = nnx.value_and_grad(_actor_loss_fn)(
        state.actor, state.critic, samples, obs_mean, obs_std, obs_clip, action_low, action_high
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


@functools.partial(nnx.jit, static_argnames=('evaluate',))
def _step_fn(
    actor_model: nnx.Module,
    observation: jnp.ndarray,
    key: jax.random.PRNGKey,
    exploration_noise: float,
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
    evaluate: bool = False,
):
    """
    A pure function to select an action.
    Its output depends only on its inputs.
    """
    # Deterministic action in [-1, 1]
    action = actor_model(observation)
    # Scale to env range
    action = _scale_to_env(action, action_low, action_high)

    # Generate noise (env units), disabled in eval
    noise = jax.random.normal(key, action.shape)
    noise_scale = jax.lax.cond(
        evaluate,
        lambda: 0.0,
        lambda: exploration_noise
    )
    act_span = (action_high - action_low)
    _n = noise * noise_scale * act_span

    noisy_action = jnp.clip(action + _n, action_low, action_high)
    return noisy_action, _n

# @functools.partial(nnx.jit, static_argnames=('gamma', 'tau', 'replay_sample_fn', 'num_steps'))
# def _multiple_grad_steps_nnx(
#     state: TrainState, 
#     key: jax.random.PRNGKey, 
#     gamma: float, 
#     tau: float, 
#     replay_sample_fn,
#     num_steps: int
# ):
#     """Performs multiple gradient update steps using nnx.fori_loop."""
    
#     def single_step(i, carry):
#         current_state, current_key = carry
#         current_key, subkey = jax.random.split(current_key)
#         updated_state, _ = _grad_step(current_state, subkey, gamma, tau, replay_sample_fn)
#         return updated_state, current_key
    
#     final_state, _ = nnx.fori_loop(0, num_steps, single_step, (state, key))
#     return final_state

class DDPG(agent.Agent):
    def __init__(self, 
                 actor: nnx.Module, 
                 critic: nnx.Module, 
                 replay: JaxReplayBuffer, 
                 buffer_state: 'BufferState', 
                 *,
                 actor_learning_rate: float = 3e-4,
                 critic_learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                #  exploration_noise: float = 0.1,
                 init_noise: float = 0.1,
                 min_noise: float = 0.01,
                 noise_decay_transitions: int = 100000,
                 steps_before_learning: int = 100,
                 steps_between_updates: int = 10,
                 learning_steps: int = 5,
                 memory_warmup: int = 100,
                 target_noise_clip: float = 0.1,
                 target_policy_noise: float = 0.1,
                 max_grad_norm: float = 1.0,
                 action_low: float = -1.0,
                 action_high: float = 1.0,
                 normalize_observations: bool = True,
                 obs_norm_clip: float = 5.0,
                 obs_norm_eps: float = 1e-8,
                 ):

        # create targets
        target_actor = copy.deepcopy(actor)
        target_critic = copy.deepcopy(critic)
        
        # Add gradient clipping to optimizers
        actor_optimizer = nnx.Optimizer(
            actor, 
            optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(actor_learning_rate)
            ), 
            wrt=nnx.Param
        )

        critic_optimizer = nnx.Optimizer(
            critic, 
            optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(critic_learning_rate)
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

        obs_stats = _init_obs_stats(obs_shape)

        self.state = TrainState(
            actor=actor,
            critic=critic,
            target_actor=target_actor,
            target_critic=target_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            buffer_state=buffer_state, # Assuming buffer has an init method
            obs_stats=obs_stats
        )

        # --- Store hyperparameters ---
        self.gamma = gamma
        self.tau = tau
        self.exploration_noise = init_noise
        self.init_noise = init_noise
        self.noise_decay_transitions = noise_decay_transitions
        self.min_noise = min_noise
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
        print("Params: \n" \
        f"   gamma {self.gamma} \n" \
        f"   tau {self.tau}\n" \
        f"   exploration_noise {self.exploration_noise}\n" \
        f"   steps_before_learning {self.steps_before_learning}\n" \
        f"   steps_between_updates {self.steps_between_updates}\n" \
        f"   learning_steps {self.learning_steps}\n" \
        f"   memory_warmup {self.memory_warmup}\n" \
        f"   target_noise_clip {self.target_noise_clip}\n" \
        f"   action_low {self.action_low}\n" \
        f"   action_high {self.action_high}\n" \
        f"   max_grad_norm {max_grad_norm}\n" \
        f"   actor_learning_rate {actor_learning_rate}\n" \
        f"   critic_learning_rate {critic_learning_rate}\n" \
        f"   target_policy_noise {target_policy_noise}\n" )


    def step(self, observation: jnp.ndarray, evaluate: bool = False, key: jax.random.PRNGKey = None) -> jnp.ndarray:
        """
        Selects an action by calling the pure, JIT-compiled step function.
        """
        # Call the standalone function, passing in the required parts from the agent's state.
        if self.normalize_observations:
            mean, std = _obs_mean_std(self.state.obs_stats, self.obs_eps)
            observation = _normalize_obs(observation, mean, std, self.obs_clip)
        
        action, noise = _step_fn(
            self.state.actor,
            observation,
            key,
            self.exploration_noise,
            self.action_low,
            self.action_high,
            evaluate,
        )
        
        return action
    
    def _decay_noise(self, steps):
        f = min(1.0, steps / self.noise_decay_transitions)
        self.exploration_noise = float(self.init_noise - (self.init_noise - self.min_noise) * f)


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
            self.state.obs_stats = _update_obs_stats(self.state.obs_stats, obs_batch)


        gradient_steps, actor_loss, critic_loss = 0, 0, 0

        self._decay_noise(steps)

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
                    self.exploration_noise,
                    self.target_policy_noise,  # Use target_policy_noise
                    self.target_noise_clip,
                    self.action_low,
                    self.action_high,
                    self.obs_eps,
                    self.obs_clip
                )
            gradient_steps += self.learning_steps
                # Perform multiple gradient steps in a single JIT-compiled call
            # self.state = _multiple_grad_steps_nnx(
            #     self.state, 
            #     key, 
            #     self.gamma, 
            #     self.tau,
            #     self.replay.sample,
            #     self.learning_steps
            # )


        return gradient_steps, actor_loss, critic_loss


    def save(self, path: Union[str, Path]):
        # Ensure the path exists
        path = Path(path).resolve()

        # Extract parameters and move to CPU
        actor_graphdef, actor_state = nnx.split(self.state.actor)
        critic_graphdef, critic_state = nnx.split(self.state.critic)
        _, target_actor_state = nnx.split(self.state.target_actor)
        _, target_critic_state = nnx.split(self.state.target_critic)

        save_data = {
            # Save graphdefs to avoid merge mismatches at load time
            'actor_graphdef': actor_graphdef,
            'critic_graphdef': critic_graphdef,

            'actor': jax.device_get(actor_state),
            'critic': jax.device_get(critic_state),
            'target_actor': jax.device_get(target_actor_state),
            'target_critic': jax.device_get(target_critic_state),
            'buffer': jax.device_get(self.state.buffer_state),
            'obs_stats': jax.device_get(self.state.obs_stats),
            'hyperparams': {
                'gamma': self.gamma,
                'tau': self.tau,
                'exploration_noise': self.exploration_noise,
                'target_noise_clip': self.target_noise_clip,
                'target_policy_noise': self.target_policy_noise,
                'action_low': self.action_low,
                'action_high': self.action_high,
                'normalize_observations': self.normalize_observations,
                'obs_norm_clip': self.obs_clip,
                'obs_norm_eps': self.obs_eps,
            }
        }

        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(path, save_data)
        print(f"Agent state saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path], actor: nnx.Module, critic: nnx.Module, replay: JaxReplayBuffer):
        """
        Loads the agent's state from a directory.
        """
        path = Path(path).resolve()
        checkpointer = ocp.PyTreeCheckpointer()
        loaded = checkpointer.restore(path)

        # Prefer saved graphdefs if present; fallback to provided placeholders.
        saved_actor_gd = loaded.get('actor_graphdef', nnx.split(actor)[0])
        saved_critic_gd = loaded.get('critic_graphdef', nnx.split(critic)[0])

        actor = nnx.merge(saved_actor_gd, loaded['actor'])
        critic = nnx.merge(saved_critic_gd, loaded['critic'])
        target_actor = None
        target_critic = None
        if 'target_actor' in loaded:
            target_actor = nnx.merge(saved_actor_gd, loaded['target_actor'])
        if 'target_critic' in loaded:
            target_critic = nnx.merge(saved_critic_gd, loaded['target_critic'])

        buffer_state = loaded.get('buffer', None)
        obs_stats = loaded.get('obs_stats', None)
        hyper = loaded.get('hyperparams', {})

        # Build agent with restored hyperparams when available
        agent = cls(
            actor=actor,
            critic=critic,
            replay=replay,
            buffer_state=buffer_state,
            gamma=float(hyper.get('gamma', 0.99)),
            tau=float(hyper.get('tau', 0.005)),
            action_low=jnp.array(hyper.get('action_low', -1.0)),
            action_high=jnp.array(hyper.get('action_high', 1.0)),
            normalize_observations=bool(hyper.get('normalize_observations', True)),
            obs_norm_clip=float(hyper.get('obs_norm_clip', 5.0)),
            obs_norm_eps=float(hyper.get('obs_norm_eps', 1e-8)),
            target_noise_clip=float(hyper.get('target_noise_clip', 0.1)),
            target_policy_noise=float(hyper.get('target_policy_noise', 0.1)),
        )

        # Restore targets and obs stats if present
        if target_actor is not None:
            agent.state.target_actor = target_actor
        if target_critic is not None:
            agent.state.target_critic = target_critic
        if obs_stats is not None:
            agent.state.obs_stats = obs_stats
        if 'exploration_noise' in hyper:
            agent.exploration_noise = float(hyper['exploration_noise'])

        print(f"Agent state loaded from {path}")
        return agent
