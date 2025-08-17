import abc
import jax
import jax.numpy as jnp
from flax import nnx
from typing import Any
import flax.struct as struct
import functools

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


class Agent(abc.ABC):
    '''Abstract class used to build agents.'''

    def initialize(self, observation_space, action_space, seed=None):
        pass
    
    @staticmethod
    @jax.jit
    def scale_to_env(x: jnp.ndarray, low: jnp.ndarray, high: jnp.ndarray):
        # x in [-1, 1] -> [low, high]
        return low + 0.5 * (x + 1.0) * (high - low)

    @staticmethod
    @functools.partial(nnx.jit, static_argnames=('evaluate',))
    def step_fn(
        actor_model: nnx.Module,
        observation: jnp.ndarray,
        key: jax.Array,
        exploration_noise: float,
        action_low: jnp.ndarray,
        action_high: jnp.ndarray,
        evaluate: bool = False,
    ):
        """
        Pure action selection for any actor-critic agent.
        - actor outputs actions in [-1, 1]
        - scale to [action_low, action_high]
        - add exploration noise in env units when not evaluating
        """
        action = actor_model(observation)  # [-1, 1]
        action = Agent.scale_to_env(action, action_low, action_high)

        noise = jax.random.normal(key, action.shape)
        noise_scale = jax.lax.cond(evaluate, lambda: 0.0, lambda: exploration_noise)
        act_span = (action_high - action_low)
        _n = noise * noise_scale * act_span

        noisy_action = jnp.clip(action + _n, action_low, action_high)
        return noisy_action, _n

    @staticmethod
    def init_obs_stats(obs_shape) -> ObsStats:
        return ObsStats(
            count=jnp.array(0.0, dtype=jnp.float32),
            sum=jnp.zeros(obs_shape, dtype=jnp.float32),
            sumsq=jnp.zeros(obs_shape, dtype=jnp.float32),
        )

    @staticmethod
    @jax.jit
    def update_obs_stats(stats: ObsStats, batch_obs: jnp.ndarray) -> ObsStats:
        # batch_obs: (B, *obs_shape)
        b = batch_obs.shape[0]
        batch_sum = jnp.sum(batch_obs, axis=0)
        batch_sumsq = jnp.sum(jnp.square(batch_obs), axis=0)
        return stats.replace(
            count=stats.count + b,
            sum=stats.sum + batch_sum,
            sumsq=stats.sumsq + batch_sumsq,
        )

    @staticmethod
    def obs_mean_std(stats: ObsStats, eps: float):
        count = jnp.maximum(stats.count, 1.0)
        mean = stats.sum / count
        var = jnp.maximum(stats.sumsq / count - jnp.square(mean), 0.0)
        std = jnp.sqrt(var + eps)
        return mean, std

    @staticmethod
    @jax.jit
    def normalize_obs(x: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray, clip: float):
        return jnp.clip((x - mean) / std, -clip, clip)

    def update(self, observations, rewards, resets, terminations, steps):
        '''Informs the agent of the latest transitions during training.'''
        pass

    def test_update(self, observations, rewards, resets, terminations, steps):
        '''Informs the agent of the latest transitions during testing.'''
        pass

    def save(self, path):
        '''Saves the agent weights during training.'''
        pass

    def load(self, path):
        '''Reloads the agent weights from a checkpoint.'''
        pass