import abc
import jax
import jax.numpy as jnp
from flax import nnx
import flax.struct as struct
import functools
import orbax.checkpoint as ocp
from typing import Any, Dict, Iterable, Optional
from pathlib import Path
import numpy as np
import cloudpickle

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

    # --------------------------
    # Checkpointing (generic)
    # --------------------------
    @abc.abstractmethod
    def _export_hyperparams(self) -> Dict[str, Any]:
        return {}

    def save(
        self,
        path: str | Path,
        *,
        format_version: int = 1,  # bump format
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        if not hasattr(self, "state"):
            raise AttributeError("Agent must define `self.state` (an nnx.Module).")

        path = Path(path).resolve()
        graphdef, state_tree = nnx.split(self.state)

        payload = {
            "format_version": format_version,
            "trainstate_graphdef": graphdef,     # serialized topology
            "trainstate_state": jax.device_get(state_tree),  # numeric pytree
            "hyperparams": self._export_hyperparams(),
            "metadata": (extra_metadata or {}),
        }
        ocp.PyTreeCheckpointer().save(path, payload)
        print(f"[Agent.save] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path, actor, critic, replay):
        path = Path(path).resolve()
        loaded = ocp.PyTreeCheckpointer().restore(path)

        ckpt_state = loaded["trainstate_state"]
        hyper = loaded.get("hyperparams", {})

        # (Optional) carry over buffer_state for shape continuity
        buffer_state = ckpt_state.get("buffer_state", None) if isinstance(ckpt_state, dict) else None

        agent = cls(actor=actor, critic=critic, replay=replay, buffer_state=buffer_state, **(hyper or {}))

        # Minimal fields commonly used at play time
        hyper = ckpt_state.get("hyperparams", {}) or {}
        agent.exploration_noise = float(hyper.get("exploration_noise", 0.0))

        # Helper: convert numpy -> jax arrays
        import numpy as _np
        def _to_jax(x):
            return jnp.asarray(x) if isinstance(x, _np.ndarray) else x

        # 2) Merge submodules individually
        def _restore_submodule(name: str):
            if not (isinstance(ckpt_state, dict) and name in ckpt_state):
                print(f"Info: '{name}' not in checkpoint; keeping live {name}.")
                return
            sub_ckpt = jax.tree.map(_to_jax, ckpt_state[name], is_leaf=lambda x: isinstance(x, _np.ndarray))
            sub_live = getattr(agent.state, name)
            gdef, _ = nnx.split(sub_live)
            try:
                restored = nnx.merge(gdef, sub_ckpt)
                setattr(agent.state, name, restored)
            except ValueError as e:
                # Architecture drift or partial state → fall back to replacing just params where possible
                print(f"Warning: merge({name}) failed ({e}). Falling back to param-only copy.")
                try:
                    dst_params = nnx.state(sub_live, nnx.Param)
                    src_params = sub_ckpt.get('params', None) if isinstance(sub_ckpt, dict) else None
                    if src_params is None:
                        print(f"Warning: no 'params' found for {name}; skipping.")
                    else:
                        # Only update params; let NNX map into its own topology.
                        nnx.update(sub_live, {'params': src_params})
                except Exception as ee:
                    print(f"Warning: param-only update for {name} failed ({ee}). Skipping.")

        for name in ("actor", "critic", "target_actor", "target_critic"):
            _restore_submodule(name)

        # 3) Restore simple values directly (replace whole object; don't partial-update)
        if isinstance(ckpt_state, dict) and "obs_stats" in ckpt_state:
            try:
                obs = ckpt_state["obs_stats"]
                # obs may be dict-like; normalize to ObsStats dataclass
                if isinstance(obs, dict):
                    agent.state.obs_stats = ObsStats(
                        count=jnp.asarray(obs["count"]),
                        sum=jnp.asarray(obs["sum"]),
                        sumsq=jnp.asarray(obs["sumsq"]),
                    )
                else:
                    # If it was saved as a struct-compatible tree, just assign it.
                    agent.state.obs_stats = jax.tree_map(_to_jax, obs, is_leaf=lambda x: isinstance(x, _np.ndarray))
            except Exception as e:
                print(f"Warning: could not restore obs_stats ({e}); using live stats.")

        # (Optional) if you really want buffer snapshot:
        if isinstance(ckpt_state, dict) and "buffer_state" in ckpt_state:
            agent.state.buffer_state = ckpt_state["buffer_state"]

        # 4) DO NOT restore optimizer internals: they’re brittle; let them be reinitialized.

        if "exploration_noise" in hyper:
            agent.exploration_noise = float(hyper["exploration_noise"])

        print(f"Agent state loaded from {path}")
        return agent