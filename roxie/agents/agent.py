import abc
import functools
import inspect
from pathlib import Path
from typing import Any, Dict, Optional

import flax.struct as struct
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

# `roxie.models.actors` imports nothing from `roxie`, so this cannot cycle.
from roxie.models.actors import distribution_entropy as _distribution_entropy


class TrainState(nnx.Module, pytree=False):
    def __init__(
        self,
        *,
        actor: nnx.Module,
        critic: nnx.Module,
        target_actor: Optional[nnx.Module],
        target_critic: Optional[nnx.Module],
        actor_optimizer: nnx.Optimizer,
        critic_optimizer: nnx.Optimizer,
        buffer_state: Any,
        obs_stats: Any,
    ):
        self.actor = actor
        self.critic = critic
        self.target_actor = target_actor
        self.target_critic = target_critic

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.buffer_state = buffer_state
        self.obs_stats = obs_stats


@struct.dataclass
class ObsStats:
    count: jnp.ndarray  # shape: ()
    sum: jnp.ndarray  # shape: obs_shape
    sumsq: jnp.ndarray  # shape: obs_shape


class Agent(abc.ABC):
    """Abstract class used to build agents."""

    @staticmethod
    @jax.jit
    def scale_to_env(x: jnp.ndarray, low: jnp.ndarray, high: jnp.ndarray):
        # x in [-1, 1] -> [low, high]
        return low + 0.5 * (x + 1.0) * (high - low)

    @staticmethod
    @functools.partial(nnx.jit, static_argnames=("evaluate",))
    def deterministic_step_fn(
        actor_model: nnx.Module,
        observation: jnp.ndarray,
        key: jax.Array,
        noise_module: nnx.Module,
        evaluate: bool = False,
    ):
        """
        Pure action selection for any actor-critic agent.
        - actor outputs actions in [-1, 1]
        - add exploration noise in env units when not evaluating
        """
        action = actor_model(observation)

        noisy_action = noise_module.add_noise(action, key, evaluate)
        noisy_action = jnp.clip(noisy_action, -1.0, 1.0)
        return noisy_action, action - noisy_action  # the noise, for logging

    @staticmethod
    @functools.partial(nnx.jit, static_argnames=("evaluate",))
    def stochastic_step_fn(
        actor_model: nnx.Module,
        observation: jnp.ndarray,
        evaluate: bool,
        key: jax.Array,
    ):
        """
        Pure action selection for any actor-critic agent.
        - actor outputs actions in [-1, 1]
        - add exploration noise in env units when not evaluating
        """
        distribution = actor_model(observation)

        if evaluate:
            # The distribution mean, never a sample, so evaluation is deterministic.
            try:
                action = distribution.mean()
            except TypeError:
                # Some distrax versions expose mean as a property.
                action = distribution.mean

            log_probs = distribution.log_prob(action)
            entropy = _distribution_entropy(distribution, key)
            return action, log_probs, entropy

        action, log_probs = distribution.sample_and_log_prob(seed=key)
        entropy = _distribution_entropy(distribution, key)
        return action, log_probs, entropy

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
        # With 0 or 1 samples the variance is identically 0, so `sqrt(var + eps)` is
        # tiny and dividing by it saturates every feature at the clip bound. Fall
        # back to the identity scale until there is a meaningful spread.
        std = jnp.where(stats.count > 1.0, jnp.sqrt(var + eps), 1.0)
        return mean, std

    @staticmethod
    @jax.jit
    def normalize_obs(x: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray, clip: float):
        return jnp.clip((x - mean) / std, -clip, clip)

    @staticmethod
    def normalize_samples(
        samples: dict,
        mean: jnp.ndarray,
        std: jnp.ndarray,
        clip: float,
        enabled: bool = True,
    ) -> dict:
        """Normalize the observation entries of a repacked sample dict.

        This is the contract every loss function relies on: losses are handed
        observations that have ALREADY been normalized and therefore take no
        `obs_mean` / `obs_std` / `obs_clip` arguments of their own (the
        on-policy path does the same once per rollout in `PPO._prepare_rollout`).
        Normalizing here also does it once per gradient step rather than once
        per loss — the actor and critic losses read the same `observations`.

        `enabled` is the agent's `normalize_observations` flag and is static at
        trace time. When it is False the samples pass through untouched, clip
        included: `obs_clip` bounds *normalized* observations, and the unused
        running stats degrade to mean 0 / std 1, so applying it anyway would
        silently squash raw observations into +/- `clip`.
        """
        if not enabled:
            return samples
        return {
            **samples,
            "observations": Agent.normalize_obs(
                samples["observations"], mean, std, clip
            ),
            "next_observations": Agent.normalize_obs(
                samples["next_observations"], mean, std, clip
            ),
        }

    # Highest update boundary already served. Class-level default so every
    # off-policy agent inherits it without touching its __init__; the first
    # firing shadows it with an instance attribute.
    _last_update_boundary = -1

    def due_for_update(self, steps: int) -> bool:
        """True at most once per `steps_between_updates` env steps past warmup.

        Do NOT write this as `(steps - steps_before_learning) % between == 0`.
        The trainer advances `steps` in strides of `parallel_envs` from a
        warmup-aligned start, so that test only ever fires if the OFFSET
        `steps_before_learning` is itself a multiple of the stride. It silently
        was not for the v1 release grid (30_000 % 256 == 48), the residue cycled
        208, 464, ... 2000 without ever reaching 0, and all six off-policy arms
        ran 5M env steps at exactly zero gradient steps.

        Tracking the last boundary served instead makes the schedule depend only
        on how many env steps have elapsed, not on whether the stride happens to
        divide the offset. A stride wider than `steps_between_updates` still
        collapses to one burst per trainer iteration (the schedule cannot run
        faster than it is called) — that is the pre-existing "rounds up to one"
        behaviour the bench configs warn about, and it is unchanged here.

        No backlog is queued: the boundary jumps to wherever `steps` now is, so
        a restored checkpoint resumes on schedule rather than firing a catch-up
        storm.
        """
        if steps < self.steps_before_learning:
            return False
        elapsed = steps - self.steps_before_learning
        boundary = self.steps_before_learning + (
            (elapsed // self.steps_between_updates) * self.steps_between_updates
        )
        if boundary <= self._last_update_boundary:
            return False
        self._last_update_boundary = boundary
        return True

    def update(self, old_states, new_states, steps, agent_rng):
        """Informs the agent of the latest transitions during training."""
        gradient_steps, actor_loss, critic_loss = 0, 0, 0
        return gradient_steps, actor_loss, critic_loss

    def test_update(self, observations, rewards, resets, terminations, steps):
        """Informs the agent of the latest transitions during testing."""
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
        format_version: int = 1,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        try:
            if not hasattr(self, "state"):
                raise AttributeError("Agent must define `self.state` (an nnx.Module).")

            path = Path(path).resolve()

            # The replay buffer dominates the state, and `device_get`'ing it to host
            # on every save spikes host RAM (orbax holds its own serialization copies
            # on top) hard enough to risk an OOM mid-write. It is not needed to
            # resume, so it is detached for the duration of the save and `load()`
            # treats it as optional. `_export_hyperparams` reads the buffer's
            # obs/action shapes, so it must be called *before* detaching.
            hyperparams = self._export_hyperparams()
            saved_buffer = getattr(self.state, "buffer_state", None)
            self.state.buffer_state = None
            try:
                graphdef, state_tree = nnx.split(self.state)

                payload = {
                    "format_version": format_version,
                    "trainstate_graphdef": graphdef,  # serialized topology
                    "trainstate_state": jax.device_get(state_tree),  # numeric pytree
                    "hyperparams": hyperparams,
                    "metadata": (extra_metadata or {}),
                }
                checkpointer = ocp.StandardCheckpointer()

                checkpointer.save(path, payload)
                # save() returns before the background write finishes. Block here so
                # the write completes while the checkpointer is still alive, rather
                # than being torn down mid-write at interpreter exit.
                checkpointer.wait_until_finished()
            finally:
                # Restore the live buffer so training continues uninterrupted.
                self.state.buffer_state = saved_buffer

            print(f"[Agent.save] Saved to {path}")
        except Exception as e:
            print(
                f"[Agent.save] Warning: could not save to {path} ({e}) \
                  Probably a basic agent without state."
            )

    @classmethod
    def load(
        cls,
        path: str | Path,
        env_obs_size: int,
        env_act_size: int,
        **config_blocks,
    ):
        """Rebuild an agent from a checkpoint.

        `config_blocks` are the yaml-side construction blocks — `actor_config`,
        `critic_config`, `memory_config`, `noise_config`, the `*_optimizer_config`
        blocks — forwarded verbatim from the run's agent config. They are taken
        as keywords rather than a fixed positional list so an agent that grows a
        new block (or drops one it never had, like SAC and `noise_config`) needs
        no change here; `play.py` passes whichever blocks the config declares.
        Everything else comes from the checkpoint's `hyperparams`.
        """
        path = Path(path).resolve()
        # Restore weights to host memory (numpy) rather than onto the device sharding
        # baked into the checkpoint: a GPU-trained run pins arrays to cuda:0, which a
        # CPU-only process could not load. Host arrays are placement-agnostic, and the
        # numpy->jax conversion below puts them on whatever device is active.
        checkpointer = ocp.PyTreeCheckpointer()
        restore_args = jax.tree.map(
            lambda _a: ocp.RestoreArgs(restore_type=np.ndarray),
            ocp.checkpoint_utils.construct_restore_args(
                checkpointer.metadata(path).item_metadata
            ),
            is_leaf=lambda a: isinstance(a, ocp.RestoreArgs),
        )
        loaded = checkpointer.restore(path, restore_args=restore_args)

        ckpt_state = loaded["trainstate_state"]
        hyper = loaded.get("hyperparams", {})
        print(hyper)

        # Collect accepted __init__ params across the whole MRO. Subclasses like
        # TD3 forward via ``*args, **kwargs``, so inspecting only ``cls.__init__``
        # would miss the parent's hyperparams (action_low/high, gamma, tau, ...)
        # and silently drop them from ``filtered_hyper``.
        valid_params = set()
        for klass in cls.__mro__:
            init = klass.__dict__.get("__init__")
            if init is not None:
                valid_params |= set(inspect.signature(init).parameters.keys())
        # Keys we set explicitly below must not also come from the checkpoint,
        # or cls(**...) would receive duplicate keyword arguments.
        explicit_keys = {"env_obs_size", "env_action_size", *config_blocks}
        filtered_hyper = {
            k: v
            for k, v in (hyper or {}).items()
            if k in valid_params and k not in explicit_keys
        }
        # Action bounds were serialized to a plain list (one entry per actuator,
        # so an env with differing ranges is not recorded as just the first
        # one's). Rebuild the array the agent had at train time rather than
        # handing the constructor a list of 0-d arrays: everything downstream —
        # scale_to_env, the noise clip, the actor's output bounds — is written
        # against an ndarray of shape (action_dim,).
        for key in ("action_low", "action_high"):
            if key in filtered_hyper:
                filtered_hyper[key] = jnp.asarray(
                    filtered_hyper[key], dtype=jnp.float32
                )

        init_kwargs = dict(
            env_obs_size=env_obs_size,
            env_action_size=env_act_size,
            # A block the config omits is dropped rather than passed as None, so
            # the constructor's own default applies.
            **{k: v for k, v in config_blocks.items() if v is not None},
            **(filtered_hyper or {}),
        )
        agent = cls(**init_kwargs)

        hyper = ckpt_state.get("hyperparams", {}) or {}
        agent.exploration_noise = float(hyper.get("exploration_noise", 0.0))

        import numpy as _np

        def _to_jax(x):
            return jnp.asarray(x) if isinstance(x, _np.ndarray) else x

        def _restore_submodule(name: str):
            if not (isinstance(ckpt_state, dict) and name in ckpt_state):
                print(f"Info: '{name}' not in checkpoint; keeping live {name}.")
                return
            sub_ckpt = jax.tree.map(
                _to_jax, ckpt_state[name], is_leaf=lambda x: isinstance(x, _np.ndarray)
            )
            sub_live = getattr(agent.state, name)
            gdef, _ = nnx.split(sub_live)
            try:
                restored = nnx.merge(gdef, sub_ckpt)
                setattr(agent.state, name, restored)
            except ValueError as e:
                # Architecture drift or partial state: fall back to params only.
                print(
                    f"Warning: merge({name}) failed ({e}). Falling back to param-only copy."
                )
                try:
                    dst_params = nnx.state(sub_live, nnx.Param)
                    src_params = (
                        sub_ckpt.get("params", None)
                        if isinstance(sub_ckpt, dict)
                        else None
                    )
                    if src_params is None:
                        print(f"Warning: no 'params' found for {name}; skipping.")
                    else:
                        # Params only; NNX maps them into its own topology.
                        nnx.update(sub_live, {"params": src_params})
                except Exception as ee:
                    print(
                        f"Warning: param-only update for {name} failed ({ee}). Skipping."
                    )

        for name in ("actor", "critic", "target_actor", "target_critic"):
            _restore_submodule(name)

        # Replaced wholesale rather than partial-updated.
        if isinstance(ckpt_state, dict) and "obs_stats" in ckpt_state:
            try:
                obs = ckpt_state["obs_stats"]
                if isinstance(obs, dict):
                    agent.state.obs_stats = ObsStats(
                        count=jnp.asarray(obs["count"]),
                        sum=jnp.asarray(obs["sum"]),
                        sumsq=jnp.asarray(obs["sumsq"]),
                    )
                else:
                    # Already a struct-compatible tree.
                    agent.state.obs_stats = jax.tree_map(
                        _to_jax, obs, is_leaf=lambda x: isinstance(x, _np.ndarray)
                    )
            except Exception as e:
                print(f"Warning: could not restore obs_stats ({e}); using live stats.")

        # Only present if the checkpoint was written with the buffer attached.
        if isinstance(ckpt_state, dict) and "buffer_state" in ckpt_state:
            agent.state.buffer_state = ckpt_state["buffer_state"]

        # Optimizer internals are deliberately not restored; they are reinitialized.

        if "exploration_noise" in hyper:
            agent.exploration_noise = float(hyper["exploration_noise"])

        print(f"Agent state loaded from {path}")
        return agent

