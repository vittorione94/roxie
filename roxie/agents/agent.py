import abc
import copy
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

from roxie.agents.utils import make_optimizer, serialize_bound
from roxie.models.actors import distribution_entropy as _distribution_entropy
from roxie.utils.checkpoint import CHECKPOINT_ITEM, checkpoint_steps


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
        """Pure action selection for a deterministic actor, which outputs
        actions in [-1, 1]. Returns the action and the applied noise."""
        action = actor_model(observation)

        noisy_action = noise_module.add_noise(action, key, evaluate)
        noisy_action = jnp.clip(noisy_action, -1.0, 1.0)
        return noisy_action, action - noisy_action

    @staticmethod
    @functools.partial(nnx.jit, static_argnames=("evaluate",))
    def stochastic_step_fn(
        actor_model: nnx.Module,
        observation: jnp.ndarray,
        evaluate: bool,
        key: jax.Array,
    ):
        """Pure action selection for a stochastic actor, which outputs a
        distribution over actions in [-1, 1]."""
        distribution = actor_model(observation)

        if evaluate:
            # The mean, never a sample, so evaluation is deterministic.
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
        # With 0 or 1 samples the variance is identically 0, so `sqrt(var + eps)`
        # is tiny and dividing by it saturates every feature at the clip bound.
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

        Losses are handed already-normalized observations and take no
        `obs_mean`/`obs_std`/`obs_clip` arguments of their own.

        When `enabled` is False the samples pass through untouched, clip
        included: `obs_clip` bounds *normalized* observations, so applying it
        anyway would squash raw observations into +/- `clip`.
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

    # --------------------------
    # Construction (shared)
    # --------------------------
    def _init_train_state(
        self,
        actor: nnx.Module,
        critic: nnx.Module,
        buffer_state: Any,
        *,
        actor_learning_rate: float,
        critic_learning_rate: float,
        max_grad_norm: float,
        actor_optimizer_config: dict = None,
        critic_optimizer_config: dict = None,
        target_actor: bool = True,
        target_critic: bool = True,
    ) -> None:
        """Assemble the pieces every actor-critic agent puts together identically.

        Sets `self.actor_learning_rate`, `self.critic_learning_rate`,
        `self.max_grad_norm` and `self.state`.

        The targets are deep copies of the live networks, so a run starts with
        `target == online`. Passing `False` leaves the slot `None`, which is what
        `restore` keys off to skip it without warning.

        The optimizer family and its own hyperparameters come from the yaml
        blocks; the learning rate and the global-norm clip stay top-level agent
        args so they remain first-class swept/logged/checkpointed knobs.
        """
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.max_grad_norm = max_grad_norm

        self.state = TrainState(
            actor=actor,
            critic=critic,
            target_actor=copy.deepcopy(actor) if target_actor else None,
            target_critic=copy.deepcopy(critic) if target_critic else None,
            actor_optimizer=make_optimizer(
                actor,
                actor_optimizer_config,
                learning_rate=actor_learning_rate,
                max_grad_norm=max_grad_norm,
            ),
            critic_optimizer=make_optimizer(
                critic,
                critic_optimizer_config,
                learning_rate=critic_learning_rate,
                max_grad_norm=max_grad_norm,
            ),
            buffer_state=buffer_state,
            # The buffer is the authority on observation width: it was allocated
            # from the prototype the transitions are written through, so stats
            # built from it cannot disagree with `add`.
            obs_stats=Agent.init_obs_stats(
                buffer_state.experience.observation.shape[-1]
            ),
        )

    # Highest update boundary already served. Class-level default so every
    # off-policy agent inherits it without touching its __init__; the first
    # firing shadows it with an instance attribute.
    _last_update_boundary = -1

    def due_for_update(self, steps: int) -> bool:
        """True at most once per `steps_between_updates` env steps past warmup.

        Do NOT write this as `(steps - steps_before_learning) % between == 0`.
        The trainer advances `steps` in strides of `parallel_envs`, so that test
        only fires when `steps_before_learning` is itself a multiple of the
        stride; otherwise the residue cycles without reaching 0 and no gradient
        step ever runs. Tracking the last boundary served makes the schedule
        depend only on elapsed env steps.

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
        """The hyperparameter block written into every checkpoint.

        Abstract because a non-learning baseline has none of the attributes read
        below and overrides this outright; a learning agent should call
        `super()` and extend the result.

        `Agent.load` filters these against the constructor's signature, so the
        names must keep matching the constructor keywords or a knob silently
        stops round-tripping through playback.
        """
        return {
            "seed": int(self.seed),
            "gamma": float(self.gamma),
            "actor_learning_rate": float(self.actor_learning_rate),
            "critic_learning_rate": float(self.critic_learning_rate),
            "max_grad_norm": float(self.max_grad_norm),
            "learning_steps": int(self.learning_steps),
            "normalize_observations": bool(self.normalize_observations),
            "obs_norm_clip": float(self.obs_clip),
            "obs_norm_eps": float(self.obs_eps),
            # Per-actuator lists, not just the first actuator's bounds.
            "action_low": serialize_bound(self.action_low),
            "action_high": serialize_bound(self.action_high),
        }

    def _replay_hyperparams(self) -> Dict[str, Any]:
        """The extra block every replay-driven agent carries.

        None of it applies on-policy, hence the split from
        `_export_hyperparams`. The env sizes are read off the buffer: the shape
        it was allocated with is what a checkpoint has to be rebuilt against,
        not whatever was passed to `__init__`.
        """
        return {
            "tau": float(self.tau),
            "env_obs_size": self.state.buffer_state.experience.observation.shape[2],
            "env_action_size": self.state.buffer_state.experience.action.shape[2],
            "steps_before_learning": int(self.steps_before_learning),
            "steps_between_updates": int(self.steps_between_updates),
            "memory_warmup": int(self.memory_warmup),
            "memory_capacity": int(self.buffer_size),
            "memory_batch_size": int(self.batch_size),
        }

    def _checkpoint_modules(self) -> Dict[str, nnx.Module]:
        """Agent-owned nnx modules that live outside `self.state`.

        `save`/`restore` serialize `self.state` wholesale, but several agents
        keep stateful modules next to it: DDPG's exploration noise, SAC's
        temperature, MPO's Lagrange duals, plus their optimizers. Dropping them
        on resume would restart exploration and the duals at their init values.

        Keys are attribute names on the agent, restored with `setattr`.
        """
        return {}

    def checkpoint_payload(
        self,
        *,
        format_version: int = 1,
        include_buffer: bool = False,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """The checkpoint's contents as host arrays; None for a stateless baseline.

        Every leaf is `device_get`'d, which is what makes the manager's async
        write safe: it reads host memory, so the trainer can resume the learner
        — whose next update donates the device buffers this came from — without
        waiting for the write to land.

        `include_buffer` is off by default: the buffer dominates the state and
        `device_get`'ing it every save can spike host RAM into an OOM mid-write.
        """
        if not hasattr(self, "state"):
            return None

        # `_export_hyperparams` reads the buffer's obs/action shapes, so it
        # must be called *before* the detach below.
        hyperparams = self._export_hyperparams()
        saved_buffer = getattr(self.state, "buffer_state", None)
        if not include_buffer:
            self.state.buffer_state = None
        try:
            return {
                "format_version": format_version,
                # No graphdef: `restore` re-derives it from the live modules it
                # is merging into.
                "trainstate_state": jax.device_get(nnx.split(self.state)[1]),
                # Split individually so restore merges them one at a time.
                "extra_state": {
                    name: jax.device_get(nnx.split(module)[1])
                    for name, module in self._checkpoint_modules().items()
                },
                "hyperparams": hyperparams,
                "last_update_boundary": int(self._last_update_boundary),
                "metadata": (extra_metadata or {}),
            }
        finally:
            self.state.buffer_state = saved_buffer

    def save(
        self,
        path: str | Path,
        *,
        format_version: int = 1,
        include_buffer: bool = False,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        """Write one self-contained checkpoint to `path`.

        Training runs do not come through here: `Trainer` saves through a
        `CheckpointManager`. This is the one-off path (tests, ad-hoc saves), and
        it writes a single directory rather than a manager's step/item pair —
        `_read_checkpoint` accepts both.
        """
        payload = self.checkpoint_payload(
            format_version=format_version,
            include_buffer=include_buffer,
            extra_metadata=extra_metadata,
        )
        if payload is None:
            print(f"[Agent.save] {type(self).__name__} has no state; skipping {path}.")
            return
        path = Path(path).resolve()
        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(path, payload)
        print(f"[Agent.save] Saved to {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        env_obs_size: int,
        env_act_size: int,
        **config_blocks,
    ):
        """Rebuild an agent from a checkpoint.

        `config_blocks` are the yaml-side construction blocks (`actor_config`,
        `critic_config`, `memory_config`, `noise_config`, the
        `*_optimizer_config` blocks), taken as keywords so an agent that grows or
        drops one needs no change here. Everything else comes from the
        checkpoint's `hyperparams`.

        This is the playback entry point (`play.py`), where the checkpoint is the
        only source of truth. To resume training, build the agent from its run
        config and call `restore` instead — there the yaml is authoritative.
        """
        path = Path(path).resolve()
        loaded = _read_checkpoint(path)

        ckpt_state = loaded["trainstate_state"]
        hyper = loaded.get("hyperparams", {})
        print(hyper)

        # Across the whole MRO: subclasses like TD3 forward via ``*args,
        # **kwargs``, so inspecting only ``cls.__init__`` would drop the
        # parent's hyperparams.
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
        # Serialized as a plain per-actuator list; everything downstream is
        # written against an ndarray of shape (action_dim,).
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
        agent.restore(path, _payload=loaded)
        return agent

    def restore(
        self,
        path: str | Path,
        *,
        restore_optimizers: bool = True,
        _payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Load a checkpoint's numeric state into this already-built agent.

        This is what resuming training goes through: the agent is constructed
        from the run's config, so the yaml stays the source of truth for every
        hyperparameter and only the numbers come from disk.

        `restore_optimizers=False` reloads the policy but starts the optimizers
        cold — a fine-tune rather than a resume, since dropping Adam's moments
        changes how the first updates after it behave.

        Returns the trainer-progress metadata the checkpoint was written with
        (`steps`, `epochs`, `episodes`, `gradient_steps`), plus
        `buffer_restored`. Fields the checkpoint lacks are absent.
        """
        path = Path(path).resolve()
        loaded = _read_checkpoint(path) if _payload is None else _payload
        ckpt_state = loaded.get("trainstate_state", {})
        if not isinstance(ckpt_state, dict):
            ckpt_state = {}

        if not hasattr(self, "state"):
            # A non-learning baseline (Constant, NormalRandom, ...).
            print(f"[Agent.restore] No `state` on {type(self).__name__}; "
                  "restoring progress metadata only.")
            return dict(loaded.get("metadata") or {}, buffer_restored=False)

        def _restore_module(owner, name: str, source: Dict[str, Any]):
            """Merge one checkpointed subtree into the live module at `name`."""
            if name not in source:
                print(f"Info: '{name}' not in checkpoint; keeping live {name}.")
                return
            sub_ckpt = _to_jax(source[name])
            sub_live = getattr(owner, name)
            gdef, _ = nnx.split(sub_live)
            try:
                setattr(owner, name, nnx.merge(gdef, sub_ckpt))
            except ValueError as e:
                # Architecture drift or partial state: fall back to params only.
                print(
                    f"Warning: merge({name}) failed ({e}). Falling back to param-only copy."
                )
                try:
                    src_params = (
                        sub_ckpt.get("params", None)
                        if isinstance(sub_ckpt, dict)
                        else None
                    )
                    if src_params is None:
                        print(f"Warning: no 'params' found for {name}; skipping.")
                    else:
                        nnx.update(sub_live, {"params": src_params})
                except Exception as ee:
                    print(
                        f"Warning: param-only update for {name} failed ({ee}). Skipping."
                    )

        names = ["actor", "critic", "target_actor", "target_critic"]
        if restore_optimizers:
            names += ["actor_optimizer", "critic_optimizer"]
        for name in names:
            # `target_actor` is None for SAC/PPO and absent from their
            # checkpoints, so a missing entry is normal there.
            if getattr(self.state, name, None) is not None:
                _restore_module(self.state, name, ckpt_state)

        extra_ckpt = loaded.get("extra_state") or {}
        for name in self._checkpoint_modules():
            # `restore_optimizers` covers the optimizers kept outside
            # `self.state` too, so a fine-tune starts all of them cold.
            if restore_optimizers or not name.endswith("optimizer"):
                _restore_module(self, name, extra_ckpt)

        # Replaced wholesale rather than partial-updated.
        if "obs_stats" in ckpt_state:
            try:
                obs = ckpt_state["obs_stats"]
                if isinstance(obs, dict):
                    self.state.obs_stats = ObsStats(
                        count=jnp.asarray(obs["count"]),
                        sum=jnp.asarray(obs["sum"]),
                        sumsq=jnp.asarray(obs["sumsq"]),
                    )
                else:  # already a struct-compatible tree
                    self.state.obs_stats = _to_jax(obs)
            except Exception as e:
                print(f"Warning: could not restore obs_stats ({e}); using live stats.")

        # Only present if the checkpoint was written with the buffer attached
        # (`save(include_buffer=True)`).
        buffer_restored = False
        if "buffer_state" in ckpt_state:
            buffer_restored = self._restore_buffer(ckpt_state["buffer_state"])

        boundary = loaded.get("last_update_boundary", None)
        if boundary is not None:
            self._last_update_boundary = int(boundary)

        metadata = dict(loaded.get("metadata") or {})
        metadata["buffer_restored"] = buffer_restored
        print(f"Agent state restored from {path}")
        return metadata

    def _restore_buffer(self, ckpt_buffer) -> bool:
        """Rebuild the replay buffer from its checkpointed leaves.

        The checkpoint stores the buffer as a plain nested dict, so flashbax
        needs its `TrajectoryBufferState` dataclass rebuilt around it. The
        freshly initialized live buffer is the template: every leaf is looked up
        by path and shape-checked against it, catching a checkpoint saved with a
        different `parallel_envs` or capacity. A mismatch keeps the empty buffer
        and returns False, so the trainer refills it through warmup.
        """
        live = getattr(self.state, "buffer_state", None)
        if live is None:
            return False

        def take(path, leaf):
            node = ckpt_buffer
            for key in path:
                node = node[_path_key(key)]
            value = jnp.asarray(node)
            if value.shape != jnp.shape(leaf):
                raise ValueError(
                    f"buffer leaf {jax.tree_util.keystr(path)} has shape "
                    f"{value.shape} in the checkpoint but {jnp.shape(leaf)} live"
                )
            return value.astype(jnp.result_type(leaf))

        try:
            self.state.buffer_state = jax.tree_util.tree_map_with_path(take, live)
        except (KeyError, TypeError, ValueError) as e:
            print(
                f"Warning: could not restore the replay buffer ({e}); "
                "keeping the empty one — the trainer will refill it."
            )
            return False
        return True


def _host_restore_args(metadata):
    """RestoreArgs pulling every leaf back as a numpy array.

    Restoring to host memory rather than the device sharding baked into the
    checkpoint is what lets a GPU-trained run load in a CPU-only process: the
    saved arrays are pinned to cuda:0, host arrays are not, and the numpy->jax
    conversion at use puts them on whatever device is active.
    """
    return jax.tree.map(
        lambda _a: ocp.RestoreArgs(restore_type=np.ndarray),
        ocp.checkpoint_utils.construct_restore_args(metadata),
        is_leaf=lambda a: isinstance(a, ocp.RestoreArgs),
    )


def _read_checkpoint(path: Path) -> Dict[str, Any]:
    """Read a checkpoint payload off disk as host (numpy) arrays.

    A run's checkpoints come from `Trainer`'s CheckpointManager, which splits a
    step into `<step>/<item>/` and files orbax's descriptor at the STEP level, so
    they are read back through a manager — a bare checkpointer pointed at the
    item directory reads fine but warns about the descriptor it cannot see.
    `Agent.save` writes a self-contained checkpoint, read directly.
    """
    step = checkpoint_steps(path)
    if step is not None and (path / CHECKPOINT_ITEM).is_dir():
        with ocp.CheckpointManager(
            path.parent, item_handlers=ocp.PyTreeCheckpointHandler()
        ) as manager:
            return manager.restore(
                step,
                args=ocp.args.PyTreeRestore(
                    restore_args=_host_restore_args(manager.item_metadata(step))
                ),
            )

    checkpointer = ocp.PyTreeCheckpointer()
    return checkpointer.restore(
        path,
        restore_args=_host_restore_args(checkpointer.metadata(path).item_metadata),
    )


def _to_jax(tree):
    """Device-put every numpy leaf of a restored subtree."""
    return jax.tree.map(
        lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x,
        tree,
        is_leaf=lambda x: isinstance(x, np.ndarray),
    )


def _path_key(key):
    """The dict key a `tree_map_with_path` path entry corresponds to.

    Buffer states nest dataclasses (attribute keys) inside dicts (dict keys) and
    orbax flattens both to plain nested dicts, so restoring needs the name from
    either kind of node.
    """
    for attr in ("key", "name", "idx"):
        if hasattr(key, attr):
            return getattr(key, attr)
    raise TypeError(f"Unsupported pytree key: {key!r}")

