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

    def _checkpoint_modules(self) -> Dict[str, nnx.Module]:
        """Agent-owned nnx modules that live OUTSIDE `self.state`.

        `self.state` is what `save`/`restore` serialize wholesale, but several
        agents keep stateful modules next to it: DDPG's exploration noise (whose
        step counter drives the decay schedule), SAC's temperature plus its
        optimizer, MPO's Lagrange duals plus theirs. A resumed run that dropped
        them would restart exploration at the initial noise scale and the duals
        at their init values — a different algorithm from the one the checkpoint
        stopped in the middle of.

        Keys are attribute names on the agent, restored with `setattr`. Default
        is empty: an agent whose whole learnable state is in `self.state` (PPO)
        overrides nothing.
        """
        return {}

    def save(
        self,
        path: str | Path,
        *,
        format_version: int = 1,
        include_buffer: bool = False,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        """Write the agent's state to `path`.

        `include_buffer` also writes the replay buffer. It defaults to False
        because the buffer dominates the state, and `device_get`'ing it to host
        on every save spikes host RAM (orbax holds its own serialization copies
        on top) hard enough to risk an OOM mid-write on a large run. Off without
        it, a resumed off-policy run has to refill the buffer through the
        trainer's warmup; on, resume is exact but every checkpoint costs the
        buffer's full size on disk. See `Trainer(save_buffer=...)`.
        """
        try:
            if not hasattr(self, "state"):
                raise AttributeError("Agent must define `self.state` (an nnx.Module).")

            path = Path(path).resolve()

            # `_export_hyperparams` reads the buffer's obs/action shapes, so it
            # must be called *before* the detach below.
            hyperparams = self._export_hyperparams()
            saved_buffer = getattr(self.state, "buffer_state", None)
            if not include_buffer:
                self.state.buffer_state = None
            try:
                graphdef, state_tree = nnx.split(self.state)

                # Modules the agent keeps outside `self.state`, each split on its
                # own so restore can merge them back one at a time against a live
                # graphdef (see `_checkpoint_modules`).
                extra_state = {
                    name: jax.device_get(nnx.split(module)[1])
                    for name, module in self._checkpoint_modules().items()
                }

                payload = {
                    "format_version": format_version,
                    "trainstate_graphdef": graphdef,  # serialized topology
                    "trainstate_state": jax.device_get(state_tree),  # numeric pytree
                    "extra_state": extra_state,
                    "hyperparams": hyperparams,
                    # Where the update schedule stands, so a resume neither
                    # re-fires the boundary it already served nor queues a
                    # catch-up storm (see `due_for_update`).
                    "last_update_boundary": int(self._last_update_boundary),
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

        This is the *playback* entry point (`play.py`): the checkpoint is the
        only source of truth, so it also dictates the hyperparameters. To resume
        TRAINING, build the agent from its run config as usual and call
        `restore` on it — there the yaml is authoritative, so a resume may
        legitimately extend `trainer.steps` or retune a knob.
        """
        path = Path(path).resolve()
        loaded = _read_checkpoint(path)

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
        agent.restore(path, _payload=loaded)
        return agent

    def restore(
        self,
        path: str | Path,
        *,
        restore_optimizers: bool = True,
        _payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Load a checkpoint's numeric state INTO this already-built agent.

        This is what resuming training goes through: the agent is constructed
        from the run's config (so the yaml stays the source of truth for every
        hyperparameter, and a resume may legitimately raise `trainer.steps` or
        retune a knob), and only the numbers come from disk — networks, targets,
        optimizer slots, observation statistics, the modules listed by
        `_checkpoint_modules`, and the replay buffer when the checkpoint carries
        one.

        `restore_optimizers=False` reloads the policy but starts the optimizers
        cold; that is a fine-tune, not a resume, so it is not the default —
        dropping Adam's moments mid-run makes the first updates after the resume
        behave nothing like the ones before it.

        Returns the trainer-progress metadata the checkpoint was written with
        (`steps`, `epochs`, `episodes`, `gradient_steps`), plus
        `buffer_restored`. Fields the checkpoint lacks are simply absent.
        """
        path = Path(path).resolve()
        loaded = _read_checkpoint(path) if _payload is None else _payload
        ckpt_state = loaded.get("trainstate_state", {})
        if not isinstance(ckpt_state, dict):
            ckpt_state = {}

        if not hasattr(self, "state"):
            # A non-learning baseline (Constant, NormalRandom, ...): there is
            # nothing numeric to load, but the run's progress metadata is still
            # what the trainer resumes its step counter from.
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
                        # Params only; NNX maps them into its own topology.
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
            # checkpoints, so a missing entry is normal, not a warning-worthy
            # loss of state.
            if getattr(self.state, name, None) is not None:
                _restore_module(self.state, name, ckpt_state)

        # Modules the agent keeps outside `self.state` (SAC's temperature, MPO's
        # duals, the exploration noise schedule). Skipped wholesale for a
        # checkpoint written before `extra_state` existed.
        extra_ckpt = loaded.get("extra_state") or {}
        for name in self._checkpoint_modules():
            # `restore_optimizers` covers every optimizer, including the ones
            # kept outside `self.state` (SAC's temperature optimizer, MPO's dual
            # optimizer), so a fine-tune starts all of them cold consistently.
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
                else:
                    # Already a struct-compatible tree.
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

        The checkpoint stores the buffer as a plain nested dict, so it cannot be
        assigned to `state.buffer_state` as-is — flashbax needs its own
        `TrajectoryBufferState` dataclass back. The live (freshly initialized)
        buffer is the template: every leaf is looked up by path and shape-checked
        against it, which is also what catches a checkpoint saved with a
        different `parallel_envs` or buffer capacity. Any mismatch keeps the
        empty live buffer and returns False, so the trainer falls back to
        refilling it through warmup rather than training on a malformed one.
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


def _read_checkpoint(path: Path) -> Dict[str, Any]:
    """Read a checkpoint payload off disk as host (numpy) arrays.

    Restoring to host memory rather than onto the device sharding baked into the
    checkpoint is what lets a GPU-trained run load in a CPU-only process: the
    saved arrays are pinned to cuda:0, host arrays are placement-agnostic, and
    the numpy->jax conversion at use puts them on whatever device is active.
    """
    checkpointer = ocp.PyTreeCheckpointer()
    restore_args = jax.tree.map(
        lambda _a: ocp.RestoreArgs(restore_type=np.ndarray),
        ocp.checkpoint_utils.construct_restore_args(
            checkpointer.metadata(path).item_metadata
        ),
        is_leaf=lambda a: isinstance(a, ocp.RestoreArgs),
    )
    return checkpointer.restore(path, restore_args=restore_args)


def _to_jax(tree):
    """Device-put every numpy leaf of a restored subtree."""
    return jax.tree.map(
        lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x,
        tree,
        is_leaf=lambda x: isinstance(x, np.ndarray),
    )


def _path_key(key):
    """The dict key a `tree_map_with_path` path entry corresponds to.

    Buffer states nest dataclasses (attribute keys) inside dicts (dict keys);
    orbax flattens both to plain nested dicts, so restoring needs the name
    whichever kind of node it came from.
    """
    for attr in ("key", "name", "idx"):
        if hasattr(key, attr):
            return getattr(key, attr)
    raise TypeError(f"Unsupported pytree key: {key!r}")

