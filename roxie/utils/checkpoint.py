"""Checkpoint management utilities for locating, reading, and restoring agent states."""

import dataclasses
import re
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

from roxie.utils.precision import FLOAT

CHECKPOINTS_DIRNAME = "checkpoints"
CHECKPOINT_ITEM = "default"

_STEP_RE = re.compile(r"^(\d+)$")


def checkpoint_steps(path: str | Path) -> int | None:
    """Extracts the env-step count encoded in a checkpoint directory name."""
    match = _STEP_RE.match(Path(path).name)
    return int(match.group(1)) if match else None


def find_checkpoint(path: str | Path) -> Path:
    """Resolves a run directory, checkpoints directory, or step path to a specific step directory.

    Args:
        path: Directory path to resolve.

    Returns:
        Path object pointing to the resolved step directory.

    Raises:
        FileNotFoundError: If no valid checkpoint directory is found.
        NotADirectoryError: If the provided path exists but is not a directory.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"No such checkpoint path: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Checkpoint path is not a directory: {path}")

    if checkpoint_steps(path) is not None:
        return path.resolve()

    for candidate in (path, path / CHECKPOINTS_DIRNAME):
        if not candidate.is_dir():
            continue
        steps = [
            (checkpoint_steps(child), child)
            for child in candidate.iterdir()
            if child.is_dir() and checkpoint_steps(child) is not None
        ]
        if steps:
            return max(steps, key=lambda item: item[0])[1].resolve()

    raise FileNotFoundError(
        f"No numbered checkpoint directory found under {path}. Pass a run "
        f"output dir, its `{CHECKPOINTS_DIRNAME}/` dir, or one `<N>` step dir."
    )


def _host_restore_args(metadata):
    """Constructs RestoreArgs to load checkpoint arrays directly into host memory."""
    return jax.tree.map(
        lambda _a: ocp.RestoreArgs(restore_type=np.ndarray),
        ocp.checkpoint_utils.construct_restore_args(metadata),
        is_leaf=lambda a: isinstance(a, ocp.RestoreArgs),
    )


def read_checkpoint(path: Path) -> Dict[str, Any]:
    """Reads a checkpoint payload off disk as host (NumPy) arrays."""
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
    """Casts host NumPy leaves to JAX device arrays in the active float precision."""
    def put(x):
        if not isinstance(x, np.ndarray):
            return x
        if np.issubdtype(x.dtype, np.floating):
            return jnp.asarray(x, dtype=FLOAT)
        return jnp.asarray(x)

    return jax.tree.map(put, tree, is_leaf=lambda x: isinstance(x, np.ndarray))


def _path_key(key):
    """Extracts the attribute or index key name from a PyTree path entry."""
    for attr in ("key", "name", "idx"):
        if hasattr(key, attr):
            return getattr(key, attr)
    raise TypeError(f"Unsupported pytree key: {key!r}")


class AgentCheckpointer:
    """Handles serialization, restoration, and validation of agent state payloads."""

    MODULE_SLOTS = (
        "actor", "critic", "target_actor", "target_critic",
        "actor_optimizer", "critic_optimizer",
    )

    def __init__(self, agent):
        self.agent = agent

    def payload(
        self,
        *,
        include_buffer: bool = False,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Extracts the agent state contents as a dictionary of host arrays."""
        agent = self.agent
        if not hasattr(agent, "state"):
            return None

        hyperparams = agent._export_hyperparams()
        saved_buffer = getattr(agent.state, "buffer_state", None)
        if not include_buffer:
            agent.state.buffer_state = None
        try:
            return {
                "format_version": 1,
                "trainstate_state": jax.device_get(nnx.split(agent.state)[1]),
                "extra_state": {
                    name: jax.device_get(nnx.split(module)[1])
                    for name, module in agent._checkpoint_modules().items()
                },
                "hyperparams": hyperparams,
                "last_update_boundary": int(agent._last_update_boundary),
                "metadata": (extra_metadata or {}),
            }
        finally:
            agent.state.buffer_state = saved_buffer

    @classmethod
    def build(
        cls,
        agent_cls,
        path: str | Path,
        env_obs_size: int,
        env_act_size: int,
        **config_blocks,
    ):
        """Constructs an agent instance and restores its state from a checkpoint."""
        path = Path(path).resolve()
        loaded = read_checkpoint(path)

        hyper = loaded.get("hyperparams", {}) or {}

        init_kwargs = dict(
            env_obs_size=env_obs_size,
            env_action_size=env_act_size,
            **{k: v for k, v in config_blocks.items() if v is not None},
        )

        for key in ("action_low", "action_high"):
            if key in hyper:
                init_kwargs[key] = jnp.asarray(hyper[key], dtype=jnp.float32)

        if agent_cls.hyperparams_cls is not None:
            fields = {f.name for f in dataclasses.fields(agent_cls.hyperparams_cls)}
            init_kwargs["hyperparams"] = agent_cls.hyperparams_cls(
                **{k: v for k, v in hyper.items() if k in fields}
            )

        agent = agent_cls(**init_kwargs)
        agent.checkpointer.restore(path, payload=loaded)
        return agent

    def restore(
        self,
        path: str | Path,
        *,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Loads numeric checkpoint state into an existing agent instance."""
        agent = self.agent
        path = Path(path).resolve()
        loaded = read_checkpoint(path) if payload is None else payload
        ckpt_state = loaded.get("trainstate_state", {})
        if not isinstance(ckpt_state, dict):
            ckpt_state = {}

        if not hasattr(agent, "state"):
            print(f"[Agent.restore] No `state` on {type(agent).__name__}; "
                  "restoring progress metadata only.")
            return dict(loaded.get("metadata") or {}, buffer_restored=False)

        self._check_obs_width(loaded, path)

        for name in self.MODULE_SLOTS:
            if getattr(agent.state, name, None) is not None:
                self._restore_module(agent.state, name, ckpt_state)

        extra_ckpt = loaded.get("extra_state") or {}
        for name in agent._checkpoint_modules():
            self._restore_module(agent, name, extra_ckpt)

        self._restore_obs_stats(ckpt_state)

        buffer_restored = False
        if "buffer_state" in ckpt_state:
            buffer_restored = self._restore_buffer(ckpt_state["buffer_state"])

        boundary = loaded.get("last_update_boundary", None)
        if boundary is not None:
            agent._last_update_boundary = int(boundary)

        metadata = dict(loaded.get("metadata") or {})
        metadata["buffer_restored"] = buffer_restored
        print(f"Agent state restored from {path}")
        return metadata

    @staticmethod
    def _restore_module(owner, name: str, source: Dict[str, Any]) -> None:
        """Merges a single checkpointed parameter subtree into a live NNX module."""
        if name not in source:
            print(f"Info: '{name}' not in checkpoint; keeping live {name}.")
            return
        gdef, _ = nnx.split(getattr(owner, name))
        setattr(owner, name, nnx.merge(gdef, _to_jax(source[name])))

    def _restore_obs_stats(self, ckpt_state: Dict[str, Any]) -> None:
        """Restores observation normalization statistics onto the agent's train state."""
        if "obs_stats" not in ckpt_state:
            return
        live = getattr(self.agent.state, "obs_stats", None)
        if live is None:
            print("Info: checkpoint carries obs_stats but this agent has none; "
                  "leaving observation normalization at its init values.")
            return
        obs = ckpt_state["obs_stats"]
        self.agent.state.obs_stats = live.replace(
            count=jnp.asarray(obs["count"], jnp.result_type(live.count)),
            sum=jnp.asarray(obs["sum"], jnp.result_type(live.sum)),
            sumsq=jnp.asarray(obs["sumsq"], jnp.result_type(live.sumsq)),
        )

    def _check_obs_width(self, loaded: Dict[str, Any], path: Path) -> None:
        """Validates that checkpoint observation dimensions match the active environment."""
        ckpt_width = (loaded.get("hyperparams") or {}).get("env_obs_size", None)
        experience = getattr(
            getattr(self.agent.state, "buffer_state", None), "experience", None,
        )
        if ckpt_width is None or experience is None:
            return
        live_width = int(jnp.shape(experience.observation)[-1])
        if int(ckpt_width) == live_width:
            return
        raise ValueError(
            f"{path} was trained on a {int(ckpt_width)}-dimensional "
            f"observation, but this env produces {live_width}."
        )

    def _restore_buffer(self, ckpt_buffer) -> bool:
        """Reconstructs Flashbax replay buffer state from host checkpoint arrays."""
        live = getattr(self.agent.state, "buffer_state", None)
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
            self.agent.state.buffer_state = jax.tree_util.tree_map_with_path(
                take, live
            )
        except (KeyError, TypeError, ValueError) as e:
            print(
                f"Warning: could not restore the replay buffer ({e}); "
                "keeping the empty one — the trainer will refill it."
            )
            return False
        return True