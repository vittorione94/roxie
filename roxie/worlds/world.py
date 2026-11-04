"""The base every world model shares: the real replay buffer, fitting, stepping.

A world model is the agent's peer in the training loop, not one of its parts.
It owns the replay buffer of REAL transitions — the rollout writes into it, the
world fits itself on windows of it, and a model-based agent starts its imagined
rollouts from observations sampled out of it — and it exposes one
differentiable `step` that a model-based agent rolls its policy through.
docs/worlds.md is the extended rationale.
"""

import abc
import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import jax.numpy as jnp
from flax import nnx, struct
from omegaconf import OmegaConf

from roxie.utils.checkpoint import WorldCheckpointer
from roxie.utils.math import obs_mean_std, update_obs_stats


class WorldState(nnx.Module):
    """Everything one model-fitting step reads and writes, as one JAX pytree.

    A pytree for the reason `TrainState` is one: a pass hands it straight to
    `jax.jit` and `lax.scan`, and the fused collect chunk carries it to write
    the buffer.
    """

    buffer_state: Any = nnx.data()
    obs_stats: Any = nnx.data()

    def __init__(
        self,
        *,
        dynamics: nnx.Module,
        optimizer: nnx.Optimizer,
        buffer_state: Any,
        obs_stats: Any,
    ):
        self.dynamics = dynamics
        self.optimizer = optimizer
        self.buffer_state = buffer_state
        self.obs_stats = obs_stats


@struct.dataclass
class WorldOutput:
    """One fitting pass's device-bound product, as `WorldModel.learn` takes it."""

    state: WorldState
    loss: jnp.ndarray
    diagnostics: Dict[str, jnp.ndarray]


def world_normalizer(obs_stats, hp) -> tuple:
    """`(mean, std)` a world reads and predicts in; identity when it is off."""
    if hp.normalize_observations:
        return obs_mean_std(obs_stats, hp.obs_norm_eps)
    width = obs_stats.sum.shape[-1]
    dtype = obs_stats.sum.dtype
    return jnp.zeros(width, dtype), jnp.ones(width, dtype)


def build_world(config, **kwargs):
    """Instantiates a world model from a Hydra configuration block.

    `_recursive_=False` for the reason `build_agent` uses it: the world builds
    its own nested blocks.
    """
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True,
        )
    else:
        config = dict(config)
    return hydra.utils.instantiate(config, _recursive_=False, **kwargs)


class WorldModel(abc.ABC):
    """Abstract class used to build world models.

    Attributes:
        state: the `WorldState`, set by the subclass's `__init__`.
        replay: the `utils.memory.ReplayManager` over the real buffer. Always a
          trajectory buffer: fitting reads windows of it.
        buffer_size: capacity the buffer was built with.
        batch_size: windows per fitting step.
        env_obs_size: observation width the world was built for.
        env_action_size: action width likewise.
    """

    hyperparams_cls = None
    hp = None
    derived_hyperparams = frozenset()
    replay = None
    buffer_size = None
    batch_size = None
    env_obs_size = None
    env_action_size = None
    diagnostics = None

    _checkpointer = None

    @abc.abstractmethod
    def step(self, dynamics, obs_mean, obs_std, obs, action) -> tuple:
        """One imagined step. Returns `(next_obs, reward, term_prob)`.

        Pure in its arguments and differentiable in `obs` and `action`, which
        is the whole contract a model-based agent relies on. `dynamics` is an
        argument rather than read off `self.state`: the agent calls this inside
        its own compiled program, where this world is a constant and its
        weights must be traced. Observations are RAW in and out — the world's
        normalization is internal, and deliberately unclipped, since a clip
        would zero the gradient of every state outside it.
        """

    @abc.abstractmethod
    def _compile_and_run(self, key, target_steps: int) -> WorldOutput:
        """Dispatches to this model's jitted fitting pass."""

    def learn(self, key, n_steps: Optional[int] = None):
        """Runs one fitting pass on the real buffer, IN PLACE on `self.state`.

        Unconditional, like `Agent.learn`: the trainer runs it whenever the
        agent's own update gate fires.

        Returns:
            `(gradient_steps, loss)`, the loss averaged over the pass.
        """
        target_steps = (
            self.hp.learning_steps if n_steps is None else int(n_steps)
        )
        output = self._compile_and_run(key, target_steps)
        self.state = output.state
        self.diagnostics.record(output.diagnostics, target_steps)
        return target_steps, output.loss

    def normalizer(self, obs_stats) -> tuple:
        """`(mean, std)` the model reads and predicts in; see `world_normalizer`."""
        return world_normalizer(obs_stats, self.hp)

    def start_sampler(self, batch_size: int):
        """Samples `batch_size` single real observations to imagine from.

        Returns:
            A static `(buffer_state, key) -> sample` callable, with a
            `(batch_size, 1)` prefix on every leaf.
        """
        return self.replay.sampler(batch_size, sequence_length=1)

    # --- the replay-owner surface the rollout writes through ----------------
    #
    # The same three spellings `Agent` answers to, so the fused chunk and the
    # warmup fill address whichever of the two owns the buffer identically.
    # The statistics differ deliberately: they are the world's own, folding
    # each observation the envs landed on once, and never feed acting.

    def buffer_transitions(self, transition, next_obs, *, state=None,
                           update_stats=True):
        """Writes one env-step batch into `state`, IN PLACE.

        Args:
            transition: as `Agent.buffer_transitions` takes it.
            next_obs: the observation the step landed on.
            state: the traced world state a fused chunk carries; omit it for
              the world's own, written with the jitted, donating add.
            update_stats: `False` leaves `obs_stats` for `absorb_obs_stats` to
              fold in over the whole chunk.
        """
        traced = state is not None
        state = state if traced else self.state
        add = self.replay.add_in_trace if traced else self.replay.add
        state.buffer_state = add(state.buffer_state, transition)
        if update_stats:
            state.obs_stats = update_obs_stats(state.obs_stats, next_obs)

    def absorb_obs_stats(self, observation, next_obs, *, state=None):
        """Folds a `(T, B, D)` block of landed observations into the statistics."""
        del observation  # each one was some earlier step's landing
        state = self.state if state is None else state
        state.obs_stats = update_obs_stats(
            state.obs_stats, next_obs.reshape(-1, next_obs.shape[-1])
        )

    def fill_buffer(self, transitions, next_obs):
        """Stores a `(T, B, ...)` warmup block, one env step per tick."""
        self.state.buffer_state = self.replay.add_block(
            self.state.buffer_state, transitions
        )
        width = next_obs.shape[-1]
        # The block's first observations plus everything it landed on: each
        # state once, the same weighting the per-chunk fold gives.
        self.state.obs_stats = update_obs_stats(
            self.state.obs_stats,
            jnp.concatenate([transitions.observation[:1].reshape(-1, width),
                             next_obs.reshape(-1, width)], axis=0),
        )

    # --- diagnostics and checkpoints ----------------------------------------

    def pop_diagnostics(self, env_steps: int = 0) -> dict:
        """This epoch's fitting diagnostics under `world/`, RESET; `{}` if none."""
        if self.diagnostics is None:
            return {}
        return self.diagnostics.pop(env_steps)

    def _export_hyperparams(self) -> Dict[str, Any]:
        """The block written into every checkpoint, env sizes included.

        The sizes are read back off the buffer, the geometry a checkpoint has
        to be rebuilt against.
        """
        experience = self.state.buffer_state.experience
        return {
            **dataclasses.asdict(self.hp),
            "env_obs_size": int(experience.observation.shape[-1]),
            "env_action_size": int(experience.action.shape[-1]),
            "memory_capacity": int(self.buffer_size),
            "memory_batch_size": int(self.batch_size),
        }

    @property
    def checkpointer(self) -> WorldCheckpointer:
        """This world's reader/writer of checkpoints, built on first use."""
        if self._checkpointer is None:
            self._checkpointer = WorldCheckpointer(self)
        return self._checkpointer

    def checkpoint_payload(self, *, include_buffer: bool = False) -> dict:
        """The world's half of a checkpoint; see `WorldCheckpointer.payload`."""
        return self.checkpointer.payload(include_buffer=include_buffer)

    def restore(self, path: str | Path) -> bool:
        """Loads a checkpoint's world half; True if the buffer came back."""
        return self.checkpointer.restore(path)
