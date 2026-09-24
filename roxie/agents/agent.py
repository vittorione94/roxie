"""The base every agent shares: acting, buffering, the update gate, checkpoints.

The docstrings here are the contract; docs/agents.md is the extended rationale
behind it.
"""

import abc
import copy
import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional

import jax.numpy as jnp
from flax import nnx, struct

from roxie.agents.utils import make_optimizer, serialize_bound
from roxie.models.actors import deterministic_action
from roxie.utils.checkpoint import AgentCheckpointer
from roxie.utils.diagnostics import DiagnosticsTracker
from roxie.utils.math import (
    finite_or_zero,
    init_obs_stats,
    normalize_obs,
    obs_mean_std,
    scale_to_env,
    update_obs_stats,
)


class TrainState(nnx.Module):
    """Everything one gradient step reads and writes, as a single JAX pytree.

    A pytree rather than an opaque nnx graph node, so a pass hands it straight
    to `jax.jit` and `lax.scan` with no `nnx.split`/`merge` at the boundary.
    """

    target_actor: Optional[nnx.Module] = nnx.data()
    target_critic: Optional[nnx.Module] = nnx.data()
    buffer_state: Any = nnx.data()
    obs_stats: Any = nnx.data()

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
class LearningOutput:
    """One learning pass's device-bound product, as `Agent.learn` takes it."""

    state: TrainState
    actor_loss: jnp.ndarray
    critic_loss: jnp.ndarray
    diagnostics: Dict[str, jnp.ndarray]
    extra_state: Any = None


class Agent(abc.ABC):
    """Abstract class used to build agents."""

    hyperparams_cls = None
    hp = None
    derived_hyperparams = frozenset()
    freeze_obs_norm_per_chunk = False
    joint_grad_clip = False

    # The slots below are declared here rather than left for a subclass's
    # `__init__` to inject, so a non-learning baseline — which gets none of
    # them — still answers rather than raising.

    # Per-actuator env action bounds, what `scale_to_env` maps [-1, 1] onto.
    action_low = None
    action_high = None

    # Set by `_init_train_state`. `state` is deliberately NOT declared here:
    # `hasattr(agent, "state")` is what tells a learning agent from a
    # play-only baseline in `checkpoint.py` and `trainer.py`, and a class-level
    # `None` would answer true for both.
    diagnostics = None

    # The `utils.memory.ReplayManager` a buffered agent builds in its
    # `__init__`, and the geometry it was built with.
    replay = None
    buffer_size = None
    batch_size = None

    # Built on first save or restore, by the `checkpointer` property.
    _checkpointer = None

    # Highest update boundary already served; see `due_for_update`.
    _last_update_boundary = -1

    @abc.abstractmethod
    def select_action(
        self,
        observation: jnp.ndarray,
        key=None,
        evaluate: bool = False,
        *,
        actor: Optional[nnx.Module] = None,
        obs_stats: Any = None,
        noise_module: Optional[nnx.Module] = None,
        critic: Optional[nnx.Module] = None,
    ) -> tuple:
        """The policy. Returns `(scaled_action, applied_noise, extras)`.

        Called bare — `select_action(obs, key)` — it acts off `self.state`. The
        keyword-only arguments override each piece of that state instead, which
        is how a fused acting chunk passes its `lax.scan` carry; every learning
        agent accepts all four and ignores the ones it has no use for.

        `extras` is behaviour the agent computed here and cannot recover later —
        empty for everything but PPO. It travels on the transition, not on
        `self`.
        """

    def learn(self, agent_rng, n_steps=None):
        """Runs one learning pass of gradient steps, IN PLACE on `self.state`.

        Unconditional — the caller gates it with `due_for_update` first.
        `n_steps` sizes the pass; `None` means `hp.learning_steps`. The
        algorithm's own work is `_compile_and_run`; everything around it is
        here.

        Returns:
            `(gradient_steps, actor_loss, critic_loss)` — the losses averaged
            over the pass, the count reported rather than assumed, since PPO's
            trust region decides it on device.

        Raises:
            NotImplementedError: on a play-only baseline.
        """
        target_steps = (
            self.hp.learning_steps if n_steps is None else int(n_steps)
        )
        steps_taken, output = self._compile_and_run(agent_rng, target_steps)

        self.state = output.state
        if output.extra_state is not None:
            self._apply_extra_state(output.extra_state)

        self.record_diagnostics(output.diagnostics, steps_taken)
        return steps_taken, output.actor_loss, output.critic_loss

    @abc.abstractmethod
    def _compile_and_run(self, agent_rng, target_steps: int):
        """Dispatches to this algorithm's jitted learning pass.

        Args:
            agent_rng: the pass's key; splitting it belongs to the pass.
            target_steps: the size `learn` resolved, ignored by PPO.

        Returns:
            `(steps_taken, LearningOutput)`. The count is a HOST int, kept out
            of the pytree so the trainer's running total stays one; PPO is the
            only agent that has to sync for it.

        Raises:
            NotImplementedError: on a play-only baseline.
        """

    def _apply_extra_state(self, extra_state) -> None:
        """Binds back what a learning pass carried outside `TrainState`."""
        raise NotImplementedError(
            f"{type(self).__name__} returned an `extra_state` without an "
            f"`_apply_extra_state` to bind it back."
        )

    def _optimizer_grad_clip(self):
        """The clip the OPTIMIZERS apply; `None` when the agent clips jointly."""
        return None if type(self).joint_grad_clip else self.hp.max_grad_norm

    def freeze_acting_norm(self):
        """The `ObsStats` acting uses for this chunk, or `None` for `state`'s.

        Called by the rollout at every chunk boundary; the host-side half of
        `freeze_obs_norm_per_chunk`.
        """
        return None

    def eval_action(self, observation, actor, obs_stats):
        """The action to score this policy by: deterministic, unexplored.

        The actor and the statistics are arguments because the eval is one
        compiled program in which this agent is a constant. Not
        `select_action(evaluate=True)`, which carries the noise module.
        """
        if self.hp.normalize_observations:
            mean, std = obs_mean_std(obs_stats, self.hp.obs_norm_eps)
            observation = normalize_obs(
                observation, mean, std, self.hp.obs_norm_clip,
            )
        # This path's own NaN scrub: `clip` would pass one straight through to
        # the physics.
        action = jnp.clip(
            finite_or_zero(deterministic_action(actor(observation))), -1.0, 1.0
        )
        return scale_to_env(action, self.action_low, self.action_high)

    def buffer_transitions(self, transition, next_obs, *, state=None,
                           update_stats=True):
        """Writes one env-step batch into `state`, IN PLACE.

        Args:
            transition: goes to the store as the caller assembled it — time
              axis, pruning and NaN scrub belong to `ReplayManager`. Its
              `terminal` is true termination only, never a time-limit
              truncation.
            next_obs: the observation the step landed on, which the buffer does
              not store but the running statistics need.
            state: the traced train state a fused chunk carries; omit it and the
              agent's own state is used with the jitted, donating add.
            update_stats: `False` writes the transition and leaves `obs_stats`
              alone, for a caller stacking the chunk's observations to fold the
              whole block in through `absorb_obs_stats` instead.
        """
        traced = state is not None
        state = state if traced else self.state
        add = self.replay.add_in_trace if traced else self.replay.add
        state.buffer_state = add(state.buffer_state, transition)
        if update_stats and self.hp.normalize_observations:
            obs_batch = jnp.concatenate([transition.observation, next_obs], axis=0)
            state.obs_stats = update_obs_stats(state.obs_stats, obs_batch)

    def absorb_obs_stats(self, observation, next_obs, *, state=None):
        """Folds a whole BLOCK of stepped observations into `state.obs_stats`.

        The bulk sibling of what `buffer_transitions` does per step, for the
        fused rollout chunk: both arguments arrive stacked over a leading time
        axis and are flattened onto it. `ObsStats` is running sums, so this
        lands where the per-step folds landed; what the deferral changes is when
        acting sees them — see `collect`.
        """
        state = self.state if state is None else state
        width = next_obs.shape[-1]
        # Three passes rather than one over a concatenation: the sums are
        # associative, and a (3 * T * B, D) temporary is the one array here
        # large enough to be worth not materializing.
        landed = next_obs.reshape(-1, width)
        stats = state.obs_stats
        # The weighting is the per-step path's: under normalization the landing
        # observation counts twice, once from the pair `buffer_transitions`
        # folds in and once from the fold the rollout does deliberately on top
        # of it. Dropping either reweights the statistics away from every run
        # logged so far. Without normalization only that second fold ever ran,
        # so the statistics still advance for anything that reads them later.
        if self.hp.normalize_observations:
            stats = update_obs_stats(stats, observation.reshape(-1, width))
            stats = update_obs_stats(stats, landed)
        state.obs_stats = update_obs_stats(stats, landed)

    def fill_buffer(self, transitions, next_obs):
        """Stores a `(T, B, ...)` BLOCK of transitions, one env step per tick.

        The bulk sibling of `buffer_transitions`, for the warmup fill: the adds
        scan inside one dispatch and the statistics take a single pass over the
        block.
        """
        self.state.buffer_state = self.replay.add_block(
            self.state.buffer_state, transitions
        )
        if self.hp.normalize_observations:
            width = transitions.observation.shape[-1]
            self.state.obs_stats = update_obs_stats(
                self.state.obs_stats,
                jnp.concatenate([transitions.observation.reshape(-1, width),
                                 next_obs.reshape(-1, width)], axis=0),
            )

    def _init_train_state(
        self,
        actor: nnx.Module,
        critic: nnx.Module,
        buffer_state: Any,
        *,
        actor_optimizer_config: dict = None,
        critic_optimizer_config: dict = None,
        target_actor: bool = True,
        target_critic: bool = True,
    ) -> None:
        """Assembles the shared actor-critic state, targets and optimizers.

        Sets `self.state` and the diagnostics tracker. Requires `self.hp` in
        place already: the learning rates and the global-norm clip are read off
        it rather than passed, while the optimizer family and its own
        hyperparameters come from the yaml blocks.

        The targets are deep copies, so a run starts with `target == online`.
        Passing `False` leaves the slot `None`, which is what `restore` keys off
        to skip it without warning.
        """
        self.diagnostics = DiagnosticsTracker(type(self).__name__.lower())

        self.state = TrainState(
            actor=actor,
            critic=critic,
            target_actor=copy.deepcopy(actor) if target_actor else None,
            target_critic=copy.deepcopy(critic) if target_critic else None,
            actor_optimizer=make_optimizer(
                actor,
                actor_optimizer_config,
                learning_rate=self.hp.actor_learning_rate,
                max_grad_norm=self._optimizer_grad_clip(),
            ),
            critic_optimizer=make_optimizer(
                critic,
                critic_optimizer_config,
                learning_rate=self.hp.critic_learning_rate,
                max_grad_norm=self._optimizer_grad_clip(),
            ),
            buffer_state=buffer_state,
            # The buffer is the authority on observation width, so stats built
            # from it cannot disagree with `add`.
            obs_stats=init_obs_stats(
                buffer_state.experience.observation.shape[-1]
            ),
        )

    def due_for_update(self, steps: int) -> bool:
        """True at most once per `steps_between_updates` env steps past warmup.

        NOT idempotent: a true answer CONSUMES the boundary it fired on, so ask
        it exactly where the pass would run. `memory_warmup` is the only gate,
        and no backlog is queued — the boundary jumps to wherever `steps` now
        is. PPO overrides this.
        """
        warmup, between = self.hp.memory_warmup, self.hp.steps_between_updates
        if steps < warmup:
            return False
        elapsed = steps - warmup
        # Floored, NOT `elapsed % between == 0`: `steps` advances in strides of
        # `parallel_envs`, and the residue can cycle without ever reaching 0.
        boundary = warmup + ((elapsed // between) * between)
        if boundary <= self._last_update_boundary:
            return False
        self._last_update_boundary = boundary
        return True

    def record_diagnostics(self, diagnostics: dict, steps: int) -> None:
        """Banks one learning pass's diagnostics, `steps` steps behind them.

        One DEVICE scalar per key (see `utils.reduce_diagnostics`), kept unread
        until the drain; see `DiagnosticsTracker.record`.
        """
        self.diagnostics.record(diagnostics, steps)

    def pop_diagnostics(self, env_steps: int = 0) -> dict:
        """This epoch's diagnostics, namespaced and RESET; `{}` if none ran.

        The trainer calls this once per epoch and logs the result under
        `train/`, passing its own `env_steps` for the rates. The reduction and
        `updates_per_env_step` are `DiagnosticsTracker.pop`'s; `buffer_frac` is
        the agent's, being the only party that knows its buffer size.
        """
        if self.diagnostics is None:
            return {}
        out = self.diagnostics.pop(env_steps)
        if out and env_steps > 0 and self.buffer_size is not None:
            out[self.diagnostics.key("buffer_frac")] = min(
                1.0, env_steps / float(self.buffer_size)
            )
        return out

    def _export_hyperparams(self) -> Dict[str, Any]:
        """The hyperparameter block written into every checkpoint.

        Flat, one key per knob, rebuilt by `Agent.load` into `hyperparams_cls`
        by field name. Every learning agent extends this through `super()`; a
        non-learning baseline, which has no `hp`, replaces it.
        """
        return {
            **dataclasses.asdict(self.hp),
            "obs_norm_clip": (
                -1.0 if self.hp.obs_norm_clip is None else self.hp.obs_norm_clip
            ),
            # Per-actuator lists, not just the first actuator's bounds.
            "action_low": serialize_bound(self.action_low),
            "action_high": serialize_bound(self.action_high),
        }

    def _replay_hyperparams(self) -> Dict[str, Any]:
        """The DERIVED block every replay-driven agent carries.

        No knobs, and the env sizes read back off the buffer rather than off
        `__init__`, because the shape it was ALLOCATED with is what a checkpoint
        has to be rebuilt against.
        """
        return {
            "env_obs_size": self.state.buffer_state.experience.observation.shape[2],
            "env_action_size": self.state.buffer_state.experience.action.shape[2],
            "memory_capacity": int(self.buffer_size),
            "memory_batch_size": int(self.batch_size),
        }

    def _checkpoint_modules(self) -> Dict[str, nnx.Module]:
        """Agent-owned nnx modules that live OUTSIDE `self.state`.

        DDPG's noise, SAC's temperature, MPO's duals and their optimizers, all
        of which resume would otherwise drop. Keys are attribute names,
        restored with `setattr`.
        """
        return {}

    @property
    def checkpointer(self) -> AgentCheckpointer:
        """This agent's reader/writer of checkpoints, built on first use.

        Everything about the on-disk format lives there; the three methods below
        are the agent-facing spelling of it.
        """
        if self._checkpointer is None:
            self._checkpointer = AgentCheckpointer(self)
        return self._checkpointer

    def checkpoint_payload(
        self,
        *,
        include_buffer: bool = False,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """The checkpoint's contents as host arrays; `None` for a baseline.

        See `AgentCheckpointer.payload`.
        """
        return self.checkpointer.payload(
            include_buffer=include_buffer, extra_metadata=extra_metadata
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        env_obs_size: int,
        env_act_size: int,
        **config_blocks,
    ):
        """Rebuilds an agent from a checkpoint; see `AgentCheckpointer.build`.

        The playback entry point, where the checkpoint is the only source of
        truth.
        """
        return AgentCheckpointer.build(
            cls, path, env_obs_size, env_act_size, **config_blocks
        )

    def restore(self, path: str | Path) -> Dict[str, Any]:
        """Loads a checkpoint's numeric state into this already-built agent.

        Returns the trainer-progress metadata it was written with; what resuming
        training goes through. See `AgentCheckpointer.restore`.
        """
        return self.checkpointer.restore(path)
