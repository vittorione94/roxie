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

from roxie.agents.utils import (
    DIAGNOSTIC_MAX_KEYS,
    DIAGNOSTIC_SUM_KEYS,
    BurstNode,
    Transition,
    make_optimizer,
    serialize_bound,
)
from roxie.models.actors import deterministic_action
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

    # How many nnx nodes this agent's fused bursts mutate, and therefore how
    # many `BurstNode` views it declares.
    _num_burst_nodes = 1

    # Held split so a burst costs a pytree pass rather than an `nnx.jit`
    # module-graph walk; see `roxie.agents.utils.SplitNodes`.
    state = BurstNode(0)

    @property
    def burst_nodes(self):
        """The `SplitNodes` a fused burst exchanges with the device.

        The acting burst (`JaxRollout`) and the gradient burst share it, so a
        window pays at most one `nnx.split` — and none at all unless host code
        materialized the live nodes in between.
        """
        return self._burst_nodes

    # Does acting read the observation statistics as they stood at the START of
    # a collect chunk, rather than as they stand at each step inside it? Only
    # PPO, which needs the mean/std byte-identical for a whole rollout or the
    # log-probs it stored at acting time stop matching its recomputed ratio.
    freeze_obs_norm_per_chunk = False

    def freeze_acting_norm(self):
        """Pin acting normalization to the current statistics.

        The host-side half of `freeze_obs_norm_per_chunk`: the rollout calls it
        at every chunk boundary, and the agent that pins (PPO) captures its
        snapshot there. A no-op for everyone else, who normalize against live
        statistics.
        """

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
        critic_model: nnx.Module = None,
    ):
        """Action selection for every stochastic-policy agent, in ONE dispatch.

        Returns ``(action, deviation_from_mode, pre_activation, log_probs,
        value)``. `action` is pre-scaling, in [-1, 1] — every stochastic actor
        here emits a `TanhNormal`, so that range is the distribution's own
        support and nothing has to bound it on the way out.
        `deviation_from_mode` is what the trainer reduces to the
        `noise/per_joint_abs` panel: these agents explore from their own policy,
        so the analogue of DDPG's injected noise is how far the sample landed
        from the mode. It is returned unconditionally (zero under `evaluate`)
        because the fused acting burst accumulates it inside a `lax.scan`, where
        a `None` would change the carry's structure.

        `critic_model` is given only by an on-policy agent, which needs the
        behaviour quantities at acting time: the value estimate, the pre-tanh
        draw, and the log-prob its ratio is recomputed against. Folding them in
        here keeps acting a single dispatch. Off-policy callers pass no critic
        and get None for all three, so neither the density nor the value ever
        enters their acting graph.

        The density is scored from the pre-tanh `u`, never re-derived from the
        action: `TanhNormal.log_prob` has to invert the squash through a clipped
        arctanh, which loses `u` entirely once it passes ~7.25 in float32.
        """
        distribution = actor_model(observation)
        mode = deterministic_action(distribution)
        # The mode's own pre-activation is the base mean; a sample's is the `u`
        # it was drawn from.
        if evaluate:
            action, pre_activation = mode, distribution.loc
        else:
            action, pre_activation = distribution.sample_from_pre(seed=key)

        if critic_model is None:
            return action, mode - action, None, None, None
        return (
            action,
            mode - action,
            pre_activation,
            distribution.log_prob_from_pre(pre_activation),
            critic_model(observation),
        )

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

    # Compiled lazily on first use, so an agent built from a checkpoint (or a
    # subclass that never buffers) pays nothing.
    _replay_add_jit = None

    def replay_add(self, buffer_state, transitions):
        """Add one env-step batch of transitions (leaves shaped (B, ...)).

        The trajectory buffer (n_step > 1) expects an explicit time axis on every
        leaf — (B, T=1, ...) for per-step adds — while the flat buffer takes the
        batch as-is. Callers go through here so they need not know which is
        active.

        Jitted and buffer-donating: flashbax's add is a dozen index computations
        and one `dynamic_update_slice` per leaf, and dispatching those eagerly
        cost 3.5 ms per env step against 0.05 ms compiled (measured on
        AcrobotSwingup / 256 envs). Donation is safe because every caller
        reassigns the slot it passed in.
        """
        if self._replay_add_jit is None:
            self._replay_add_jit = jax.jit(self._replay_add, donate_argnums=(0,))
        return self._replay_add_jit(buffer_state, transitions)

    def _replay_add(self, buffer_state, transitions):
        """The un-jitted body of `replay_add`, so a caller already inside a
        trace (the fused acting burst) skips the nested `pjit`.

        `n_step` defaults to 1 for an agent that declares no TD horizon (MPO,
        whose target is 1-step), which is the flat-buffer layout — no time axis.
        """
        if getattr(self, "n_step", 1) > 1:
            transitions = jax.tree.map(lambda x: x[:, None], transitions)
        return self.replay.add(buffer_state, transitions)

    @staticmethod
    def _pruned_transition(buffer_state, **fields):
        """A `Transition` carrying only the fields this buffer allocated.

        `transition_prototype` leaves a field off as `None` — an empty pytree
        node — and an add whose tree has a leaf where the store has None does not
        typecheck. MPO is the one agent that omits `truncation` (its 1-step
        target reads only `terminal`), so rather than have it reimplement the
        whole buffering path to drop one field, every caller builds through here
        and the buffer's own layout decides. Same idiom as `JaxRollout.warmup`,
        which prunes its scanned fill against this prototype.
        """
        proto = buffer_state.experience
        return Transition(**{
            name: (value if getattr(proto, name, None) is not None else None)
            for name, value in fields.items()
        })

    def buffer_transitions(
        self, state, prev_obs, action, reward, termination, truncation, next_obs,
        extras=None,
    ):
        """Write one env-step batch into `state`, IN PLACE.

        `state` is explicit rather than `self.state` so the same body serves the
        imperative per-step path and the fused acting burst, where the train
        state is a `lax.scan` carry rather than the agent's live attribute.

        `terminal` is true termination only: marking a time-limit truncation
        terminal zeroes its bootstrap and collapses Q at the cutoff, for every
        env at once since they hit the limit in lockstep. `truncation` is stored
        separately so n-step windows stop there too — in the flat stream the item
        after any done is a reset state.

        `extras` are the per-step fields an agent's `select_action` computed and
        its buffer stores alongside the standard five — PPO's behaviour log-prob
        and value estimate. They have to travel from acting to buffering as a
        VALUE because on the fused path both run inside one `lax.scan`, where
        the `self.last_*` attribute the per-step loop uses would be a traced
        value escaping its trace.
        """
        experiences = self._pruned_transition(
            state.buffer_state,
            observation=prev_obs,
            action=action,
            reward=reward,
            terminal=termination,
            truncation=truncation,
            **(extras or {}),
        )
        state.buffer_state = self._replay_add(state.buffer_state, experiences)

        if self.normalize_observations:
            obs_batch = jnp.concatenate([prev_obs, next_obs], axis=0)
            state.obs_stats = Agent.update_obs_stats(state.obs_stats, obs_batch)

    def add_transitions(
        self, prev_obs, action, reward, termination, truncation, next_obs,
        extras=None,
    ):
        """`buffer_transitions` against the agent's live state.

        Split out so the async learner (which owns `self.state` on its own
        thread) can add transitions whose action travelled with them through the
        hand-off queue, rather than reading `self.last_action`, which the acting
        thread overwrites every step.
        """
        # Not `_replay_add`: dispatched from Python once per env step, so it
        # wants the jitted, donating variant.
        experiences = self._pruned_transition(
            self.state.buffer_state,
            observation=prev_obs,
            action=action,
            reward=reward,
            terminal=termination,
            truncation=truncation,
            **(extras or {}),
        )
        self.state.buffer_state = self.replay_add(
            self.state.buffer_state, experiences
        )
        if self.normalize_observations:
            obs_batch = jnp.concatenate([prev_obs, next_obs], axis=0)
            self.state.obs_stats = Agent.update_obs_stats(
                self.state.obs_stats, obs_batch
            )

    def add(self, prev_obs, timestep):
        self.add_transitions(
            prev_obs,
            self.last_action,
            timestep.reward,
            timestep.terminated,
            timestep.truncated,
            timestep.obs,
            # The per-step counterpart to the fused path's carried `extras`:
            # `step` leaves them here for the `add` that follows.
            getattr(self, "last_extras", None),
        )

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
        # Every learning agent comes through here, so this is where the
        # diagnostics accumulator is born rather than in each `__init__`.
        self._diag_bursts = []

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
            # The buffer is the authority on observation width, so stats built
            # from it cannot disagree with `add`.
            obs_stats=Agent.init_obs_stats(
                buffer_state.experience.observation.shape[-1]
            ),
        )

    # Highest update boundary already served. Class-level so every off-policy
    # agent inherits it; the first firing shadows it per instance.
    _last_update_boundary = -1

    def due_for_update(self, steps: int) -> bool:
        """True at most once per `steps_between_updates` env steps past warmup.

        `memory_warmup` is the gate, and the only one: learning starts once the
        prefill is in the buffer. Sampling before that is not merely premature —
        flashbax allocates with `jnp.empty_like`, so a batch drawn below the
        buffer's `min_length` is uninitialized memory rather than an error.

        There used to be a second knob, `steps_before_learning`, and every
        config set it equal to `memory_warmup`. It could not do anything else:
        the trainer runs the warmup to completion before the loop's first call,
        so any value at or below the warmup gave a bit-identical schedule, and
        any value above it meant acting with an untrained policy while refusing
        to learn from the result.

        Do NOT write this as `(steps - memory_warmup) % between == 0`. The
        trainer advances `steps` in strides of `parallel_envs`, so that test
        only fires when the offset is itself a multiple of the stride;
        otherwise the residue cycles without reaching 0 and no gradient step
        ever runs. Tracking the last boundary served makes the schedule depend
        only on elapsed env steps.

        No backlog is queued: the boundary jumps to wherever `steps` now is, so
        a restored checkpoint resumes on schedule rather than firing a catch-up
        storm.
        """
        if steps < self.memory_warmup:
            return False
        elapsed = steps - self.memory_warmup
        boundary = self.memory_warmup + (
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

    # ---- Diagnostics ------------------------------------------------------
    #
    # Every learning agent reports the health of its own update through the
    # same two calls: `record_diagnostics` at the end of a gradient burst, and
    # `pop_diagnostics`, which the trainer drains once per epoch. What differs
    # between agents is only the KEYS — TD3's saturation and value-inflation
    # metrics, SAC's temperature and entropy, MPO's duals and KLs, PPO's trust
    # region — never the plumbing.

    # Set by `_init_train_state`. Class level so a non-learning baseline, which
    # never gets one, still answers `pop_diagnostics` rather than raising.
    _diag_bursts = ()
    # Run-to-date gradient steps recorded, so the rates below read as levels
    # rather than per-epoch spikes.
    _diag_steps = 0
    # What the last drain covered, for an override reporting a per-burst rate.
    _diag_drained_bursts = 0
    _diag_drained_steps = 0

    # Replay capacity, set by the agents that have one (DDPG, SAC, MPO and
    # their subclasses). None means "not replay-driven", which is what suppresses
    # the buffer-fill level for PPO and the baselines — an absent attribute
    # rather than a hook nobody implements.
    buffer_size = None

    def record_diagnostics(self, diagnostics: dict, steps: int) -> None:
        """Bank one gradient burst's already-reduced diagnostics.

        `diagnostics` holds one DEVICE scalar per key (see
        `utils.reduce_diagnostics`) and is kept unread until the drain, so a
        burst never forces a host sync on the loop's critical path. `steps` is
        how many gradient steps it covered, and weights the epoch mean.

        Appended from whichever thread owns the learner; `list.append` is atomic
        under the GIL, as is the swap in `pop_diagnostics`. The list is bounded
        by the bursts of one epoch, since the trainer drains it at every epoch
        boundary.
        """
        self._diag_bursts.append((int(steps), diagnostics))
        self._diag_steps += int(steps)

    def pop_diagnostics(self, env_steps: int = 0) -> dict:
        """This epoch's diagnostics, namespaced and reset.

        The trainer calls this on every agent once per epoch and logs the result
        under `train/`. Returns `{}` when no gradient burst ran, so nothing is
        logged rather than a misleading zero — which is indistinguishable from a
        converged metric and renders as a value rather than a gap.

        Keys are reduced across the epoch's bursts by the same rule that reduced
        each burst across its steps (`DIAGNOSTIC_MAX_KEYS` / `SUM`, otherwise a
        gradient-step-weighted mean), then prefixed with the agent's own name.

        `env_steps` is the trainer's step count, passed in rather than snapshotted
        during `update` so the rates below also hold on the async path, where the
        learner runs bursts off-thread and `update` is never called.
        """
        bursts, self._diag_bursts = self._diag_bursts, []
        if not bursts:
            return {}

        self._diag_drained_bursts = len(bursts)
        self._diag_drained_steps = sum(steps for steps, _ in bursts)

        prefix = type(self).__name__.lower()
        weights = jnp.asarray([float(steps) for steps, _ in bursts])
        out = {}
        # Every burst of one epoch comes from the same agent, so the first one's
        # keys are all of them.
        for key in bursts[0][1]:
            values = jnp.stack([jnp.asarray(burst[key]) for _, burst in bursts])
            if key in DIAGNOSTIC_MAX_KEYS:
                reduced = jnp.max(values)
            elif key in DIAGNOSTIC_SUM_KEYS:
                reduced = jnp.sum(values)
            else:
                reduced = jnp.sum(values * weights) / jnp.sum(weights)
            out[f"{prefix}/{key}"] = float(reduced)

        if env_steps > 0:
            # The realized replay ratio. Off the configured
            # `learning_steps / steps_between_updates` means the update gate is
            # not firing as intended — how the release_v1 walker grid was found
            # to have run at zero gradient steps.
            out[f"{prefix}/updates_per_env_step"] = self._diag_steps / env_steps
            if self.buffer_size is not None:
                # Below 1.0 the sampler draws from a narrower window than
                # configured, changing the off-policyness of every batch.
                out[f"{prefix}/buffer_frac"] = min(
                    1.0, env_steps / float(self.buffer_size)
                )
        return out

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

        # Reads the buffer's obs/action shapes, so it must run before the
        # detach below.
        hyperparams = self._export_hyperparams()
        saved_buffer = getattr(self.state, "buffer_state", None)
        if not include_buffer:
            self.state.buffer_state = None
        try:
            return {
                "format_version": format_version,
                # No graphdef: `restore` re-derives it from the live modules.
                "trainstate_state": jax.device_get(nnx.split(self.state)[1]),
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

        # Across the whole MRO: subclasses forward via ``*args, **kwargs``, so
        # inspecting only ``cls.__init__`` would drop the parent's hyperparams.
        valid_params = set()
        for klass in cls.__mro__:
            init = klass.__dict__.get("__init__")
            if init is not None:
                valid_params |= set(inspect.signature(init).parameters.keys())
        # Also coming from the checkpoint would make cls(**...) receive
        # duplicate keyword arguments.
        explicit_keys = {"env_obs_size", "env_action_size", *config_blocks}
        filtered_hyper = {
            k: v
            for k, v in (hyper or {}).items()
            if k in valid_params and k not in explicit_keys
        }
        # Serialized as a plain list; everything downstream expects an ndarray.
        for key in ("action_low", "action_high"):
            if key in filtered_hyper:
                filtered_hyper[key] = jnp.asarray(
                    filtered_hyper[key], dtype=jnp.float32
                )

        init_kwargs = dict(
            env_obs_size=env_obs_size,
            env_action_size=env_act_size,
            # A block the config omits is dropped rather than passed as None,
            # so the constructor's own default applies.
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
            # checkpoints, so a missing entry is normal.
            if getattr(self.state, name, None) is not None:
                _restore_module(self.state, name, ckpt_state)

        extra_ckpt = loaded.get("extra_state") or {}
        for name in self._checkpoint_modules():
            if restore_optimizers or not name.endswith("optimizer"):
                _restore_module(self, name, extra_ckpt)

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

        # Only present if saved with `include_buffer=True`.
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

