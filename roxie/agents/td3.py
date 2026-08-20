import functools

import hydra
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from roxie.agents.agent import Agent, TrainState
from roxie.agents.ddpg import DDPG
from roxie.agents.utils import network_rngs, repack_samples
from roxie.losses.actor_losses import td3_actor_loss_fn
from roxie.losses.critic_losses import td3_critic_loss_fn
from roxie.models.critics import TwinCritic

# Diagnostic keys produced by `td3_actor_loss_fn`'s aux. Kept here (rather than
# inferred) because `nnx.cond`'s skipped branch has to synthesize a structurally
# identical dict on the steps where the delayed policy update does not run.
ACTOR_DIAGNOSTIC_KEYS = (
    "pre_act_abs",
    "pre_act_max",
    "pre_act_penalty",
    "sat_frac",
    "tanh_grad",
    "actor_q",
)

# Metrics reduced with max rather than mean across the fused burst: an epoch
# mean of a per-batch maximum would wash out exactly the outlier it is there to
# expose.
MAX_REDUCED_KEYS = frozenset({"pre_act_max"})


# Single TD3 gradient step. Not jitted on its own — called inside the jitted
# `_grad_steps` below so N steps fuse into one compiled program. `update_actor`
# is a Python bool (static at trace time): when False the actor and target
# updates are simply not traced, implementing the delayed-policy-update trick.
def _grad_step(
    state: TrainState,
    key: jax.random.PRNGKey,
    gamma: float,
    tau: float,
    replay_sample_fn,
    target_policy_noise: float,
    target_noise_clip: float,
    action_low: float,
    action_high: float,
    obs_mean: jnp.ndarray,
    obs_std: jnp.ndarray,
    obs_clip: float,
    normalize: bool,
    update_actor,
    pre_activation_coef: float,
    n_step: int = 1,
):
    """One TD3 step. `obs_mean`/`obs_std` are hoisted in by `_grad_steps` (the
    stats are loop-constant). `update_actor` is a *traced* boolean here: the actor
    and target updates run under `nnx.cond` so the delayed-policy-update trick
    survives the `lax.scan` (where the step index is no longer static).
    `n_step` is the TD horizon (NOT the scan length)."""
    # `repack_samples` folds the n-step return, bootstrap coefficient, and
    # bootstrap obs into the dict, so the critic loss never sees gamma/terminals.
    key, noise_key = jax.random.split(key)
    samples = replay_sample_fn(state.buffer_state, key)
    re_packed_samples = repack_samples(samples, gamma, n_step)
    # Normalize once, here: the critic and (delayed) actor losses read the same
    # `observations`, and neither normalizes (see `Agent.normalize_samples`).
    re_packed_samples = Agent.normalize_samples(
        re_packed_samples, obs_mean, obs_std, obs_clip, normalize
    )

    # Critic update (twin critic, clipped double-Q target), on every step.
    (critic_loss, critic_aux), critic_grads = nnx.value_and_grad(
        td3_critic_loss_fn, has_aux=True
    )(
        state.critic,
        state.target_actor,
        state.target_critic,
        re_packed_samples,
        noise_key,
        target_policy_noise,
        target_noise_clip,
        action_low,
        action_high,
    )
    state.critic_optimizer.update(state.critic, critic_grads)

    # Delayed actor + target updates, only on steps where `update_actor` is True.
    # Under `lax.scan` the step index is traced, so this must be a runtime branch
    # (`nnx.cond`) rather than a Python `if`.
    def _actor_update(state):
        (actor_loss, actor_aux), actor_grads = nnx.value_and_grad(
            td3_actor_loss_fn, has_aux=True
        )(
            state.actor,
            state.critic,
            re_packed_samples,
            action_low,
            action_high,
            pre_activation_coef,
        )
        state.actor_optimizer.update(state.actor, actor_grads)

        # Soft-update both target networks alongside the policy.
        new_actor_tensors = nnx.state(state.actor, nnx.Param)
        old_actor_tensors = nnx.state(state.target_actor, nnx.Param)
        new_target_actor_tensors = optax.incremental_update(
            new_tensors=new_actor_tensors, old_tensors=old_actor_tensors, step_size=tau
        )

        new_critic_tensors = nnx.state(state.critic, nnx.Param)
        old_critic_tensors = nnx.state(state.target_critic, nnx.Param)
        new_target_critic_tensors = optax.incremental_update(
            new_tensors=new_critic_tensors,
            old_tensors=old_critic_tensors,
            step_size=tau,
        )

        nnx.update(state.target_actor, new_target_actor_tensors)
        nnx.update(state.target_critic, new_target_critic_tensors)
        return actor_loss, actor_aux

    def _skip_actor_update(state):
        zero = jnp.array(0.0, dtype=critic_loss.dtype)
        # `nnx.cond` requires both branches to return the same pytree, so the
        # skipped branch has to mirror the aux dict exactly. The zeros are never
        # averaged in: `_grad_steps` divides the actor sums by the number of
        # *update* steps, matching how `actor_loss` is already handled.
        return zero, {k: zero for k in ACTOR_DIAGNOSTIC_KEYS}

    actor_loss, actor_aux = nnx.cond(
        update_actor, _actor_update, _skip_actor_update, state
    )

    return (
        TrainState(
            actor=state.actor,
            critic=state.critic,
            actor_optimizer=state.actor_optimizer,
            target_actor=state.target_actor,
            target_critic=state.target_critic,
            critic_optimizer=state.critic_optimizer,
            buffer_state=state.buffer_state,
            obs_stats=state.obs_stats,
        ),
        actor_loss,
        critic_loss,
        actor_aux,
        critic_aux,
    )


# Fused N-step update. The body is compiled once and run `n_steps` times on-device
# via `lax.scan` rather than unrolled, which would blow up compile time and HLO
# size at large `n_steps`. The delayed-policy-update schedule is precomputed as a
# boolean mask scanned over alongside the per-step keys. Only the trainable graph state is
# carried; `buffer_state` and the normalization params are loop-constant.
@functools.partial(
    nnx.jit,
    static_argnames=(
        "gamma", "tau", "replay_sample_fn", "n_steps", "policy_delay", "n_step",
        "normalize",
    ),
    # Donate the train state (arg 0): its large read-only replay buffer is threaded
    # unchanged through the scan, so without donation XLA allocates a full second
    # copy of it every update. The caller reassigns self.state from the result.
    donate_argnums=(0,),
)
def _grad_steps(
    state: TrainState,
    key: jax.random.PRNGKey,
    n_steps: int,
    gamma: float,
    tau: float,
    replay_sample_fn,
    target_policy_noise: float,
    target_noise_clip: float,
    action_low: float,
    action_high: float,
    obs_eps: float,
    obs_clip: float,
    normalize: bool,
    policy_delay: int,
    pre_activation_coef: float,
    n_step: int = 1,
):
    # Hoist the (loop-constant) normalization params out of the scan body.
    obs_mean, obs_std = Agent.obs_mean_std(state.obs_stats, obs_eps)

    # Pre-split per-step keys and precompute the delayed-update schedule.
    keys = jax.random.split(key, n_steps)
    update_mask = (jnp.arange(n_steps) % policy_delay) == 0

    graphdef, scan_state = nnx.split(state)

    def body(scan_state, xs):
        step_key, update_actor = xs
        st = nnx.merge(graphdef, scan_state)
        st, actor_loss, critic_loss, actor_aux, critic_aux = _grad_step(
            st,
            step_key,
            gamma,
            tau,
            replay_sample_fn,
            target_policy_noise,
            target_noise_clip,
            action_low,
            action_high,
            obs_mean,
            obs_std,
            obs_clip,
            normalize,
            update_actor,
            pre_activation_coef,
            n_step,
        )
        _, scan_state = nnx.split(st)
        return scan_state, (actor_loss, critic_loss, actor_aux, critic_aux)

    scan_state, (actor_losses, critic_losses, actor_aux, critic_aux) = jax.lax.scan(
        body, scan_state, (keys, update_mask)
    )
    state = nnx.merge(graphdef, scan_state)

    # Actor loss only on update steps → average over those; critic over all steps.
    n_actor_updates = jnp.maximum(jnp.sum(update_mask), 1)
    actor_loss = jnp.sum(actor_losses) / n_actor_updates

    def _reduce(per_step, key, denom):
        if key in MAX_REDUCED_KEYS:
            return jnp.max(per_step)
        return jnp.sum(per_step) / denom

    # Reduced on-device to one scalar per metric, so the host only ever sees a
    # handful of numbers per burst instead of an (n_steps,) array per metric.
    diagnostics = {
        **{k: _reduce(v, k, n_actor_updates) for k, v in actor_aux.items()},
        **{k: _reduce(v, k, n_steps) for k, v in critic_aux.items()},
    }
    return state, actor_loss, jnp.mean(critic_losses), diagnostics


class TD3(DDPG):
    """Twin Delayed DDPG.

    Extends DDPG with the three TD3 stabilizers: clipped double-Q critics,
    target policy smoothing (already present in DDPG's target construction),
    and delayed policy/target updates. Everything else — action selection,
    replay handling, observation normalization — is inherited unchanged.
    """

    def __init__(self, *args, policy_delay: int = 2, **kwargs):
        self.policy_delay = int(policy_delay)
        super().__init__(*args, **kwargs)
        # Per-burst diagnostic scalars, drained once per epoch by the trainer via
        # `pop_diagnostics`. Kept as device arrays and only reduced to floats at
        # drain time, so no burst pays a host sync on the loop's critical path.
        self._diag_bursts: list[dict] = []
        self._diag_grad_steps = 0
        self._diag_env_steps = 0

    def _make_critic(self, critic_config, env_obs_size, env_action_size):
        """Two Q heads behind a TwinCritic. The heads get distinct seed offsets
        so they don't start identical — the double-Q min is worthless if they
        do."""
        critic1 = hydra.utils.instantiate(
            critic_config,
            in_features=env_obs_size + env_action_size,
            rngs=network_rngs(self.seed, offset=2),
        )
        critic2 = hydra.utils.instantiate(
            critic_config,
            in_features=env_obs_size + env_action_size,
            rngs=network_rngs(self.seed, offset=4),
        )
        return TwinCritic(critic1, critic2)

    def learn(self, agent_rng, n_steps=None):
        """One unconditional burst (TD3 variant: threads `policy_delay` into the
        fused grad step). See DDPG.learn for the sync/async sharing rationale."""
        burst_steps = self.learning_steps if n_steps is None else int(n_steps)
        self.state, actor_loss, critic_loss, diagnostics = _grad_steps(
            self.state,
            agent_rng,
            burst_steps,
            self.gamma,
            self.tau,
            self.replay.sample,
            self.target_policy_noise,
            self.target_noise_clip,
            self.action_low,
            self.action_high,
            self.obs_eps,
            self.obs_clip,
            self.normalize_observations,
            self.policy_delay,
            self.pre_activation_coef,
            n_step=self.n_step,
        )
        # Appended from whichever thread owns the learner (sync loop or the async
        # learner thread); list.append is atomic under the GIL and the trainer
        # drains on the same cadence it reads the losses.
        self._diag_bursts.append(diagnostics)
        self._diag_grad_steps += burst_steps
        return actor_loss, critic_loss

    def update(self, steps, agent_rng):
        gradient_steps, actor_loss, critic_loss = 0, 0, 0

        if self.due_for_update(steps):
            actor_loss, critic_loss = self.learn(agent_rng)
            gradient_steps += self.learning_steps

        self._diag_env_steps = steps
        return gradient_steps, actor_loss, critic_loss

    def pop_diagnostics(self) -> dict:
        """Return this epoch's TD3 health metrics and reset the accumulators.

        Optional agent hook (see `Trainer`): PPO's counterpart reports whether
        the trust region is intact; TD3's reports whether the two things that
        actually break it are intact — the actor's tanh saturation and the
        critic's value inflation.

        Actor block. `td3/tanh_grad` is the mean d(tanh u)/du, i.e. the factor
        every actor gradient is multiplied by; as it falls toward 0 the policy
        freezes into a bang-bang controller and the deterministic eval score
        detaches from the (noise-softened) training score. `td3/pre_act_abs` and
        `td3/sat_frac` are the same story in the units you tune with, and
        `td3/pre_act_penalty` times `pre_activation_coef` is directly comparable
        against `td3/actor_q` — if that product is orders of magnitude below the
        Q term, the saturation hinge is inert and the coefficient is too small.

        Critic block. `td3/q_buffer` vs `td3/q_target` and `td3/twin_gap` track
        value inflation; `td3/target_act_rail_frac` says how much of the
        bootstrap is being evaluated at action-space corners, and
        `td3/target_smooth_clip_frac` whether `target_noise_clip` is inert (~0)
        or has collapsed the smoothing Gaussian into two spikes (~1).

        Returns ``{}`` when no gradient burst ran this epoch, so the trainer
        logs nothing rather than a misleading zero.
        """
        if not self._diag_bursts:
            return {}
        bursts, self._diag_bursts = self._diag_bursts, []

        out = {}
        for key in bursts[0]:
            values = jnp.stack([jnp.asarray(b[key]) for b in bursts])
            reduced = jnp.max(values) if key in MAX_REDUCED_KEYS else jnp.mean(values)
            out[f"td3/{key}"] = float(reduced)

        # Run-to-date (not per-epoch) counters, so these read as a level rather than
        # a rate that jitters with epoch boundaries. Only available on the sync path:
        # the async learner drives `learn` directly and never calls `update`, so
        # there is no env-step count to divide by.
        if self._diag_env_steps > 0:
            # Realized replay ratio, the single number `steps_between_updates` and
            # `learning_steps` jointly control.
            out["td3/updates_per_env_step"] = (
                self._diag_grad_steps / self._diag_env_steps
            )
            # How much of the replay buffer holds real data. Below 1.0 the sampler
            # draws from a window narrower than configured, which changes the
            # effective off-policyness of every batch.
            out["td3/buffer_frac"] = min(
                1.0, self._diag_env_steps / float(self.buffer_size)
            )
        return out

    def _export_hyperparams(self) -> dict:
        params = super()._export_hyperparams()
        params["policy_delay"] = int(self.policy_delay)
        return params
