import functools

import hydra
import jax
import jax.numpy as jnp
from flax import nnx

from roxie.agents.agent import Agent, TrainState
from roxie.agents.ddpg import DDPG
from roxie.agents.utils import (
    fused_grad_steps,
    network_rngs,
    repack_samples,
    soft_update,
)
from roxie.losses.actor_losses import td3_actor_loss_fn
from roxie.losses.critic_losses import td3_critic_loss_fn
from roxie.models.critics import TwinCritic

# Listed here rather than inferred because `nnx.cond`'s skipped branch has to
# synthesize a structurally identical dict when the actor update does not run.
ACTOR_DIAGNOSTIC_KEYS = (
    "pre_act_abs",
    "pre_act_max",
    "pre_act_penalty",
    "sat_frac",
    "tanh_grad",
    "actor_q",
)

# Reduced with max rather than mean across the fused burst: an epoch mean of a
# per-batch maximum would wash out exactly the outlier it is there to expose.
MAX_REDUCED_KEYS = frozenset({"pre_act_max"})


def _grad_step(
    state: TrainState,
    key: jax.random.PRNGKey,
    update_actor,
    *,
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
    pre_activation_coef: float,
    n_step: int = 1,
):
    """One TD3 step, called inside the jitted `_grad_steps` below so N steps fuse
    into one compiled program. `update_actor` is a *traced* boolean: the actor
    and target updates run under `nnx.cond` so the delayed-policy-update trick
    survives the `lax.scan`, where the step index is no longer static. `n_step`
    is the TD horizon, not the scan length."""
    # `repack_samples` folds the n-step return, bootstrap coefficient, and
    # bootstrap obs into the dict, so the critic loss never sees gamma/terminals.
    key, noise_key = jax.random.split(key)
    samples = replay_sample_fn(state.buffer_state, key)
    re_packed_samples = repack_samples(samples, gamma, n_step)
    # Normalized once here: the critic and (delayed) actor losses read the same
    # `observations`, and neither normalizes.
    re_packed_samples = Agent.normalize_samples(
        re_packed_samples, obs_mean, obs_std, obs_clip, normalize
    )

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

        # Both targets move with the policy, not with the critic.
        soft_update(state.target_actor, state.actor, tau)
        soft_update(state.target_critic, state.critic, tau)
        return actor_loss, actor_aux

    def _skip_actor_update(state):
        zero = jnp.array(0.0, dtype=critic_loss.dtype)
        # The zeros are never averaged in: `_grad_steps` divides the actor sums
        # by the number of *update* steps.
        return zero, {k: zero for k in ACTOR_DIAGNOSTIC_KEYS}

    actor_loss, actor_aux = nnx.cond(
        update_actor, _actor_update, _skip_actor_update, state
    )

    return actor_loss, critic_loss, actor_aux, critic_aux


# `fused_grad_steps` compiles the body once and runs it `n_steps` times
# on-device, so a burst costs one host dispatch rather than one per step.
@functools.partial(
    nnx.jit,
    static_argnames=(
        "gamma", "tau", "replay_sample_fn", "n_steps", "policy_delay", "n_step",
        "normalize",
    ),
    # The replay buffer rides unchanged through the scan; without donation XLA
    # allocates a full second copy of it every update.
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
    # Loop-constant, so hoisted out of the scan body.
    obs_mean, obs_std = Agent.obs_mean_std(state.obs_stats, obs_eps)

    # The delayed-update schedule, scanned over alongside the keys.
    update_mask = (jnp.arange(n_steps) % policy_delay) == 0

    state, (actor_losses, critic_losses, actor_aux, critic_aux) = fused_grad_steps(
        state,
        key,
        n_steps,
        functools.partial(
            _grad_step,
            gamma=gamma,
            tau=tau,
            replay_sample_fn=replay_sample_fn,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            action_low=action_low,
            action_high=action_high,
            obs_mean=obs_mean,
            obs_std=obs_std,
            obs_clip=obs_clip,
            normalize=normalize,
            pre_activation_coef=pre_activation_coef,
            n_step=n_step,
        ),
        extras=(update_mask,),
    )

    # Actor loss only on update steps → average over those; critic over all steps.
    n_actor_updates = jnp.maximum(jnp.sum(update_mask), 1)
    actor_loss = jnp.sum(actor_losses) / n_actor_updates

    def _reduce(per_step, key, denom):
        if key in MAX_REDUCED_KEYS:
            return jnp.max(per_step)
        return jnp.sum(per_step) / denom

    # Reduced on-device so the host sees one scalar per metric per burst rather
    # than an (n_steps,) array.
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
        # Kept as device arrays and only reduced to floats at drain time, so no
        # burst pays a host sync on the loop's critical path.
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
        """One unconditional burst, threading `policy_delay` into the fused grad
        step. See DDPG.learn for the sync/async sharing rationale."""
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
        # Appended from whichever thread owns the learner; list.append is atomic
        # under the GIL.
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

        Optional agent hook (see `Trainer`), tracking the two things that break
        TD3: the actor's tanh saturation and the critic's value inflation.

        `td3/tanh_grad` is the mean d(tanh u)/du that every actor gradient is
        multiplied by; as it falls toward 0 the policy freezes into a bang-bang
        controller. `td3/pre_act_penalty` times `pre_activation_coef` is
        comparable against `td3/actor_q` — orders of magnitude below it means the
        saturation hinge is inert.

        `td3/q_buffer` vs `td3/q_target` and `td3/twin_gap` track value
        inflation; `td3/target_act_rail_frac` says how much of the bootstrap is
        evaluated at action-space corners, and `td3/target_smooth_clip_frac`
        whether `target_noise_clip` is inert (~0) or has collapsed the smoothing
        Gaussian into two spikes (~1).

        Returns ``{}`` when no gradient burst ran this epoch.
        """
        if not self._diag_bursts:
            return {}
        bursts, self._diag_bursts = self._diag_bursts, []

        out = {}
        for key in bursts[0]:
            values = jnp.stack([jnp.asarray(b[key]) for b in bursts])
            reduced = jnp.max(values) if key in MAX_REDUCED_KEYS else jnp.mean(values)
            out[f"td3/{key}"] = float(reduced)

        # Run-to-date rather than per-epoch, so these read as a level. Sync path
        # only: the async learner drives `learn` directly and never calls
        # `update`, so there is no env-step count to divide by.
        if self._diag_env_steps > 0:
            # Realized replay ratio, which `steps_between_updates` and
            # `learning_steps` jointly control.
            out["td3/updates_per_env_step"] = (
                self._diag_grad_steps / self._diag_env_steps
            )
            # Below 1.0 the sampler draws from a narrower window than configured,
            # changing the effective off-policyness of every batch.
            out["td3/buffer_frac"] = min(
                1.0, self._diag_env_steps / float(self.buffer_size)
            )
        return out

    def _export_hyperparams(self) -> dict:
        params = super()._export_hyperparams()
        params["policy_delay"] = int(self.policy_delay)
        return params
