import functools

import hydra
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from roxie.agents.agent import Agent, TrainState
from roxie.agents.d4pg import D4PG
from roxie.agents.utils import network_rngs, repack_samples
from roxie.losses.actor_losses import td4_actor_loss_fn
from roxie.losses.critic_losses import td4_critic_loss_fn
from roxie.models.critics import TwinCritic


# Single TD4 gradient step. Not jitted on its own — called inside the jitted
# `_grad_steps` below so N steps fuse into one compiled program. This is D4PG's
# step (categorical critic on the fixed support `atoms`) with TD3's twin critic
# and delayed policy update bolted on.
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
    atoms: jnp.ndarray,
    update_actor,
    n_step: int = 1,
):
    """One TD4 step. `obs_mean`/`obs_std` and `atoms` are hoisted in by
    `_grad_steps` (all loop-constant). `update_actor` is a *traced* boolean: the
    actor and target updates run under `nnx.cond` so the delayed-policy-update
    trick survives the `lax.scan` (where the step index is no longer static).
    `n_step` is the TD horizon (NOT the scan length `n_steps`)."""
    # `repack_samples` folds the n-step return, bootstrap coefficient, and
    # bootstrap obs into the dict — exactly the ingredients the categorical
    # projection needs to shift the support.
    key, noise_key = jax.random.split(key)
    samples = replay_sample_fn(state.buffer_state, key)
    re_packed_samples = repack_samples(samples, gamma, n_step)
    # Normalize once, here: the critic and (delayed) actor losses read the same
    # `observations`, and neither normalizes (see `Agent.normalize_samples`).
    re_packed_samples = Agent.normalize_samples(
        re_packed_samples, obs_mean, obs_std, obs_clip, normalize
    )

    # Critic update: cross-entropy against the projected target categorical of the
    # pessimistic target head, on every step.
    critic_loss, critic_grads = nnx.value_and_grad(td4_critic_loss_fn)(
        state.critic,
        state.target_actor,
        state.target_critic,
        re_packed_samples,
        noise_key,
        target_policy_noise,
        target_noise_clip,
        action_low,
        action_high,
        atoms,
    )
    state.critic_optimizer.update(state.critic, critic_grads)

    # Delayed actor + target updates, only on steps where `update_actor` is True.
    # Under `lax.scan` the step index is traced, so this must be a runtime branch
    # (`nnx.cond`) rather than a Python `if`.
    def _actor_update(state):
        actor_loss, actor_grads = nnx.value_and_grad(td4_actor_loss_fn)(
            state.actor,
            state.critic,
            re_packed_samples,
            action_low,
            action_high,
            atoms,
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
        return actor_loss

    def _skip_actor_update(state):
        return jnp.array(0.0, dtype=critic_loss.dtype)

    actor_loss = nnx.cond(update_actor, _actor_update, _skip_actor_update, state)

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
    )


# Fused N-step update. The body is compiled once and run `n_steps` times on-device
# via `lax.scan` rather than unrolled, which would blow up compile time and HLO
# size at large `n_steps`. The delayed-policy-update schedule is precomputed as a
# boolean mask scanned over alongside the per-step keys. Only the trainable graph state is
# carried; `buffer_state`, the normalization params, and `atoms` are loop-constant.
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
    atoms: jnp.ndarray,
    policy_delay: int,
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
        st, actor_loss, critic_loss = _grad_step(
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
            atoms,
            update_actor,
            n_step,
        )
        _, scan_state = nnx.split(st)
        return scan_state, (actor_loss, critic_loss)

    scan_state, (actor_losses, critic_losses) = jax.lax.scan(
        body, scan_state, (keys, update_mask)
    )
    state = nnx.merge(graphdef, scan_state)

    # Actor loss only on update steps → average over those; critic over all steps.
    actor_loss = jnp.sum(actor_losses) / jnp.maximum(jnp.sum(update_mask), 1)
    return state, actor_loss, jnp.mean(critic_losses)


class TD4(D4PG):
    """Twin Delayed D4PG — D4PG with TD3's twin critic and delayed updates.

    Extends D4PG with two of TD3's three stabilizers: a *pair* of categorical
    critics with a clipped double-Q bootstrap, and delayed policy/target
    updates. (The third, target policy smoothing, already exists in the shared
    target construction; D4PG configs just set the noise to 0 — TD4 turns it
    back on by default.)

    The clipped double-Q carries over to distributions by picking, per sample,
    the target head with the lower *expected* value and using its whole
    categorical as the bootstrap — an elementwise min over atom probabilities
    would not be a distribution. Both heads are then fit to that same projected
    target; the actor follows head 1's mean. See `td4_critic_loss_fn`.

    Everything else — the support, action selection, replay handling,
    observation normalization — is inherited from D4PG/DDPG unchanged, so
    `v_min`/`v_max` must still bracket the achievable n-step-discounted return.
    """

    def __init__(self, *args, policy_delay: int = 2, **kwargs):
        self.policy_delay = int(policy_delay)
        super().__init__(*args, **kwargs)

    def _make_critic(self, critic_config, env_obs_size, env_action_size):
        """Two categorical critics behind a TwinCritic. `num_atoms` is injected
        from the agent args so the critic yaml doesn't have to repeat it; the
        two heads get distinct seed offsets so they don't start identical (the
        double-Q min is worthless if they are)."""
        critic1 = hydra.utils.instantiate(
            critic_config,
            in_features=env_obs_size + env_action_size,
            num_atoms=self.num_atoms,
            rngs=network_rngs(self.seed, offset=2),
        )
        critic2 = hydra.utils.instantiate(
            critic_config,
            in_features=env_obs_size + env_action_size,
            num_atoms=self.num_atoms,
            rngs=network_rngs(self.seed, offset=4),
        )
        return TwinCritic(critic1, critic2)

    def learn(self, agent_rng, n_steps=None):
        """One unconditional burst (TD4 variant: threads both the categorical
        `atoms` support and `policy_delay` into the fused grad step). See
        DDPG.learn for the sync/async sharing rationale."""
        self.state, actor_loss, critic_loss = _grad_steps(
            self.state,
            agent_rng,
            self.learning_steps if n_steps is None else int(n_steps),
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
            self.atoms,
            self.policy_delay,
            n_step=self.n_step,
        )
        return actor_loss, critic_loss

    def _export_hyperparams(self) -> dict:
        params = super()._export_hyperparams()
        params["policy_delay"] = int(self.policy_delay)
        return params
