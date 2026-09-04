import functools

import hydra
import jax
import jax.numpy as jnp
from flax import nnx

from roxie.agents.agent import Agent, TrainState
from roxie.agents.d4pg import D4PG
from roxie.agents.utils import (
    fused_grad_steps,
    graph_jit,
    network_rngs,
    reduce_diagnostics,
    repack_samples,
    soft_update,
)
from roxie.losses.actor_losses import skipped_actor_aux, td4_actor_loss_fn
from roxie.losses.critic_losses import td4_critic_loss_fn
from roxie.models.critics import TwinCritic


# D4PG's step (categorical critic on the fixed support `atoms`) with TD3's twin
# critic and delayed policy update bolted on. Not jitted on its own — called
# inside `_grad_steps` so N steps fuse into one compiled program.
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
    atoms: jnp.ndarray,
    pre_activation_coef: float,
    n_step: int = 1,
):
    """One TD4 step. `obs_mean`/`obs_std` and `atoms` are hoisted in by
    `_grad_steps` (all loop-constant). `update_actor` is a *traced* boolean: the
    actor and target updates run under `nnx.cond` so the delayed-policy-update
    trick survives the `lax.scan` (where the step index is no longer static).
    `n_step` is the TD horizon (NOT the scan length `n_steps`)."""
    # `repack_samples` folds the n-step return, bootstrap coefficient and
    # bootstrap obs into the dict — what shifts the categorical support.
    key, noise_key = jax.random.split(key)
    samples = replay_sample_fn(state.buffer_state, key)
    re_packed_samples = repack_samples(samples, gamma, n_step)
    # Normalized once here: the critic and actor losses read the same
    # `observations`.
    re_packed_samples = Agent.normalize_samples(
        re_packed_samples, obs_mean, obs_std, obs_clip, normalize
    )

    (critic_loss, critic_aux), critic_grads = nnx.value_and_grad(
        td4_critic_loss_fn, has_aux=True
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
        atoms,
    )
    state.critic_optimizer.update(state.critic, critic_grads)

    # Under `lax.scan` the step index is traced, so the delayed update has to
    # be a runtime branch rather than a Python `if`.
    def _actor_update(state):
        (actor_loss, actor_aux), actor_grads = nnx.value_and_grad(
            td4_actor_loss_fn, has_aux=True
        )(
            state.actor,
            state.critic,
            re_packed_samples,
            action_low,
            action_high,
            atoms,
            pre_activation_coef,
        )
        state.actor_optimizer.update(state.actor, actor_grads)

        # Both targets move with the policy, not with the critic.
        soft_update(state.target_actor, state.actor, tau)
        soft_update(state.target_critic, state.critic, tau)
        return actor_loss, actor_aux

    def _skip_actor_update(state):
        zero = jnp.array(0.0, dtype=critic_loss.dtype)
        return zero, skipped_actor_aux(critic_loss.dtype, "pre_act_penalty")

    actor_loss, actor_aux = nnx.cond(
        update_actor, _actor_update, _skip_actor_update, state
    )

    return actor_loss, critic_loss, actor_aux, critic_aux


@functools.partial(
    graph_jit,
    static_argnames=(
        "gamma", "tau", "replay_sample_fn", "n_steps", "policy_delay", "n_step",
        "normalize",
    ),
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
    pre_activation_coef: float,
    atoms: jnp.ndarray,
    policy_delay: int,
    n_step: int = 1,
):
    obs_mean, obs_std = Agent.obs_mean_std(state.obs_stats, obs_eps)

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
            atoms=atoms,
            pre_activation_coef=pre_activation_coef,
            n_step=n_step,
        ),
        extras=(update_mask,),
    )

    # Actor loss only on update steps → average over those; critic over all steps.
    # The diagnostics split the same way: the skipped steps contributed zeros.
    n_actor_updates = jnp.maximum(jnp.sum(update_mask), 1)
    actor_loss = jnp.sum(actor_losses) / n_actor_updates
    diagnostics = {
        **reduce_diagnostics(actor_aux, n_actor_updates),
        **reduce_diagnostics(critic_aux, n_steps),
    }
    return state, actor_loss, jnp.mean(critic_losses), diagnostics


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
        burst_steps = self.learning_steps if n_steps is None else int(n_steps)
        actor_loss, critic_loss, diagnostics = _grad_steps(
            self._burst_nodes,
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
            self.pre_activation_coef,
            self.atoms,
            self.policy_delay,
            n_step=self.n_step,
        )
        self.record_diagnostics(diagnostics, burst_steps)
        return actor_loss, critic_loss

    def _export_hyperparams(self) -> dict:
        params = super()._export_hyperparams()
        params["policy_delay"] = int(self.policy_delay)
        return params
