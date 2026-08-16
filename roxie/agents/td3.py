import functools

import hydra
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from roxie.agents.agent import Agent, TrainState
from roxie.agents.ddpg import DDPG
from roxie.agents.utils import repack_samples
from roxie.losses.actor_losses import td3_actor_loss_fn
from roxie.losses.critic_losses import td3_critic_loss_fn
from roxie.models.critics import TwinCritic


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
    update_actor,
    pre_activation_coef: float,
    n_step: int = 1,
):
    """One TD3 step. `obs_mean`/`obs_std` are hoisted in by `_grad_steps` (the
    stats are loop-constant). `update_actor` is a *traced* boolean here: the actor
    and target updates run under `nnx.cond` so the delayed-policy-update trick
    survives the `lax.scan` (where the step index is no longer static).
    `n_step` is the TD horizon (NOT the scan length)."""
    # 1. Sample from the replay buffer. `repack_samples` folds the n-step
    # return, bootstrap coefficient, and bootstrap obs into the dict, so the
    # critic loss no longer sees gamma/terminals directly.
    key, noise_key = jax.random.split(key)
    samples = replay_sample_fn(state.buffer_state, key)
    re_packed_samples = repack_samples(samples, gamma, n_step)

    # 2. Critic update (twin critic, clipped double-Q target) — every step
    critic_loss, critic_grads = nnx.value_and_grad(td3_critic_loss_fn)(
        state.critic,
        state.target_actor,
        state.target_critic,
        re_packed_samples,
        noise_key,
        target_policy_noise,
        target_noise_clip,
        action_low,
        action_high,
        obs_mean,
        obs_std,
        obs_clip,
    )
    state.critic_optimizer.update(state.critic, critic_grads)

    # 3. Delayed actor + target updates — only on steps where `update_actor` is
    # True. Under `lax.scan` the step index is traced, so this must be a runtime
    # branch (`nnx.cond`) rather than a Python `if`.
    def _actor_update(state):
        actor_loss, actor_grads = nnx.value_and_grad(td3_actor_loss_fn)(
            state.actor,
            state.critic,
            re_packed_samples,
            obs_mean,
            obs_std,
            obs_clip,
            action_low,
            action_high,
            pre_activation_coef,
        )
        state.actor_optimizer.update(state.actor, actor_grads)

        # Soft-update both target networks alongside the policy update
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
# via `lax.scan` (instead of unrolling, which blows up compile time / HLO size at
# large `n_steps`). The delayed-policy-update schedule is precomputed as a boolean
# mask scanned over alongside the per-step keys. Only the trainable graph state is
# carried; `buffer_state` and the normalization params are loop-constant.
@functools.partial(
    nnx.jit,
    static_argnames=(
        "gamma", "tau", "replay_sample_fn", "n_steps", "policy_delay", "n_step",
    ),
    # Donate the train state (arg 0): its large read-only replay buffer is
    # threaded unchanged through the scan, so without donation XLA allocates a
    # full second copy of the buffer (~1.4GB for 500k obs) every update. The
    # caller reassigns self.state from the result, so donating is safe.
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
            update_actor,
            pre_activation_coef,
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

    def _make_critic(self, critic_config, env_obs_size, env_action_size):
        critic_rngs_1 = nnx.Rngs(params=2, dropout=3)
        critic_rngs_2 = nnx.Rngs(params=4, dropout=5)
        critic1 = hydra.utils.instantiate(
            critic_config,
            in_features=env_obs_size + env_action_size,
            rngs=critic_rngs_1,
        )
        critic2 = hydra.utils.instantiate(
            critic_config,
            in_features=env_obs_size + env_action_size,
            rngs=critic_rngs_2,
        )
        return TwinCritic(critic1, critic2)

    def learn(self, agent_rng, n_steps=None):
        """One unconditional burst (TD3 variant: threads `policy_delay` into the
        fused grad step). See DDPG.learn for the sync/async sharing rationale."""
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
            self.policy_delay,
            self.pre_activation_coef,
            n_step=self.n_step,
        )
        return actor_loss, critic_loss

    def update(self, steps, agent_rng):
        gradient_steps, actor_loss, critic_loss = 0, 0, 0

        if (
            steps >= self.steps_before_learning
            and (steps - self.steps_before_learning) % self.steps_between_updates == 0
        ):
            actor_loss, critic_loss = self.learn(agent_rng)
            gradient_steps += self.learning_steps

        return gradient_steps, actor_loss, critic_loss

    def _export_hyperparams(self) -> dict:
        params = super()._export_hyperparams()
        params["policy_delay"] = int(self.policy_delay)
        return params
