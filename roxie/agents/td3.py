import functools

import hydra
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from roxie.agents.agent import Agent, TrainState
from roxie.agents.ddpg import DDPG
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
    obs_eps: float,
    obs_clip: float,
    update_actor: bool,
):
    # 1. Sample from the replay buffer
    key, noise_key = jax.random.split(key)
    samples = replay_sample_fn(state.buffer_state, key)

    re_packed_samples = {
        "observations": samples.experience.first.observation,
        "actions": samples.experience.first.action,
        "rewards": samples.experience.first.reward,
        "next_observations": samples.experience.second.observation,
        "terminals": samples.experience.first.terminal,
    }

    obs_mean, obs_std = Agent.obs_mean_std(state.obs_stats, obs_eps)

    # 2. Critic update (twin critic, clipped double-Q target) — every step
    critic_loss, critic_grads = nnx.value_and_grad(td3_critic_loss_fn)(
        state.critic,
        state.target_actor,
        state.target_critic,
        re_packed_samples,
        gamma,
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

    # 3. Delayed actor + target updates — only every `policy_delay` steps
    actor_loss = 0.0
    if update_actor:
        actor_loss, actor_grads = nnx.value_and_grad(td3_actor_loss_fn)(
            state.actor,
            state.critic,
            re_packed_samples,
            obs_mean,
            obs_std,
            obs_clip,
            action_low,
            action_high,
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


# Fused N-step update. `n_steps` and `policy_delay` are static, so the Python
# loop unrolls at trace time into a single compiled program with the delayed
# actor updates baked in.
@functools.partial(
    nnx.jit,
    static_argnames=("gamma", "tau", "replay_sample_fn", "n_steps", "policy_delay"),
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
):
    actor_loss = critic_loss = 0.0
    for i in range(n_steps):
        key, step_key = jax.random.split(key)
        update_actor = (i % policy_delay) == 0
        state, step_actor_loss, critic_loss = _grad_step(
            state,
            step_key,
            gamma,
            tau,
            replay_sample_fn,
            target_policy_noise,
            target_noise_clip,
            action_low,
            action_high,
            obs_eps,
            obs_clip,
            update_actor,
        )
        if update_actor:
            actor_loss = step_actor_loss
    return state, actor_loss, critic_loss


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

    def update(self, steps, agent_rng):
        gradient_steps, actor_loss, critic_loss = 0, 0, 0

        if (
            steps >= self.steps_before_learning
            and (steps - self.steps_before_learning) % self.steps_between_updates == 0
        ):
            self.state, actor_loss, critic_loss = _grad_steps(
                self.state,
                agent_rng,
                self.learning_steps,
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
            )
            gradient_steps += self.learning_steps

        return gradient_steps, actor_loss, critic_loss

    def _export_hyperparams(self) -> dict:
        params = super()._export_hyperparams()
        params["policy_delay"] = int(self.policy_delay)
        return params
