import functools
from typing import Any

import hydra
import jax
import jax.numpy as jnp
from flax import nnx

from roxie.agents.agent import Agent, TrainState
from roxie.agents.utils import (
    build_replay,
    fused_grad_steps,
    network_rngs,
    reduce_diagnostics,
    repack_samples,
    soft_update,
    transition_prototype,
)
from roxie.losses.actor_losses import ddpg_actor_loss_fn
from roxie.losses.critic_losses import ddpg_critic_loss_fn
from roxie.utils.math import normalize_obs, scale_to_env


def _grad_step(
    state: TrainState,
    key: jax.random.PRNGKey,
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
    """One gradient step, mutating `state` in place. Not jitted on its own — it
    runs inside `_grad_steps` below, which fuses N steps into one compiled
    program. `n_step` is the TD horizon, not the scan length.
    """
    key, noise_key = jax.random.split(key)
    samples = replay_sample_fn(state.buffer_state, key)
    re_packed_samples = repack_samples(samples, gamma, n_step)
    re_packed_samples = Agent.normalize_samples(
        re_packed_samples, obs_mean, obs_std, obs_clip, normalize
    )

    (critic_loss, critic_aux), critic_grads = nnx.value_and_grad(
        ddpg_critic_loss_fn, has_aux=True
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

    (actor_loss, actor_aux), actor_grads = nnx.value_and_grad(
        ddpg_actor_loss_fn, has_aux=True
    )(
        state.actor,
        state.critic,
        re_packed_samples,
        action_low,
        action_high,
        pre_activation_coef,
    )
    state.actor_optimizer.update(state.actor, actor_grads)

    soft_update(state.target_actor, state.actor, tau)
    soft_update(state.target_critic, state.critic, tau)

    return actor_loss, critic_loss, actor_aux, critic_aux


@functools.partial(
    jax.jit,
    static_argnames=(
        "gamma", "tau", "replay_sample_fn", "n_steps", "n_step", "normalize",
    ),
    # The replay buffer threads unchanged through the scan; without donation XLA
    # allocates a second copy of it per burst.
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
    pre_activation_coef: float,
    n_step: int = 1,
):
    obs_mean, obs_std = Agent.obs_mean_std(state.obs_stats, obs_eps)

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
    )

    # Averaged over the fused steps for less noisy logging. No `policy_delay`
    # here, so the actor and critic ran on every one of them and share a
    # denominator.
    diagnostics = {
        **reduce_diagnostics(actor_aux, n_steps),
        **reduce_diagnostics(critic_aux, n_steps),
    }
    return state, jnp.mean(actor_losses), jnp.mean(critic_losses), diagnostics


class DDPG(Agent):
    def __init__(
        self,
        env_obs_size: int,
        env_action_size: int,
        action_low: jnp.ndarray,
        action_high: jnp.ndarray,
        actor_config: dict,
        critic_config: dict,
        memory_config: dict,
        noise_config: dict,
        *,
        actor_optimizer_config: dict = None,
        critic_optimizer_config: dict = None,
        seed: int = 0,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        steps_between_updates: int = 10,
        learning_steps: int = 5,
        memory_warmup: int = 100,
        n_step: int = 1,
        target_noise_clip: float = 0.1,
        target_policy_noise: float = 0.1,
        max_grad_norm: float = 1.0,
        pre_activation_coef: float = 0.0,
        normalize_observations: bool = True,
        obs_norm_clip: float = 5.0,
        obs_norm_eps: float = 1e-8,
    ):

        # Set before `_make_critic` so subclasses can derive seed offsets.
        self.seed = int(seed)

        actor = hydra.utils.instantiate(
            actor_config,
            in_features=env_obs_size,
            action_dim=env_action_size,
            rngs=network_rngs(self.seed, offset=0),
        )

        critic = self._make_critic(critic_config, env_obs_size, env_action_size)

        self.n_step = int(n_step)
        replay = build_replay(memory_config, self.n_step)
        self.batch_size = memory_config.sample_batch_size
        self.buffer_size = memory_config.max_length

        buffer_state = replay.init(
            transition_prototype(env_obs_size, env_action_size)
        )

        self.noise_module = hydra.utils.instantiate(
            noise_config, action_shape=(env_action_size,)
        )

        self._init_train_state(
            actor,
            critic,
            buffer_state,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            max_grad_norm=max_grad_norm,
            actor_optimizer_config=actor_optimizer_config,
            critic_optimizer_config=critic_optimizer_config,
        )

        self.gamma = gamma
        self.tau = tau
        self.action_low = action_low
        self.action_high = action_high
        self.replay = replay
        self.steps_between_updates = steps_between_updates
        self.learning_steps = learning_steps
        self.memory_warmup = memory_warmup
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise
        self.normalize_observations = normalize_observations
        self.obs_clip = float(obs_norm_clip)
        self.obs_eps = float(obs_norm_eps)
        # Consumed by TD3's actor loss; 0.0 = the textbook DPG objective. On
        # the base so every deterministic-actor agent round-trips it.
        self.pre_activation_coef = float(pre_activation_coef)

        print(f"{type(self).__name__} agent initialized.")
        print("Noise module hyperparameters:", self.noise_module.hyperparameters())
        print("Hyper Params:", self._export_hyperparams())

    def _checkpoint_modules(self) -> dict:
        # Carries the step counter its decay schedule reads, so without it a
        # resumed run would explore at the initial scale again.
        return {"noise_module": self.noise_module}

    def _make_critic(self, critic_config, env_obs_size, env_action_size):
        """Build the critic network. Overridden by TD3 to return a TwinCritic."""
        return hydra.utils.instantiate(
            critic_config,
            in_features=env_obs_size + env_action_size,
            rngs=network_rngs(self.seed, offset=2),
        )

    def select_action(
        self,
        actor: nnx.Module,
        obs_stats: Any,
        observation: jnp.ndarray,
        key: jax.random.PRNGKey,
        evaluate: bool = False,
        noise_module: nnx.Module = None,
        critic: nnx.Module = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
        """Pure action selection from an explicit actor + obs stats.

        Taking the actor and the stats as arguments rather than reading
        ``self.state`` is what lets the fused acting burst run this against a
        ``lax.scan`` carry, and the async learner's acting thread select from a
        behaviour snapshot. Returns ``(scaled_action, applied_noise, extras)``;
        `extras` is empty — only PPO stores anything beyond the standard five.

        ``noise_module`` is explicit for the same reason the actor is: the burst
        carries the module through the scan, and its decay counter has to
        advance on the carry rather than on ``self``. ``critic`` is part of the
        same shared signature and ignored — only an on-policy agent stores a
        value estimate at acting time.
        """
        del critic
        if self.normalize_observations:
            mean, std = Agent.obs_mean_std(obs_stats, self.obs_eps)
            observation = normalize_obs(observation, mean, std, self.obs_clip)

        action, noise = Agent.deterministic_step_fn(
            actor,
            observation,
            key,
            self.noise_module if noise_module is None else noise_module,
            evaluate,
        )
        return (
            scale_to_env(action, self.action_low, self.action_high),
            noise,
            {},
        )

    def step(
        self,
        observation: jnp.ndarray,
        evaluate: bool = False,
        key: jax.random.PRNGKey = None,
    ) -> jnp.ndarray:
        """Selects an action by calling the pure, JIT-compiled step function."""
        self.last_action, noise, self.last_extras = self.select_action(
            self.state.actor, self.state.obs_stats, observation, key, evaluate,
        )
        self.last_noise = noise

        return self.last_action

    def learn(self, agent_rng, n_steps=None):
        """Run one unconditional burst of ``n_steps`` (default ``learning_steps``)
        fused gradient steps, updating ``self.state`` in place; returns
        ``(actor_loss, critic_loss)``.

        Shared by the (gated) sync ``update`` and the async learner, which calls
        it directly from its own thread — the sole owner of ``self.state`` there,
        so the buffer-donating ``_grad_steps`` stays valid unchanged. The async
        learner passes a small ``n_steps`` so the GPU stream frees up frequently
        for the acting thread's forward pass.
        """
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
            self.pre_activation_coef,
            n_step=self.n_step,
        )
        self.record_diagnostics(diagnostics, burst_steps)
        return actor_loss, critic_loss

    def update(self, steps, agent_rng):
        gradient_steps, actor_loss, critic_loss = 0, 0, 0

        if self.due_for_update(steps):
            actor_loss, critic_loss = self.learn(agent_rng)
            gradient_steps += self.learning_steps

        return gradient_steps, actor_loss, critic_loss

    def _export_hyperparams(self) -> dict:
        params = super()._export_hyperparams()
        params.update(self._replay_hyperparams())
        params.update(
            {
                "n_step": int(self.n_step),
                "target_policy_noise": float(self.target_policy_noise),
                "target_noise_clip": float(self.target_noise_clip),
                "pre_activation_coef": float(self.pre_activation_coef),
            }
        )
        return params
