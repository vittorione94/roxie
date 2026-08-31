import functools
from typing import Any

import hydra
import jax
import jax.numpy as jnp
from flax import nnx

from roxie.agents.agent import Agent, TrainState
from roxie.agents.utils import (
    Transition,
    build_replay,
    fused_grad_steps,
    network_rngs,
    repack_samples,
    soft_update,
    transition_prototype,
)
from roxie.losses.actor_losses import ddpg_actor_loss_fn
from roxie.losses.critic_losses import ddpg_critic_loss_fn


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
    n_step: int = 1,
):
    """One gradient step, mutating `state` in place. Not jitted on its own — it
    runs inside `_grad_steps` below, which fuses N steps into one compiled
    program. `n_step` is the TD horizon, not the scan length.
    """
    # `repack_samples` folds the Bellman target ingredients (n-step return,
    # bootstrap coefficient, bootstrap obs) into the dict for both buffer
    # layouts, so the critic loss never sees gamma/terminals.
    key, noise_key = jax.random.split(key)
    samples = replay_sample_fn(state.buffer_state, key)
    re_packed_samples = repack_samples(samples, gamma, n_step)
    # Normalized once here: both losses below read the same `observations`, and
    # neither of them normalizes.
    re_packed_samples = Agent.normalize_samples(
        re_packed_samples, obs_mean, obs_std, obs_clip, normalize
    )

    critic_loss, critic_grads = nnx.value_and_grad(ddpg_critic_loss_fn)(
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

    actor_loss, actor_grads = nnx.value_and_grad(ddpg_actor_loss_fn)(
        state.actor,
        state.critic,
        re_packed_samples,
        action_low,
        action_high,
    )
    state.actor_optimizer.update(state.actor, actor_grads)

    soft_update(state.target_actor, state.actor, tau)
    soft_update(state.target_critic, state.critic, tau)

    return actor_loss, critic_loss


# `fused_grad_steps` compiles the body once and runs it `n_steps` times
# on-device, so a burst costs one host dispatch rather than one per step.
@functools.partial(
    nnx.jit,
    static_argnames=(
        "gamma", "tau", "replay_sample_fn", "n_steps", "n_step", "normalize",
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
    n_step: int = 1,
):
    # Loop-constant, so hoisted out of the scan body.
    obs_mean, obs_std = Agent.obs_mean_std(state.obs_stats, obs_eps)

    state, (actor_losses, critic_losses) = fused_grad_steps(
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
            n_step=n_step,
        ),
    )

    # Averaged over the fused steps for less noisy logging.
    return state, jnp.mean(actor_losses), jnp.mean(critic_losses)


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
        steps_before_learning: int = 100,
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

        # Set before `_make_critic` so overriding subclasses can derive their
        # own seed offsets.
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
        self.steps_before_learning = steps_before_learning
        self.steps_between_updates = steps_between_updates
        self.learning_steps = learning_steps
        self.memory_warmup = memory_warmup
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise
        self.normalize_observations = normalize_observations
        self.obs_clip = float(obs_norm_clip)
        self.obs_eps = float(obs_norm_eps)
        # Weight on the actor's pre-tanh saturation penalty, consumed by TD3's
        # actor loss; 0.0 = the textbook DPG objective. On the base so every
        # deterministic-actor agent round-trips it identically.
        self.pre_activation_coef = float(pre_activation_coef)

        print(f"{type(self).__name__} agent initialized.")
        print("Noise module hyperparameters:", self.noise_module.hyperparameters())
        print("Hyper Params:", self._export_hyperparams())

    def _checkpoint_modules(self) -> dict:
        # The noise module carries the step counter its decay schedule reads, so
        # without it a resumed run would explore at the initial scale again.
        return {"noise_module": self.noise_module}

    def _make_critic(self, critic_config, env_obs_size, env_action_size):
        """Build the critic network. Overridden by TD3 to return a TwinCritic."""
        return hydra.utils.instantiate(
            critic_config,
            in_features=env_obs_size + env_action_size,
            rngs=network_rngs(self.seed, offset=2),
        )

    def replay_add(self, buffer_state, transitions):
        """Add one env-step batch of transitions (leaves shaped (B, ...)).

        The trajectory buffer (n_step > 1) expects an explicit time axis on every
        leaf — (B, T=1, ...) for per-step adds — while the flat buffer takes the
        batch as-is. Callers go through here so they need not know which is
        active.
        """
        if self.n_step > 1:
            transitions = jax.tree.map(lambda x: x[:, None], transitions)
        return self.replay.add(buffer_state, transitions)

    def select_action(
        self,
        actor: nnx.Module,
        obs_stats: Any,
        observation: jnp.ndarray,
        key: jax.random.PRNGKey,
        evaluate: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Pure action selection from an explicit actor + obs stats.

        Factored out of ``step`` so the async learner's acting thread can select
        actions from a behaviour actor snapshot — decoupled from the learner's
        live ``self.state.actor`` — through the same normalization + noise path.
        Returns ``(scaled_action, applied_noise)``.
        """
        if self.normalize_observations:
            mean, std = Agent.obs_mean_std(obs_stats, self.obs_eps)
            observation = Agent.normalize_obs(observation, mean, std, self.obs_clip)

        action, noise = Agent.deterministic_step_fn(
            actor, observation, key, self.noise_module, evaluate,
        )
        return Agent.scale_to_env(action, self.action_low, self.action_high), noise

    def step(
        self,
        observation: jnp.ndarray,
        evaluate: bool = False,
        key: jax.random.PRNGKey = None,
    ) -> jnp.ndarray:
        """Selects an action by calling the pure, JIT-compiled step function."""
        self.last_action, noise = self.select_action(
            self.state.actor, self.state.obs_stats, observation, key, evaluate,
        )
        # Post-clip noise, per env per joint. Kept on device; the trainer reduces
        # it to a per-joint epoch mean.
        self.last_noise = noise

        return self.last_action

    def add_transitions(
        self, prev_obs, action, reward, termination, truncation, next_obs
    ):
        """Buffer one env-step batch given an explicit action.

        Split out of ``add`` so the async learner (which owns ``self.state`` on
        its own thread) can add transitions whose action travelled with them
        through the hand-off queue, rather than reading ``self.last_action``,
        which the acting thread overwrites every step.
        """
        # `terminal` is true termination only: marking a time-limit truncation
        # terminal zeroes its bootstrap and collapses Q at the cutoff, for every
        # env at once since they hit the limit in lockstep. `truncation` is
        # stored separately so n-step windows stop there too — in the flat
        # stream the item after any done is a reset state.
        experiences = Transition(
            observation=prev_obs,
            action=action,
            reward=reward,
            terminal=termination,
            truncation=truncation,
        )

        self.state.buffer_state = self.replay_add(self.state.buffer_state, experiences)

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
        )

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
            n_step=self.n_step,
        )
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
