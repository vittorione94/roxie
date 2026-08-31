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
    make_optimizer,
    network_rngs,
    repack_samples,
    soft_update,
    transition_prototype,
)
from roxie.losses.actor_losses import sac_actor_loss_fn, sac_alpha_loss_fn
from roxie.losses.critic_losses import sac_critic_loss_fn
from roxie.models.critics import TwinCritic


class LogAlpha(nnx.Module):
    def __init__(self, init_value=0.0):
        self.log_alpha = nnx.Param(jnp.array(init_value, dtype=jnp.float32))


@functools.partial(nnx.jit, static_argnames=("evaluate",))
def _sac_step_fn(actor_model, observation, evaluate, key):
    """Squashed action plus the deviation from the policy mode.

    The second return mirrors DDPG's ``(action, noise)`` contract so the trainer
    logs an exploration magnitude uniformly. SAC has no noise module — its
    exploration *is* the sampling — so the analogue is how far the sample landed
    from tanh(mean), in the same normalized [-1, 1] action units.
    """
    distribution = actor_model(observation)
    try:
        mean = distribution.mean()
    except TypeError:
        # Some distrax versions expose mean as a property
        mean = distribution.mean
    mode = jnp.tanh(mean)

    if evaluate:
        return mode, jnp.zeros_like(mode)

    action = jnp.tanh(distribution.sample(seed=key))
    return action, mode - action


# Not jitted on its own — called inside `_grad_steps` below so N steps fuse into
# one compiled program. `n_step` is the TD horizon, not the scan length.
# `update_actor` is a *traced* boolean: under `lax.scan` the step index is not
# static, so the delayed policy update has to be a runtime branch.
def _grad_step(
    nodes,
    key: jax.random.PRNGKey,
    update_actor,
    *,
    gamma: float,
    tau: float,
    replay_sample_fn,
    target_entropy: float,
    auto_alpha: bool,
    action_low: float,
    action_high: float,
    obs_mean: jnp.ndarray,
    obs_std: jnp.ndarray,
    obs_clip: float,
    normalize: bool,
    n_step: int = 1,
):
    # The temperature and its optimizer travel with the train state so their
    # updates survive the fused burst.
    state, log_alpha_module, alpha_optimizer = nodes

    # `repack_samples` folds the n-step return, bootstrap coefficient, and
    # bootstrap obs into the dict, so the critic loss never sees gamma/terminals.
    key, sample_key, actor_key, critic_key = jax.random.split(key, 4)
    samples = replay_sample_fn(state.buffer_state, sample_key)
    re_packed_samples = repack_samples(samples, gamma, n_step)
    # Normalized once here: the critic and (delayed) actor losses read the same
    # `observations`, and neither normalizes.
    re_packed_samples = Agent.normalize_samples(
        re_packed_samples, obs_mean, obs_std, obs_clip, normalize
    )

    alpha = jnp.exp(log_alpha_module.log_alpha.value)

    critic_loss, critic_grads = nnx.value_and_grad(sac_critic_loss_fn)(
        state.critic,
        state.actor,
        state.target_critic,
        re_packed_samples,
        alpha,
        critic_key,
        action_low,
        action_high,
    )
    state.critic_optimizer.update(state.critic, critic_grads)

    # Alpha rides with the actor: its gradient is a function of the actor's
    # log-probs, so updating it on a step where the policy did not move would
    # re-apply the same gradient.
    def _actor_update(operand):
        st, la, aopt = operand
        (actor_loss, log_probs), actor_grads = nnx.value_and_grad(
            sac_actor_loss_fn, has_aux=True
        )(
            st.actor,
            st.critic,
            alpha,
            re_packed_samples,
            actor_key,
            action_low,
            action_high,
        )
        st.actor_optimizer.update(st.actor, actor_grads)

        if auto_alpha:
            _, alpha_grads = nnx.value_and_grad(sac_alpha_loss_fn)(
                la,
                jax.lax.stop_gradient(log_probs),
                target_entropy,
            )
            aopt.update(la, alpha_grads)
        return actor_loss

    def _skip_actor_update(operand):
        # `nnx.cond` requires both branches to return the same pytree. The zero
        # is never averaged in: `_grad_steps` divides the actor-loss sum by the
        # number of *update* steps.
        return jnp.array(0.0, dtype=critic_loss.dtype)

    operand = (state, log_alpha_module, alpha_optimizer)
    if update_actor is None:  # policy_delay == 1: no branch in the program
        actor_loss = _actor_update(operand)
    else:
        actor_loss = nnx.cond(
            update_actor, _actor_update, _skip_actor_update, operand
        )

    # Runs on every step regardless of `policy_delay`: the target tracks the
    # critic, not the policy.
    soft_update(state.target_critic, state.critic, tau)

    return actor_loss, critic_loss


# `fused_grad_steps` compiles the body once and runs it `n_steps` times
# on-device, so a burst costs one host dispatch rather than one per step.
@functools.partial(
    nnx.jit,
    static_argnames=(
        "gamma", "tau", "replay_sample_fn", "n_steps",
        "target_entropy", "auto_alpha", "policy_delay", "n_step", "normalize",
    ),
    # The replay buffer rides unchanged through the scan; without donation XLA
    # allocates a full second copy of it every update.
    donate_argnums=(0,),
)
def _grad_steps(
    state: TrainState,
    log_alpha_module: LogAlpha,
    alpha_optimizer: nnx.Optimizer,
    key: jax.random.PRNGKey,
    n_steps: int,
    gamma: float,
    tau: float,
    replay_sample_fn,
    target_entropy: float,
    auto_alpha: bool,
    action_low: float,
    action_high: float,
    obs_eps: float,
    obs_clip: float,
    normalize: bool = True,
    policy_delay: int = 1,
    n_step: int = 1,
):
    # Loop-constant, so hoisted out of the scan body.
    obs_mean, obs_std = Agent.obs_mean_std(state.obs_stats, obs_eps)

    # `policy_delay` is static, so at 1 the mask is dropped altogether and
    # `_grad_step` takes its unconditional path. `None` is an empty pytree node,
    # so it rides along in `xs` without contributing a scanned leaf.
    update_mask = (
        None if policy_delay <= 1 else (jnp.arange(n_steps) % policy_delay) == 0
    )

    # Carried as one tuple so `log_alpha` and its Adam slots keep updating across
    # the fused steps.
    (state, log_alpha_module, alpha_optimizer), (
        actor_losses,
        critic_losses,
    ) = fused_grad_steps(
        (state, log_alpha_module, alpha_optimizer),
        key,
        n_steps,
        functools.partial(
            _grad_step,
            gamma=gamma,
            tau=tau,
            replay_sample_fn=replay_sample_fn,
            target_entropy=target_entropy,
            auto_alpha=auto_alpha,
            action_low=action_low,
            action_high=action_high,
            obs_mean=obs_mean,
            obs_std=obs_std,
            obs_clip=obs_clip,
            normalize=normalize,
            n_step=n_step,
        ),
        extras=(update_mask,),
    )

    # Actor loss only on update steps → average over those; critic over all.
    n_actor_updates = (
        n_steps if update_mask is None else jnp.maximum(jnp.sum(update_mask), 1)
    )
    return (
        state,
        log_alpha_module,
        alpha_optimizer,
        jnp.sum(actor_losses) / n_actor_updates,
        jnp.mean(critic_losses),
    )


class SAC(Agent):
    def __init__(
        self,
        env_obs_size: int,
        env_action_size: int,
        action_low: jnp.ndarray,
        action_high: jnp.ndarray,
        actor_config: dict,
        critic_config: dict,
        memory_config: dict,
        *,
        actor_optimizer_config: dict = None,
        critic_optimizer_config: dict = None,
        alpha_optimizer_config: dict = None,
        seed: int = 0,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        alpha_learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        init_log_alpha: float = 0.0,
        auto_alpha: bool = True,
        target_entropy: float = None,
        steps_before_learning: int = 100,
        steps_between_updates: int = 10,
        learning_steps: int = 5,
        memory_warmup: int = 100,
        n_step: int = 1,
        policy_delay: int = 1,
        max_grad_norm: float = 1.0,
        normalize_observations: bool = True,
        obs_norm_clip: float = 5.0,
        obs_norm_eps: float = 1e-8,
    ):
        self.seed = int(seed)

        actor = hydra.utils.instantiate(
            actor_config,
            in_features=env_obs_size,
            action_dim=env_action_size,
            rngs=network_rngs(self.seed, offset=0),
        )

        # Distinct seed offsets so the two heads don't start identical.
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
        twin_critic = TwinCritic(critic1, critic2)

        self.n_step = int(n_step)
        replay = build_replay(memory_config, self.n_step)
        self.batch_size = memory_config.sample_batch_size
        self.buffer_size = memory_config.max_length
        buffer_state = replay.init(
            transition_prototype(env_obs_size, env_action_size)
        )

        self.log_alpha_module = LogAlpha(init_log_alpha)
        self.auto_alpha = auto_alpha
        self.target_entropy = (
            target_entropy if target_entropy is not None else -float(env_action_size)
        )
        self.alpha_learning_rate = alpha_learning_rate

        # Deliberately unclipped: a global-norm clip on a 1-element tree just
        # rescales the step and would fight the dual's own convergence.
        self.alpha_optimizer = make_optimizer(
            self.log_alpha_module,
            alpha_optimizer_config,
            learning_rate=self.alpha_learning_rate,
        )

        # No target actor in SAC: the policy is re-sampled every step, so the
        # target only ever needs to slow down the critic.
        self._init_train_state(
            actor,
            twin_critic,
            buffer_state,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            max_grad_norm=max_grad_norm,
            actor_optimizer_config=actor_optimizer_config,
            critic_optimizer_config=critic_optimizer_config,
            target_actor=False,
        )

        self.gamma = gamma
        self.tau = tau
        self.action_low = action_low
        self.action_high = action_high
        self.replay = replay
        self.steps_before_learning = steps_before_learning
        self.steps_between_updates = steps_between_updates
        self.learning_steps = learning_steps
        # Actor (and temperature) updates run on 1 of every `policy_delay`
        # gradient steps; the critic updates on all of them.
        self.policy_delay = int(policy_delay)
        self.memory_warmup = memory_warmup
        self.normalize_observations = normalize_observations
        self.obs_clip = float(obs_norm_clip)
        self.obs_eps = float(obs_norm_eps)

        print("SAC agent initialized.")
        print("Hyper Params:", self._export_hyperparams())

    def _checkpoint_modules(self) -> dict:
        # These live outside `self.state`; dropping them on resume would re-run
        # the temperature annealing against a trained policy.
        return {
            "log_alpha_module": self.log_alpha_module,
            "alpha_optimizer": self.alpha_optimizer,
        }

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
        live ``self.state.actor`` — through the same normalization path. Returns
        ``(scaled_action, deviation_from_mode)``.
        """
        if self.normalize_observations:
            mean, std = Agent.obs_mean_std(obs_stats, self.obs_eps)
            observation = Agent.normalize_obs(observation, mean, std, self.obs_clip)

        action, noise = _sac_step_fn(actor, observation, evaluate, key)
        return Agent.scale_to_env(action, self.action_low, self.action_high), noise

    def step(
        self,
        observation: jnp.ndarray,
        evaluate: bool = False,
        key: jax.random.PRNGKey = None,
    ) -> jnp.ndarray:
        self.last_action, noise = self.select_action(
            self.state.actor, self.state.obs_stats, observation, key, evaluate,
        )
        # Kept on device; the trainer reduces it to a per-joint epoch mean.
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
        # `terminal` is true termination only: a time-limit truncation must still
        # bootstrap the next-state value. `truncation` is stored separately so
        # n-step windows stop there too — in the flat stream the item after any
        # done is the next episode's reset state.
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

        Shared by the gated sync ``update`` and the async learner, which calls it
        from its own thread — the sole owner of ``self.state`` there, so the
        buffer-donating ``_grad_steps`` stays valid unchanged.
        """
        (
            self.state,
            self.log_alpha_module,
            self.alpha_optimizer,
            actor_loss,
            critic_loss,
        ) = _grad_steps(
            self.state,
            self.log_alpha_module,
            self.alpha_optimizer,
            agent_rng,
            self.learning_steps if n_steps is None else int(n_steps),
            self.gamma,
            self.tau,
            self.replay.sample,
            self.target_entropy,
            self.auto_alpha,
            self.action_low,
            self.action_high,
            self.obs_eps,
            self.obs_clip,
            self.normalize_observations,
            self.policy_delay,
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
                "policy_delay": int(self.policy_delay),
                "alpha_learning_rate": float(self.alpha_learning_rate),
                # The live temperature, not the constructor's: a checkpoint has
                # to resume where the annealing got to, not where it started.
                "init_log_alpha": float(self.log_alpha_module.log_alpha.value),
                "auto_alpha": bool(self.auto_alpha),
                "target_entropy": float(self.target_entropy),
            }
        )
        return params
