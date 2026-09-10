import functools
from typing import Any

import hydra
import jax
import jax.numpy as jnp
from flax import nnx

from roxie.agents.agent import Agent, TrainState
from roxie.agents.utils import (
    BurstNode,
    build_replay,
    fused_grad_steps,
    graph_jit,
    make_optimizer,
    network_rngs,
    reduce_diagnostics,
    repack_samples,
    soft_update,
    transition_prototype,
)
from roxie.losses.actor_losses import (
    sac_actor_loss_fn,
    sac_alpha_loss_fn,
    skipped_actor_aux,
)
from roxie.losses.critic_losses import sac_critic_loss_fn
from roxie.models.critics import TwinCritic
from roxie.utils.math import normalize_obs, scale_to_env


class LogAlpha(nnx.Module):
    def __init__(self, init_value=0.0):
        self.log_alpha = nnx.Param(jnp.array(init_value, dtype=jnp.float32))


# `update_actor` is a *traced* boolean: under `lax.scan` the step index is not
# static, so the delayed policy update is a runtime branch.
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
    state, log_alpha_module, alpha_optimizer = nodes

    key, sample_key, actor_key, critic_key = jax.random.split(key, 4)
    samples = replay_sample_fn(state.buffer_state, sample_key)
    re_packed_samples = repack_samples(samples, gamma, n_step)
    re_packed_samples = Agent.normalize_samples(
        re_packed_samples, obs_mean, obs_std, obs_clip, normalize
    )

    alpha = jnp.exp(log_alpha_module.log_alpha.value)

    (critic_loss, critic_aux), critic_grads = nnx.value_and_grad(
        sac_critic_loss_fn, has_aux=True
    )(
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
        (actor_loss, (log_probs, actor_aux)), actor_grads = nnx.value_and_grad(
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
        return actor_loss, actor_aux

    def _skip_actor_update(operand):
        # `nnx.cond` requires both branches to return the same pytree.
        zero = jnp.array(0.0, dtype=critic_loss.dtype)
        return zero, skipped_actor_aux(critic_loss.dtype, "alpha", "entropy")

    operand = (state, log_alpha_module, alpha_optimizer)
    if update_actor is None:  # policy_delay == 1: no branch in the program
        actor_loss, actor_aux = _actor_update(operand)
    else:
        actor_loss, actor_aux = nnx.cond(
            update_actor, _actor_update, _skip_actor_update, operand
        )

    # Every step regardless of `policy_delay`: the target tracks the critic,
    # not the policy.
    soft_update(state.target_critic, state.critic, tau)

    return actor_loss, critic_loss, actor_aux, critic_aux


@functools.partial(
    graph_jit,
    static_argnames=(
        "gamma", "tau", "replay_sample_fn", "n_steps",
        "target_entropy", "auto_alpha", "policy_delay", "n_step", "normalize",
    ),
    # The temperature and its optimizer are mutated alongside the train state,
    # so all three ride in one split.
    num_nodes=3,
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
    obs_mean, obs_std = Agent.obs_mean_std(state.obs_stats, obs_eps)

    # `policy_delay` is static, so at 1 the mask is dropped and `_grad_step`
    # takes its unconditional path. `None` is an empty pytree node, so it rides
    # along in `xs` without contributing a scanned leaf.
    update_mask = (
        None if policy_delay <= 1 else (jnp.arange(n_steps) % policy_delay) == 0
    )

    (state, log_alpha_module, alpha_optimizer), (
        actor_losses,
        critic_losses,
        actor_aux,
        critic_aux,
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

    n_actor_updates = (
        n_steps if update_mask is None else jnp.maximum(jnp.sum(update_mask), 1)
    )
    diagnostics = {
        **reduce_diagnostics(actor_aux, n_actor_updates),
        **reduce_diagnostics(critic_aux, n_steps),
    }
    return (
        state,
        log_alpha_module,
        alpha_optimizer,
        jnp.sum(actor_losses) / n_actor_updates,
        jnp.mean(critic_losses),
        diagnostics,
    )


class SAC(Agent):
    _num_burst_nodes = 3
    # Mutated by every gradient burst, so held in the same split as the train
    # state.
    log_alpha_module = BurstNode(1)
    alpha_optimizer = BurstNode(2)

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
        target_entropy_scale: float = 1.0,
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
        # -dim(A) is the SAC paper's heuristic: one nat per actuator. Because the
        # policy is tanh-squashed, that entropy floor doubles as a saturation
        # CEILING — the -sum log(1 - a^2) Jacobian term diverges at the rails, so
        # a policy cannot both rail and hold the target. With many actuators the
        # constraint binds on the sum and dims trade (some rail, others stay
        # diffuse); at dim(A) = 1 there is no slack, and the mean is pinned near
        # tanh(1). Tasks that want bang-bang torque need a scale > 1.
        self.target_entropy_scale = float(target_entropy_scale)
        self.target_entropy = (
            target_entropy
            if target_entropy is not None
            else -self.target_entropy_scale * float(env_action_size)
        )
        self.alpha_learning_rate = alpha_learning_rate

        # Deliberately unclipped: a global-norm clip on a 1-element tree just
        # rescales the step and would fight the dual's own convergence.
        self.alpha_optimizer = make_optimizer(
            self.log_alpha_module,
            alpha_optimizer_config,
            learning_rate=self.alpha_learning_rate,
        )

        # No target actor: the policy is re-sampled every step, so the target
        # only ever needs to slow down the critic.
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
        self.steps_between_updates = steps_between_updates
        self.learning_steps = learning_steps
        self.policy_delay = int(policy_delay)
        self.memory_warmup = memory_warmup
        self.normalize_observations = normalize_observations
        self.obs_clip = float(obs_norm_clip)
        self.obs_eps = float(obs_norm_eps)

        print("SAC agent initialized.")
        print("Hyper Params:", self._export_hyperparams())

    def _checkpoint_modules(self) -> dict:
        # Outside `self.state`; dropping them on resume would re-run the
        # temperature annealing against a trained policy.
        return {
            "log_alpha_module": self.log_alpha_module,
            "alpha_optimizer": self.alpha_optimizer,
        }

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
        behaviour snapshot. Returns ``(scaled_action, deviation_from_mode,
        extras)``; `extras` is empty — only PPO stores anything beyond the
        standard five.

        ``noise_module`` and ``critic`` are part of the shared signature the
        fused acting burst calls through, and ignored here: SAC explores from
        its own stochastic policy, and only an on-policy agent stores a value
        estimate at acting time.
        """
        del noise_module, critic
        if self.normalize_observations:
            mean, std = Agent.obs_mean_std(obs_stats, self.obs_eps)
            observation = normalize_obs(observation, mean, std, self.obs_clip)

        # No bounding on top: SAC's actor is a `TanhNormal`, which squashes
        # its own samples and owns the tanh log-prob correction its losses
        # need.
        action, noise, _, _, _ = Agent.stochastic_step_fn(
            actor, observation, evaluate, key,
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
        self.last_action, noise, self.last_extras = self.select_action(
            self.state.actor, self.state.obs_stats, observation, key, evaluate,
        )
        self.last_noise = noise
        return self.last_action

    def learn(self, agent_rng, n_steps=None):
        """Run one unconditional burst of ``n_steps`` (default ``learning_steps``)
        fused gradient steps, updating ``self.state`` in place; returns
        ``(actor_loss, critic_loss)``.

        Shared by the gated sync ``update`` and the async learner, which calls it
        from its own thread — the sole owner of ``self.state`` there, so the
        buffer-donating ``_grad_steps`` stays valid unchanged.
        """
        burst_steps = self.learning_steps if n_steps is None else int(n_steps)
        actor_loss, critic_loss, diagnostics = _grad_steps(
            self._burst_nodes,
            agent_rng,
            burst_steps,
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
                "policy_delay": int(self.policy_delay),
                "alpha_learning_rate": float(self.alpha_learning_rate),
                # The live temperature, not the constructor's: a resume has to
                # pick up where the annealing got to.
                "init_log_alpha": float(self.log_alpha_module.log_alpha.value),
                "auto_alpha": bool(self.auto_alpha),
                # Both: `target_entropy` is what the dual actually chased, and a
                # resume passes it explicitly so the scale is inert on playback —
                # but it still has to survive the round trip to describe the run.
                "target_entropy": float(self.target_entropy),
                "target_entropy_scale": float(self.target_entropy_scale),
            }
        )
        return params
