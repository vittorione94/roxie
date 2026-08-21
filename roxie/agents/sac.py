import copy
import functools
from typing import Any

import flashbax
import hydra
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from roxie.agents.agent import Agent, TrainState
from roxie.agents.utils import (
    Transition,
    build_optimizer,
    network_rngs,
    repack_samples,
    serialize_bound,
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
    can log an exploration magnitude uniformly across agents. SAC has no noise
    module — its exploration *is* the sampling — so the analogue is how far the
    sample landed from tanh(mean), in the same normalized [-1, 1] action units.
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


# Single SAC gradient step. Not jitted on its own — called inside the jitted
# `_grad_steps` below so N steps fuse into one compiled program. `obs_mean` /
# `obs_std` are hoisted in by the caller (the stats are loop-constant), and
# `n_step` is the TD horizon (NOT the scan length). `update_actor` is a *traced*
# boolean: under `lax.scan` the step index is not static, so the delayed policy
# update has to be a runtime branch (`nnx.cond`).
def _grad_step(
    state: TrainState,
    log_alpha_module: LogAlpha,
    alpha_optimizer: nnx.Optimizer,
    key: jax.random.PRNGKey,
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
    update_actor,
    n_step: int = 1,
):
    # `repack_samples` folds the n-step return, bootstrap coefficient, and
    # bootstrap obs into the dict, so the critic loss never sees gamma/terminals.
    key, sample_key, actor_key, critic_key = jax.random.split(key, 4)
    samples = replay_sample_fn(state.buffer_state, sample_key)
    re_packed_samples = repack_samples(samples, gamma, n_step)
    # Normalize once, here: the critic and (delayed) actor losses read the same
    # `observations`, and neither normalizes (see `Agent.normalize_samples`).
    re_packed_samples = Agent.normalize_samples(
        re_packed_samples, obs_mean, obs_std, obs_clip, normalize
    )

    alpha = jnp.exp(log_alpha_module.log_alpha.value)

    # Critic update (twin critic, clipped double-Q soft target).
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

    # Actor + temperature update, only on steps where `update_actor` is True.
    # `policy_delay` 1 (the default) reproduces textbook SAC; >1 is the
    # REDQ/DroQ-style trade of policy freshness for critic updates per second.
    #
    # Alpha rides with the actor rather than the critic: its gradient is a
    # function of the actor's log-probs, so updating it on a step where the
    # policy did not move would just re-apply the same gradient.
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

        # `auto_alpha` is a Python bool (static at trace time), so when it is
        # False this simply is not traced.
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
    if update_actor is None:
        # policy_delay == 1: the caller skips the mask entirely, so the default
        # configuration never pays for a branch it always takes.
        actor_loss = _actor_update(operand)
    else:
        actor_loss = nnx.cond(
            update_actor, _actor_update, _skip_actor_update, operand
        )

    # Soft update of the target critics (SAC has no target actor). Runs on every
    # step regardless of `policy_delay`: it tracks the critic, not the policy.
    new_critic_tensors = nnx.state(state.critic, nnx.Param)
    old_critic_tensors = nnx.state(state.target_critic, nnx.Param)
    new_target_critic_tensors = optax.incremental_update(
        new_tensors=new_critic_tensors,
        old_tensors=old_critic_tensors,
        step_size=tau,
    )
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


# Fused N-step update. The body is compiled once and run `n_steps` times on-device
# via `lax.scan`, so a burst costs one host dispatch rather than one per gradient
# step. The entropy temperature and its optimizer ride along in the scan carry
# alongside the train state; `buffer_state` and the normalization params are
# loop-constant.
@functools.partial(
    nnx.jit,
    static_argnames=(
        "gamma", "tau", "replay_sample_fn", "n_steps",
        "target_entropy", "auto_alpha", "policy_delay", "n_step", "normalize",
    ),
    # Donate the train state (arg 0): its large read-only replay buffer is threaded
    # unchanged through the scan, so without donation XLA allocates a full second
    # copy of it every update. The caller reassigns self.state from the result.
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
    # Hoist the (loop-constant) normalization params out of the scan body.
    obs_mean, obs_std = Agent.obs_mean_std(state.obs_stats, obs_eps)

    # Pre-split per-step keys and precompute the delayed-update schedule.
    # `policy_delay` is static, so at 1 the mask is dropped altogether and
    # `_grad_step` takes its unconditional path — no `nnx.cond` in the compiled
    # program at all. `None` is an empty pytree node, so it rides along in `xs`
    # without contributing a scanned leaf.
    keys = jax.random.split(key, n_steps)
    update_mask = (
        None if policy_delay <= 1 else (jnp.arange(n_steps) % policy_delay) == 0
    )

    # Split into a static graph definition + the trainable pytree state. Only
    # the state is carried through the scan; the graphdef is closed over. The
    # three graph nodes are split as one tuple so `log_alpha` and its Adam slots
    # stay in the carry and keep updating across the fused steps.
    graphdef, scan_state = nnx.split((state, log_alpha_module, alpha_optimizer))

    def body(scan_state, xs):
        step_key, update_actor = xs
        st, la, aopt = nnx.merge(graphdef, scan_state)
        st, actor_loss, critic_loss = _grad_step(
            st,
            la,
            aopt,
            step_key,
            gamma,
            tau,
            replay_sample_fn,
            target_entropy,
            auto_alpha,
            action_low,
            action_high,
            obs_mean,
            obs_std,
            obs_clip,
            normalize,
            update_actor,
            n_step,
        )
        _, scan_state = nnx.split((st, la, aopt))
        return scan_state, (actor_loss, critic_loss)

    scan_state, (actor_losses, critic_losses) = jax.lax.scan(
        body, scan_state, (keys, update_mask)
    )
    state, log_alpha_module, alpha_optimizer = nnx.merge(graphdef, scan_state)

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

        # Stochastic actor (outputs distribution for reparameterized sampling)
        actor = hydra.utils.instantiate(
            actor_config,
            in_features=env_obs_size,
            action_dim=env_action_size,
            rngs=network_rngs(self.seed, offset=0),
        )

        # Twin Q-networks to reduce overestimation bias. Distinct seed offsets
        # so the two heads don't start identical.
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

        # Replay buffer. `truncation` is stored alongside `terminal` so n-step
        # windows can stop at episode boundaries the terminal flag doesn't mark
        # (clip-end / time-limit truncations).
        prototype = Transition(
            observation=jnp.zeros(env_obs_size, dtype=jnp.float32),
            action=jnp.zeros(env_action_size, dtype=jnp.float32),
            reward=jnp.zeros((), dtype=jnp.float32),
            terminal=jnp.zeros((), dtype=jnp.bool_),
            truncation=jnp.zeros((), dtype=jnp.bool_),
        )
        self.n_step = int(n_step)
        if self.n_step > 1:
            # n-step targets need n_step+1 consecutive items per sample: use a
            # trajectory buffer (period=1 = windows at every offset). The yaml
            # keeps the flat-buffer schema (max_length/min_length are TOTAL
            # transitions); convert to flashbax's per-row time-axis lengths.
            add_batch = int(memory_config.add_batch_size)
            replay = flashbax.make_trajectory_buffer(
                add_batch_size=add_batch,
                sample_batch_size=int(memory_config.sample_batch_size),
                sample_sequence_length=self.n_step + 1,
                period=1,
                min_length_time_axis=max(
                    self.n_step + 1, int(memory_config.min_length) // add_batch
                ),
                max_length_time_axis=int(memory_config.max_length) // add_batch,
            )
        else:
            replay = hydra.utils.instantiate(memory_config)
        self.batch_size = memory_config.sample_batch_size
        self.buffer_size = memory_config.max_length
        buffer_state = replay.init(prototype)

        # Target critic (no target actor in SAC)
        target_twin_critic = copy.deepcopy(twin_critic)

        # Learnable entropy temperature
        self.log_alpha_module = LogAlpha(init_log_alpha)
        self.auto_alpha = auto_alpha
        self.target_entropy = (
            target_entropy if target_entropy is not None else -float(env_action_size)
        )

        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.alpha_learning_rate = alpha_learning_rate
        self.max_grad_norm = max_grad_norm

        actor_optimizer = nnx.Optimizer(
            actor,
            build_optimizer(
                actor_optimizer_config,
                learning_rate=self.actor_learning_rate,
                max_grad_norm=self.max_grad_norm,
            ),
            wrt=nnx.Param,
        )

        critic_optimizer = nnx.Optimizer(
            twin_critic,
            build_optimizer(
                critic_optimizer_config,
                learning_rate=self.critic_learning_rate,
                max_grad_norm=self.max_grad_norm,
            ),
            wrt=nnx.Param,
        )

        # The temperature is a single scalar, so it is deliberately NOT clipped:
        # a global-norm clip on a 1-element tree just rescales the step and
        # would fight the dual's own convergence.
        self.alpha_optimizer = nnx.Optimizer(
            self.log_alpha_module,
            build_optimizer(
                alpha_optimizer_config, learning_rate=self.alpha_learning_rate
            ),
            wrt=nnx.Param,
        )

        obs_shape = buffer_state.experience.observation.shape[-1]
        obs_stats = Agent.init_obs_stats(obs_shape)

        self.state = TrainState(
            actor=actor,
            critic=twin_critic,
            target_actor=None,
            target_critic=target_twin_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            buffer_state=buffer_state,
            obs_stats=obs_stats,
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
        # gradient steps; the critic updates on all of them. 1 = textbook SAC.
        self.policy_delay = int(policy_delay)
        self.memory_warmup = memory_warmup
        self.normalize_observations = normalize_observations
        self.obs_clip = float(obs_norm_clip)
        self.obs_eps = float(obs_norm_eps)

        print("SAC agent initialized.")
        print("Hyper Params:", self._export_hyperparams())

    def _checkpoint_modules(self) -> dict:
        # The entropy temperature and its Adam slots live outside `self.state`,
        # so a resume that dropped them would restart alpha from
        # `init_log_alpha` — re-running the whole temperature annealing against
        # an already-trained policy.
        return {
            "log_alpha_module": self.log_alpha_module,
            "alpha_optimizer": self.alpha_optimizer,
        }

    def replay_add(self, buffer_state, transitions):
        """Add one env-step batch of transitions (leaves shaped (B, ...)).

        The trajectory buffer (n_step > 1) expects an explicit time axis on
        every leaf — (B, T=1, ...) for per-step adds — while the flat buffer
        takes the batch as-is. Callers (self.add, the trainer's warmup fill) go
        through here so they never need to know which layout is active.
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
        """Pure action selection from an EXPLICIT actor + obs stats.

        Factored out of ``step`` so the async learner's acting thread can select
        actions from a *behaviour* actor snapshot — decoupled from the learner's
        live, mutating ``self.state.actor`` — through the exact same
        normalization path. Returns ``(scaled_action, deviation_from_mode)``.
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
        # How far the sampled action landed from the policy mode, in normalized
        # [-1, 1] action units. Kept on device; the trainer reduces it to a
        # per-joint epoch mean for logging. Zero when evaluating.
        self.last_noise = noise
        return self.last_action

    def add_transitions(
        self, prev_obs, action, reward, termination, truncation, next_obs
    ):
        """Buffer one env-step batch given an EXPLICIT action.

        Split out of ``add`` so the async learner (which owns ``self.state`` on
        its own thread) can add transitions whose action travelled with them
        through the hand-off queue, rather than reading ``self.last_action``
        (which the acting thread overwrites every step).
        """
        experiences = Transition(
            observation=prev_obs,
            action=action,
            # True termination only, not `done` (termination OR truncation): a
            # time-limit truncation must still bootstrap the next-state value.
            reward=reward,
            terminal=termination,
            # Stored separately so n-step windows can stop at truncations too —
            # in the flat stream the item after ANY done is the next episode's
            # reset state, so a window must never accumulate across one.
            truncation=truncation,
        )

        self.state.buffer_state = self.replay_add(self.state.buffer_state, experiences)

        if self.normalize_observations:
            obs_batch = jnp.concatenate([prev_obs, next_obs], axis=0)
            self.state.obs_stats = Agent.update_obs_stats(
                self.state.obs_stats, obs_batch
            )

    def add(self, prev_states, states):
        self.add_transitions(
            prev_states.obs,
            self.last_action,
            states.reward,
            states.info["termination"],
            states.info["truncation"],
            states.obs,
        )

    def learn(self, agent_rng, n_steps=None):
        """Run one unconditional burst of ``n_steps`` (default ``learning_steps``)
        fused gradient steps, updating ``self.state`` in place; returns
        ``(actor_loss, critic_loss)``.

        Shared by the (gated) sync ``update`` and the async learner, which calls
        it directly from its own thread — the sole owner of ``self.state`` there,
        so the buffer-donating ``_grad_steps`` stays valid unchanged.
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
        return {
            "seed": int(self.seed),
            "gamma": float(self.gamma),
            "tau": float(self.tau),
            "actor_learning_rate": float(self.actor_learning_rate),
            "critic_learning_rate": float(self.critic_learning_rate),
            "alpha_learning_rate": float(self.alpha_learning_rate),
            "max_grad_norm": float(self.max_grad_norm),
            "env_obs_size": self.state.buffer_state.experience.observation.shape[2],
            "env_action_size": self.state.buffer_state.experience.action.shape[2],
            "init_log_alpha": float(self.log_alpha_module.log_alpha.value),
            "auto_alpha": bool(self.auto_alpha),
            "target_entropy": float(self.target_entropy),
            "steps_before_learning": int(self.steps_before_learning),
            "steps_between_updates": int(self.steps_between_updates),
            "learning_steps": int(self.learning_steps),
            "policy_delay": int(self.policy_delay),
            "n_step": int(self.n_step),
            "memory_warmup": int(self.memory_warmup),
            "memory_capacity": int(self.buffer_size),
            "memory_batch_size": int(self.batch_size),
            "normalize_observations": bool(self.normalize_observations),
            "obs_norm_clip": float(self.obs_clip),
            "obs_norm_eps": float(self.obs_eps),
            "action_low": serialize_bound(self.action_low),
            "action_high": serialize_bound(self.action_high),
        }
