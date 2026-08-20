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
from roxie.losses.actor_losses import ddpg_actor_loss_fn
from roxie.losses.critic_losses import ddpg_critic_loss_fn


# Pure single gradient step. Not jitted on its own — it is called inside the
# jitted `_grad_steps` below so that N steps fuse into one compiled program.
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
    n_step: int = 1,
):
    """Performs one full gradient update step and returns the new state.

    `obs_mean`/`obs_std` are passed in (not recomputed): `obs_stats` is constant
    across the update loop, so they are hoisted out by `_grad_steps`.
    `n_step` is the TD horizon (NOT the scan length `n_steps` in _grad_steps).
    """
    # `repack_samples` folds the Bellman target ingredients (n-step return,
    # per-sample bootstrap coefficient, bootstrap observation) into the dict for
    # both buffer layouts, so the critic loss never sees gamma/terminals.
    key, noise_key = jax.random.split(key)
    samples = replay_sample_fn(state.buffer_state, key)
    re_packed_samples = repack_samples(samples, gamma, n_step)
    # Normalize once, here: both losses below read the same `observations`, and
    # neither of them normalizes (see `Agent.normalize_samples`).
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

    # Soft update of both target networks.
    new_actor_tensors = nnx.state(state.actor, nnx.Param)
    old_actor_tensors = nnx.state(state.target_actor, nnx.Param)

    new_target_actor_tensors = optax.incremental_update(
        new_tensors=new_actor_tensors, old_tensors=old_actor_tensors, step_size=tau
    )

    new_critic_tensors = nnx.state(state.critic, nnx.Param)
    old_critic_tensors = nnx.state(state.target_critic, nnx.Param)
    new_target_critic_tensors = optax.incremental_update(
        new_tensors=new_critic_tensors, old_tensors=old_critic_tensors, step_size=tau
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


# Fused N-step update. The body is compiled once and run `n_steps` times on-device
# via `lax.scan` rather than unrolled, which would blow up compile time and HLO
# size at large `n_steps`. Only the trainable graph state is carried;
# `buffer_state` and the normalization params are loop-constant and closed over.
@functools.partial(
    nnx.jit,
    static_argnames=(
        "gamma", "tau", "replay_sample_fn", "n_steps", "n_step", "normalize",
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
    n_step: int = 1,
):
    # Hoist the (loop-constant) normalization params out of the scan body.
    obs_mean, obs_std = Agent.obs_mean_std(state.obs_stats, obs_eps)

    # Pre-split all per-step keys so they can be scanned over as `xs`.
    keys = jax.random.split(key, n_steps)

    # Split into a static graph definition + the trainable pytree state. Only the
    # state is carried through the scan; the graphdef is closed over.
    graphdef, scan_state = nnx.split(state)

    def body(scan_state, step_key):
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
            n_step,
        )
        _, scan_state = nnx.split(st)
        return scan_state, (actor_loss, critic_loss)

    scan_state, (actor_losses, critic_losses) = jax.lax.scan(body, scan_state, keys)
    state = nnx.merge(graphdef, scan_state)

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

        # Network-init seed. Set before `_make_critic` (called below and
        # overridden by TD3/TD4) so subclasses can derive their own offsets.
        self.seed = int(seed)

        actor = hydra.utils.instantiate(
            actor_config,
            in_features=env_obs_size,
            action_dim=env_action_size,
            rngs=network_rngs(self.seed, offset=0),
        )

        # Overridable so TD3 can swap in a TwinCritic.
        critic = self._make_critic(critic_config, env_obs_size, env_action_size)

        # `truncation` is stored alongside `terminal` so n-step windows can stop at
        # episode boundaries the terminal flag doesn't mark (clip-end / time-limit).
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

        noise_module = hydra.utils.instantiate(
            noise_config, action_shape=(env_action_size,)
        )

        target_actor = copy.deepcopy(actor)
        target_critic = copy.deepcopy(critic)

        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate = actor_learning_rate
        self.max_grad_norm = max_grad_norm
        self.noise_module = noise_module

        # Optimizer family + its hyperparameters come from yaml; the learning
        # rate and the global-norm clip stay top-level agent args.
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
            critic,
            build_optimizer(
                critic_optimizer_config,
                learning_rate=self.critic_learning_rate,
                max_grad_norm=self.max_grad_norm,
            ),
            wrt=nnx.Param,
        )

        obs_shape = buffer_state.experience.observation.shape[-1]
        obs_stats = Agent.init_obs_stats(obs_shape)

        self.state = TrainState(
            actor=actor,
            critic=critic,
            target_actor=target_actor,
            target_critic=target_critic,
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
        self.memory_warmup = memory_warmup
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise
        self.normalize_observations = normalize_observations
        self.obs_clip = float(obs_norm_clip)
        self.obs_eps = float(obs_norm_eps)
        # Weight on the actor's pre-tanh saturation penalty, consumed by TD3's actor
        # loss. Stored on the base so every DeterministicActor agent round-trips it
        # through checkpoints identically. 0.0 = the textbook DPG objective.
        self.pre_activation_coef = float(pre_activation_coef)

        print(f"{type(self).__name__} agent initialized.")
        print("Noise module hyperparameters:", self.noise_module.hyperparameters())
        print("Hyper Params:", self._export_hyperparams())

    def _make_critic(self, critic_config, env_obs_size, env_action_size):
        """Build the critic network. Overridden by TD3 to return a TwinCritic."""
        return hydra.utils.instantiate(
            critic_config,
            in_features=env_obs_size + env_action_size,
            rngs=network_rngs(self.seed, offset=2),
        )

    def replay_add(self, buffer_state, transitions):
        """Add one env-step batch of transitions (leaves shaped (B, ...)).

        The trajectory buffer (n_step > 1) expects an explicit time axis on
        every leaf — (B, T=1, ...) for per-step adds — while the flat buffer
        takes the batch as-is. Callers (self.add, the trainer's warmup fill)
        go through here so they never need to know which layout is active.
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
        normalization + noise path. Returns ``(scaled_action, applied_noise)``.
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
        """
        Selects an action by calling the pure, JIT-compiled step function.
        """
        self.last_action, noise = self.select_action(
            self.state.actor, self.state.obs_stats, observation, key, evaluate,
        )
        # Effective exploration noise actually applied this step (post-clip), in
        # normalized [-1, 1] action units -- one value per env per joint. Kept on
        # device; the trainer reduces it to a per-joint epoch mean for logging.
        # Zero when evaluating (add_noise is a no-op there).
        self.last_noise = noise

        return self.last_action

    def add_transitions(
        self, prev_obs, action, reward, termination, truncation, next_obs
    ):
        """Buffer one env-step batch given an EXPLICIT action.

        Split out of ``add`` so the async learner (which owns ``self.state`` on
        its own thread) can add transitions whose action travelled with them
        through the hand-off queue, rather than reading ``self.last_action``
        (which the acting thread overwrites every step). Mutates
        ``self.state.buffer_state`` / ``self.state.obs_stats`` in place.
        """
        experiences = Transition(
            observation=prev_obs,
            action=action,
            reward=reward,
            # The true termination signal, NOT `done` (= termination OR truncation).
            # A time-limit truncation must still bootstrap the next-state value in
            # the Bellman target; marking it terminal zeroes the bootstrap and
            # collapses Q at the cutoff — for every env at once, since they hit the
            # time limit in lockstep.
            terminal=termination,
            # Stored separately so n-step windows can stop at truncations too —
            # in the flat stream the item after ANY done is the next episode's
            # reset state, so a window must never accumulate across one.
            truncation=truncation,
        )

        self.state.buffer_state = self.replay_add(self.state.buffer_state, experiences)

        # Normalization stats see both the current and the next observation.
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
        so the buffer-donating ``_grad_steps`` stays valid unchanged. The async
        learner passes a small ``n_steps`` (chunk) so the GPU stream frees up
        frequently for the acting thread's forward pass, letting the CPU physics
        overlap the learning instead of stalling behind one large fused burst.
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
        return {
            "seed": int(self.seed),
            "gamma": float(self.gamma),
            "tau": float(self.tau),
            "actor_learning_rate": float(self.actor_learning_rate),
            "critic_learning_rate": float(self.critic_learning_rate),
            "max_grad_norm": float(self.max_grad_norm),
            "env_obs_size": self.state.buffer_state.experience.observation.shape[2],
            "env_action_size": self.state.buffer_state.experience.action.shape[2],
            "target_policy_noise": float(self.target_policy_noise),
            "pre_activation_coef": float(self.pre_activation_coef),
            "target_noise_clip": float(self.target_noise_clip),
            "steps_before_learning": int(self.steps_before_learning),
            "steps_between_updates": int(self.steps_between_updates),
            "learning_steps": int(self.learning_steps),
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
