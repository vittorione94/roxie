import copy
import functools
import math

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
    serialize_bound,
)
from roxie.losses.actor_losses import mpo_actor_loss_fn
from roxie.losses.critic_losses import mpo_critic_loss_fn


def _inv_softplus(y: float) -> float:
    """Inverse of softplus so a raw param initializes to a desired positive value."""
    return float(math.log(math.expm1(y)))


class MPODualParams(nnx.Module):
    """Lagrange dual variables for MPO, stored in raw (pre-softplus) space.

    - ``log_temperature``: the E-step temperature (scalar) for the KL bound.
    - ``log_alpha_mean`` / ``log_alpha_stddev``: per-dimension M-step KL
      multipliers for the decoupled mean / covariance trust regions.
    """

    def __init__(
        self,
        action_dim: int,
        init_temperature: float = 1.0,
        init_alpha_mean: float = 1.0,
        init_alpha_stddev: float = 1.0,
    ):
        self.log_temperature = nnx.Param(
            jnp.asarray(_inv_softplus(init_temperature), dtype=jnp.float32)
        )
        self.log_alpha_mean = nnx.Param(
            jnp.full((action_dim,), _inv_softplus(init_alpha_mean), dtype=jnp.float32)
        )
        self.log_alpha_stddev = nnx.Param(
            jnp.full((action_dim,), _inv_softplus(init_alpha_stddev), dtype=jnp.float32)
        )


@functools.partial(nnx.jit, static_argnames=("evaluate",))
def _mpo_step_fn(actor_model, observation, evaluate, key):
    """Action selection for MPO: mean when evaluating, otherwise a sample.

    Actions are a plain Gaussian bounded by clipping to [-1, 1] (no tanh
    squashing), consistent with how the loss treats them.
    """
    distribution = actor_model(observation)
    if evaluate:
        try:
            action = distribution.mean()
        except TypeError:
            action = distribution.mean
    else:
        action = distribution.sample(seed=key)
    return jnp.clip(action, -1.0, 1.0)


# Single MPO gradient step. Not jitted on its own — called inside the jitted
# `_mpo_grad_steps` below so N steps fuse into one compiled program. `obs_mean` /
# `obs_std` are hoisted in by the caller (the stats are loop-constant).
def _mpo_grad_step(
    state: TrainState,
    dual_params: MPODualParams,
    dual_optimizer: nnx.Optimizer,
    key: jax.random.PRNGKey,
    gamma: float,
    tau: float,
    replay_sample_fn,
    num_action_samples: int,
    epsilon: float,
    epsilon_mean: float,
    epsilon_stddev: float,
    action_low: float,
    action_high: float,
    obs_mean: jnp.ndarray,
    obs_std: jnp.ndarray,
    obs_clip: float,
    normalize: bool,
):
    key, sample_key, critic_key, actor_key = jax.random.split(key, 4)

    samples = replay_sample_fn(state.buffer_state, sample_key)
    re_packed_samples = {
        "observations": samples.experience.first.observation,
        "actions": samples.experience.first.action,
        "rewards": samples.experience.first.reward,
        "next_observations": samples.experience.second.observation,
        "terminals": samples.experience.first.terminal,
    }
    # Normalize once, here: both losses below read the same `observations`, and
    # neither of them normalizes (see `Agent.normalize_samples`).
    re_packed_samples = Agent.normalize_samples(
        re_packed_samples, obs_mean, obs_std, obs_clip, normalize
    )

    # Critic update: policy evaluation under the target policy.
    critic_loss, critic_grads = nnx.value_and_grad(mpo_critic_loss_fn)(
        state.critic,
        state.target_actor,
        state.target_critic,
        re_packed_samples,
        gamma,
        critic_key,
        num_action_samples,
        action_low,
        action_high,
    )
    state.critic_optimizer.update(state.critic, critic_grads)

    # Actor + dual update (E-step + M-step), differentiated jointly w.r.t. the
    # policy and the Lagrange duals.
    (actor_loss, aux), (actor_grads, dual_grads) = nnx.value_and_grad(
        mpo_actor_loss_fn, argnums=(0, 1), has_aux=True
    )(
        state.actor,
        dual_params,
        state.target_actor,
        state.critic,
        re_packed_samples,
        actor_key,
        num_action_samples,
        epsilon,
        epsilon_mean,
        epsilon_stddev,
        action_low,
        action_high,
    )
    state.actor_optimizer.update(state.actor, actor_grads)
    dual_optimizer.update(dual_params, dual_grads)

    # Soft-update both target networks. The target actor is the "old" policy the
    # E-step samples from, so it tracks the online policy slowly.
    new_actor_tensors = nnx.state(state.actor, nnx.Param)
    old_actor_tensors = nnx.state(state.target_actor, nnx.Param)
    nnx.update(
        state.target_actor,
        optax.incremental_update(new_actor_tensors, old_actor_tensors, tau),
    )

    new_critic_tensors = nnx.state(state.critic, nnx.Param)
    old_critic_tensors = nnx.state(state.target_critic, nnx.Param)
    nnx.update(
        state.target_critic,
        optax.incremental_update(new_critic_tensors, old_critic_tensors, tau),
    )

    new_state = TrainState(
        actor=state.actor,
        critic=state.critic,
        target_actor=state.target_actor,
        target_critic=state.target_critic,
        actor_optimizer=state.actor_optimizer,
        critic_optimizer=state.critic_optimizer,
        buffer_state=state.buffer_state,
        obs_stats=state.obs_stats,
    )
    return new_state, actor_loss, critic_loss


# Fused N-step update. The body is compiled once and run `n_steps` times
# on-device via `lax.scan`, so a burst costs one host dispatch rather than one per
# gradient step. The Lagrange duals and their optimizer ride along in the carry
# alongside
# the train state; `buffer_state` and the normalization params are loop-constant.
@functools.partial(
    nnx.jit,
    static_argnames=(
        "gamma", "tau", "replay_sample_fn", "num_action_samples", "n_steps",
        "normalize",
    ),
    # Donate the train state (arg 0): its large read-only replay buffer is threaded
    # unchanged through the scan, so without donation XLA allocates a full second
    # copy of it every update. The caller reassigns self.state from the result.
    donate_argnums=(0,),
)
def _mpo_grad_steps(
    state: TrainState,
    dual_params: MPODualParams,
    dual_optimizer: nnx.Optimizer,
    key: jax.random.PRNGKey,
    n_steps: int,
    gamma: float,
    tau: float,
    replay_sample_fn,
    num_action_samples: int,
    epsilon: float,
    epsilon_mean: float,
    epsilon_stddev: float,
    action_low: float,
    action_high: float,
    obs_eps: float,
    obs_clip: float,
    normalize: bool,
):
    # Hoist the (loop-constant) normalization params out of the scan body.
    obs_mean, obs_std = Agent.obs_mean_std(state.obs_stats, obs_eps)

    # Pre-split all per-step keys so they can be scanned over as `xs`.
    keys = jax.random.split(key, n_steps)

    # Split into a static graph definition + the trainable pytree state. Only
    # the state is carried through the scan; the graphdef is closed over. The
    # three graph nodes are split as one tuple so the duals and their Adam slots
    # stay in the carry and keep updating across the fused steps.
    graphdef, scan_state = nnx.split((state, dual_params, dual_optimizer))

    def body(scan_state, step_key):
        st, duals, dopt = nnx.merge(graphdef, scan_state)
        st, actor_loss, critic_loss = _mpo_grad_step(
            st,
            duals,
            dopt,
            step_key,
            gamma,
            tau,
            replay_sample_fn,
            num_action_samples,
            epsilon,
            epsilon_mean,
            epsilon_stddev,
            action_low,
            action_high,
            obs_mean,
            obs_std,
            obs_clip,
            normalize,
        )
        _, scan_state = nnx.split((st, duals, dopt))
        return scan_state, (actor_loss, critic_loss)

    scan_state, (actor_losses, critic_losses) = jax.lax.scan(body, scan_state, keys)
    state, dual_params, dual_optimizer = nnx.merge(graphdef, scan_state)

    # Average over the burst for less noisy logging.
    return (
        state,
        dual_params,
        dual_optimizer,
        jnp.mean(actor_losses),
        jnp.mean(critic_losses),
    )


class MPO(Agent):
    """Maximum a Posteriori Policy Optimization (Abdolmaleki et al., 2018).

    https://arxiv.org/abs/1806.06920

    Off-policy actor-critic with a Gaussian policy. Each update alternates:

    - E-step: estimate a nonparametric improved policy by reweighting actions
      sampled from the target policy with ``softmax(Q / temperature)``; the
      temperature solves a convex dual of a hard KL bound (``epsilon``).
    - M-step: project that improved policy back onto the parametric Gaussian by
      weighted maximum likelihood under a decoupled KL trust region on the mean
      (``epsilon_mean``) and the covariance (``epsilon_stddev``).

    The temperature and the two KL multipliers are learned Lagrange duals
    (``MPODualParams``) optimized jointly with the policy.
    """

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
        dual_optimizer_config: dict = None,
        seed: int = 0,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        dual_learning_rate: float = 1e-2,
        gamma: float = 0.99,
        tau: float = 5e-3,
        num_action_samples: int = 20,
        epsilon: float = 0.1,
        epsilon_mean: float = 1e-3,
        epsilon_stddev: float = 1e-5,
        init_temperature: float = 1.0,
        init_alpha_mean: float = 1.0,
        init_alpha_stddev: float = 1.0,
        steps_before_learning: int = 100,
        steps_between_updates: int = 10,
        learning_steps: int = 5,
        memory_warmup: int = 100,
        max_grad_norm: float = 1.0,
        normalize_observations: bool = True,
        obs_norm_clip: float = 5.0,
        obs_norm_eps: float = 1e-8,
    ):
        self.seed = int(seed)

        # Gaussian policy (mean + diagonal std).
        actor = hydra.utils.instantiate(
            actor_config,
            in_features=env_obs_size,
            action_dim=env_action_size,
            rngs=network_rngs(self.seed, offset=0),
        )

        # Single Q critic (as in the original MPO paper).
        critic = hydra.utils.instantiate(
            critic_config,
            in_features=env_obs_size + env_action_size,
            rngs=network_rngs(self.seed, offset=2),
        )

        # Replay buffer.
        prototype = Transition(
            observation=jnp.zeros(env_obs_size, dtype=jnp.float32),
            action=jnp.zeros(env_action_size, dtype=jnp.float32),
            reward=jnp.zeros((), dtype=jnp.float32),
            terminal=jnp.zeros((), dtype=jnp.bool_),
        )
        replay = hydra.utils.instantiate(memory_config)
        self.batch_size = memory_config.sample_batch_size
        self.buffer_size = memory_config.max_length
        buffer_state = replay.init(prototype)

        # Target networks: the target actor is the old policy the E-step samples.
        target_actor = copy.deepcopy(actor)
        target_critic = copy.deepcopy(critic)

        # Learnable Lagrange duals (temperature + decoupled KL multipliers).
        self.dual_params = MPODualParams(
            action_dim=env_action_size,
            init_temperature=init_temperature,
            init_alpha_mean=init_alpha_mean,
            init_alpha_stddev=init_alpha_stddev,
        )

        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.dual_learning_rate = dual_learning_rate
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
            critic,
            build_optimizer(
                critic_optimizer_config,
                learning_rate=self.critic_learning_rate,
                max_grad_norm=self.max_grad_norm,
            ),
            wrt=nnx.Param,
        )
        # The Lagrange duals are a handful of scalars, deliberately unclipped:
        # a global-norm clip over them would just rescale the dual ascent step
        # and fight the KL bounds it is supposed to enforce.
        self.dual_optimizer = nnx.Optimizer(
            self.dual_params,
            build_optimizer(
                dual_optimizer_config, learning_rate=self.dual_learning_rate
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
        self.num_action_samples = int(num_action_samples)
        self.epsilon = float(epsilon)
        self.epsilon_mean = float(epsilon_mean)
        self.epsilon_stddev = float(epsilon_stddev)
        self.init_temperature = float(init_temperature)
        self.init_alpha_mean = float(init_alpha_mean)
        self.init_alpha_stddev = float(init_alpha_stddev)
        self.action_low = action_low
        self.action_high = action_high
        self.replay = replay
        self.steps_before_learning = steps_before_learning
        self.steps_between_updates = steps_between_updates
        self.learning_steps = learning_steps
        self.memory_warmup = memory_warmup
        self.normalize_observations = normalize_observations
        self.obs_clip = float(obs_norm_clip)
        self.obs_eps = float(obs_norm_eps)

        print("MPO agent initialized.")
        print("Hyper Params:", self._export_hyperparams())

    def _checkpoint_modules(self) -> dict:
        # The Lagrange duals and their optimizer live outside `self.state`. They
        # are learned, and they are what enforce the KL trust region: resuming
        # with them reset to `init_temperature` / `init_alpha_*` would re-open
        # the trust region on an already-converged policy.
        return {
            "dual_params": self.dual_params,
            "dual_optimizer": self.dual_optimizer,
        }

    def step(
        self,
        observation: jnp.ndarray,
        evaluate: bool = False,
        key: jax.random.PRNGKey = None,
    ) -> jnp.ndarray:
        if self.normalize_observations:
            mean, std = Agent.obs_mean_std(self.state.obs_stats, self.obs_eps)
            observation = Agent.normalize_obs(observation, mean, std, self.obs_clip)

        action = _mpo_step_fn(self.state.actor, observation, evaluate, key)
        self.last_action = Agent.scale_to_env(action, self.action_low, self.action_high)
        return self.last_action

    def add(self, prev_states, states):
        experiences = Transition(
            observation=prev_states.obs,
            action=self.last_action,
            reward=states.reward,
            # True termination only (not time-limit truncation) so truncated
            # transitions still bootstrap in the Bellman target.
            terminal=states.info["termination"],
        )
        self.state.buffer_state = self.replay.add(self.state.buffer_state, experiences)

        if self.normalize_observations:
            obs_batch = jnp.concatenate([prev_states.obs, states.obs], axis=0)
            self.state.obs_stats = Agent.update_obs_stats(
                self.state.obs_stats, obs_batch
            )

    def _learn(self, agent_rng, n_steps=None):
        """Run one unconditional burst of ``n_steps`` (default ``learning_steps``)
        fused gradient steps, updating ``self.state`` in place; returns
        ``(actor_loss, critic_loss)``.

        Deliberately named `_learn`, not `learn`: the trainer treats a public
        `learn` as the signal that an agent can be driven by the async learner
        (`Trainer._run`), which additionally requires `select_action` and
        `add_transitions`. MPO implements neither, so it stays on the sync path
        until it does.
        """
        (
            self.state,
            self.dual_params,
            self.dual_optimizer,
            actor_loss,
            critic_loss,
        ) = _mpo_grad_steps(
            self.state,
            self.dual_params,
            self.dual_optimizer,
            agent_rng,
            self.learning_steps if n_steps is None else int(n_steps),
            self.gamma,
            self.tau,
            self.replay.sample,
            self.num_action_samples,
            self.epsilon,
            self.epsilon_mean,
            self.epsilon_stddev,
            self.action_low,
            self.action_high,
            self.obs_eps,
            self.obs_clip,
            self.normalize_observations,
        )
        return actor_loss, critic_loss

    def update(self, steps, agent_rng):
        gradient_steps, actor_loss, critic_loss = 0, 0, 0

        if self.due_for_update(steps):
            actor_loss, critic_loss = self._learn(agent_rng)
            gradient_steps += self.learning_steps

        return gradient_steps, actor_loss, critic_loss

    def _export_hyperparams(self) -> dict:
        return {
            "seed": int(self.seed),
            "gamma": float(self.gamma),
            "tau": float(self.tau),
            "actor_learning_rate": float(self.actor_learning_rate),
            "critic_learning_rate": float(self.critic_learning_rate),
            "dual_learning_rate": float(self.dual_learning_rate),
            "max_grad_norm": float(self.max_grad_norm),
            "env_obs_size": self.state.buffer_state.experience.observation.shape[2],
            "env_action_size": self.state.buffer_state.experience.action.shape[2],
            "num_action_samples": int(self.num_action_samples),
            "epsilon": float(self.epsilon),
            "epsilon_mean": float(self.epsilon_mean),
            "epsilon_stddev": float(self.epsilon_stddev),
            "init_temperature": float(self.init_temperature),
            "init_alpha_mean": float(self.init_alpha_mean),
            "init_alpha_stddev": float(self.init_alpha_stddev),
            "steps_before_learning": int(self.steps_before_learning),
            "steps_between_updates": int(self.steps_between_updates),
            "learning_steps": int(self.learning_steps),
            "memory_warmup": int(self.memory_warmup),
            "memory_capacity": int(self.buffer_size),
            "memory_batch_size": int(self.batch_size),
            "normalize_observations": bool(self.normalize_observations),
            "obs_norm_clip": float(self.obs_clip),
            "obs_norm_eps": float(self.obs_eps),
            "action_low": serialize_bound(self.action_low),
            "action_high": serialize_bound(self.action_high),
        }
