import copy
import functools
import math

import hydra
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from roxie.agents.agent import Agent, TrainState
from roxie.agents.utils import Transition, serialize_bound
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


@functools.partial(
    nnx.jit,
    static_argnames=("gamma", "tau", "replay_sample_fn", "num_action_samples"),
)
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
    obs_eps: float,
    obs_clip: float,
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

    obs_mean, obs_std = Agent.obs_mean_std(state.obs_stats, obs_eps)

    # 1. Critic update (policy evaluation under the target policy).
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
        obs_mean,
        obs_std,
        obs_clip,
    )
    state.critic_optimizer.update(state.critic, critic_grads)

    # 2. Actor + dual update (E-step + M-step). Differentiate jointly w.r.t. the
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
        obs_mean,
        obs_std,
        obs_clip,
    )
    state.actor_optimizer.update(state.actor, actor_grads)
    dual_optimizer.update(dual_params, dual_grads)

    # 3. Soft-update both target networks. The target actor is the "old" policy
    # the E-step samples from, so it tracks the online policy slowly.
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
    return new_state, dual_params, dual_optimizer, actor_loss, critic_loss


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
        actor_rngs = nnx.Rngs(params=0, dropout=1)
        critic_rngs = nnx.Rngs(params=2, dropout=3)

        # Gaussian policy (mean + diagonal std).
        actor = hydra.utils.instantiate(
            actor_config,
            in_features=env_obs_size,
            action_dim=env_action_size,
            rngs=actor_rngs,
        )

        # Single Q critic (as in the original MPO paper).
        critic = hydra.utils.instantiate(
            critic_config,
            in_features=env_obs_size + env_action_size,
            rngs=critic_rngs,
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
            optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.adam(self.actor_learning_rate),
            ),
            wrt=nnx.Param,
        )
        critic_optimizer = nnx.Optimizer(
            critic,
            optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.adam(self.critic_learning_rate),
            ),
            wrt=nnx.Param,
        )
        self.dual_optimizer = nnx.Optimizer(
            self.dual_params,
            optax.adam(self.dual_learning_rate),
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

    def update(self, steps, agent_rng):
        gradient_steps, actor_loss, critic_loss = 0, 0, 0

        if (
            steps >= self.steps_before_learning
            and (steps - self.steps_before_learning) % self.steps_between_updates == 0
        ):
            for _ in range(self.learning_steps):
                agent_rng, key = jax.random.split(agent_rng, 2)
                (
                    self.state,
                    self.dual_params,
                    self.dual_optimizer,
                    actor_loss,
                    critic_loss,
                ) = _mpo_grad_step(
                    self.state,
                    self.dual_params,
                    self.dual_optimizer,
                    key,
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
                )
            gradient_steps += self.learning_steps

        return gradient_steps, actor_loss, critic_loss

    def _export_hyperparams(self) -> dict:
        return {
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
