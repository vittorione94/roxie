import functools
import math
from typing import Any

import hydra
import jax
import jax.numpy as jnp
from flax import nnx

from roxie.agents.agent import Agent, TrainState
from roxie.agents.utils import (
    BurstNode,
    fused_grad_steps,
    graph_jit,
    make_optimizer,
    network_rngs,
    reduce_diagnostics,
    soft_update,
    transition_prototype,
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

    Returns ``(action, deviation_from_mode)``. The second value is what the
    trainer reduces to the `noise/per_joint_abs` panel; MPO explores from its own
    stochastic policy, so — exactly as in SAC — it is the sampled action's
    distance from the mode rather than an injected perturbation. It is returned
    unconditionally because the fused acting burst accumulates it inside a
    `lax.scan`, where a `None` would change the carry's structure.
    """
    distribution = actor_model(observation)
    try:
        mean = distribution.mean()
    except TypeError:
        # Some distrax versions expose mean as a property.
        mean = distribution.mean
    mode = jnp.clip(mean, -1.0, 1.0)

    if evaluate:
        return mode, jnp.zeros_like(mode)

    action = jnp.clip(distribution.sample(seed=key), -1.0, 1.0)
    return action, mode - action


# Not jitted on its own — called inside `_mpo_grad_steps` so N steps fuse into
# one compiled program.
def _mpo_grad_step(
    nodes,
    key: jax.random.PRNGKey,
    *,
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
    state, dual_params, dual_optimizer = nodes

    key, sample_key, critic_key, actor_key = jax.random.split(key, 4)

    samples = replay_sample_fn(state.buffer_state, sample_key)
    re_packed_samples = {
        "observations": samples.experience.first.observation,
        "actions": samples.experience.first.action,
        "rewards": samples.experience.first.reward,
        "next_observations": samples.experience.second.observation,
        "terminals": samples.experience.first.terminal,
    }
    # Normalized once here: both losses read the same `observations`.
    re_packed_samples = Agent.normalize_samples(
        re_packed_samples, obs_mean, obs_std, obs_clip, normalize
    )

    (critic_loss, critic_aux), critic_grads = nnx.value_and_grad(
        mpo_critic_loss_fn, has_aux=True
    )(
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

    # E-step + M-step, differentiated jointly w.r.t. the policy and the duals.
    (actor_loss, actor_aux), (actor_grads, dual_grads) = nnx.value_and_grad(
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

    # The target actor is the "old" policy the E-step samples from.
    soft_update(state.target_actor, state.actor, tau)
    soft_update(state.target_critic, state.critic, tau)

    return actor_loss, critic_loss, actor_aux, critic_aux


@functools.partial(
    graph_jit,
    static_argnames=(
        "gamma", "tau", "replay_sample_fn", "num_action_samples", "n_steps",
        "normalize",
    ),
    # The duals and their optimizer are mutated alongside the train state, so
    # all three ride in one split.
    num_nodes=3,
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
    obs_mean, obs_std = Agent.obs_mean_std(state.obs_stats, obs_eps)

    (state, dual_params, dual_optimizer), (
        actor_losses,
        critic_losses,
        actor_aux,
        critic_aux,
    ) = fused_grad_steps(
        (state, dual_params, dual_optimizer),
        key,
        n_steps,
        functools.partial(
            _mpo_grad_step,
            gamma=gamma,
            tau=tau,
            replay_sample_fn=replay_sample_fn,
            num_action_samples=num_action_samples,
            epsilon=epsilon,
            epsilon_mean=epsilon_mean,
            epsilon_stddev=epsilon_stddev,
            action_low=action_low,
            action_high=action_high,
            obs_mean=obs_mean,
            obs_std=obs_std,
            obs_clip=obs_clip,
            normalize=normalize,
        ),
    )

    # No `policy_delay` in MPO: the actor, the duals and the critic all move on
    # every fused step, so they share a denominator.
    diagnostics = {
        **reduce_diagnostics(actor_aux, n_steps),
        **reduce_diagnostics(critic_aux, n_steps),
    }
    return (
        state,
        dual_params,
        dual_optimizer,
        jnp.mean(actor_losses),
        jnp.mean(critic_losses),
        diagnostics,
    )


class MPO(Agent):
    _num_burst_nodes = 3
    # Mutated by every gradient burst, so held in the same split as the train
    # state.
    dual_params = BurstNode(1)
    dual_optimizer = BurstNode(2)

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

        actor = hydra.utils.instantiate(
            actor_config,
            in_features=env_obs_size,
            action_dim=env_action_size,
            rngs=network_rngs(self.seed, offset=0),
        )

        critic = hydra.utils.instantiate(
            critic_config,
            in_features=env_obs_size + env_action_size,
            rngs=network_rngs(self.seed, offset=2),
        )

        # No `truncation`: MPO's target is 1-step, so nothing reads past the
        # stored `terminal`.
        replay = hydra.utils.instantiate(memory_config)
        self.batch_size = memory_config.sample_batch_size
        self.buffer_size = memory_config.max_length
        buffer_state = replay.init(
            transition_prototype(env_obs_size, env_action_size, truncation=False)
        )

        self.dual_params = MPODualParams(
            action_dim=env_action_size,
            init_temperature=init_temperature,
            init_alpha_mean=init_alpha_mean,
            init_alpha_stddev=init_alpha_stddev,
        )
        self.dual_learning_rate = dual_learning_rate

        # Deliberately unclipped: a global-norm clip over a handful of scalars
        # would rescale the dual ascent step and fight the KL bounds.
        self.dual_optimizer = make_optimizer(
            self.dual_params,
            dual_optimizer_config,
            learning_rate=self.dual_learning_rate,
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
        # Outside `self.state`, and what enforce the KL trust region: resuming
        # with them reset would re-open it on an already-converged policy.
        return {
            "dual_params": self.dual_params,
            "dual_optimizer": self.dual_optimizer,
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
        ``lax.scan`` carry, so a whole ``steps_between_updates`` window of acting
        costs one host dispatch instead of one per env step. Returns
        ``(scaled_action, deviation_from_mode, extras)``; `extras` is the
        per-step fields the buffer stores beyond the standard five, and is empty
        here — only PPO has any.

``critic`` is accepted and ignored too: only an on-policy agent
        stores a value estimate at acting time.

        ``noise_module`` is accepted and ignored: MPO explores from its own
        stochastic policy and carries no noise module. The argument is part of
        the shared signature the fused acting burst calls through.
        """
        del noise_module, critic
        if self.normalize_observations:
            mean, std = Agent.obs_mean_std(obs_stats, self.obs_eps)
            observation = Agent.normalize_obs(observation, mean, std, self.obs_clip)

        action, noise = _mpo_step_fn(actor, observation, evaluate, key)
        return (
            Agent.scale_to_env(action, self.action_low, self.action_high),
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

    def _learn(self, agent_rng, n_steps=None):
        """Run one unconditional burst of ``n_steps`` (default ``learning_steps``)
        fused gradient steps, updating ``self.state`` in place; returns
        ``(actor_loss, critic_loss)``.

        Deliberately named `_learn`, not `learn`: the trainer treats a public
        `learn` as the signal that an agent can be driven by the async learner
        (`Trainer._run`). MPO now satisfies the rest of that contract
        (`select_action` / `add_transitions`, which is also what puts it on the
        fused acting path), so this name is the ONLY thing keeping it
        synchronous — and it should stay that way until an async run is actually
        validated against the sync curves. Renaming it is the whole opt-in.
        """
        burst_steps = self.learning_steps if n_steps is None else int(n_steps)
        actor_loss, critic_loss, diagnostics = _mpo_grad_steps(
            self._burst_nodes,
            agent_rng,
            burst_steps,
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
        self.record_diagnostics(diagnostics, burst_steps)
        return actor_loss, critic_loss

    def update(self, steps, agent_rng):
        gradient_steps, actor_loss, critic_loss = 0, 0, 0

        if self.due_for_update(steps):
            actor_loss, critic_loss = self._learn(agent_rng)
            gradient_steps += self.learning_steps

        return gradient_steps, actor_loss, critic_loss

    def _export_hyperparams(self) -> dict:
        params = super()._export_hyperparams()
        params.update(self._replay_hyperparams())
        params.update(
            {
                "dual_learning_rate": float(self.dual_learning_rate),
                "num_action_samples": int(self.num_action_samples),
                "epsilon": float(self.epsilon),
                "epsilon_mean": float(self.epsilon_mean),
                "epsilon_stddev": float(self.epsilon_stddev),
                "init_temperature": float(self.init_temperature),
                "init_alpha_mean": float(self.init_alpha_mean),
                "init_alpha_stddev": float(self.init_alpha_stddev),
            }
        )
        return params
