"""Proximal Policy Optimization (PPO) agent."""

import dataclasses
import functools
from typing import NamedTuple

import hydra
import jax
import jax.numpy as jnp
import rlax
from flax import nnx

from roxie.agents.agent import Agent, LearningOutput, TrainState
from roxie.agents.hyperparams import PPOHyperparams, build_hyperparams
from roxie.agents.utils import transition_prototype
from roxie.losses.actor_losses import ppo_loss_fn
from roxie.losses.critic_losses import ppo_critic_loss_fn
from roxie.models.actors import stochastic_step_fn
from roxie.utils import precision
from roxie.utils.math import normalize_obs, obs_mean_std, scale_to_env
from roxie.utils.memory import ReplayManager


_AUX_KEYS = (
    "approx_kl",
    "clip_frac",
    "entropy",
    "policy_std",
    "sat_frac",
    "tanh_grad",
)


class AdvantageScale(nnx.Module):
    """Debiased running std of the GAE advantage, carried across rollouts.

    Attributes:
        mean_sq: Decayed accumulator of each rollout's advantage variance.
        count: Decayed accumulator of the weights, which debiases `mean_sq`.
    """

    def __init__(self):
        self.mean_sq = nnx.Param(jnp.zeros((), dtype=precision.FLOAT))
        self.count = nnx.Param(jnp.zeros((), dtype=precision.FLOAT))


class GradStats(NamedTuple):
    """Running totals a rollout's minibatch passes accumulate.

    Attributes:
        steps: Minibatch passes actually taken before the trust region tripped.
        actor_loss_sum: Summed actor loss over those passes.
        critic_loss_sum: Summed critic loss over those passes.
        aux_sums: Summed `ppo_loss_fn` diagnostics, keyed by `_AUX_KEYS`.
        early_stops: Passes whose `approx_kl` exceeded `target_kl`.
        last_approx_kl: `approx_kl` from the most recent pass.
        last_clip_frac: `clip_frac` from the most recent pass.
    """

    steps: jnp.ndarray
    actor_loss_sum: jnp.ndarray
    critic_loss_sum: jnp.ndarray
    aux_sums: dict
    early_stops: jnp.ndarray
    last_approx_kl: jnp.ndarray
    last_clip_frac: jnp.ndarray


class StepCarry(NamedTuple):
    """The epoch and minibatch scans' shared carry.

    Attributes:
        state: The train state being updated in place.
        rng: The key stream both scans split from.
        stopped: Sticky flag; once set, every remaining pass is a no-op.
        stats: The running totals.
    """

    state: TrainState
    rng: jax.Array
    stopped: jnp.ndarray
    stats: GradStats


@functools.partial(
    jax.jit,
    static_argnames=("hp", "replay_get_fn"),
)
def _prepare_rollout(
    state: TrainState,
    adv_scale: AdvantageScale,
    hp: PPOHyperparams,
    replay_get_fn,
    obs_mean: jnp.ndarray,
    obs_std: jnp.ndarray,
):
    """Dequeues a rollout, computes GAE, and flattens it to a transition batch.

    Returns:
        `(state, adv_scale, norm_obs, pre_actions, old_log_probs, returns_t,
        adv_t, rollout_stats)` — every tensor flat over `(env, time)` and trimmed
        to the GAE horizon, so `_ppo_grad_steps` shuffles transitions.
    """
    state.buffer_state, data = replay_get_fn(state.buffer_state)
    data = getattr(data, "experience", data)

    norm_obs = (
        normalize_obs(data.observation, obs_mean, obs_std, hp.obs_norm_clip)
        if hp.normalize_observations
        else data.observation
    )

    values = data.value
    v_t = values[:, :-1]
    term = data.terminal[:, :-1].astype(jnp.float32)
    trunc = data.truncation[:, :-1].astype(jnp.float32)

    discount_t = hp.gamma * (1.0 - term) * (1.0 - trunc)
    r_t = jnp.where(trunc > 0, v_t, data.reward[:, :-1])

    adv_t = jax.vmap(
        rlax.truncated_generalized_advantage_estimation,
        in_axes=(0, 0, None, 0),
    )(r_t, discount_t, hp.gae_lambda, values)

    returns_t = adv_t + v_t

    residual_var = jnp.var(returns_t - v_t)

    # Flattened over `(env, time)` so the update's minibatches are draws from the
    # rollout's TRANSITIONS: permuting the env axis alone hands each minibatch
    # whole trajectories, every step of which shares a start state and a policy.
    # The policy tensors are sliced to the GAE horizon here rather than in the
    # losses, so one permutation serves every leaf.
    def flat(x):
        return x[:, :-1].reshape((-1,) + x.shape[2:])

    norm_obs = flat(norm_obs)
    pre_actions = flat(data.pre_action)
    old_log_probs = flat(data.log_probs)
    returns_t = returns_t.reshape(-1)
    adv_t = adv_t.reshape(-1)

    # Normalizing by this rollout's own spread rescales a rollout with no
    # reward in it to unit variance, so PPO takes a full trust-region step on
    # value-function noise.
    centered = adv_t - jnp.mean(adv_t)
    decay = hp.adv_norm_decay
    mean_sq = decay * adv_scale.mean_sq[...] + (1.0 - decay) * jnp.mean(
        jnp.square(centered)
    )
    count = decay * adv_scale.count[...] + (1.0 - decay)
    adv_scale.mean_sq[...] = mean_sq
    adv_scale.count[...] = count
    scale = jnp.sqrt(mean_sq / jnp.maximum(count, 1e-8))

    rollout_stats = {
        "adv_abs": jnp.mean(jnp.abs(adv_t)),
        "adv_scale": scale,
        "value_ev": 1.0 - residual_var / (jnp.var(returns_t) + 1e-8),
        "trunc_frac": jnp.mean(trunc),
        "term_frac": jnp.mean(term),
        "reward_abs": jnp.mean(jnp.abs(data.reward[:, :-1])),
    }

    adv_t = centered / (scale + 1e-8)

    return (
        state,
        adv_scale,
        norm_obs,
        pre_actions,
        old_log_probs,
        returns_t,
        adv_t,
        rollout_stats,
    )


def _clip_jointly(grads, max_norm):
    """Scales a tuple of gradient trees by ONE global norm."""
    if not max_norm:
        return grads
    sq = sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads))
    # `+ 1e-6` and the clamp at 1 are `torch.nn.utils.clip_grad_norm_`'s own.
    scale = jnp.minimum(1.0, max_norm / (jnp.sqrt(sq) + 1e-6))
    return jax.tree.map(lambda g: g * scale, grads)


def _ppo_grad_step(
    state: TrainState,
    key: jax.Array,
    norm_obs: jnp.ndarray,
    pre_actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    returns_t: jnp.ndarray,
    adv_t: jnp.ndarray,
    *,
    hp: PPOHyperparams,
):
    """Executes a single minibatch gradient step for actor and critic networks."""
    (actor_loss, aux), actor_grads = nnx.value_and_grad(
        ppo_loss_fn, has_aux=True
    )(
        actor_model=state.actor,
        observations=norm_obs,
        pre_actions=pre_actions,
        old_log_probs=old_log_probs,
        advantages=adv_t,
        clip_epsilon=hp.clip_eps,
        entropy_coef=hp.entropy_coef,
        key=key,
    )
    critic_loss, critic_grads = nnx.value_and_grad(ppo_critic_loss_fn)(
        state.critic,
        observations=norm_obs,
        returns=returns_t,
    )

    # Both gradients before either update, so the clip below sees the same pair
    # the reference's single backward pass produces.
    actor_grads, critic_grads = _clip_jointly(
        (actor_grads, critic_grads), hp.max_grad_norm
    )
    state.actor_optimizer.update(state.actor, actor_grads)
    state.critic_optimizer.update(state.critic, critic_grads)

    return actor_loss, critic_loss, aux


def _init_grad_stats() -> GradStats:
    f0, i0 = jnp.zeros((), jnp.float32), jnp.zeros((), jnp.int32)
    return GradStats(
        steps=i0,
        actor_loss_sum=f0,
        critic_loss_sum=f0,
        aux_sums={key: f0 for key in _AUX_KEYS},
        early_stops=i0,
        last_approx_kl=f0,
        last_clip_frac=f0,
    )


def _step_minibatch(carry: StepCarry, mb_data: tuple, hp: PPOHyperparams) -> StepCarry:
    """Executes a single minibatch update if target KL has not been reached."""
    state, rng, _stopped, stats = carry
    rng, step_key = jax.random.split(rng)

    actor_loss, critic_loss, aux = _ppo_grad_step(state, step_key, *mb_data, hp=hp)

    approx_kl, clip_frac = aux["approx_kl"], aux["clip_frac"]
    trip = (approx_kl > hp.target_kl) if hp.target_kl is not None else jnp.bool_(False)

    new_stats = GradStats(
        steps=stats.steps + 1,
        actor_loss_sum=stats.actor_loss_sum + actor_loss,
        critic_loss_sum=stats.critic_loss_sum + critic_loss,
        aux_sums={key: stats.aux_sums[key] + aux[key] for key in _AUX_KEYS},
        early_stops=stats.early_stops + trip.astype(jnp.int32),
        last_approx_kl=approx_kl,
        last_clip_frac=clip_frac,
    )
    return StepCarry(state, rng, trip, new_stats)


@functools.partial(
    jax.jit,
    static_argnames=("hp", "minibatch_size"),
    donate_argnums=(0,),
)
def _ppo_grad_steps(
    state: TrainState,
    key: jax.Array,
    norm_obs: jnp.ndarray,
    pre_actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    returns_t: jnp.ndarray,
    adv_t: jnp.ndarray,
    hp: PPOHyperparams,
    minibatch_size: int,
):
    """Runs full minibatch epoch schedules for a prepared rollout on-device."""
    rollout = (norm_obs, pre_actions, old_log_probs, returns_t, adv_t)
    init_carry = StepCarry(
        state=state,
        rng=key,
        stopped=jnp.bool_(False),
        stats=_init_grad_stats(),
    )

    def epoch_fn(carry: StepCarry, _):
        def run_epoch(c: StepCarry) -> StepCarry:
            c_rng, perm_key = jax.random.split(c.rng)
            perm = jax.random.permutation(perm_key, norm_obs.shape[0])

            shuffled_mbs = jax.tree.map(
                lambda leaf: leaf[perm].reshape(
                    (hp.num_minibatches, minibatch_size) + leaf.shape[1:]
                ),
                rollout,
            )

            def minibatch_fn(mb_carry: StepCarry, mb_data: tuple):
                out_carry = jax.lax.cond(
                    mb_carry.stopped,
                    lambda c: c,
                    lambda c: _step_minibatch(c, mb_data, hp),
                    mb_carry,
                )
                return out_carry, None

            c_shuffled = StepCarry(c.state, c_rng, c.stopped, c.stats)
            final_mb_carry, _ = jax.lax.scan(minibatch_fn, c_shuffled, shuffled_mbs)
            return final_mb_carry

        return jax.lax.cond(carry.stopped, lambda c: c, run_epoch, carry), None

    final_carry, _ = jax.lax.scan(
        epoch_fn, init_carry, None, length=hp.learning_steps
    )

    stats = final_carry.stats
    steps = stats.steps
    return steps, LearningOutput(
        state=final_carry.state,
        actor_loss=stats.actor_loss_sum / steps,
        critic_loss=stats.critic_loss_sum / steps,
        diagnostics={
            **{key: value / steps for key, value in stats.aux_sums.items()},
            "kl_early_stops": stats.early_stops,
        },
        extra_state=(stats.last_approx_kl, stats.last_clip_frac),
    )


class PPO(Agent):
    """Proximal Policy Optimization agent.

    Reference:
        Schulman et al., 2017: https://arxiv.org/abs/1707.06347
    """

    hyperparams_cls = PPOHyperparams
    derived_hyperparams = frozenset({"steps_between_updates"})
    freeze_obs_norm_per_chunk = True
    # `max_grad_norm` is one norm over the actor and the critic together, as the
    # reference has it; `_clip_jointly` applies it and the optimizers carry none.
    joint_grad_clip = True

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
        hyperparams: PPOHyperparams = None,
        actor_optimizer_config: dict = None,
        critic_optimizer_config: dict = None,
    ):
        self.hp = build_hyperparams(type(self).hyperparams_cls, hyperparams)

        actor = hydra.utils.instantiate(actor_config)
        critic = hydra.utils.instantiate(critic_config)

        prototype = transition_prototype(
            env_obs_size, env_action_size, on_policy=True
        )
        self.replay = ReplayManager(
            hydra.utils.instantiate(memory_config), prototype, time_axis=True,
        )
        self.add_sequence_length = memory_config.add_sequence_length
        self.max_length_time_axis = memory_config.max_length_time_axis
        self.batch_size = memory_config.add_batch_size
        self.sample_sequence_length = int(memory_config.sample_sequence_length)
        self.hp = dataclasses.replace(
            self.hp,
            steps_between_updates=self.sample_sequence_length * int(self.batch_size),
        )

        # A minibatch is a draw from the rollout's TRANSITIONS, not from its
        # trajectories, and GAE spends the last step of every trajectory on the
        # bootstrap — so the pool is `envs * (T - 1)` wide, not `envs`.
        self.rollout_transitions = int(self.batch_size) * (
            self.sample_sequence_length - 1
        )
        if self.rollout_transitions % self.hp.num_minibatches != 0:
            raise ValueError(
                f"num_minibatches ({self.hp.num_minibatches}) must divide the "
                f"rollout's {self.rollout_transitions} transitions "
                f"(add_batch_size {self.batch_size} x sample_sequence_length - 1"
                f" {self.sample_sequence_length - 1})"
            )
        self.minibatch_size = self.rollout_transitions // self.hp.num_minibatches

        buffer_state = self.replay.init()

        self._init_train_state(
            actor,
            critic,
            buffer_state,
            actor_optimizer_config=actor_optimizer_config,
            critic_optimizer_config=critic_optimizer_config,
            target_actor=False,
            target_critic=False,
        )

        self.adv_scale = AdvantageScale()
        self.last_approx_kl = 0.0
        self.last_clip_frac = 0.0
        self.action_low = action_low
        self.action_high = action_high
        self._obs_stats_snapshot = None
        self._obs_norm = None

        print("PPO agent initialized.")

    def select_action(
        self,
        observation: jnp.ndarray,
        key: jax.Array = None,
        evaluate: bool = False,
        *,
        actor: nnx.Module = None,
        obs_stats=None,
        noise_module: nnx.Module = None,
        critic: nnx.Module = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
        """Selects an action and records behavior policy log probabilities and value estimates."""
        del noise_module
        if self.hp.normalize_observations:
            stats = self._frozen_obs_stats() if obs_stats is None else obs_stats
            mean, std = obs_mean_std(stats, self.hp.obs_norm_eps)
            observation = normalize_obs(
                observation, mean, std, self.hp.obs_norm_clip
            )

        action, noise, pre_action, log_probs, value = stochastic_step_fn(
            self.state.actor if actor is None else actor,
            observation,
            evaluate,
            key,
            critic_model=self.state.critic if critic is None else critic,
        )
        return (
            scale_to_env(action, self.action_low, self.action_high),
            noise,
            {
                "log_probs": log_probs,
                "value": value.squeeze(-1),
                "pre_action": pre_action,
            },
        )

    def freeze_acting_norm(self):
        """Pins normalization statistics for the current rollout chunk."""
        return self._frozen_obs_stats()

    def _frozen_obs_stats(self):
        """Returns the cached snapshot of observation statistics for the current rollout."""
        if self._obs_stats_snapshot is None:
            self._obs_stats_snapshot = jax.tree.map(
                jnp.copy, self.state.obs_stats
            )
        return self._obs_stats_snapshot

    def _frozen_obs_norm(self):
        """Computes and caches mean and standard deviation arrays from the frozen snapshot."""
        if self._obs_norm is None:
            self._obs_norm = obs_mean_std(
                self._frozen_obs_stats(), self.hp.obs_norm_eps
            )
        return self._obs_norm

    def due_for_update(self, steps: int) -> bool:
        """Determines if the environment step count has crossed a rollout update boundary."""
        between = int(self.hp.steps_between_updates)
        if steps < between:
            return False
        boundary = (steps // between) * between
        if boundary <= self._last_update_boundary:
            return False
        self._last_update_boundary = boundary
        return True

    def _compile_and_run(self, agent_rng: jax.Array, target_steps: int):
        """Trains on the pending queue rollout using PPO loss objectives."""
        del target_steps
        obs_mean, obs_std = self._frozen_obs_norm()

        (
            self.state,
            self.adv_scale,
            norm_obs,
            pre_actions,
            old_log_probs,
            returns_t,
            adv_t,
            rollout_stats,
        ) = _prepare_rollout(
            self.state,
            self.adv_scale,
            hp=self.hp,
            replay_get_fn=self.replay.sample,
            obs_mean=obs_mean,
            obs_std=obs_std,
        )

        self._obs_stats_snapshot = None
        self._obs_norm = None

        steps_taken, output = _ppo_grad_steps(
            self.state,
            agent_rng,
            norm_obs,
            pre_actions,
            old_log_probs,
            returns_t,
            adv_t,
            self.hp,
            self.minibatch_size,
        )
        # The one sync on the learning path, and unavoidable: the trust region
        # decided this count on device, and every host-side consumer of it —
        # the trainer's running total, the diagnostics weighting — is an int.
        return steps_taken.item(), dataclasses.replace(
            output, diagnostics={**output.diagnostics, **rollout_stats}
        )

    def _checkpoint_modules(self) -> dict:
        """Returns extra stateful modules outside `TrainState` to checkpoint."""
        return {"adv_scale": self.adv_scale}

    def _apply_extra_state(self, extra_state) -> None:
        self.last_approx_kl, self.last_clip_frac = extra_state

    def pop_diagnostics(self, env_steps: int = 0) -> dict:
        """Flushes diagnostics and appends per-rollout work statistics."""
        passes = self.diagnostics.pending_learning_passes
        steps = self.diagnostics.pending_steps

        out = super().pop_diagnostics(env_steps)
        if out:
            per_rollout = float(steps) / passes
            out["ppo/steps_per_rollout"] = per_rollout
            out["ppo/epochs_per_rollout"] = per_rollout / self.hp.num_minibatches
        return out

    def _export_hyperparams(self) -> dict:
        params = super()._export_hyperparams()
        params.update(
            {
                "target_kl": -1.0 if self.hp.target_kl is None else self.hp.target_kl,
                "minibatch_size": int(self.minibatch_size),
                "memory_capacity": int(self.max_length_time_axis),
                "memory_batch_size": int(self.batch_size),
            }
        )
        return params