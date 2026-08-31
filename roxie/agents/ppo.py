import functools

import hydra
import jax
import jax.numpy as jnp
from flax import nnx

from roxie.agents.agent import Agent, TrainState
from roxie.agents.utils import Transition, network_rngs, transition_prototype
from roxie.losses.actor_losses import ppo_loss_fn
from roxie.losses.critic_losses import ppo_critic_loss_fn


def _compute_gae(rewards, values, termination, truncation, gamma, gae_lambda):
    """GAE for one trajectory that distinguishes termination from truncation.

    ``rewards``/``termination``/``truncation`` have length T-1; ``values`` has
    length T, with ``values[-1]`` the bootstrap. A termination zeroes the
    next-state bootstrap; a truncation is not a terminal, so its delta is zeroed
    and the recursion stops at the cut. Mirrors Brax's ``compute_gae``.
    """
    v_t = values[:-1]
    v_tp1 = values[1:]
    cont = 1.0 - termination           # value-bootstrap mask
    trunc_mask = 1.0 - truncation      # drop truncated step + stop recursion
    deltas = (rewards + gamma * cont * v_tp1 - v_t) * trunc_mask

    def scan_fn(acc, x):
        delta, cont_t, trunc_mask_t = x
        acc = delta + gamma * gae_lambda * cont_t * trunc_mask_t * acc
        return acc, acc

    _, adv = jax.lax.scan(
        scan_fn, jnp.zeros(()), (deltas, cont, trunc_mask), reverse=True
    )
    return adv


# Split out of the gradient step so the same rollout can be trained on
# `learning_steps` times; with a single pass the ratio is 1 and the clip never
# binds.
@functools.partial(
    nnx.jit,
    static_argnames=(
        "gamma",
        "gae_lambda",
        "replay_get_fn",
        "normalize",
    ),
)
def _prepare_rollout(
    state: TrainState,
    gamma: float,
    gae_lambda: float,
    replay_get_fn,
    obs_clip: float,
    normalize: bool,
    obs_mean: jnp.ndarray,
    obs_std: jnp.ndarray,
):
    """Dequeue one rollout and return the tensors every epoch trains on.

    Sampling a flashbax trajectory *queue* advances ``read_index``, so this call
    removes the transitions; they survive only in the arrays returned here.
    """
    state.buffer_state, data  = replay_get_fn(state.buffer_state)
    data = getattr(data, "experience", data)

    # All leaves are (NUM_ENVS, BATCH_SIZE, ...).
    re_packed_samples = {
        "observations": data.observation,
        "actions": data.action,
        "log_probs": data.log_probs,
        "rewards": data.reward,
        "values": data.value,
        "terminations": data.terminal,
        "truncations": data.truncation,
    }

    # `obs_mean`/`obs_std` are the snapshot the behaviour policy ran under: live
    # stats would make the clip and KL early stop fire on normalization drift.
    norm_obs = (
        Agent.normalize_obs(re_packed_samples["observations"], obs_mean, obs_std, obs_clip)
        if normalize
        else re_packed_samples["observations"]
    )

    term = re_packed_samples["terminations"].astype(jnp.float32)
    trunc = re_packed_samples["truncations"].astype(jnp.float32)
    gae_fn = jax.vmap(
        lambda r, v, te, tr: _compute_gae(r, v, te, tr, gamma, gae_lambda),
        in_axes=(0, 0, 0, 0),  # batch over envs
    )
    adv_t = gae_fn(
        re_packed_samples["rewards"][:, :-1],
        re_packed_samples["values"],
        term[:, :-1],
        trunc[:, :-1],
    )

    # Value target from the RAW advantage, on the reward's natural scale: a
    # standardized target would ask the critic to fit unit-variance noise.
    returns_t = adv_t + re_packed_samples["values"][:, :-1]

    # ...and only now standardize, for the actor alone.
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

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
        norm_obs,
        re_packed_samples["actions"],
        re_packed_samples["log_probs"],
        returns_t,
        adv_t,
    )


@jax.jit
def _shuffle_envs(perm: jnp.ndarray, *arrays: jnp.ndarray):
    """Reorder a rollout's leading (env) axis by ``perm``.

    Minibatches cut the env axis only, never time: GAE, the ratio and the value
    target are per-trajectory, so a trajectory has to stay whole inside one
    minibatch. Applied once per epoch, leaving each minibatch a static slice.
    """
    return tuple(arr[perm] for arr in arrays)


@functools.partial(
    nnx.jit,
    static_argnames=(
        "clip_eps",
        "entropy_coef",
    ),
)
def _grad_step(
    state: TrainState,
    key: jax.random.PRNGKey,
    norm_obs: jnp.ndarray,
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    returns_t: jnp.ndarray,
    adv_t: jnp.ndarray,
    clip_eps: float,
    entropy_coef: float,
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
):
    """One gradient step for the PPO agent on an already prepared rollout.

    Called ``learning_steps`` times per rollout. ``old_log_probs`` stays pinned
    to the behaviour policy across those calls, so the ratio drifts away from 1.
    """
    # Aux diagnostics are evaluated at the parameters entering this step, so they
    # describe the drift accumulated by the passes so far.
    (actor_loss, (approx_kl, clip_frac)), actor_grads = nnx.value_and_grad(
        ppo_loss_fn, has_aux=True
    )(
            actor_model=state.actor,
            observations=norm_obs,
            actions_buf=actions,
            old_log_probs=old_log_probs,
            action_low=action_low,
            action_high=action_high,
            advantages=adv_t,
            clip_epsilon=clip_eps,
            entropy_coef=entropy_coef,
            key=key
        )
    state.actor_optimizer.update(state.actor, actor_grads)

    critic_loss, critic_grads = nnx.value_and_grad(ppo_critic_loss_fn)(
            state.critic,
            observations=norm_obs,
            returns=returns_t,
        )
    state.critic_optimizer.update(state.critic, critic_grads)

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
        approx_kl,
        clip_frac,
    )


class PPO(Agent):
    """Proximal Policy Optimization agent."""

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
        seed: int = 0,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        target_kl: float | None = 0.02,
        learning_steps: int = 5,
        num_minibatches: int = 1,
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
            in_features=env_obs_size,
            rngs=network_rngs(self.seed, offset=2),
        )

        print("env_obs_size:", env_obs_size)
        print("env_action_size:", env_action_size)
        # `on_policy` adds the behaviour log-prob and value estimate stored at
        # acting time, which the ratio and GAE are computed against.
        prototype = transition_prototype(
            env_obs_size, env_action_size, on_policy=True
        )
        replay = hydra.utils.instantiate(memory_config)
        self.add_sequence_length = memory_config.add_sequence_length
        self.max_length_time_axis = memory_config.max_length_time_axis
        self.batch_size = memory_config.add_batch_size

        # An uneven last minibatch would have a different shape and retrigger a
        # full XLA compile of `_grad_step` on every rollout.
        if self.batch_size % num_minibatches != 0:
            raise ValueError(
                f"num_minibatches ({num_minibatches}) must divide the number of "
                f"envs / memory.add_batch_size ({self.batch_size})"
            )
        self.num_minibatches = int(num_minibatches)
        self.minibatch_size = self.batch_size // self.num_minibatches

        buffer_state = replay.init(prototype)

        # Without donation flashbax's `add` copies the whole queue on every env
        # step. Safe: the input dies as `add` reassigns `self.state.buffer_state`.
        self._jit_replay_add = jax.jit(replay.add, donate_argnums=(0,))

        # No target networks: PPO is on-policy and never bootstraps off a frozen
        # copy — the trust region does the job targets do elsewhere.
        self._init_train_state(
            actor,
            critic,
            buffer_state,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            max_grad_norm=max_grad_norm,
            actor_optimizer_config=actor_optimizer_config,
            critic_optimizer_config=critic_optimizer_config,
            target_actor=False,
            target_critic=False,
        )

        # `static_argnames` on the jitted kernels: changing any of these mid-run
        # retriggers an XLA compile.
        self.gamma = gamma
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.entropy_coef = float(entropy_coef)
        # Checked on the host rather than baked into the jit: the early stop
        # breaks a Python loop, costing one device sync per gradient pass.
        self.target_kl = None if target_kl is None else float(target_kl)

        self.last_approx_kl = 0.0
        self.last_clip_frac = 0.0
        self._kl_sum = 0.0
        self._clip_frac_sum = 0.0
        self._kl_iters = 0
        self._kl_early_stops = 0
        self._rollouts = 0
        self.action_low = action_low
        self.action_high = action_high
        self.replay = replay
        self.learning_steps = learning_steps
        self.normalize_observations = normalize_observations
        self.obs_clip = float(obs_norm_clip)
        self.obs_eps = float(obs_norm_eps)
        # Normalization is part of the policy: `step()` and `_prepare_rollout`
        # need byte-identical mean/std for a whole rollout or the stored
        # log-probs stop matching. `None` means "stale, recompute"; `update()`
        # invalidates it once a rollout is consumed.
        self._obs_norm = None

        print("PPO agent initialized.")

    def step(
        self,
        observation: jnp.ndarray,
        evaluate: bool = False,
        key: jax.random.PRNGKey = None,
    ):
        """Selects an action by calling the pure, JIT-compiled step function."""
        if self.normalize_observations:
            mean, std = self._frozen_obs_norm()
            observation = Agent.normalize_obs(observation, mean, std, self.obs_clip)

        action, self.last_log_prob, _ = Agent.stochastic_step_fn(
            self.state.actor,
            observation,
            evaluate,
            key,
        )

        self.last_values = self.state.critic(observation)
        self.last_action = Agent.scale_to_env(action, self.action_low, self.action_high)

        return self.last_action

    def _frozen_obs_norm(self):
        """Mean/std the current rollout is pinned to, recomputing if stale.

        Kept as device arrays so acting never syncs to host just to normalize.
        `obs_stats` keeps accumulating underneath and is only read at rollout
        boundaries, so the statistics lag by one rollout.
        """
        if self._obs_norm is None:
            self._obs_norm = Agent.obs_mean_std(self.state.obs_stats, self.obs_eps)
        return self._obs_norm

    def add(self, prev_obs, timestep):
        # A length-1 time axis is inserted to match the trajectory buffer's
        # (NUM_ENVS, TIME, ...) layout.
        experiences = Transition(
            observation=prev_obs[:, None, :],
            action=self.last_action[:, None, :],
            reward=timestep.reward[:, None],
            # Stored separately rather than as one `done`: GAE bootstraps the
            # value at a truncation but zeroes it at a termination.
            terminal=timestep.terminated[:, None],
            log_probs=self.last_log_prob[:, None],
            value=self.last_values,
            truncation=timestep.truncated[:, None],
        )
        self.state.buffer_state = self._jit_replay_add(
            self.state.buffer_state, experiences
        )

        if self.normalize_observations:
            obs_batch = jnp.concatenate([prev_obs, timestep.obs], axis=0)
            self.state.obs_stats = Agent.update_obs_stats(
                self.state.obs_stats, obs_batch
            )

    def update(self, steps, agent_rng):
        gradient_steps = 0
        actor_loss_sum, critic_loss_sum = 0.0, 0.0

        # Runs once a full rollout is queued.
        while self.replay.can_sample(self.state.buffer_state):
            # Read before the dequeue so `norm_obs` reproduces what `step()` fed
            # the actor and the ratio is 1 on the first pass.
            obs_mean, obs_std = self._frozen_obs_norm()

            (
                self.state,
                norm_obs,
                actions,
                old_log_probs,
                returns_t,
                adv_t,
            ) = _prepare_rollout(
                state=self.state,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                replay_get_fn=self.replay.sample,
                obs_clip=self.obs_clip,
                normalize=self.normalize_observations,
                obs_mean=obs_mean,
                obs_std=obs_std,
            )

            # Rollout consumed: the next one may safely act under fresher stats.
            self._obs_norm = None

            self._rollouts += 1
            mb_size = self.minibatch_size
            stop_epochs = False
            for _ in range(self.learning_steps):
                agent_rng, perm_key = jax.random.split(agent_rng, 2)
                perm = jax.random.permutation(perm_key, self.batch_size)
                mb_obs, mb_act, mb_logp, mb_ret, mb_adv = _shuffle_envs(
                    perm, norm_obs, actions, old_log_probs, returns_t, adv_t
                )

                for m in range(self.num_minibatches):
                    sl = slice(m * mb_size, (m + 1) * mb_size)
                    agent_rng, key = jax.random.split(agent_rng, 2)

                    self.state, actor_loss, critic_loss, approx_kl, clip_frac = _grad_step(
                        state=self.state,
                        key=key,
                        norm_obs=mb_obs[sl],
                        actions=mb_act[sl],
                        old_log_probs=mb_logp[sl],
                        returns_t=mb_ret[sl],
                        adv_t=mb_adv[sl],
                        clip_eps=self.clip_eps,
                        entropy_coef=self.entropy_coef,
                        action_low=self.action_low,
                        action_high=self.action_high,
                    )
                    gradient_steps += 1

                    # The `float()` costs no extra sync: `approx_kl` below
                    # already blocks on the same `_grad_step` output.
                    actor_loss_sum += float(actor_loss)
                    critic_loss_sum += float(critic_loss)

                    self.last_approx_kl = float(approx_kl)
                    self.last_clip_frac = float(clip_frac)
                    self._kl_sum += self.last_approx_kl
                    self._clip_frac_sum += self.last_clip_frac
                    self._kl_iters += 1

                    # `approx_kl` is total drift from the behaviour policy, not
                    # the last step's increment, and is 0 on the first pass so at
                    # least one always runs. The budget is per rollout: breaking
                    # abandons the rest of it (as in CleanRL).
                    if self.target_kl is not None and self.last_approx_kl > self.target_kl:
                        self._kl_early_stops += 1
                        stop_epochs = True
                        break

                if stop_epochs:
                    break

        if gradient_steps == 0:
            return 0, 0.0, 0.0
        return (
            gradient_steps,
            actor_loss_sum / gradient_steps,
            critic_loss_sum / gradient_steps,
        )

    def pop_diagnostics(self) -> dict:
        """Return the epoch's trust-region metrics and reset the accumulators.

        Optional agent hook: the trainer calls it via ``getattr``. Returns ``{}``
        when no gradient pass ran, so nothing is logged rather than a misleading
        zero. `ppo/clip_frac` near 1 means the surrogate is saturated.
        """
        if self._kl_iters == 0:
            return {}
        out = {
            "ppo/approx_kl": self._kl_sum / self._kl_iters,
            "ppo/clip_frac": self._clip_frac_sum / self._kl_iters,
            "ppo/kl_early_stops": float(self._kl_early_stops),
            # How much of the possible learning_steps * num_minibatches budget
            # the trust region allowed.
            "ppo/steps_per_rollout": self._kl_iters / max(self._rollouts, 1),
            "ppo/epochs_per_rollout": (
                self._kl_iters / self.num_minibatches / max(self._rollouts, 1)
            ),
        }
        self._kl_sum = 0.0
        self._clip_frac_sum = 0.0
        self._kl_iters = 0
        self._kl_early_stops = 0
        self._rollouts = 0
        return out

    def _export_hyperparams(self) -> dict:
        # Not `_replay_hyperparams`: on-policy, so no tau and no update schedule,
        # and the queue's size comes from its own pair of config keys.
        params = super()._export_hyperparams()
        params.update(
            {
                "gae_lambda": float(self.gae_lambda),
                "clip_eps": float(self.clip_eps),
                "entropy_coef": float(self.entropy_coef),
                # -1 is the on-disk spelling of `None`: the block has to stay
                # plain scalars.
                "target_kl": -1.0 if self.target_kl is None else float(self.target_kl),
                "num_minibatches": int(self.num_minibatches),
                "minibatch_size": int(self.minibatch_size),
                "memory_capacity": int(self.max_length_time_axis),
                "memory_batch_size": int(self.batch_size),
            }
        )
        return params
