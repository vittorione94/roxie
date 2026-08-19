import functools

import hydra
import jax
import jax.numpy as jnp
from flax import nnx

from roxie.agents.agent import Agent, TrainState
from roxie.agents.utils import (
    Transition,
    build_optimizer,
    network_rngs,
    serialize_bound,
)
from roxie.losses.actor_losses import ppo_loss_fn
from roxie.losses.critic_losses import ppo_critic_loss_fn


def _compute_gae(rewards, values, termination, truncation, gamma, gae_lambda):
    """GAE for one trajectory that distinguishes termination from truncation.

    ``rewards``/``termination``/``truncation`` have length T-1 (steps 0..T-2);
    ``values`` has length T (V(s_0..s_{T-1}), with ``values[-1]`` the bootstrap).

    A genuine termination zeroes the next-state bootstrap via ``(1 - termination)``
    (the episode truly ended). A truncation (time-limit / clip-end) is NOT a
    terminal: its delta is zeroed and the backward recursion is stopped at the
    cut, so the next episode's return never leaks across the boundary and no
    false ``V(s')=0`` target is injected. Mirrors Brax's ``compute_gae``. The
    bootstrap and the step before each boundary use ``values[t+1]``, which is the
    true continuation everywhere except at a boundary step -- where it is the
    reset state's value but is always masked out (by ``trunc_mask`` for a
    truncation, by ``1 - termination`` for a termination).
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


# Drains one on-policy rollout from the queue and precomputes everything that
# stays fixed for the whole update (normalized obs, old log-probs, GAE returns
# and advantages). Split out of the gradient step so the same rollout can be
# trained on `learning_steps` times: PPO's clipped surrogate exists precisely to
# license several passes over one batch. With a single pass `logp_new` equals
# `old_log_probs` exactly, the ratio is identically 1, the clip never binds and
# the objective degenerates to the vanilla policy gradient (i.e. A2C).
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
    replay_get_fn,  # Function to get the on-policy data
    obs_clip: float,
    normalize: bool,
    obs_mean: jnp.ndarray,
    obs_std: jnp.ndarray,
):
    """Dequeue one rollout and return the tensors every epoch trains on.

    Sampling from a flashbax trajectory *queue* advances ``read_index``, so this
    call is what removes the transitions from the buffer; they survive only in
    the arrays returned here and are dropped once the caller's loop is done.
    """
    # 1. Get the most recent on-policy data from the buffer
    # This function is expected to be `replay.get_recent_window`
    state.buffer_state, data  = replay_get_fn(state.buffer_state)

    # everything is wrapped in an experience attribute
    data = getattr(data, "experience", data)

    re_packed_samples = {
        "observations": data.observation, # (NUM_ENVS, BATCH_SIZE, OBS_SIZE)
        "actions": data.action, # (NUM_ENVS, BATCH_SIZE, ACT_SIZE)
        "log_probs": data.log_probs, # (NUM_ENVS, BATCH_SIZE)
        "rewards": data.reward, # (NUM_ENVS, BATCH_SIZE)
        "values": data.value, # (NUM_ENVS, BATCH_SIZE)
        "terminations": data.terminal,   # genuine termination only (NUM_ENVS, BATCH_SIZE)
        "truncations": data.truncation,  # time-limit / clip-end truncation
    }

    # `obs_mean`/`obs_std` are the snapshot the BEHAVIOUR policy ran under (see
    # `PPO._obs_norm`), not the live running stats. Re-deriving them from
    # `state.obs_stats` here would normalize with statistics that kept moving
    # for the whole rollout, so re-evaluating step 0 would not reproduce the
    # log-prob stored with it: the ratio would already differ from 1 before any
    # gradient step, and the clip / KL early stop would fire on normalization
    # drift instead of policy drift.
    norm_obs = (
        Agent.normalize_obs(re_packed_samples["observations"], obs_mean, obs_std, obs_clip)
        if normalize
        else re_packed_samples["observations"]
    )

    # 4. Generalized advantage estimation, distinguishing termination from
    # truncation (see _compute_gae): a terminal zeroes the value bootstrap; a
    # truncation merely cuts the trajectory (drop the step, stop the recursion)
    # rather than being treated as a hard terminal that collapses the target.
    term = re_packed_samples["terminations"].astype(jnp.float32)
    trunc = re_packed_samples["truncations"].astype(jnp.float32)
    gae_fn = jax.vmap(
        lambda r, v, te, tr: _compute_gae(r, v, te, tr, gamma, gae_lambda),
        in_axes=(0, 0, 0, 0),  # Batch over envs (first dimension)
    )
    adv_t = gae_fn(
        re_packed_samples["rewards"][:, :-1],
        re_packed_samples["values"],
        term[:, :-1],
        trunc[:, :-1],
    )

    # 4b. Value target FIRST, from the RAW advantage: returns = A_raw + V_old is
    # the GAE estimate of the true return and lives on the reward's natural
    # scale. The standardization below is a policy-gradient device -- it only
    # rescales the step direction and leaves the gradient's sign structure
    # intact -- but it is destructive for a regression target: feeding the
    # normalized advantage to the critic asks it to fit `V_old + unit-variance
    # noise`, whose best achievable MSE is var(A_norm) = 1.0 no matter how good
    # the critic is. Keep the two quantities separate.
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

    Minibatches cut the ENV axis only, never time: GAE, the ratio and the value
    target are all per-trajectory with a ``[:, :-1]`` slice (see ``ppo_loss_fn``
    / ``ppo_critic_loss_fn``), so a trajectory has to stay whole inside one
    minibatch. Envs are i.i.d. rollouts of the same policy, so shuffling across
    them is where the decorrelation actually lives.

    Applied once per epoch so each minibatch is then a cheap static slice.
    """
    return tuple(arr[perm] for arr in arrays)


# This is the core computational kernel that will be JIT-compiled.
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
    to the behaviour policy across those calls, so the ratio drifts away from 1
    and the clipped surrogate actually binds.
    This function is JIT-compiled for performance.
    """
    # Actor update. `has_aux` carries the trust-region diagnostics out of the
    # loss without a second forward pass; they are measured at the CURRENT
    # parameters, i.e. they describe the drift accumulated by the passes so far.
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

    # Critic update
    critic_loss, critic_grads = nnx.value_and_grad(ppo_critic_loss_fn)(
            state.critic,
            observations=norm_obs,
            returns=returns_t,
        )
    state.critic_optimizer.update(state.critic, critic_grads)

    # 5. Return the new, updated state object
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

        # Instantiate actor
        actor = hydra.utils.instantiate(
            actor_config,
            in_features=env_obs_size,
            action_dim=env_action_size,
            rngs=network_rngs(self.seed, offset=0),
        )

        # Instantiate critic
        critic = hydra.utils.instantiate(
            critic_config,
            in_features=env_obs_size,
            rngs=network_rngs(self.seed, offset=2),
        )

        # Instantiate replay buffer
        print("env_obs_size:", env_obs_size)
        print("env_action_size:", env_action_size)
        prototype = Transition(
            observation=jnp.zeros(env_obs_size, dtype=jnp.float32),
            action=jnp.zeros(env_action_size, dtype=jnp.float32),
            reward=jnp.zeros((), dtype=jnp.float32),
            terminal=jnp.zeros((), dtype=jnp.bool_),
            log_probs=jnp.zeros((), dtype=jnp.float32),
            value=jnp.zeros((), dtype=jnp.float32),
            truncation=jnp.zeros((), dtype=jnp.bool_),
        )
        replay = hydra.utils.instantiate(memory_config)
        self.add_sequence_length = memory_config.add_sequence_length
        self.max_length_time_axis = memory_config.max_length_time_axis
        self.batch_size = memory_config.add_batch_size

        # Minibatches split the env axis, so the split must be exact: an uneven
        # last minibatch would have a different shape and retrigger a full XLA
        # compile of `_grad_step` on every rollout.
        if self.batch_size % num_minibatches != 0:
            raise ValueError(
                f"num_minibatches ({num_minibatches}) must divide the number of "
                f"envs / memory.add_batch_size ({self.batch_size})"
            )
        self.num_minibatches = int(num_minibatches)
        self.minibatch_size = self.batch_size // self.num_minibatches

        buffer_state = replay.init(prototype)

        # flashbax's `add` is a pure (queue_state, batch) -> queue_state
        # function, but calling it eagerly runs it as a standalone XLA program
        # with the queue as a live input, so the whole queue is COPIED on every
        # env step — at parallel_envs=1000 and obs 1069 that is a 0.29 GB
        # read+write per step (2.3 GB before the capacity fix, which is what
        # made `agent.add` the single most expensive item in the CPU training
        # loop: 43 ms/step against 41 ms for the physics). Jitting with donation
        # turns it into an in-place scatter. Donation is safe for the same
        # reason it is in DDPG's `_grad_steps`: the input is dead the moment
        # `add` reassigns `self.state.buffer_state` below.
        self._jit_replay_add = jax.jit(replay.add, donate_argnums=(0,))

        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate = actor_learning_rate
        self.max_grad_norm = max_grad_norm

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

        # Init observation stats from buffer state's observation shape
        obs_shape = buffer_state.experience.observation.shape[
            -1
        ]  # Exclude batch dimension
        obs_stats = Agent.init_obs_stats(obs_shape)

        self.state = TrainState(
            actor=actor,
            critic=critic,
            target_actor=None,
            target_critic=None,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            buffer_state=buffer_state,
            obs_stats=obs_stats,
        )

        # Store hyperparameters. These four are `static_argnames` on the jitted
        # kernels, so they are baked into the compiled program -- floats from the
        # config are fine, but they must not change during a run or every change
        # retriggers an XLA compile.
        self.gamma = gamma
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.entropy_coef = float(entropy_coef)
        # `target_kl` is a plain Python float (or None to disable) and is checked
        # on the host, NOT baked into the jit -- the early stop breaks a Python
        # loop, so reading it costs one device sync per gradient pass. That is
        # per-rollout traffic, not per-env-step, so it stays off the hot path.
        self.target_kl = None if target_kl is None else float(target_kl)

        # Epoch accumulators for the trust-region metrics (drained by the trainer
        # via `pop_diagnostics`), plus the most recent values.
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
        # Observation normalization is part of the policy, so it has to be a
        # FIXED function for the whole rollout: `step()` and `_prepare_rollout`
        # must use byte-identical mean/std or the stored log-probs stop matching
        # a re-evaluation of the same states. `None` means "stale, recompute
        # from obs_stats on next use"; `update()` invalidates it once a rollout
        # has been consumed, which is the only safe refresh point (and also
        # makes a checkpoint-restored `obs_stats` pick itself up).
        self._obs_norm = None

        print("PPO agent initialized.")

    def step(
        self,
        observation: jnp.ndarray,
        evaluate: bool = False,
        key: jax.random.PRNGKey = None,
    ):
        """
        Selects an action by calling the pure, JIT-compiled step function.
        """
        # Call the standalone function, passing in the required parts from the agent's state.
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

        Kept as device arrays: the snapshot is taken with a pure device op, so
        acting never syncs to host just to normalize. The running `obs_stats`
        keep accumulating underneath -- they are only *read* at rollout
        boundaries, so the statistics still track the current policy's state
        distribution, just one rollout behind instead of mid-trajectory.
        """
        if self._obs_norm is None:
            self._obs_norm = Agent.obs_mean_std(self.state.obs_stats, self.obs_eps)
        return self._obs_norm

    def add(self, prev_states, states):
        # Add sequence dimension (length=1) to match trajectory buffer format
        experiences = Transition(
            observation=prev_states.obs[:, None, :],  # (NUM_ENVS,) -> (NUM_ENVS, 1, obs_dim)
            action=self.last_action[:, None, :],      # (NUM_ENVS, act_dim) -> (NUM_ENVS, 1, act_dim)
            reward=states.reward[:, None],            # (NUM_ENVS,) -> (NUM_ENVS, 1)
            # Store termination and truncation separately, NOT `done` (= either):
            # GAE bootstraps the value at a truncation but zeroes it at a true
            # termination (see _compute_gae). Folding them into one `done` would
            # treat a time-limit/clip-end cut as a hard terminal and bias returns.
            terminal=states.info["termination"][:, None],   # (NUM_ENVS,) -> (NUM_ENVS, 1)
            log_probs=self.last_log_prob[:, None],    # (NUM_ENVS,) -> (NUM_ENVS, 1)
            value=self.last_values,          # (NUM_ENVS, 1)
            truncation=states.info["truncation"][:, None],  # (NUM_ENVS,) -> (NUM_ENVS, 1)
        )
        # store in memory (jitted + donating — see _jit_replay_add in __init__)
        self.state.buffer_state = self._jit_replay_add(
            self.state.buffer_state, experiences
        )
        # print(self.state.buffer_state.experience.observation.shape) --> (NUM_ENVS, TIME, OBS_SPACE)

        # Update observation normalization stats with both current and next observations
        if self.normalize_observations:
            obs_batch = jnp.concatenate([prev_states.obs, states.obs], axis=0)
            self.state.obs_stats = Agent.update_obs_stats(
                self.state.obs_stats, obs_batch
            )

    def update(self, steps, agent_rng):
        # Losses are summed over every gradient pass actually executed and
        # averaged at the end. Returning the last minibatch's value instead
        # would make `loss/actor` / `loss/critic` a snapshot of one arbitrary
        # slice of one epoch -- and, when the KL early stop fires, specifically
        # the threshold-crossing batch -- rather than a summary of the update.
        gradient_steps = 0
        actor_loss_sum, critic_loss_sum = 0.0, 0.0

        # A full rollout is queued, so we can start learning
        while self.replay.can_sample(self.state.buffer_state):
            # Normalizer snapshot the rollout was ACTED under. Read before the
            # dequeue and passed in explicitly, so `norm_obs` reproduces exactly
            # what `step()` fed the actor and `old_log_probs` is a true
            # behaviour log-prob (ratio == 1 on the first pass).
            obs_mean, obs_std = self._frozen_obs_norm()

            # Dequeue the rollout ONCE (this is what drops it from the queue),
            # then take `learning_steps` gradient passes over those same arrays.
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
                replay_get_fn=self.replay.sample,  # Pass the sample method itself,
                obs_clip=self.obs_clip,
                normalize=self.normalize_observations,
                obs_mean=obs_mean,
                obs_std=obs_std,
            )

            # The rollout is now consumed, so the next one may safely act under
            # fresher statistics: drop the snapshot and let `_frozen_obs_norm`
            # rebuild it from everything `add()` has accumulated since.
            self._obs_norm = None

            self._rollouts += 1
            mb_size = self.minibatch_size
            stop_epochs = False
            for _ in range(self.learning_steps):
                # Fresh env permutation each epoch, applied once so the
                # minibatches below are cheap static slices.
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
                    # Count real steps only. This used to be an unconditional
                    # `+= learning_steps` outside the loop, which reported updates
                    # on every env step even when none ran -- inflating the logged
                    # gradient_steps and, because the trainer appends losses
                    # whenever gradient_steps > 0, averaging a placeholder 0 loss
                    # into every epoch that had no update.
                    gradient_steps += 1

                    # Accumulate this pass's losses. The `float()` costs no
                    # extra sync: `approx_kl` below already blocks on the same
                    # `_grad_step` output.
                    actor_loss_sum += float(actor_loss)
                    critic_loss_sum += float(critic_loss)

                    # Trust-region diagnostics for this step, exposed for the
                    # trainer to log (same opt-in getattr pattern as `last_noise`).
                    self.last_approx_kl = float(approx_kl)
                    self.last_clip_frac = float(clip_frac)
                    self._kl_sum += self.last_approx_kl
                    self._clip_frac_sum += self.last_clip_frac
                    self._kl_iters += 1

                    # KL early stop. `approx_kl` is measured at the parameters
                    # ENTERING this step, against the behaviour policy, so it is
                    # the TOTAL drift accumulated over this rollout so far -- not
                    # the increment from the last step. It is exactly 0 on the
                    # very first step (ratio == 1) and we always take at least
                    # one. Note the budget is per rollout, not per epoch: with
                    # minibatching the drift is spent across `learning_steps *
                    # num_minibatches` steps, so more minibatches means the budget
                    # is reached in fewer EPOCHS, each covering less data.
                    # Breaking abandons the rest of the rollout entirely, which is
                    # the CleanRL behaviour.
                    if self.target_kl is not None and self.last_approx_kl > self.target_kl:
                        self._kl_early_stops += 1
                        stop_epochs = True
                        break

                if stop_epochs:
                    break

        # The rollout was consumed from the queue by `_prepare_rollout` and the
        # local arrays go out of scope here: nothing is reused across updates.
        #
        # Mean over the passes actually executed (which may span several
        # rollouts if more than one was queued). With no pass, the trainer
        # gates on `gradient_steps > 0` and never reads the losses, so the 0.0
        # is never logged.
        if gradient_steps == 0:
            return 0, 0.0, 0.0
        return (
            gradient_steps,
            actor_loss_sum / gradient_steps,
            critic_loss_sum / gradient_steps,
        )

    def pop_diagnostics(self) -> dict:
        """Return the epoch's trust-region metrics and reset the accumulators.

        Optional agent hook: the trainer calls it via ``getattr`` so agents that
        do not define it simply contribute nothing. Returns ``{}`` when no
        gradient pass ran this epoch, so the trainer logs nothing rather than a
        misleading zero.

        `ppo/approx_kl` is the drift per gradient pass and `ppo/clip_frac` the
        share of the batch outside the clip range -- if clip_frac sits near 1 the
        surrogate is saturated and the update is no longer a policy gradient.
        `ppo/passes_per_rollout` shows what `learning_steps` actually achieved
        once the KL early stop is taken into account.
        """
        if self._kl_iters == 0:
            return {}
        out = {
            "ppo/approx_kl": self._kl_sum / self._kl_iters,
            "ppo/clip_frac": self._clip_frac_sum / self._kl_iters,
            "ppo/kl_early_stops": float(self._kl_early_stops),
            # Gradient steps actually taken per rollout, out of a possible
            # learning_steps * num_minibatches. Reads directly as "how much of
            # the configured budget the trust region allowed".
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
        # Keep this minimal and JSON-serializable
        return {
            "seed": int(self.seed),
            "gamma": float(self.gamma),
            "gae_lambda": float(self.gae_lambda),
            "clip_eps": float(self.clip_eps),
            "entropy_coef": float(self.entropy_coef),
            "target_kl": -1.0 if self.target_kl is None else float(self.target_kl),
            "actor_learning_rate": float(self.actor_learning_rate),
            "critic_learning_rate": float(self.critic_learning_rate),
            "max_grad_norm": float(self.max_grad_norm),
            "learning_steps": int(self.learning_steps),
            "num_minibatches": int(self.num_minibatches),
            "minibatch_size": int(self.minibatch_size),
            "memory_capacity": int(self.max_length_time_axis),
            "memory_batch_size": int(self.batch_size),
            "normalize_observations": bool(self.normalize_observations),
            "obs_norm_clip": float(self.obs_clip),
            "obs_norm_eps": float(self.obs_eps),
            # bounds can be scalar or arrays → use your helpers
            "action_low": serialize_bound(self.action_low),
            "action_high": serialize_bound(self.action_high),
        }
