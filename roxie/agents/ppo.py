import functools

import hydra
import jax
import jax.numpy as jnp
from flax import nnx

from roxie.agents.agent import Agent, TrainState
from roxie.agents.utils import (
    Transition,
    graph_jit,
    network_rngs,
    transition_prototype,
)
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


@functools.partial(nnx.jit, static_argnames=("evaluate",))
def _ppo_step_fn(actor_model, critic_model, observation, evaluate, key):
    """Everything acting produces, in ONE dispatch.

    The value estimate used to be a separate eager ``self.state.critic(obs)``
    call outside the jitted actor step, so every env step cost two dispatches
    (plus an un-jitted nnx module call) instead of one.

    Returns ``(action, log_probs, value, deviation_from_mode)``. `action` is
    pre-scaling, in [-1, 1] — that is the space its log-prob is taken in, and
    the ratio at update time has to match it. The last value feeds the trainer's
    `noise/per_joint_abs` panel; PPO explores from its own stochastic policy, so
    as in SAC/MPO it is the sample's distance from the mode.
    """
    distribution = actor_model(observation)
    try:
        mode = distribution.mean()
    except TypeError:
        # Some distrax versions expose mean as a property.
        mode = distribution.mean

    if evaluate:
        action = mode
        log_probs = distribution.log_prob(action)
    else:
        action, log_probs = distribution.sample_and_log_prob(seed=key)

    return action, log_probs, critic_model(observation), mode - action


# Split out of the gradient step so the same rollout can be trained on
# `learning_steps` times.
@functools.partial(
    graph_jit,
    static_argnames=(
        "gamma",
        "gae_lambda",
        "replay_get_fn",
        "normalize",
    ),
    # Read-only on the buffer side and called once per rollout.
    donate=False,
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

    # The snapshot the behaviour policy ran under: live stats would make the
    # clip and KL early stop fire on normalization drift.
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

    # From the RAW advantage: a standardized target would ask the critic to fit
    # unit-variance noise.
    returns_t = adv_t + re_packed_samples["values"][:, :-1]

    # Standardized for the actor alone.
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


# Not jitted on its own — called inside `_grad_steps` so a whole rollout's
# passes fuse into one compiled program.
def _grad_step(
    state: TrainState,
    key: jax.random.PRNGKey,
    norm_obs: jnp.ndarray,
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    returns_t: jnp.ndarray,
    adv_t: jnp.ndarray,
    *,
    clip_eps: float,
    entropy_coef: float,
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
):
    """One gradient step for the PPO agent on an already prepared rollout,
    MUTATING ``state`` in place.

    Runs ``learning_steps * num_minibatches`` times per rollout.
    ``old_log_probs`` stays pinned to the behaviour policy across those calls,
    so the ratio drifts away from 1.
    """
    # Aux diagnostics are evaluated at the parameters entering this step, so
    # they describe the drift accumulated by the passes so far.
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

    return actor_loss, critic_loss, approx_kl, clip_frac


# The whole rollout's passes as ONE host dispatch, with the `target_kl` early
# stop as a sticky flag in the scan carry rather than a Python `break` on a
# host-synced `approx_kl`. Exactly equivalent, not an approximation: the check
# happens after the update, `lax.cond` is a real branch outside vmap so skipped
# steps cost nothing, and the RNG splits live inside the taken branch so the key
# stream advances exactly as far as the Python loop advanced it.
@functools.partial(
    graph_jit,
    static_argnames=(
        "learning_steps", "num_minibatches", "minibatch_size",
        "clip_eps", "entropy_coef", "target_kl",
    ),
)
def _grad_steps(
    state: TrainState,
    key: jax.random.PRNGKey,
    norm_obs: jnp.ndarray,
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    returns_t: jnp.ndarray,
    adv_t: jnp.ndarray,
    learning_steps: int,
    num_minibatches: int,
    minibatch_size: int,
    clip_eps: float,
    entropy_coef: float,
    target_kl: float,
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
):
    """Run one prepared rollout's full `learning_steps` x `num_minibatches`
    schedule as a single compiled program, stopping early on `target_kl`.

    Returns `(state, key, gradient_steps, actor_loss_sum, critic_loss_sum,
    kl_sum, clip_frac_sum, early_stops, last_kl, last_clip_frac)` — sums rather
    than means so the caller can fold several rollouts together, and the
    advanced `key` so the stream threads on exactly as the Python loop's did.
    """
    graphdef, node_carry = nnx.split(state)
    rollout = (norm_obs, actions, old_log_probs, returns_t, adv_t)

    f0, i0 = jnp.zeros((), jnp.float32), jnp.zeros((), jnp.int32)
    # (steps, actor_sum, critic_sum, kl_sum, clip_sum, early_stops,
    #  last_kl, last_clip)
    init_stats = (i0, f0, f0, f0, f0, i0, f0, f0)

    def epoch(carry, _):
        def run(carry):
            node_carry, rng, stopped, stats = carry
            rng, perm_key = jax.random.split(rng)
            # Cuts the ENV axis only, never time: GAE, the ratio and the value
            # target are per-trajectory, so a trajectory must stay whole inside
            # one minibatch.
            perm = jax.random.permutation(perm_key, norm_obs.shape[0])
            shuffled = tuple(leaf[perm] for leaf in rollout)

            def minibatch(carry, m):
                def run_mb(carry):
                    node_carry, rng, _stopped, stats = carry
                    rng, step_key = jax.random.split(rng)
                    mb = tuple(
                        jax.lax.dynamic_slice_in_dim(
                            leaf, m * minibatch_size, minibatch_size, axis=0,
                        )
                        for leaf in shuffled
                    )
                    inner = nnx.merge(graphdef, node_carry)
                    actor_loss, critic_loss, approx_kl, clip_frac = _grad_step(
                        inner, step_key, *mb,
                        clip_eps=clip_eps, entropy_coef=entropy_coef,
                        action_low=action_low, action_high=action_high,
                    )
                    # The updates wrote through to `inner`, so re-splitting
                    # gives the post-step carry.
                    _, node_carry = nnx.split(inner)

                    steps, a_sum, c_sum, kl_sum, cf_sum, stops, _, _ = stats
                    # Total drift from the behaviour policy, not this step's
                    # increment, and 0 on the first pass so at least one always
                    # runs. Tripping abandons the rest of the rollout.
                    trip = (
                        jnp.zeros((), jnp.bool_) if target_kl is None
                        else approx_kl > target_kl
                    )
                    stats_out = (
                        steps + 1,
                        a_sum + actor_loss,
                        c_sum + critic_loss,
                        kl_sum + approx_kl,
                        cf_sum + clip_frac,
                        stops + trip.astype(jnp.int32),
                        approx_kl,
                        clip_frac,
                    )
                    return node_carry, rng, trip, stats_out

                return jax.lax.cond(carry[2], lambda c: c, run_mb, carry), None

            carry, _ = jax.lax.scan(
                minibatch, (node_carry, rng, stopped, stats),
                jnp.arange(num_minibatches),
            )
            return carry

        return jax.lax.cond(carry[2], lambda c: c, run, carry), None

    (node_carry, key, _stopped, stats), _ = jax.lax.scan(
        epoch,
        (node_carry, key, jnp.zeros((), jnp.bool_), init_stats),
        None,
        length=learning_steps,
    )
    return (nnx.merge(graphdef, node_carry), key, *stats)


class PPO(Agent):
    """Proximal Policy Optimization agent."""

    # The mean/std the behaviour policy ran under must stay byte-identical for
    # a whole rollout: `_prepare_rollout` re-normalizes the stored observations
    # to recompute the ratio, and drift there would make the clip and the KL
    # early stop fire on normalization noise rather than on policy change. So
    # the fused acting burst hoists the snapshot out of its scan instead of
    # reading the carry's live statistics.
    freeze_obs_norm_per_chunk = True

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
        # `on_policy` adds the behaviour log-prob and value estimate the ratio
        # and GAE are computed against.
        prototype = transition_prototype(
            env_obs_size, env_action_size, on_policy=True
        )
        replay = hydra.utils.instantiate(memory_config)
        self.add_sequence_length = memory_config.add_sequence_length
        self.max_length_time_axis = memory_config.max_length_time_axis
        self.batch_size = memory_config.add_batch_size
        # What `can_sample` waits for, NOT the queue's capacity.
        self.sample_sequence_length = int(memory_config.sample_sequence_length)
        # PPO gates its own updates on `can_sample`, but `Trainer._collect_steps`
        # sizes the fused acting chunk from this — without it PPO collects one
        # env step per dispatch.
        self.steps_between_updates = (
            self.sample_sequence_length * int(self.batch_size)
        )

        # An uneven last minibatch would retrigger a full XLA compile of
        # `_grad_step` on every rollout.
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

        # No target networks: the trust region does the job targets do elsewhere.
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
        self.target_kl = None if target_kl is None else float(target_kl)

        # The last rollout's trust-region readings, for a caller wanting them
        # between epochs; the epoch aggregates go through `record_diagnostics`.
        self.last_approx_kl = 0.0
        self.last_clip_frac = 0.0
        self.action_low = action_low
        self.action_high = action_high
        self.replay = replay
        self.learning_steps = learning_steps
        self.normalize_observations = normalize_observations
        self.obs_clip = float(obs_norm_clip)
        self.obs_eps = float(obs_norm_eps)
        # `step()` and `_prepare_rollout` need byte-identical mean/std for a
        # whole rollout or the stored log-probs stop matching. `None` means
        # "stale, recompute"; `update()` invalidates both once one is consumed.
        self._obs_stats_snapshot = None
        self._obs_norm = None

        print("PPO agent initialized.")

    def select_action(
        self,
        actor: nnx.Module,
        obs_stats,
        observation: jnp.ndarray,
        key: jax.random.PRNGKey,
        evaluate: bool = False,
        noise_module: nnx.Module = None,
        critic: nnx.Module = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
        """Pure action selection from an explicit actor, critic and obs stats.

        Taking all three as arguments rather than reading ``self.state`` is what
        lets the fused acting burst run this against a ``lax.scan`` carry, so a
        whole rollout of acting costs one host dispatch instead of one per env
        step. Returns ``(scaled_action, deviation_from_mode, extras)``.

        ``extras`` carries the behaviour log-prob and value estimate, which are
        properties of the policy AT ACTING TIME and cannot be recovered later:
        the ratio PPO clips is against exactly these. They reach the buffer as a
        value rather than via ``self.last_*`` because on the fused path acting
        and buffering are both inside one trace.

        ``obs_stats`` MUST be the rollout's frozen snapshot, not live statistics
        — see ``freeze_obs_norm_per_chunk``. ``noise_module`` is accepted and
        ignored: PPO explores from its own stochastic policy.
        """
        del noise_module
        if self.normalize_observations:
            mean, std = Agent.obs_mean_std(obs_stats, self.obs_eps)
            observation = Agent.normalize_obs(observation, mean, std, self.obs_clip)

        action, log_probs, value, noise = _ppo_step_fn(
            actor, self.state.critic if critic is None else critic,
            observation, evaluate, key,
        )
        return (
            Agent.scale_to_env(action, self.action_low, self.action_high),
            noise,
            # Pre-scaling: the log-prob is taken in [-1, 1], where the ratio at
            # update time is recomputed.
            {"log_probs": log_probs, "value": value},
        )

    def step(
        self,
        observation: jnp.ndarray,
        evaluate: bool = False,
        key: jax.random.PRNGKey = None,
    ):
        """Selects an action by calling the pure, JIT-compiled step function."""
        self.last_action, noise, self.last_extras = self.select_action(
            self.state.actor, self._frozen_obs_stats(), observation, key,
            evaluate, critic=self.state.critic,
        )
        self.last_noise = noise
        return self.last_action

    def freeze_acting_norm(self):
        """Pin acting normalization to the statistics as they stand NOW.

        Called by the rollout at each chunk boundary. On the fused path nothing
        else would capture it: `step` is never called there, so without this the
        snapshot would first be taken at `update()` time — from statistics the
        whole rollout had already moved, which is the drift the freeze exists to
        prevent. Idempotent within a rollout: `update()` is what clears the pin.
        """
        self._frozen_obs_stats()

    def _frozen_obs_stats(self):
        """The `ObsStats` the current rollout is pinned to, snapshotting if
        stale. `obs_stats` keeps accumulating underneath and is only re-read at
        rollout boundaries, so the statistics lag by one rollout.

        A COPY, because the fused acting burst DONATES the train state: holding
        a reference to `self.state.obs_stats` across a chunk hands `update()` a
        deleted buffer.
        """
        if self._obs_stats_snapshot is None:
            self._obs_stats_snapshot = jax.tree.map(
                jnp.copy, self.state.obs_stats
            )
        return self._obs_stats_snapshot

    def _frozen_obs_norm(self):
        """Mean/std derived from that snapshot, cached.

        Kept as device arrays so acting never syncs to host just to normalize.
        """
        if self._obs_norm is None:
            self._obs_norm = Agent.obs_mean_std(
                self._frozen_obs_stats(), self.obs_eps
            )
        return self._obs_norm

    @staticmethod
    def _rollout_transition(
        prev_obs, action, reward, termination, truncation, extras,
    ):
        """One env-step batch in the trajectory queue's (NUM_ENVS, TIME, ...)
        layout — a length-1 time axis on every leaf.

        `value` alone is already (NUM_ENVS, 1) straight from the critic, so it
        is the one field that must NOT be expanded again.
        """
        return Transition(
            observation=prev_obs[:, None, :],
            action=action[:, None, :],
            reward=reward[:, None],
            # Stored separately rather than as one `done`: GAE bootstraps the
            # value at a truncation but zeroes it at a termination.
            terminal=termination[:, None],
            log_probs=extras["log_probs"][:, None],
            value=extras["value"],
            truncation=truncation[:, None],
        )

    def buffer_transitions(
        self, state, prev_obs, action, reward, termination, truncation, next_obs,
        extras=None,
    ):
        """Write one env-step batch into `state`, IN PLACE — the traced path.

        Overridden rather than inherited because the on-policy queue's layout
        genuinely differs from the off-policy buffers': it wants an explicit
        time axis, and it stores the two behaviour quantities `extras` carries.
        `self.replay.add`, not the jitted `_jit_replay_add`, because the caller
        is already inside a trace and would only nest a `pjit` in it.
        """
        state.buffer_state = self.replay.add(
            state.buffer_state,
            self._rollout_transition(
                prev_obs, action, reward, termination, truncation, extras,
            ),
        )
        if self.normalize_observations:
            obs_batch = jnp.concatenate([prev_obs, next_obs], axis=0)
            state.obs_stats = Agent.update_obs_stats(state.obs_stats, obs_batch)

    def add(self, prev_obs, timestep):
        # The per-step path. Uses the jitted, donating add: dispatched from
        # Python once per env step, it would otherwise copy the whole queue.
        self.state.buffer_state = self._jit_replay_add(
            self.state.buffer_state,
            self._rollout_transition(
                prev_obs, self.last_action, timestep.reward,
                timestep.terminated, timestep.truncated, self.last_extras,
            ),
        )

        if self.normalize_observations:
            obs_batch = jnp.concatenate([prev_obs, timestep.obs], axis=0)
            self.state.obs_stats = Agent.update_obs_stats(
                self.state.obs_stats, obs_batch
            )

    def update(self, steps, agent_rng):
        gradient_steps = 0
        actor_loss_sum, critic_loss_sum = 0.0, 0.0

        while self.replay.can_sample(self.state.buffer_state):
            # Read before the dequeue so `norm_obs` reproduces what `step()` fed
            # the actor and the ratio is 1 on the first pass.
            obs_mean, obs_std = self._frozen_obs_norm()

            (
                norm_obs,
                actions,
                old_log_probs,
                returns_t,
                adv_t,
            ) = _prepare_rollout(
                self._burst_nodes,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                replay_get_fn=self.replay.sample,
                obs_clip=self.obs_clip,
                normalize=self.normalize_observations,
                obs_mean=obs_mean,
                obs_std=obs_std,
            )

            # Rollout consumed: the next one may act under fresher stats.
            self._obs_stats_snapshot = None
            self._obs_norm = None

            (
                agent_rng,
                steps_taken,
                actor_sum,
                critic_sum,
                kl_sum,
                clip_frac_sum,
                early_stops,
                last_kl,
                last_clip_frac,
            ) = _grad_steps(
                self._burst_nodes,
                agent_rng,
                norm_obs,
                actions,
                old_log_probs,
                returns_t,
                adv_t,
                self.learning_steps,
                self.num_minibatches,
                self.minibatch_size,
                self.clip_eps,
                self.entropy_coef,
                self.target_kl,
                self.action_low,
                self.action_high,
            )

            # The one host sync for the whole rollout.
            steps_taken = int(steps_taken)
            gradient_steps += steps_taken
            actor_loss_sum += float(actor_sum)
            critic_loss_sum += float(critic_sum)

            self.last_approx_kl = float(last_kl)
            self.last_clip_frac = float(last_clip_frac)

            # One record per rollout, weighted by the minibatch steps the trust
            # region actually allowed — which is what makes the epoch mean below
            # a per-step mean rather than a mean of per-rollout means.
            if steps_taken:
                self.record_diagnostics(
                    {
                        "approx_kl": float(kl_sum) / steps_taken,
                        "clip_frac": float(clip_frac_sum) / steps_taken,
                        "kl_early_stops": float(early_stops),
                    },
                    steps_taken,
                )

        if gradient_steps == 0:
            return 0, 0.0, 0.0
        return (
            gradient_steps,
            actor_loss_sum / gradient_steps,
            critic_loss_sum / gradient_steps,
        )

    def pop_diagnostics(self, env_steps: int = 0) -> dict:
        """The base drain plus the two metrics that are per ROLLOUT, not per step.

        `record_diagnostics` is called once per rollout, so the drained burst
        count is the rollout count and the drained step count is the minibatch
        budget the trust region allowed across them. Everything else — including
        `ppo/clip_frac`, which near 1 means the surrogate is saturated — reduces
        the same way it does for every other agent.
        """
        out = super().pop_diagnostics(env_steps)
        if out:
            per_rollout = self._diag_drained_steps / self._diag_drained_bursts
            out["ppo/steps_per_rollout"] = per_rollout
            out["ppo/epochs_per_rollout"] = per_rollout / self.num_minibatches
        return out

    def _export_hyperparams(self) -> dict:
        # Not `_replay_hyperparams`: on-policy, so no tau and no update
        # schedule, and the queue sizes itself from its own config keys.
        params = super()._export_hyperparams()
        params.update(
            {
                "gae_lambda": float(self.gae_lambda),
                "clip_eps": float(self.clip_eps),
                "entropy_coef": float(self.entropy_coef),
                # -1 is the on-disk spelling of `None`: scalars only.
                "target_kl": -1.0 if self.target_kl is None else float(self.target_kl),
                "num_minibatches": int(self.num_minibatches),
                "minibatch_size": int(self.minibatch_size),
                "memory_capacity": int(self.max_length_time_axis),
                "memory_batch_size": int(self.batch_size),
            }
        )
        return params
