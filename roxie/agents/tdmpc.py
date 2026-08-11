import functools

import hydra
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from roxie.agents.agent import Agent, TrainState
from roxie.agents.ddpg import DDPG
from roxie.agents.planner import plan, plan_jit
from roxie.agents.utils import unpack_sequence
from roxie.losses.world_losses import tdmpc_model_loss_fn, tdmpc_policy_loss_fn
from roxie.models.critics import TwinCritic
from roxie.models.world import TOLD


# Single TD-MPC gradient step. Not jitted on its own — called inside the jitted
# `_grad_steps` below so N steps fuse into one compiled program. Unlike the
# actor-critic agents there is no separate critic step: encoder, dynamics,
# reward and both Q heads are trained by one joint objective, and the policy
# prior follows on the latents that objective already produced.
def _grad_step(
    state: TrainState,
    key: jax.random.PRNGKey,
    gamma: float,
    tau: float,
    replay_sample_fn,
    action_low,
    action_high,
    obs_mean: jnp.ndarray,
    obs_std: jnp.ndarray,
    obs_clip: float,
    rho: float,
    reward_coef: float,
    value_coef: float,
    consistency_coef: float,
    horizon: int,
):
    """One TD-MPC step. `obs_mean`/`obs_std` are hoisted in by `_grad_steps`
    (the stats are loop-constant). `horizon` is both the imagination length and
    the replay window length (the buffer stores `horizon + 1` step windows)."""
    # 1. Sample a window from the replay buffer. `unpack_sequence` keeps the
    # time axis and derives the per-step masks that stop the model from being
    # trained across an episode boundary.
    samples = replay_sample_fn(state.buffer_state, key)
    sequence = unpack_sequence(samples, gamma, horizon)

    # 2. Joint world-model update. `state.critic` is the whole TOLD model, so
    # this one optimizer step moves encoder, dynamics, reward and both Q heads.
    (model_loss, aux), model_grads = nnx.value_and_grad(
        tdmpc_model_loss_fn, has_aux=True
    )(
        state.critic,
        state.target_critic,
        state.actor,
        sequence,
        obs_mean,
        obs_std,
        obs_clip,
        action_low,
        action_high,
        rho,
        reward_coef,
        value_coef,
        consistency_coef,
    )
    state.critic_optimizer.update(state.critic, model_grads)

    # 3. Policy prior update on the (detached) imagined latents, against the
    # just-updated model — the ordering TD-MPC uses.
    policy_loss, policy_grads = nnx.value_and_grad(tdmpc_policy_loss_fn)(
        state.actor,
        state.critic,
        aux["latents"],
        aux["policy_weights"],
        action_low,
        action_high,
    )
    state.actor_optimizer.update(state.actor, policy_grads)

    # 4. Soft-update the target model. TD-MPC EMAs the *whole* model (the target
    # encoder produces both the TD bootstrap latent and the consistency target),
    # which is exactly what a single incremental_update over TOLD's params does.
    new_model_tensors = nnx.state(state.critic, nnx.Param)
    old_model_tensors = nnx.state(state.target_critic, nnx.Param)
    nnx.update(
        state.target_critic,
        optax.incremental_update(
            new_tensors=new_model_tensors, old_tensors=old_model_tensors, step_size=tau
        ),
    )
    # The target policy is not read by either loss (the TD target uses the
    # ONLINE policy, per the paper). It is kept as an EMA of pi purely so the
    # checkpoint layout stays identical to every other agent here.
    new_actor_tensors = nnx.state(state.actor, nnx.Param)
    old_actor_tensors = nnx.state(state.target_actor, nnx.Param)
    nnx.update(
        state.target_actor,
        optax.incremental_update(
            new_tensors=new_actor_tensors, old_tensors=old_actor_tensors, step_size=tau
        ),
    )

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
        policy_loss,
        model_loss,
    )


# Fused N-step update, same shape as the other agents': the body is compiled
# once and run `n_steps` times on-device via `lax.scan`. Only the trainable
# graph state is carried; `buffer_state` and the normalization params are
# loop-constant.
@functools.partial(
    nnx.jit,
    static_argnames=(
        "gamma", "tau", "replay_sample_fn", "n_steps", "horizon",
    ),
    # Donate the train state (arg 0): its large read-only replay buffer is
    # threaded unchanged through the scan, so without donation XLA allocates a
    # full second copy of the buffer every update. The caller reassigns
    # self.state from the result, so donating is safe.
    donate_argnums=(0,),
)
def _grad_steps(
    state: TrainState,
    key: jax.random.PRNGKey,
    n_steps: int,
    gamma: float,
    tau: float,
    replay_sample_fn,
    action_low,
    action_high,
    obs_eps: float,
    obs_clip: float,
    rho: float,
    reward_coef: float,
    value_coef: float,
    consistency_coef: float,
    horizon: int,
):
    # Hoist the (loop-constant) normalization params out of the scan body.
    obs_mean, obs_std = Agent.obs_mean_std(state.obs_stats, obs_eps)

    keys = jax.random.split(key, n_steps)
    graphdef, scan_state = nnx.split(state)

    def body(scan_state, step_key):
        st = nnx.merge(graphdef, scan_state)
        st, policy_loss, model_loss = _grad_step(
            st,
            step_key,
            gamma,
            tau,
            replay_sample_fn,
            action_low,
            action_high,
            obs_mean,
            obs_std,
            obs_clip,
            rho,
            reward_coef,
            value_coef,
            consistency_coef,
            horizon,
        )
        _, scan_state = nnx.split(st)
        return scan_state, (policy_loss, model_loss)

    scan_state, (policy_losses, model_losses) = jax.lax.scan(body, scan_state, keys)
    state = nnx.merge(graphdef, scan_state)

    return state, jnp.mean(policy_losses), jnp.mean(model_losses)


class TDMPC(DDPG):
    """Temporal Difference Learning for Model Predictive Control.

    Hansen, Wang & Su 2022 — https://td-mpc.github.io/

    Learns a latent world model (TOLD: encoder, dynamics, reward, twin Q) with
    no reconstruction term, so the latent keeps only what predicting reward and
    value requires. Actions come from short-horizon MPPI planning in that latent
    space, with a learned policy prior seeding the candidate set and a terminal Q
    standing in for return beyond the planning horizon.

    How it maps onto this repo's abstractions:

    * ``TrainState.critic`` holds the whole TOLD model and ``TrainState.actor``
      the policy prior, so checkpointing, target soft-updates and the fused
      ``_grad_steps`` all work unchanged.
    * The replay buffer is the trajectory buffer DDPG already builds for
      ``n_step > 1``; ``horizon`` sets its window (``horizon + 1`` items).
    * ``critic_learning_rate`` is the world model's, ``actor_learning_rate``
      the policy prior's.

    Not compatible with ``trainer.async_learner``: planning reads the world
    model, and the async learner only hands the acting thread a snapshot of the
    actor, so acting would race the learner's live model. Leave it off.
    """

    def __init__(
        self,
        *args,
        horizon: int = 5,
        latent_dim: int = 50,
        # Planner
        num_samples: int = 256,
        num_elites: int = 32,
        num_policy_trajectories: int = 24,
        num_iterations: int = 6,
        temperature: float = 0.5,
        momentum: float = 0.1,
        min_std: float = 0.05,
        max_std: float = 2.0,
        # Loss
        rho: float = 0.5,
        reward_coef: float = 0.5,
        value_coef: float = 0.1,
        consistency_coef: float = 2.0,
        **kwargs,
    ):
        self.horizon = int(horizon)
        self.latent_dim = int(latent_dim)

        self.num_samples = int(num_samples)
        self.num_elites = int(num_elites)
        self.num_policy_trajectories = int(num_policy_trajectories)
        self.num_iterations = int(num_iterations)
        self.temperature = float(temperature)
        self.momentum = float(momentum)
        self.min_std = float(min_std)
        self.max_std = float(max_std)

        self.rho = float(rho)
        self.reward_coef = float(reward_coef)
        self.value_coef = float(value_coef)
        self.consistency_coef = float(consistency_coef)

        # The replay window is owned by `horizon`, not by `n_step`: DDPG builds
        # a trajectory buffer of `n_step + 1` items, which is exactly the
        # `horizon + 1` observations one imagined rollout needs. There is no
        # n-step return anywhere in TD-MPC — every imagined step gets its own
        # 1-step TD target — so `n_step` is only ever the window length here.
        kwargs["n_step"] = self.horizon

        super().__init__(*args, **kwargs)

        # Per-env plan carried between steps (MPPI warm start), in [-1, 1].
        # Training and evaluation keep separate plans: eval runs interleave with
        # training steps and must not clobber the acting plan.
        self._plan_mean = None
        self._eval_plan_mean = None
        self.last_plan_std = None

    # --------------------------
    # Construction
    # --------------------------
    def _make_actor(self, actor_config, env_obs_size, env_action_size):
        """The policy prior maps a LATENT to an action, not an observation."""
        # Stashed here (this runs first inside DDPG.__init__) because the plan
        # buffer needs the action dim, and `action_high` may be a bare scalar.
        self.env_action_size = int(env_action_size)
        return hydra.utils.instantiate(
            actor_config,
            in_features=self.latent_dim,
            action_dim=env_action_size,
            rngs=nnx.Rngs(params=0, dropout=1),
        )

    def _make_critic(self, critic_config, env_obs_size, env_action_size):
        """Build the whole TOLD world model from the nested `critic` config.

        The four sub-nets live under `critic:` in the yaml so that `train.py`'s
        existing actor/critic/memory/noise plumbing carries them across
        untouched — TD-MPC needs no new config passthrough.
        """
        encoder = hydra.utils.instantiate(
            critic_config.encoder,
            in_features=env_obs_size,
            latent_dim=self.latent_dim,
            rngs=nnx.Rngs(params=2, dropout=3),
        )
        dynamics = hydra.utils.instantiate(
            critic_config.dynamics,
            latent_dim=self.latent_dim,
            action_dim=env_action_size,
            rngs=nnx.Rngs(params=4, dropout=5),
        )
        reward = hydra.utils.instantiate(
            critic_config.reward,
            latent_dim=self.latent_dim,
            action_dim=env_action_size,
            rngs=nnx.Rngs(params=6, dropout=7),
        )
        # Twin Q over latents — the existing QCritic/TwinCritic, unmodified.
        q1 = hydra.utils.instantiate(
            critic_config.q,
            in_features=self.latent_dim + env_action_size,
            rngs=nnx.Rngs(params=8, dropout=9),
        )
        q2 = hydra.utils.instantiate(
            critic_config.q,
            in_features=self.latent_dim + env_action_size,
            rngs=nnx.Rngs(params=10, dropout=11),
        )
        return TOLD(
            encoder=encoder, dynamics=dynamics, reward=reward, critic=TwinCritic(q1, q2)
        )

    # --------------------------
    # Acting
    # --------------------------
    def _get_plan_mean(self, batch: int, action_dim: int, evaluate: bool):
        attr = "_eval_plan_mean" if evaluate else "_plan_mean"
        mean = getattr(self, attr)
        if mean is None or mean.shape[0] != batch:
            mean = jnp.zeros((batch, self.horizon, action_dim), dtype=jnp.float32)
            setattr(self, attr, mean)
        return mean

    def select_action(self, actor, obs_stats, observation, key, evaluate=False):
        """Plan an action from an EXPLICIT policy prior + obs stats.

        Signature matches DDPG's so the trainer's call sites are unchanged, but
        the world model is read from ``self.state.critic`` — which is why the
        async learner (whose whole point is that the acting thread reads only
        the passed-in snapshot) is not supported for this agent.

        Returns ``(scaled_action, applied_noise)``; the second value is the
        planner's residual noise, kept for the trainer's noise logging.
        """
        if self.normalize_observations:
            mean, std = Agent.obs_mean_std(obs_stats, self.obs_eps)
            observation = Agent.normalize_obs(observation, mean, std, self.obs_clip)

        model = self.state.critic
        z = model.encode(observation)

        prev_mean = self._get_plan_mean(z.shape[0], self.env_action_size, evaluate)
        action, next_mean, plan_std, noise = plan_jit(
            model,
            actor,
            z,
            prev_mean,
            key,
            self.action_low,
            self.action_high,
            horizon=self.horizon,
            num_samples=self.num_samples,
            num_elites=self.num_elites,
            num_policy_trajectories=self.num_policy_trajectories,
            num_iterations=self.num_iterations,
            gamma=self.gamma,
            temperature=self.temperature,
            momentum=self.momentum,
            min_std=self.min_std,
            max_std=self.max_std,
            evaluate=evaluate,
        )
        setattr(self, "_eval_plan_mean" if evaluate else "_plan_mean", next_mean)
        self.last_plan_std = plan_std

        return Agent.scale_to_env(action, self.action_low, self.action_high), noise

    def add_transitions(
        self, prev_obs, action, reward, termination, truncation, next_obs
    ):
        """Buffer the transition, then drop the plan of any env that just ended.

        A plan warm-started from the previous episode is meaningless after a
        reset, and would bias the first steps of the new one. Done as a device
        op (`jnp.where`), never a host branch, to keep the trainer's async GPU
        pipeline unbroken.
        """
        super().add_transitions(
            prev_obs, action, reward, termination, truncation, next_obs
        )
        if self._plan_mean is not None:
            done = jnp.logical_or(termination, truncation)
            self._plan_mean = jnp.where(
                done[:, None, None], 0.0, self._plan_mean
            )

    # --------------------------
    # Learning
    # --------------------------
    def learn(self, agent_rng, n_steps=None):
        """One unconditional burst of fused TD-MPC gradient steps.

        Returns ``(policy_loss, model_loss)`` in the (actor_loss, critic_loss)
        slots the trainer logs.
        """
        self.state, policy_loss, model_loss = _grad_steps(
            self.state,
            agent_rng,
            self.learning_steps if n_steps is None else int(n_steps),
            self.gamma,
            self.tau,
            self.replay.sample,
            self.action_low,
            self.action_high,
            self.obs_eps,
            self.obs_clip,
            self.rho,
            self.reward_coef,
            self.value_coef,
            self.consistency_coef,
            self.horizon,
        )
        return policy_loss, model_loss

    def update(self, steps, agent_rng):
        gradient_steps, actor_loss, critic_loss = 0, 0, 0

        if (
            steps >= self.steps_before_learning
            and (steps - self.steps_before_learning) % self.steps_between_updates == 0
        ):
            actor_loss, critic_loss = self.learn(agent_rng)
            gradient_steps += self.learning_steps

        return gradient_steps, actor_loss, critic_loss

    # --------------------------
    # Evaluation
    # --------------------------
    def eval_action_fn(self):
        """Planning-based greedy action for the trainer's compiled eval rollout.

        The default eval path calls ``actor(obs)``, which for TD-MPC would
        evaluate the bare policy prior rather than the planner — a different
        (and much weaker) agent than the one being trained.

        Returns a callable ``(actor, critic, obs_stats, obs, carry, key) ->
        (action_in_env_units, new_carry)``, where the carry is the MPPI warm
        start. The networks arrive as arguments rather than being closed over so
        the trainer can compile the rollout once and still evaluate the *current*
        weights every epoch.
        """

        def act(actor, model, obs_stats, obs, plan_mean, key):
            if self.normalize_observations:
                mean, std = Agent.obs_mean_std(obs_stats, self.obs_eps)
                obs = Agent.normalize_obs(obs, mean, std, self.obs_clip)
            action, next_mean, _, _ = plan(
                model,
                actor,
                model.encode(obs),
                plan_mean,
                key,
                self.action_low,
                self.action_high,
                horizon=self.horizon,
                num_samples=self.num_samples,
                num_elites=self.num_elites,
                num_policy_trajectories=self.num_policy_trajectories,
                num_iterations=self.num_iterations,
                gamma=self.gamma,
                temperature=self.temperature,
                momentum=self.momentum,
                min_std=self.min_std,
                max_std=self.max_std,
                evaluate=True,
            )
            return (
                Agent.scale_to_env(action, self.action_low, self.action_high),
                next_mean,
            )

        return act

    def initial_plan_mean(self, batch: int):
        """Zero warm start for a fresh eval rollout of `batch` environments."""
        return jnp.zeros(
            (batch, self.horizon, self.env_action_size), dtype=jnp.float32
        )

    def reset_plan(self, evaluate: bool = False):
        """Drop the carried MPPI plan.

        Called by the trainer at the start of each eval rollout: episodes there
        are unrelated to the previous eval's, so a stale warm start would bias
        the first planning steps of every run.
        """
        setattr(self, "_eval_plan_mean" if evaluate else "_plan_mean", None)

    def _export_hyperparams(self) -> dict:
        params = super()._export_hyperparams()
        params.update(
            horizon=int(self.horizon),
            latent_dim=int(self.latent_dim),
            num_samples=int(self.num_samples),
            num_elites=int(self.num_elites),
            num_policy_trajectories=int(self.num_policy_trajectories),
            num_iterations=int(self.num_iterations),
            temperature=float(self.temperature),
            momentum=float(self.momentum),
            min_std=float(self.min_std),
            max_std=float(self.max_std),
            rho=float(self.rho),
            reward_coef=float(self.reward_coef),
            value_coef=float(self.value_coef),
            consistency_coef=float(self.consistency_coef),
        )
        return params
