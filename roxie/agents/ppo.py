import functools

import hydra
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from roxie.agents.agent import Agent, TrainState
from roxie.agents.utils import Transition, serialize_bound
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


# This is the core computational kernel that will be JIT-compiled.
@functools.partial(
    nnx.jit,
    static_argnames=(
        "gamma",
        "gae_lambda",
        "clip_eps",
        "entropy_coef",
        "replay_get_fn",
    ),
)
def _grad_step(
    state: TrainState,
    key: jax.random.PRNGKey,
    gamma: float,
    gae_lambda: float,
    clip_eps: float,
    entropy_coef: float,
    replay_get_fn,  # Function to get the on-policy data
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
    obs_clip: float,
    obs_eps: float,
):
    """
    Performs a single gradient update step for the PPO agent.
    This function is JIT-compiled for performance.
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

    obs_mean, obs_std = Agent.obs_mean_std(state.obs_stats, obs_eps)
    norm_obs = Agent.normalize_obs(re_packed_samples["observations"], obs_mean, obs_std, obs_clip)

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
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    # Actor update
    actor_loss, actor_grads = nnx.value_and_grad(ppo_loss_fn)(
            actor_model=state.actor,
            observations=norm_obs,
            actions_buf=re_packed_samples["actions"],
            old_log_probs=re_packed_samples["log_probs"],
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
            values=re_packed_samples["values"],
            advantages=adv_t,
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
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        gamma: float = 0.99,
        learning_steps: int = 5,
        max_grad_norm: float = 1.0,
        normalize_observations: bool = True,
        obs_norm_clip: float = 5.0,
        obs_norm_eps: float = 1e-8,
    ):

        actor_rngs = nnx.Rngs(params=0, dropout=1)
        critic_rngs = nnx.Rngs(params=0, dropout=1)

        # Instantiate actor
        actor = hydra.utils.instantiate(
            actor_config,
            in_features=env_obs_size,
            action_dim=env_action_size,
            rngs=actor_rngs,
        )

        # Instantiate critic
        critic = hydra.utils.instantiate(
            critic_config, in_features=env_obs_size, rngs=critic_rngs
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

        buffer_state = replay.init(prototype)

        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate = actor_learning_rate
        self.max_grad_norm = max_grad_norm

        # Add gradient clipping to optimizers
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

        # Store hyperparameters
        self.gamma = gamma
        self.action_low = action_low
        self.action_high = action_high
        self.replay = replay
        self.learning_steps = learning_steps
        self.normalize_observations = normalize_observations
        self.obs_clip = float(obs_norm_clip)
        self.obs_eps = float(obs_norm_eps)

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
            mean, std = Agent.obs_mean_std(self.state.obs_stats, self.obs_eps)
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
        # store in memory
        self.state.buffer_state = self.replay.add(
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
        gradient_steps, actor_loss, critic_loss = 0, 0, 0

        # The buffer is full, so we can start learning
        while self.replay.can_sample(self.state.buffer_state):
            agent_rng, key = jax.random.split(agent_rng, 2)

            self.state, actor_loss, critic_loss = _grad_step(
                state=self.state,
                key=key,
                gamma=self.gamma,
                gae_lambda=0.95, # TODO: make configurable
                clip_eps=0.2,    # TODO: make configurable
                entropy_coef=0.01, # TODO: make configurable
                replay_get_fn=self.replay.sample,  # Pass the sample method itself,
                action_low=self.action_low,
                action_high=self.action_high,
                obs_clip=self.obs_clip,
                obs_eps=self.obs_eps,
            )
        
        gradient_steps += self.learning_steps

        # empty the buffer after learning
        return gradient_steps, actor_loss, critic_loss

    def _export_hyperparams(self) -> dict:
        # Keep this minimal and JSON-serializable
        return {
            "gamma": float(self.gamma),
            "actor_learning_rate": float(self.actor_learning_rate),
            "critic_learning_rate": float(self.critic_learning_rate),
            "max_grad_norm": float(self.max_grad_norm),
            "learning_steps": int(self.learning_steps),
            "memory_capacity": int(self.max_length_time_axis),
            "memory_batch_size": int(self.batch_size),
            "normalize_observations": bool(self.normalize_observations),
            "obs_norm_clip": float(self.obs_clip),
            "obs_norm_eps": float(self.obs_eps),
            # bounds can be scalar or arrays → use your helpers
            "action_low": serialize_bound(self.action_low),
            "action_high": serialize_bound(self.action_high),
        }
