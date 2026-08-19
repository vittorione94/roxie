import functools

import hydra
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from roxie.agents.agent import Agent, TrainState
from roxie.agents.ddpg import DDPG
from roxie.agents.utils import network_rngs, repack_samples
from roxie.losses.actor_losses import d4pg_actor_loss_fn
from roxie.losses.critic_losses import d4pg_critic_loss_fn


# Single D4PG gradient step. Not jitted on its own — called inside the jitted
# `_grad_steps` below so N steps fuse into one compiled program. Identical to
# DDPG's step except the losses go through the categorical critic, which needs
# the fixed support `atoms`.
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
    atoms: jnp.ndarray,
    n_step: int = 1,
):
    """One D4PG step. `obs_mean`/`obs_std` are hoisted in by `_grad_steps` (the
    stats are loop-constant), as is `atoms`. `n_step` is the TD horizon (NOT the
    scan length `n_steps` in _grad_steps)."""
    # 1. Sample from the replay buffer. `repack_samples` folds the n-step
    # return, bootstrap coefficient, and bootstrap obs into the dict — exactly
    # the ingredients the categorical projection needs to shift the support.
    key, noise_key = jax.random.split(key)
    samples = replay_sample_fn(state.buffer_state, key)
    re_packed_samples = repack_samples(samples, gamma, n_step)
    # Normalize once, here: both losses below read the same `observations`, and
    # neither of them normalizes (see `Agent.normalize_samples`).
    re_packed_samples = Agent.normalize_samples(
        re_packed_samples, obs_mean, obs_std, obs_clip, normalize
    )

    # 2. Critic update (cross-entropy against the projected target categorical)
    critic_loss, critic_grads = nnx.value_and_grad(d4pg_critic_loss_fn)(
        state.critic,
        state.target_actor,
        state.target_critic,
        re_packed_samples,
        noise_key,
        target_policy_noise,
        target_noise_clip,
        action_low,
        action_high,
        atoms,
    )
    state.critic_optimizer.update(state.critic, critic_grads)

    # 3. Actor update (DPG through the categorical's expected value)
    actor_loss, actor_grads = nnx.value_and_grad(d4pg_actor_loss_fn)(
        state.actor,
        state.critic,
        re_packed_samples,
        action_low,
        action_high,
        atoms,
    )
    state.actor_optimizer.update(state.actor, actor_grads)

    # 4. Update target networks using soft updates
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


# Fused N-step update. The body is compiled once and run `n_steps` times on-device
# via `lax.scan` (instead of unrolling the Python loop, which at large `n_steps`
# blows up compile time and the HLO graph). Only the trainable graph state is
# carried; `buffer_state`, the normalization params, and `atoms` are constant
# across the loop and closed over.
@functools.partial(
    nnx.jit,
    static_argnames=(
        "gamma", "tau", "replay_sample_fn", "n_steps", "n_step", "normalize",
    ),
    # Donate the train state (arg 0): its large read-only replay buffer is
    # threaded unchanged through the scan, so without donation XLA allocates a
    # full second copy of the buffer (~1.4GB for 500k obs) every update. The
    # caller reassigns self.state from the result, so donating is safe.
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
    atoms: jnp.ndarray,
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
            atoms,
            n_step,
        )
        _, scan_state = nnx.split(st)
        return scan_state, (actor_loss, critic_loss)

    scan_state, (actor_losses, critic_losses) = jax.lax.scan(body, scan_state, keys)
    state = nnx.merge(graphdef, scan_state)

    # Average over the update steps for less noisy logging.
    return state, jnp.mean(actor_losses), jnp.mean(critic_losses)


class D4PG(DDPG):
    """Distributed Distributional DDPG (Barth-Maron et al. 2018).

    Extends DDPG with a categorical (C51-style) distributional critic on a
    fixed support of `num_atoms` atoms over [v_min, v_max]: the critic is
    trained by cross-entropy against the L2-projected Bellman target
    distribution, and the actor follows the gradient of the categorical's
    expected value. Pair with `n_step > 1` (the paper uses N=5) — the n-step
    machinery is inherited from DDPG. Everything else — action selection,
    replay handling, observation normalization — is inherited unchanged.

    Of the paper's four ingredients (distributional critic, n-step returns,
    distributed actors, prioritized replay), this implements the first two;
    parallel envs stand in for distributed actors and replay stays uniform.

    `v_min`/`v_max` must bracket the achievable n-step-discounted return —
    a support that clips real returns saturates the edge atoms and biases Q.
    """

    def __init__(
        self,
        *args,
        v_min: float = -150.0,
        v_max: float = 150.0,
        num_atoms: int = 51,
        **kwargs,
    ):
        # Set before super().__init__: _make_critic (called from DDPG.__init__)
        # needs num_atoms.
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.num_atoms = int(num_atoms)
        self.atoms = jnp.linspace(self.v_min, self.v_max, self.num_atoms)
        super().__init__(*args, **kwargs)

    def _make_critic(self, critic_config, env_obs_size, env_action_size):
        """Build the categorical critic. `num_atoms` is injected from the agent
        args so the critic yaml doesn't have to repeat it."""
        return hydra.utils.instantiate(
            critic_config,
            in_features=env_obs_size + env_action_size,
            num_atoms=self.num_atoms,
            rngs=network_rngs(self.seed, offset=2),
        )

    def learn(self, agent_rng, n_steps=None):
        """One unconditional burst (D4PG variant: threads the categorical
        `atoms` support into the fused grad step). See DDPG.learn for the
        sync/async sharing rationale."""
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
            self.atoms,
            n_step=self.n_step,
        )
        return actor_loss, critic_loss

    def update(self, steps, agent_rng):
        gradient_steps, actor_loss, critic_loss = 0, 0, 0

        if (
            steps >= self.steps_before_learning
            and (steps - self.steps_before_learning) % self.steps_between_updates == 0
        ):
            actor_loss, critic_loss = self.learn(agent_rng)
            gradient_steps += self.learning_steps

        return gradient_steps, actor_loss, critic_loss

    def _export_hyperparams(self) -> dict:
        params = super()._export_hyperparams()
        params["v_min"] = float(self.v_min)
        params["v_max"] = float(self.v_max)
        params["num_atoms"] = int(self.num_atoms)
        return params
