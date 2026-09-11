import functools

import hydra
import jax
import jax.numpy as jnp
from flax import nnx

from roxie.agents.agent import Agent, TrainState
from roxie.agents.ddpg import DDPG
from roxie.agents.utils import (
    fused_grad_steps,
    network_rngs,
    reduce_diagnostics,
    repack_samples,
    soft_update,
)
from roxie.losses.actor_losses import d4pg_actor_loss_fn
from roxie.losses.critic_losses import d4pg_critic_loss_fn


# DDPG's step, except the losses go through the categorical critic and so need
# the fixed support `atoms`.
def _grad_step(
    state: TrainState,
    key: jax.random.PRNGKey,
    *,
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
    pre_activation_coef: float,
    n_step: int = 1,
):
    """One D4PG step, mutating `state` in place. `obs_mean`/`obs_std` are hoisted
    in by `_grad_steps` (the stats are loop-constant), as is `atoms`. `n_step` is
    the TD horizon (NOT the scan length `n_steps` in _grad_steps)."""
    key, noise_key = jax.random.split(key)
    samples = replay_sample_fn(state.buffer_state, key)
    re_packed_samples = repack_samples(samples, gamma, n_step)
    re_packed_samples = Agent.normalize_samples(
        re_packed_samples, obs_mean, obs_std, obs_clip, normalize
    )

    (critic_loss, critic_aux), critic_grads = nnx.value_and_grad(
        d4pg_critic_loss_fn, has_aux=True
    )(
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

    (actor_loss, actor_aux), actor_grads = nnx.value_and_grad(
        d4pg_actor_loss_fn, has_aux=True
    )(
        state.actor,
        state.critic,
        re_packed_samples,
        action_low,
        action_high,
        atoms,
        pre_activation_coef,
    )
    state.actor_optimizer.update(state.actor, actor_grads)

    soft_update(state.target_actor, state.actor, tau)
    soft_update(state.target_critic, state.critic, tau)

    return actor_loss, critic_loss, actor_aux, critic_aux


@functools.partial(
    jax.jit,
    static_argnames=(
        "gamma", "tau", "replay_sample_fn", "n_steps", "n_step", "normalize",
    ),
    # The replay buffer threads unchanged through the scan; without donation XLA
    # allocates a second copy of it per burst.
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
    pre_activation_coef: float,
    atoms: jnp.ndarray,
    n_step: int = 1,
):
    obs_mean, obs_std = Agent.obs_mean_std(state.obs_stats, obs_eps)

    state, (actor_losses, critic_losses, actor_aux, critic_aux) = fused_grad_steps(
        state,
        key,
        n_steps,
        functools.partial(
            _grad_step,
            gamma=gamma,
            tau=tau,
            replay_sample_fn=replay_sample_fn,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            action_low=action_low,
            action_high=action_high,
            obs_mean=obs_mean,
            obs_std=obs_std,
            obs_clip=obs_clip,
            normalize=normalize,
            atoms=atoms,
            pre_activation_coef=pre_activation_coef,
            n_step=n_step,
        ),
    )

    diagnostics = {
        **reduce_diagnostics(actor_aux, n_steps),
        **reduce_diagnostics(critic_aux, n_steps),
    }
    return state, jnp.mean(actor_losses), jnp.mean(critic_losses), diagnostics


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
        # Set before super().__init__: `_make_critic` needs it.
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
        burst_steps = self.learning_steps if n_steps is None else int(n_steps)
        self.state, actor_loss, critic_loss, diagnostics = _grad_steps(
            self.state,
            agent_rng,
            burst_steps,
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
            self.pre_activation_coef,
            self.atoms,
            n_step=self.n_step,
        )
        self.record_diagnostics(diagnostics, burst_steps)
        return actor_loss, critic_loss

    def _export_hyperparams(self) -> dict:
        params = super()._export_hyperparams()
        params["v_min"] = float(self.v_min)
        params["v_max"] = float(self.v_max)
        params["num_atoms"] = int(self.num_atoms)
        return params
