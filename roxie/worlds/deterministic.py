"""A deterministic, observation-space world model fit on k-step windows."""

import functools

import hydra
import jax
import jax.numpy as jnp
from flax import nnx

from roxie.agents.hyperparams import build_hyperparams
from roxie.agents.utils import (
    build_replay,
    fused_grad_steps,
    make_optimizer,
    reduce_diagnostics,
    transition_prototype,
)
from roxie.losses.world_losses import world_rollout_loss_fn
from roxie.utils.diagnostics import DiagnosticsTracker
from roxie.utils.math import init_obs_stats
from roxie.utils.memory import ReplayManager
from roxie.worlds.hyperparams import WorldHyperparams
from roxie.worlds.world import (
    WorldModel,
    WorldOutput,
    WorldState,
    world_normalizer,
)


def _world_grad_step(
    state: WorldState,
    key: jax.Array,
    *,
    hp: WorldHyperparams,
    sample_fn,
    obs_mean: jnp.ndarray,
    obs_std: jnp.ndarray,
):
    """Fits the model on one batch of real `k + 1`-step windows."""
    window = sample_fn(state.buffer_state, key).experience
    (loss, aux), grads = nnx.value_and_grad(
        world_rollout_loss_fn, has_aux=True
    )(
        state.dynamics,
        window,
        obs_mean,
        obs_std,
        hp.reward_coef,
        hp.termination_coef,
    )
    state.optimizer.update(state.dynamics, grads)
    return loss, aux


@functools.partial(
    jax.jit,
    static_argnames=("hp", "sample_fn", "n_steps"),
    # The buffer rides in the state and is the largest array in the run;
    # donating it avoids a transient second copy of it every pass.
    donate_argnums=(0,),
)
def _world_grad_steps(
    state: WorldState,
    key: jax.Array,
    n_steps: int,
    hp: WorldHyperparams,
    sample_fn,
):
    """Runs one fitting pass of world-model gradient steps via `jax.lax.scan`."""
    # Pinned for the pass, as an agent pins its own: the targets of every step
    # are then measured in one space.
    obs_mean, obs_std = world_normalizer(state.obs_stats, hp)
    state, (losses, aux) = fused_grad_steps(
        state,
        key,
        n_steps,
        functools.partial(
            _world_grad_step,
            hp=hp,
            sample_fn=sample_fn,
            obs_mean=obs_mean,
            obs_std=obs_std,
        ),
    )
    return WorldOutput(
        state=state,
        loss=jnp.mean(losses),
        diagnostics=reduce_diagnostics(aux, n_steps),
    )


class DeterministicWorld(WorldModel):
    """One deterministic network predicting the next observation, reward, done.

    The SuperMimic world model (Liao et al., 2026), in observation space: the
    environments here expose no simulator state or domain-randomization
    context, so `s = obs` and there is no `xi` to condition on.

    Reference:
        Liao et al., 2026: https://arxiv.org/abs/2610.28480
    """

    hyperparams_cls = WorldHyperparams

    def __init__(
        self,
        env_obs_size: int,
        env_action_size: int,
        dynamics_config: dict,
        memory_config: dict,
        *,
        hyperparams: WorldHyperparams = None,
        optimizer_config: dict = None,
    ):
        self.hp = build_hyperparams(type(self).hyperparams_cls, hyperparams)
        self.env_obs_size = int(env_obs_size)
        self.env_action_size = int(env_action_size)

        dynamics = hydra.utils.instantiate(dynamics_config)

        # `window` steps fitted means `window + 1` observations per sample —
        # exactly the trajectory buffer `build_replay` makes for an n-step
        # return of the same length.
        self.replay = ReplayManager(
            build_replay(memory_config, self.hp.window),
            transition_prototype(self.env_obs_size, self.env_action_size),
            time_axis=True,
        )
        self.batch_size = memory_config.sample_batch_size
        self.buffer_size = memory_config.max_length

        self.state = WorldState(
            dynamics=dynamics,
            optimizer=make_optimizer(
                dynamics,
                optimizer_config,
                learning_rate=self.hp.learning_rate,
                max_grad_norm=self.hp.max_grad_norm,
            ),
            buffer_state=self.replay.init(),
            obs_stats=init_obs_stats(self.env_obs_size),
        )
        self.diagnostics = DiagnosticsTracker("world")

        print(f"{type(self).__name__} world model initialized.")
        print("World Params:", self._export_hyperparams())

    def step(self, dynamics, obs_mean, obs_std, obs, action):
        """One imagined step through `dynamics`; see `WorldModel.step`."""
        z = (obs - obs_mean) / obs_std
        delta, reward, logit = dynamics(z, action)
        return (z + delta) * obs_std + obs_mean, reward, jax.nn.sigmoid(logit)

    def _compile_and_run(self, key, target_steps: int) -> WorldOutput:
        """Runs one fused fitting pass on the real buffer."""
        return _world_grad_steps(
            self.state, key, target_steps, self.hp, self.replay.sample,
        )
