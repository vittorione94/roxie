"""Dynamics networks a world model predicts the environment's next step with."""

from typing import Callable, Sequence

import jax.numpy as jnp
from flax import nnx

from roxie.models.critics import MLPBlock


class DynamicsModel(nnx.Module):
    """A deterministic one-step model: `(obs, action) -> (delta, reward, logit)`.

    Works in the world's NORMALIZED observation space and predicts the change
    in it rather than the next observation itself, so an untrained model starts
    out predicting "nothing moves" instead of an arbitrary point. Likewise it
    starts out predicting "nothing ends": at a zero logit every imagined step
    would carry a 50% chance of termination, and a 32-step rollout would keep
    ~1e-10 of its weight past the horizon until the termination fit caught up.

    Attributes:
        obs_size: Width of the observation it reads and predicts the change of.
        action_size: Width of the action it conditions on.
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        features: Sequence[int],
        *,
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.relu,
        use_layer_norm: bool = True,
        output_init_scale: float = 0.01,
        termination_bias_init: float = -5.0,
    ):
        self.obs_size = int(obs_size)
        self.action_size = int(action_size)
        self.trunk = MLPBlock(
            self.obs_size + self.action_size,
            features,
            rngs=rngs,
            activation_fn=activation_fn,
            use_layer_norm=use_layer_norm,
        )
        width = self.trunk.out_features
        self.delta_head = nnx.Linear(
            width,
            self.obs_size,
            rngs=rngs,
            kernel_init=nnx.initializers.variance_scaling(
                output_init_scale, "fan_in", "truncated_normal"
            ),
            bias_init=nnx.initializers.zeros_init(),
        )
        self.reward_head = nnx.Linear(width, 1, rngs=rngs)
        self.termination_head = nnx.Linear(
            width,
            1,
            rngs=rngs,
            bias_init=nnx.initializers.constant(termination_bias_init),
        )

    def __call__(
        self, norm_obs: jnp.ndarray, action: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Returns `(delta, reward, termination_logit)` for one step.

        `delta` is in normalized units, and `reward` and the logit have the
        trailing unit axis squeezed off.
        """
        x = self.trunk(jnp.concatenate([norm_obs, action], axis=-1))
        return (
            self.delta_head(x),
            self.reward_head(x)[..., 0],
            self.termination_head(x)[..., 0],
        )
