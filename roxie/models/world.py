"""Latent world-model networks for TD-MPC (Hansen et al. 2022).

The pieces of the "Task-Oriented Latent Dynamics" (TOLD) model:

* ``Encoder``          observation -> latent z
* ``LatentDynamics``   (z, action) -> next latent z'
* ``RewardPredictor``  (z, action) -> predicted reward
* twin Q               (z, action) -> Q(z, a), reused from ``roxie.models.critics``

``TOLD`` bundles all four into a single ``nnx.Module`` so the agent can keep it
in the ``critic`` slot of ``TrainState``: checkpointing (``nnx.split``), the
``optax.incremental_update`` soft-target pattern, and the async learner's
snapshotting all operate on it uniformly, with no special-casing.

Nothing here is TD-MPC-specific beyond the shapes — the policy prior pi is a
plain ``DeterministicActor`` over latents, and the Q heads are the existing
``QCritic``/``TwinCritic`` with ``in_features=latent_dim + action_dim``.
"""

from typing import Callable, Optional, Sequence

import jax.numpy as jnp
from flax import nnx


def simnorm(x: jnp.ndarray, group_size: int) -> jnp.ndarray:
    """Simplicial normalization (TD-MPC2, Hansen et al. 2024).

    Splits the last axis into groups of ``group_size`` and softmaxes within each
    group, so the latent lives on a product of simplices. Bounded by
    construction, which is what keeps the latent from drifting during long
    imagined rollouts. Off by default — TD-MPC1 uses an unconstrained latent.
    """
    shape = x.shape
    x = x.reshape(*shape[:-1], shape[-1] // group_size, group_size)
    x = nnx.softmax(x, axis=-1)
    return x.reshape(*shape)


class _MLPTrunk(nnx.Module):
    """Hidden stack shared by the world-model heads.

    ``Linear -> [LayerNorm] -> activation -> [Dropout]`` per entry in
    ``features``. Factored out because three of the heads below need the exact
    same body; the existing critics/actors predate it and are left untouched.
    """

    def __init__(
        self,
        in_features: int,
        features: Sequence[int],
        *,
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.elu,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
    ):
        self.use_layer_norm = use_layer_norm
        self.activation_fn = activation_fn

        hidden_layers = []
        norm_layers = []

        current_features = in_features
        for feat in features:
            hidden_layers.append(nnx.Linear(current_features, feat, rngs=rngs))
            if self.use_layer_norm:
                norm_layers.append(nnx.LayerNorm(feat, rngs=rngs))
            current_features = feat

        self.hidden_layers = nnx.List(hidden_layers)
        if self.use_layer_norm:
            self.norm_layers = nnx.List(norm_layers)

        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.out_features = current_features

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.norm_layers[i](x)
            x = self.activation_fn(x)
            x = self.dropout(x, deterministic=not training)
        return x


class Encoder(nnx.Module):
    """Observation -> latent state.

    The latent is trained purely through the reward / value / consistency
    losses (there is no reconstruction term), so ``latent_dim`` is free to be
    much smaller than the observation.
    """

    def __init__(
        self,
        in_features: int,
        features: Sequence[int],
        latent_dim: int,
        *,
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.elu,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        simnorm_group: Optional[int] = None,
    ):
        self.latent_dim = latent_dim
        self.simnorm_group = simnorm_group
        if simnorm_group is not None and latent_dim % simnorm_group != 0:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be divisible by "
                f"simnorm_group ({simnorm_group})"
            )

        self.trunk = _MLPTrunk(
            in_features,
            features,
            rngs=rngs,
            activation_fn=activation_fn,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
        )
        self.output_layer = nnx.Linear(self.trunk.out_features, latent_dim, rngs=rngs)

    def __call__(self, observations: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        z = self.output_layer(self.trunk(observations, training=training))
        if self.simnorm_group is not None:
            z = simnorm(z, self.simnorm_group)
        return z


class LatentDynamics(nnx.Module):
    """(latent, action) -> next latent.

    The output must live in the same space as the encoder's, so it carries the
    same ``simnorm_group`` setting — mismatching the two would make the latent
    consistency loss compare points from two different geometries.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        features: Sequence[int],
        *,
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.elu,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        simnorm_group: Optional[int] = None,
    ):
        self.latent_dim = latent_dim
        self.simnorm_group = simnorm_group
        if simnorm_group is not None and latent_dim % simnorm_group != 0:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be divisible by "
                f"simnorm_group ({simnorm_group})"
            )

        self.trunk = _MLPTrunk(
            latent_dim + action_dim,
            features,
            rngs=rngs,
            activation_fn=activation_fn,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
        )
        self.output_layer = nnx.Linear(self.trunk.out_features, latent_dim, rngs=rngs)

    def __call__(
        self, latents: jnp.ndarray, actions: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        x = jnp.concatenate([latents, actions], axis=-1)
        z = self.output_layer(self.trunk(x, training=training))
        if self.simnorm_group is not None:
            z = simnorm(z, self.simnorm_group)
        return z


class RewardPredictor(nnx.Module):
    """(latent, action) -> predicted single-step reward, shape (..., 1).

    Trailing singleton axis matches ``QCritic`` so the loss can treat predicted
    rewards and Q-values the same way.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        features: Sequence[int],
        *,
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.elu,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
    ):
        self.trunk = _MLPTrunk(
            latent_dim + action_dim,
            features,
            rngs=rngs,
            activation_fn=activation_fn,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
        )
        self.output_layer = nnx.Linear(self.trunk.out_features, 1, rngs=rngs)

    def __call__(
        self, latents: jnp.ndarray, actions: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        x = jnp.concatenate([latents, actions], axis=-1)
        return self.output_layer(self.trunk(x, training=training))


class TOLD(nnx.Module):
    """Task-Oriented Latent Dynamics model: encoder + dynamics + reward + twin Q.

    A container, not a computation: it exists so the four components travel as
    one ``nnx.Module`` through ``TrainState.critic``, and so the target model is
    a single ``copy.deepcopy`` soft-updated in one ``optax.incremental_update``
    (TD-MPC EMAs the whole model, encoder included, not just the Q heads).
    """

    def __init__(
        self,
        encoder: nnx.Module,
        dynamics: nnx.Module,
        reward: nnx.Module,
        critic: nnx.Module,
    ):
        self.encoder = encoder
        self.dynamics = dynamics
        self.reward = reward
        self.critic = critic

    def encode(self, observations: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        return self.encoder(observations, training=training)

    def next(
        self, latents: jnp.ndarray, actions: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        return self.dynamics(latents, actions, training=training)

    def predict_reward(
        self, latents: jnp.ndarray, actions: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        return self.reward(latents, actions, training=training)

    def q(
        self, latents: jnp.ndarray, actions: jnp.ndarray, training: bool = False
    ) -> tuple:
        """Twin Q-values ``(q1, q2)``, each shaped (..., 1)."""
        return self.critic(latents, actions, training=training)

    def step(
        self, latents: jnp.ndarray, actions: jnp.ndarray, training: bool = False
    ) -> tuple:
        """One imagined step: ``(next_latent, predicted_reward)``.

        The planner's inner loop — kept here so the rollout stays a single call
        per step rather than two independent traversals of the same input.
        """
        return (
            self.dynamics(latents, actions, training=training),
            self.reward(latents, actions, training=training),
        )
