"""Critic neural network models for value estimation."""

from typing import Callable, Sequence

import jax.numpy as jnp
from flax import nnx

from roxie.models.actors import torch_linear_init
from roxie.utils.precision import FLOAT, as_float, linear_kwargs


class MLPBlock(nnx.Module):
    """Shared multi-layer perceptron block for feature extraction."""

    def __init__(
        self,
        in_features: int,
        features: Sequence[int],
        *,
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.relu,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        torch_init: bool = False,
    ):
        self.use_layer_norm = use_layer_norm
        self.activation_fn = activation_fn

        hidden_layers = []
        norm_layers = []

        current_features = in_features
        for feat in features:
            extra = {}
            if torch_init:
                k, b = torch_linear_init(current_features)
                extra = {"kernel_init": k, "bias_init": b}
            hidden_layers.append(
                nnx.Linear(current_features, feat, rngs=rngs,
                           **linear_kwargs(), **extra)
            )
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
            x = as_float(layer(x))
            if self.use_layer_norm:
                x = self.norm_layers[i](x)
            x = self.activation_fn(x)
            x = self.dropout(x, deterministic=not training)
        return x


class QCritic(nnx.Module):
    """Q-value critic mapping (observation, action) -> Q(s, a)."""

    def __init__(
        self,
        in_features: int,
        features: Sequence[int],
        *,
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.relu,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
    ):
        self.mlp = MLPBlock(
            in_features,
            features,
            rngs=rngs,
            activation_fn=activation_fn,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
        )
        self.output_layer = nnx.Linear(self.mlp.out_features, 1, rngs=rngs)

    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        x = self.mlp(x, training=training)
        return self.output_layer(x)


class DistributionalQCritic(nnx.Module):
    """Categorical distributional Q-critic mapping (observation, action) -> logits over return atoms."""

    def __init__(
        self,
        in_features: int,
        features: Sequence[int],
        v_min: float,
        v_max: float,
        num_atoms: int,
        *,
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.relu,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
    ):
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.num_atoms = int(num_atoms)
        self.atoms = jnp.linspace(
            self.v_min, self.v_max, self.num_atoms, dtype=FLOAT
        )

        self.mlp = MLPBlock(
            in_features,
            features,
            rngs=rngs,
            activation_fn=activation_fn,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
        )
        self.output_layer = nnx.Linear(self.mlp.out_features, num_atoms, rngs=rngs)

    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        x = self.mlp(x, training=training)
        return self.output_layer(x)


class TwinCritic(nnx.Module):
    """Dual independent critics evaluated together for clipped double-Q estimation."""

    def __init__(self, critic1: nnx.Module, critic2: nnx.Module):
        self.critic1 = critic1
        self.critic2 = critic2

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, **kwargs) -> tuple:
        q1 = self.critic1(observations, actions, **kwargs)
        q2 = self.critic2(observations, actions, **kwargs)
        return q1, q2


class VCritic(nnx.Module):
    """State-value critic mapping observation -> V(s)."""

    def __init__(
        self,
        in_features: int,
        features: Sequence[int],
        *,
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.relu,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        torch_init: bool = False,
    ):
        self.mlp = MLPBlock(
            in_features,
            features,
            rngs=rngs,
            activation_fn=activation_fn,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
            torch_init=torch_init,
        )
        head = {}
        if torch_init:
            k, b = torch_linear_init(self.mlp.out_features)
            head = {"kernel_init": k, "bias_init": b}
        self.output_layer = nnx.Linear(self.mlp.out_features, 1, rngs=rngs, **head)

    def __call__(self, observations: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = self.mlp(observations, training=training)
        return self.output_layer(x)