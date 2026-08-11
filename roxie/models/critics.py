from typing import Callable, Sequence

import jax.numpy as jnp
from flax import nnx


class QCritic(nnx.Module):
    """Q-value critic: maps (state, action) → Q(s, a)."""

    def __init__(
        self,
        in_features: int,
        features: Sequence[int],
        *,  # Make rngs a keyword-only argument
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.relu,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
    ):
        """
"""
        # --- Store static configuration ---
        self.use_layer_norm = use_layer_norm
        self.activation_fn = activation_fn

        # --- Define stateful layers ---
        hidden_layers = []
        norm_layers = []

        # Create hidden layers dynamically
        current_features = in_features
        for feat in features:
            hidden_layers.append(nnx.Linear(current_features, feat, rngs=rngs))
            if self.use_layer_norm:
                norm_layers.append(nnx.LayerNorm(feat, rngs=rngs))
            current_features = feat

        self.hidden_layers = nnx.List(hidden_layers)
        if self.use_layer_norm:
            self.norm_layers = nnx.List(norm_layers)

        # Dropout and output layers
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.output_layer = nnx.Linear(current_features, 1, rngs=rngs)

    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.norm_layers[i](x)
            x = self.activation_fn(x)
            x = self.dropout(x, deterministic=not training)

        x = self.output_layer(x)
        return x


class DistributionalQCritic(nnx.Module):
    """Categorical distributional critic (C51-style, used by D4PG).

    Maps (state, action) → logits over `num_atoms` fixed return atoms. The
    support itself (v_min/v_max/atoms) lives in the agent, not here — the
    network only parameterizes the categorical weights.
    """

    def __init__(
        self,
        in_features: int,
        features: Sequence[int],
        num_atoms: int,
        *,  # Make rngs a keyword-only argument
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.relu,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
    ):
        # --- Store static configuration ---
        self.use_layer_norm = use_layer_norm
        self.activation_fn = activation_fn

        # --- Define stateful layers ---
        hidden_layers = []
        norm_layers = []

        # Create hidden layers dynamically
        current_features = in_features
        for feat in features:
            hidden_layers.append(nnx.Linear(current_features, feat, rngs=rngs))
            if self.use_layer_norm:
                norm_layers.append(nnx.LayerNorm(feat, rngs=rngs))
            current_features = feat

        self.hidden_layers = nnx.List(hidden_layers)
        if self.use_layer_norm:
            self.norm_layers = nnx.List(norm_layers)

        # Dropout and output layers
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.output_layer = nnx.Linear(current_features, num_atoms, rngs=rngs)

    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.norm_layers[i](x)
            x = self.activation_fn(x)
            x = self.dropout(x, deterministic=not training)

        x = self.output_layer(x)
        return x


class TwinCritic(nnx.Module):
    def __init__(self, critic1: nnx.Module, critic2: nnx.Module):
        self.critic1 = critic1
        self.critic2 = critic2

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, **kwargs) -> tuple:
        q1 = self.critic1(observations, actions, **kwargs)
        q2 = self.critic2(observations, actions, **kwargs)
        return q1, q2


class VCritic(nnx.Module):
    """State-value critic: maps observations → V(s)."""

    def __init__(self,
        in_features: int,
        features: Sequence[int],
        *,  # Make rngs a keyword-only argument
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.relu,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,):
        # --- Store static configuration ---
        self.use_layer_norm = use_layer_norm
        self.activation_fn = activation_fn

        # --- Define stateful layers ---
        hidden_layers = []
        norm_layers = []

        # Create hidden layers dynamically
        current_features = in_features
        for feat in features:
            hidden_layers.append(nnx.Linear(current_features, feat, rngs=rngs))
            if self.use_layer_norm:
                norm_layers.append(nnx.LayerNorm(feat, rngs=rngs))
            current_features = feat

        self.hidden_layers = nnx.List(hidden_layers)
        if self.use_layer_norm:
            self.norm_layers = nnx.List(norm_layers)

        # Dropout and output layers
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.output_layer = nnx.Linear(current_features, 1, rngs=rngs)

    def __call__(self, observations: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = observations
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.norm_layers[i](x)
            x = self.activation_fn(x)
            x = self.dropout(x, deterministic=not training)

        x = self.output_layer(x)
        return x
