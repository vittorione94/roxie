from typing import Callable, Optional, Sequence

import distrax
import jax.numpy as jnp
from flax import nnx


class DeterministicActor(nnx.Module):
    def __init__(
        self,
        in_features: int,
        features: Sequence[int],
        action_dim: int,
        *,  # Make rngs a keyword-only argument
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.relu,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
    ):
        # --- Store static configuration ---
        self.use_layer_norm = use_layer_norm
        self.activation_fn = activation_fn
        self.action_dim = action_dim

        # --- Define stateful layers ---
        self.hidden_layers = []
        if self.use_layer_norm:
            self.norm_layers = []

        # Create hidden layers dynamically
        current_features = in_features
        for feat in features:
            self.hidden_layers.append(nnx.Linear(current_features, feat, rngs=rngs))
            if self.use_layer_norm:
                self.norm_layers.append(nnx.LayerNorm(feat, rngs=rngs))
            current_features = feat

        # Dropout and output layers
        # self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.output_layer = nnx.Linear(current_features, action_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Use the layers defined in __init__
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.norm_layers[i](x)
            x = self.activation_fn(x)
            # x = self.dropout(x, deterministic=not training)

        x = self.output_layer(x)
        # Scale output to action space range, e.g., [-1, 1]
        x = nnx.tanh(x)
        return x


# --- Actor for SAC/PPO ---
class StochasticActor(nnx.Module):
    def __init__(
        self,
        in_features: int,
        features: Sequence[int],
        action_dim: int,
        *,  # Make rngs a keyword-only argument
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.relu,
        use_layer_norm: bool = False,
        std_min: float = 1e-4,
        std_max: float = 1.0,
    ):
        # --- Store static configuration ---
        self.use_layer_norm = use_layer_norm
        self.activation_fn = activation_fn
        self.action_dim = action_dim
        self.std_min = std_min
        self.std_max = std_max

        # --- Define stateful layers ---
        self.hidden_layers = []
        if self.use_layer_norm:
            self.norm_layers = []

        # Create hidden layers dynamically
        current_features = in_features
        for feat in features:
            self.hidden_layers.append(nnx.Linear(current_features, feat, rngs=rngs))
            if self.use_layer_norm:
                self.norm_layers.append(nnx.LayerNorm(feat, rngs=rngs))
            current_features = feat

        # Output layers for mean and log_std
        self.output_layer = nnx.Linear(current_features, action_dim, rngs=rngs)
        self.log_std_layer = nnx.Linear(current_features, action_dim, rngs=rngs)

        self.distribution = distrax.MultivariateNormalDiag

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Use the layers defined in __init__
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.norm_layers[i](x)
            x = self.activation_fn(x)

        mean = self.output_layer(x)
        std = nnx.softplus(self.log_std_layer(x)) + 1e-5  # Ensure std is positive

        # clamp std to avoid numerical issues
        std = jnp.clip(std, a_min=self.std_min, a_max=self.std_max)

        return self.distribution(mean, std)
