from typing import Sequence, Callable
from flax.experimental import nnx
import jax.numpy as jnp

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
    self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
    self.output_layer = nnx.Linear(current_features, action_dim, rngs=rngs)

  def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
    # Use the layers defined in __init__
    for i, layer in enumerate(self.hidden_layers):
        x = layer(x)
        if self.use_layer_norm:
            x = self.norm_layers[i](x)
        x = self.activation_fn(x)
        x = self.dropout(x, deterministic=not training)
    
    x = self.output_layer(x)
    # Scale output to action space range, e.g., [-1, 1]
    x = nnx.tanh(x) 
    return x
    

# --- Actor for SAC/PPO ---
# class StochasticActor(nnx.Module):
#     features: Sequence[int]
#     action_dim: int
#     log_std_min: Optional[float] = -20
#     log_std_max: Optional[float] = 2

#     @nnx.compact
#     def __call__(self, state: jnp.ndarray) -> distrax.Distribution:
#         x = state
#         for feat in self.features:
#             x = nnx.Dense(feat)(x)
#             x = nnx.relu(x)
        
#         # Output parameters for a Gaussian distribution
#         # One output for the mean, one for the log_std
#         mean = nnx.Dense(self.action_dim, name="mean")(x)
#         log_std = nnx.Dense(self.action_dim, name="log_std")(x)

#         # Clamp log_std for numerical stability
#         log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)

#         # Create and return a Tanh-squashed Gaussian distribution
#         return distrax.Transformed(
#             distrax.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std)),
#             distrax.Block(distrax.Tanh(), ndims=1)
#         )