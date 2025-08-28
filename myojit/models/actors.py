from typing import Sequence, Callable, Optional
from flax import nnx
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
    def __init__(self):
        pass

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        pass
