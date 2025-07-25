from flax import nnx
from typing import Sequence, Callable
import jax.numpy as jnp

class DeterministicCritic(nnx.Module):
    features: Sequence[int]
    activation_fn: Callable = nnx.relu
    use_layer_norm: bool = False
    dropout_rate: float = 0.0

    @nnx.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray, training: bool) -> jnp.ndarray:
        # Concatenate state and action
        x = jnp.concatenate([state, action], axis=-1)
        
        for feat in self.features:
            x = nnx.Dense(feat)(x)
            if self.use_layer_norm:
                x = nnx.LayerNorm()(x)
            x = self.activation_fn(x)
            x = nnx.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
            
        x = nnx.Dense(1)(x)
        return jnp.squeeze(x, axis=-1)