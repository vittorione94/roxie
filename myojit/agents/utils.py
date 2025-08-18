import jax
import jax.numpy as jnp
from flax import nnx
from typing import Any, Dict, Iterable
import numbers
import numpy as np

# Helpers to serialize/deserialize bounds minimally
def serialize_bound(x):
    x = jax.device_get(x)
    if isinstance(x, (jnp.ndarray, np.ndarray)):
        return float(x) if x.shape == () else np.asarray(x, dtype=np.float32).tolist()[0] # TODO: check if this index is correct
    if hasattr(x, "item"):
        return x.item()
    return float(x)

def deserialize_bound(x):
    # Accept scalar or list -> jnp.array
    return jnp.asarray(x, dtype=jnp.float32)
