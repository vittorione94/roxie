import jax
import jax.numpy as jnp
import numpy as np


# Helpers to serialize/deserialize bounds minimally
def serialize_bound(x):
    x = jax.device_get(x)
    if isinstance(x, (jnp.ndarray, np.ndarray)):
        return float(x) if x.shape == () else np.asarray(x, dtype=np.float32).tolist()
    if hasattr(x, "item"):
        return x.item()
    return float(x)

def deserialize_bound(x):
    # Accept scalar or list -> jnp.array
    return jnp.asarray(x, dtype=jnp.float32)
