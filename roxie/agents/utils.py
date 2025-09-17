from typing import Any, Dict, Iterable, Optional

import flax.struct as struct
import jax
import jax.numpy as jnp
import numpy as np


@struct.dataclass
class Transition:
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    terminal: jnp.ndarray
    log_probs: Optional[jnp.ndarray] = None  # Optional, used in some algorithms
    value: Optional[jnp.ndarray] = None      # Optional, used in some algorithms


# Helpers to serialize/deserialize bounds minimally
def serialize_bound(x):
    x = jax.device_get(x)
    if isinstance(x, (jnp.ndarray, np.ndarray)):
        return (
            float(x) if x.shape == () else np.asarray(x, dtype=np.float32).tolist()[0]
        )  # TODO: check if this index is correct
    if hasattr(x, "item"):
        return x.item()
    return float(x)
