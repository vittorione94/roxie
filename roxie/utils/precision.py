"""Floating-point precision configuration and dtype casting utilities."""

import os

import jax.numpy as jnp
import numpy as np

# Global canonical float dtypes for learner computations.
FLOAT = jnp.float32
NP_FLOAT = np.float32

# Flags enabling oneDNN custom calls for CPU bfloat16 performance.
ONEDNN_BF16_FLAGS = (
    "--xla_cpu_experimental_onednn_custom_call "
    "--xla_cpu_experimental_onednn_fusion_type=dot"
)


def matmul_dtype() -> jnp.dtype:
    """Determines the floating-point precision for hidden linear layer matmuls.

    Reads `ROXIE_MATMUL_DTYPE` from the environment. Returns `jnp.bfloat16` 
    when set to 'bf16' or 'bfloat16', and defaults to `FLOAT` (`jnp.float32`).

    Returns:
        The JAX dtype for matrix multiplication operations.
    """
    name = os.environ.get("ROXIE_MATMUL_DTYPE", "").strip().lower()
    if name in ("bf16", "bfloat16"):
        return jnp.bfloat16
    return FLOAT


def linear_kwargs() -> dict:
    """Generates `dtype` and `param_dtype` arguments for hidden `nnx.Linear` layers.

    Returns:
        A dictionary containing `dtype` and `param_dtype` keyword arguments.
    """
    return {"dtype": matmul_dtype(), "param_dtype": FLOAT}


def as_float(x) -> jnp.ndarray:
    """Casts input array to the primary learner precision (`FLOAT`).

    Args:
        x: Array-like input.

    Returns:
        A JAX array cast to `jnp.float32`.
    """
    return jnp.asarray(x, dtype=FLOAT)