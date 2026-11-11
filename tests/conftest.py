import os

# Before `import jax`, as XLA reads it at init: autotuning hangs on Blackwell
# when compiling MJX-shaped kernels.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402
from flax import nnx  # noqa: E402

# CPU-only, so the suite never takes the GPU from a training run. A config call,
# not JAX_PLATFORMS: a plugin may import jax before conftest runs.
jax.config.update("jax_platforms", "cpu")


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def rngs():
    return nnx.Rngs(params=0, dropout=1)


@pytest.fixture
def obs_dim():
    return 8


@pytest.fixture
def action_dim():
    return 3


@pytest.fixture
def batch_size():
    return 16


@pytest.fixture
def hidden_features():
    return [64, 64]
