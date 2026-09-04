import os

# Before `import jax`: XLA reads this at initialization. Autotuning hangs
# indefinitely on Blackwell when compiling MJX-shaped kernels, so a bare
# `pytest tests/` on the GPU stalls with no output at all.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402
from flax import nnx  # noqa: E402

# The suite is CPU-only, always. Nothing here is large enough for the GPU to
# pay for its own dispatch, and a bare `pytest tests/` must never take the
# device away from a training run that is using it -- these are the same
# workstation.
#
# `jax.config.update`, not `os.environ["JAX_PLATFORMS"]`: the env var is read
# once, at `import jax`, and conftest is not reliably the first module to
# import it (a plugin or a test module may get there first), so an environ
# write here can land too late to have any effect. The config call is applied
# whenever it runs. Unlike XLA_FLAGS above, which XLA reads lazily.
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
