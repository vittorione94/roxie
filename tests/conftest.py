import os

# Before `import jax`: XLA reads this at initialization. Autotuning hangs
# indefinitely on Blackwell (RTX 5080) when compiling MJX-shaped kernels, and a
# bare `pytest tests/` on the GPU stalls with no output at all rather than
# failing — the same reason `train.py` and `play.py` set it. `setdefault`, so an
# explicit XLA_FLAGS in the environment still wins.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402
from flax import nnx  # noqa: E402


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
