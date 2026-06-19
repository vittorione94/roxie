import jax
import jax.numpy as jnp
import pytest
from flax import nnx


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
