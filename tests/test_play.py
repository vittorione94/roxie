"""The steps `roxie/play.py` runs before it opens a window."""

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture
def play(monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("XLA_FLAGS", "")
    from roxie import play

    return play


class _Agent:
    def __init__(self):
        self.calls = 0

    def select_action(self, obs, key, evaluate):
        self.calls += 1
        return jnp.zeros((1, 2), dtype=jnp.float32), None, None


class _State:
    obs = jnp.zeros(3, dtype=jnp.float32)


def test_warming_up_runs_a_full_step_of_the_world(play):
    stepped = []
    world = play._World(
        observe=lambda obs: obs, ghost=None, model=None, data=None, step_dt=0.01,
        reset=lambda key: _State(),
        step=lambda state, action: stepped.append(action) or state,
        env_nq=0, env_nbody=0,
    )
    agent = _Agent()
    play._warm_up(world, agent, jax.random.PRNGKey(0))
    assert agent.calls == 1
    assert stepped[0].shape == (2,)
