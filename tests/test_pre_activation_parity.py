"""`pre_activation_coef` has to be live on every deterministic arm.

experiments/dmc/agent/ddpg_bench.yaml calls the four deterministic arms "a clean
algorithm-only A/B" on the strength of them carrying the same
`pre_activation_coef`. They did carry the same value — but only TD3's actor loss
took the argument, so DDPG, D4PG and TD4 ran at an effective 0 while TD3 paid a
0.1 penalty nothing else paid. On AcrobotSwingup that was worth roughly 2x in
score, and nothing in the run reported it: the value round-tripped through the
config, the checkpoint and the hyperparameter log either way.

So the invariant is tested by DIFFERENTIATION, not by reading the configs: run
the same burst at two coefficients and require the actor to have moved
differently.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from omegaconf import OmegaConf

from roxie.agents.d4pg import D4PG
from roxie.agents.ddpg import DDPG
from roxie.agents.td3 import TD3
from roxie.agents.td4 import TD4

OBS, ACT, ENVS = 6, 3, 8

# The hinge is at |u| = 1 and the benchmark's actor starts at
# `output_init_scale` 0.01, so an untouched actor emits |u| ~ 0.1 and the
# penalty is identically zero — a test that skipped this would pass on an agent
# that ignores the knob entirely.
SATURATED_BIAS = 3.0

CATEGORICAL = dict(v_min=-5.0, v_max=120.0, num_atoms=51)

AGENTS = [
    ("ddpg", DDPG, "roxie.models.critics.QCritic", {}),
    ("td3", TD3, "roxie.models.critics.QCritic", {"policy_delay": 2}),
    ("d4pg", D4PG, "roxie.models.critics.DistributionalQCritic", CATEGORICAL),
    ("td4", TD4, "roxie.models.critics.DistributionalQCritic",
     {**CATEGORICAL, "policy_delay": 2}),
]


def _cfg(mapping):
    return OmegaConf.create(mapping)


def _build(cls, critic_target, coef, extra):
    agent = cls(
        env_obs_size=OBS,
        env_action_size=ACT,
        action_low=-jnp.ones(ACT),
        action_high=jnp.ones(ACT),
        actor_config=_cfg({
            "_target_": "roxie.models.actors.DeterministicActor",
            "features": [32, 32], "use_layer_norm": True,
            "output_init_scale": 0.01,
        }),
        critic_config=_cfg({
            "_target_": critic_target, "features": [32, 32],
            "use_layer_norm": True,
        }),
        memory_config=_cfg({
            "_target_": "flashbax.buffers.make_flat_buffer",
            "max_length": 2048, "min_length": 64, "sample_batch_size": 32,
            "add_sequences": False, "add_batch_size": ENVS,
        }),
        noise_config=_cfg({
            "_target_": "roxie.exploration.noisy.GaussianNoise",
            "initial_noise_scale": 0.1,
        }),
        learning_steps=4,
        pre_activation_coef=coef,
        **extra,
    )
    for module in (agent.state.actor, agent.state.target_actor):
        module.output_layer.bias.value = jnp.full((ACT,), SATURATED_BIAS)

    rng = np.random.default_rng(0)
    for _ in range(40):
        agent.add_transitions(
            jnp.asarray(rng.standard_normal((ENVS, OBS)), jnp.float32),
            jnp.asarray(rng.standard_normal((ENVS, ACT)), jnp.float32),
            jnp.asarray(rng.standard_normal(ENVS), jnp.float32),
            jnp.zeros(ENVS, jnp.bool_),
            jnp.zeros(ENVS, jnp.bool_),
            jnp.asarray(rng.standard_normal((ENVS, OBS)), jnp.float32),
        )
    for i in range(8):
        agent.learn(jax.random.PRNGKey(100 + i))
    return jax.tree.leaves(nnx.state(agent.state.actor, nnx.Param))


@pytest.mark.parametrize("name,cls,critic_target,extra", AGENTS)
def test_the_saturation_penalty_reaches_the_actor(name, cls, critic_target, extra):
    off = _build(cls, critic_target, 0.0, extra)
    on = _build(cls, critic_target, 100.0, extra)
    moved = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(off, on))
    assert moved > 1e-6, (
        f"{name}: pre_activation_coef 0 and 100 trained the actor to the same "
        f"weights, so the knob is dead for this agent"
    )
