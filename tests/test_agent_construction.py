"""Every agent config must actually build an agent.

`tests/test_agent_configs.py` checks the config against the constructor
*signature*; this file runs the construction train.py performs, so a block whose
`_target_` resolves but whose arguments don't (a critic missing `num_atoms`, a
buffer whose lengths don't divide, an optax kwarg that moved) fails here rather
than after the env has been built on the cluster.
"""

import dataclasses
import inspect
from pathlib import Path

import jax.numpy as jnp
import optax
import pytest
from flax import nnx
from hydra.utils import get_class
from omegaconf import OmegaConf

from roxie.agents.utils import build_agent, build_optimizer, network_rngs

REPO = Path(__file__).resolve().parent.parent

OBS, ACT = 16, 6
# PPO splits `add_batch_size * (sample_sequence_length - 1)` transitions into
# `num_minibatches`, and the shrink below caps the sequence at 8 — so this has
# to leave `ENVS * 7` divisible by every `num_minibatches` the repo's PPO
# configs name (4).
ENVS = 40

CONFIG_FILES = sorted(
    list((REPO / "roxie" / "configs" / "agent").glob("*.yaml"))
    + list(REPO.glob("experiments/*/agent/*.yaml"))
)


def _accepts(cls, name: str) -> bool:
    return any(
        name in inspect.signature(k.__dict__["__init__"]).parameters
        for k in cls.__mro__
        if "__init__" in k.__dict__
    )


def _build(agent_cfg):
    """Construct one agent exactly as train.py does, with the replay buffers
    shrunk — this is a wiring check, not a memory test."""
    # `obs_size` / `action_size` / `obs_action_size` stand in for what
    # `loader.publish_env_shapes` writes back once the real env is built; the
    # network blocks interpolate them.
    cfg = OmegaConf.create(
        {
            "env": {
                "parallel_envs": ENVS,
                "obs_size": OBS,
                "action_size": ACT,
                "obs_action_size": OBS + ACT,
            },
            "agent": agent_cfg,
        }
    ).agent
    memory = cfg.get("memory_config", None)
    if memory is not None:
        for key, value in (("max_length", 2000), ("min_length", 100),
                           ("max_length_time_axis", 64)):
            if key in memory:
                memory[key] = value
        if "sample_sequence_length" in memory:
            memory.sample_sequence_length = min(memory.sample_sequence_length, 8)

    cls = get_class(cfg._target_)
    kwargs = dict(
        env_obs_size=OBS,
        env_action_size=ACT,
        action_low=jnp.full((ACT,), -1.0, dtype=jnp.float32),
        action_high=jnp.full((ACT,), 1.0, dtype=jnp.float32),
    )
    if _accepts(cls, "noise_config"):
        kwargs["noise_config"] = OmegaConf.load(
            REPO / "roxie" / "configs" / "noise" / "gaussian.yaml"
        )
    return build_agent(cfg, **kwargs)


@pytest.mark.parametrize(
    "path", CONFIG_FILES, ids=lambda p: str(p.relative_to(REPO))
)
def test_agent_config_builds_an_agent_that_can_be_played_back(path):
    """Construction, and the round trip that construction has to support.

    One build answers both: that the blocks instantiate at all (a critic missing
    `num_atoms`, a buffer whose lengths do not divide, an optax kwarg that
    moved), and that every hyperparameter the agent accepts comes back out of
    `_export_hyperparams`.

    `Agent.load` rebuilds a checkpoint by filtering the exported block against
    `hyperparams_cls`'s fields, so a field that is accepted but never exported
    is silently dropped on playback: the agent comes back with the dataclass
    default instead of the value it trained with, and nothing warns. That
    failure is invisible at save time and only shows up as a playback that
    scores differently from the run it came from.

    `dataclasses.asdict` in the base export is what makes this hold by
    construction; the check stays because the agents that override the export
    (SAC, PPO) could still drop a key on the way past.
    """
    agent = _build(OmegaConf.load(path))
    assert agent is not None

    if agent.hyperparams_cls is None:
        # A non-learning baseline (Constant, NormalRandom, ...) replaces
        # `_export_hyperparams` and has nothing to round-trip.
        return

    exported = set(agent._export_hyperparams())
    knobs = {f.name for f in dataclasses.fields(agent.hyperparams_cls)}
    assert not (knobs - exported), (
        f"{type(agent).__name__} accepts {sorted(knobs - exported)} but never "
        "exports them; they will not survive `Agent.load`."
    )


def _params(module) -> jnp.ndarray:
    import jax

    return jnp.concatenate(
        [jnp.ravel(x) for x in jax.tree.leaves(nnx.state(module, nnx.Param))]
    )


def test_seed_controls_network_init():
    """`seed` must actually reach the parameters — otherwise a seed sweep would
    silently run the same initialization N times."""
    cfg = OmegaConf.load(REPO / "roxie" / "configs" / "agent" / "td3.yaml")

    def build(seed):
        c = cfg.copy()
        c.hyperparams.seed = seed
        return _build(c)

    a, a_again, b = build(0), build(0), build(100)
    assert jnp.allclose(_params(a.state.actor), _params(a_again.state.actor))
    assert not jnp.allclose(_params(a.state.actor), _params(b.state.actor))

    # Twin heads must not start identical: a clipped double-Q min over two
    # identical critics is just a single critic.
    assert not jnp.allclose(
        _params(a.state.critic.critic1), _params(a.state.critic.critic2)
    )


def test_default_optimizer_matches_the_previous_hardcoded_chain():
    """`seed: 0` + the default adam block must reproduce pre-refactor numerics,
    so existing sweep results stay comparable to new runs."""
    params = {"w": jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)}
    grads = {"w": jnp.array([10.0, -4.0, 0.5], dtype=jnp.float32)}

    def one_step(tx):
        return tx.update(grads, tx.init(params), params)[0]["w"]

    hardcoded = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
    from_yaml = build_optimizer(
        OmegaConf.create({"_target_": "optax.adam", "b1": 0.9, "b2": 0.999, "eps": 1e-8}),
        learning_rate=3e-4,
        max_grad_norm=1.0,
    )
    assert jnp.allclose(one_step(hardcoded), one_step(from_yaml))

    # `None` (an agent built straight from Python, with no optimizer block)
    # keeps the same behaviour.
    assert jnp.allclose(
        one_step(hardcoded),
        one_step(build_optimizer(None, learning_rate=3e-4, max_grad_norm=1.0)),
    )


def test_optimizer_block_selects_family_and_schedule():
    params = {"w": jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)}
    grads = {"w": jnp.array([10.0, -4.0, 0.5], dtype=jnp.float32)}

    def one_step(tx):
        return tx.update(grads, tx.init(params), params)[0]["w"]

    adam = build_optimizer(
        OmegaConf.create({"_target_": "optax.adam"}),
        learning_rate=3e-4, max_grad_norm=1.0,
    )
    adamw = build_optimizer(
        OmegaConf.create({"_target_": "optax.adamw", "weight_decay": 1e-2}),
        learning_rate=3e-4, max_grad_norm=1.0,
    )
    assert not jnp.allclose(one_step(adam), one_step(adamw))

    # A block that declares its own `learning_rate` wins over the agent's
    # scalar arg — that is how a schedule is passed.
    scheduled = build_optimizer(
        OmegaConf.create({
            "_target_": "optax.adam",
            "learning_rate": {
                "_target_": "optax.cosine_decay_schedule",
                "init_value": 1e-2,
                "decay_steps": 1000,
            },
        }),
        learning_rate=3e-4, max_grad_norm=1.0,
    )
    assert jnp.max(jnp.abs(one_step(scheduled))) > 1e-3  # 1e-2 lr, not 3e-4

    # Falsy max_grad_norm drops the clip entirely: plain SGD lands on -lr * grad.
    unclipped = build_optimizer(
        OmegaConf.create({"_target_": "optax.sgd"}),
        learning_rate=0.1, max_grad_norm=None,
    )
    assert jnp.allclose(one_step(unclipped), jnp.array(
        [-1.0, 0.4, -0.05], dtype=jnp.float32
    ))


def test_network_rngs_offsets_are_distinct():
    import jax

    key = lambda offset: jax.random.key_data(
        network_rngs(0, offset).params.key[...]
    )
    assert not jnp.array_equal(key(0), key(2))
    assert not jnp.array_equal(key(2), key(4))
    # The same agent seed must give the same stream — `seed` is the only input.
    assert jnp.array_equal(
        key(0), jax.random.key_data(network_rngs(0, 0).params.key[...])
    )
