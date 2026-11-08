"""Every agent must learn from pixels when the config gives its networks eyes.

Vision is a property of the networks alone — the observation stays one flat
vector through the env, the replay buffer, the running statistics and the
losses, with the image flattened at its front — so what has to hold is narrow
but has to hold everywhere: the encoder reads the right slice, every network
class wires it in front of its MLP, every shipped agent config reaches it from
the `vision` group, and gradients from every agent's losses land in it. The
last test drives a pixel env through the real trainer, which is where a
flat-observation assumption anywhere else would surface.
"""

import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx, struct
from omegaconf import OmegaConf

from roxie.environment import functional
from roxie.environment.loader import (
    PlaygroundFuncEnv,
    load_playground_env,
    publish_env_shapes,
)
from roxie.environment.vector import JaxVectorEnv
from roxie.models.actors import DeterministicActor, StochasticActor
from roxie.models.critics import DistributionalQCritic, QCritic, VCritic
from roxie.models.vision import PixelEncoder
from tests import harness

REPO = Path(__file__).resolve().parent.parent

# Small enough to compile in a blink, big enough that both convs and the
# proprioceptive tail are real: 12 -> 5 -> 3 spatially.
PIXELS = (12, 12, 2)
PROPRIO = 4
OBS = math.prod(PIXELS) + PROPRIO
VISION = {
    "pixel_shape": list(PIXELS),
    "channels": [4, 4],
    "kernel_sizes": [3, 3],
    "strides": [2, 1],
    "latent_dim": 8,
}


def _encoder(**overrides):
    return PixelEncoder(**{**VISION, **overrides}, rngs=nnx.Rngs(0))


def _obs(key, lead=(16,), width=OBS):
    return jax.random.normal(key, lead + (width,), dtype=jnp.float32)


# The encoder


class TestPixelEncoder:
    def test_replaces_the_image_with_a_bounded_latent(self, rng_key):
        encoder = _encoder()
        x = _obs(rng_key)
        out = encoder(x)

        assert out.shape == (16, VISION["latent_dim"] + PROPRIO)
        assert encoder.out_features(OBS) == out.shape[-1]
        latent = out[:, : VISION["latent_dim"]]
        assert jnp.all(jnp.abs(latent) <= 1.0), "the latent is tanh-bounded"
        np.testing.assert_array_equal(out[:, VISION["latent_dim"]:], x[:, -PROPRIO:])

    def test_the_latent_reads_the_leading_block_and_nothing_else(self, rng_key):
        """The image is the FRONT of the flat observation. Moving the tail must
        leave the latent alone; moving one pixel must not."""
        encoder = _encoder()
        x = _obs(rng_key)
        latent = lambda obs: encoder(obs)[:, : VISION["latent_dim"]]  # noqa: E731

        np.testing.assert_array_equal(
            latent(x), latent(x.at[:, -PROPRIO:].add(100.0))
        )
        assert not np.allclose(latent(x), latent(x.at[:, 0].add(10.0)))

    def test_any_leading_axes_encode_like_a_flat_batch(self, rng_key):
        """MPO tiles observations to `(samples, batch, D)`; the eval path feeds
        a single unbatched one. Both must match the flat batch, row for row."""
        encoder = _encoder()
        x = _obs(rng_key, lead=(3, 5))
        flat = encoder(x.reshape(15, OBS))

        np.testing.assert_allclose(
            encoder(x).reshape(15, -1), flat, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(encoder(x[1, 2]), flat[7], rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize(
        "overrides,match",
        [
            ({"pixel_shape": None}, "publishes no `pixel_shape`"),
            ({"pixel_shape": [12, 12]}, r"\(H, W, C\)"),
            ({"strides": [2]}, "one entry per conv layer"),
            ({"kernel_sizes": [7, 7]}, "shrinks"),
        ],
    )
    def test_refuses_a_geometry_it_cannot_build(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            _encoder(**overrides)

    def test_refuses_an_input_too_narrow_to_hold_the_image(self):
        with pytest.raises(ValueError, match="cannot hold"):
            _encoder().out_features(math.prod(PIXELS) - 1)


# The network classes


def _networks(vision):
    """One of every network class an agent config can name, all on one input."""
    rngs = lambda: nnx.Rngs(params=0, dropout=1)  # noqa: E731
    act = harness.ACT
    return {
        "DeterministicActor": DeterministicActor(
            OBS, [16], act, rngs=rngs(), vision=vision
        ),
        "StochasticActor": StochasticActor(OBS, [16], act, rngs=rngs(), vision=vision),
        "QCritic": QCritic(OBS + act, [16], rngs=rngs(), vision=vision),
        "DistributionalQCritic": DistributionalQCritic(
            OBS + act, [16], -10.0, 10.0, 11, rngs=rngs(), vision=vision
        ),
        "VCritic": VCritic(OBS, [16], rngs=rngs(), vision=vision),
    }


def _scalar_output(name, network, obs, actions):
    if name == "StochasticActor":
        return jnp.sum(network(obs).loc)
    if name in ("QCritic", "DistributionalQCritic"):
        return jnp.sum(network(obs, actions))
    return jnp.sum(network(obs))


@pytest.mark.parametrize("name", list(_networks(None)))
def test_every_network_trains_its_encoder_end_to_end(name, rng_key):
    """Gradient from the network's own output must reach the first conv kernel:
    an encoder that is built but bypassed, or detached, fails here."""
    network = _networks(VISION)[name]
    obs_key, act_key = jax.random.split(rng_key)
    obs = _obs(obs_key)
    actions = jax.random.uniform(act_key, (16, harness.ACT), jnp.float32, -1, 1)

    grads = nnx.grad(
        lambda net: _scalar_output(name, net, obs, actions)
    )(network)
    kernel_grad = grads.encoder.conv_layers[0].kernel[...]
    assert jnp.all(jnp.isfinite(kernel_grad))
    assert jnp.any(kernel_grad != 0.0)


@pytest.mark.parametrize("name", list(_networks(None)))
def test_without_vision_a_network_is_a_plain_mlp(name):
    """`vision: null` is every shipped config's default, and must build exactly
    the network that existed before vision did: no encoder, no params for one."""
    plain = _networks(None)[name]
    assert plain.encoder is None
    params = nnx.state(plain, nnx.Param)
    assert "encoder" not in params


# Every agent


def _agent_networks(agent):
    """The actor and every critic head, online and target, that the agent owns."""
    state = agent.state
    out = {"actor": state.actor, "target_actor": state.target_actor}
    for prefix, critic in (("critic", state.critic),
                           ("target_critic", state.target_critic)):
        if critic is None:
            continue
        if hasattr(critic, "critic1"):
            out[f"{prefix}1"] = critic.critic1
            out[f"{prefix}2"] = critic.critic2
        else:
            out[prefix] = critic
    return {k: v for k, v in out.items() if v is not None}


@pytest.fixture(scope="module", params=harness.AGENTS)
def pixel_agent(request):
    """A vision agent built from its shipped yaml, warmed on pixel observations,
    with its untrained encoders snapshotted — built once per agent."""
    name = request.param
    agent = harness.build(name, vision=VISION, obs_size=OBS)
    obs = _obs(jax.random.PRNGKey(1), lead=(harness.ENVS,))
    steps = harness.drive(agent, harness.WARMUP_ITERS, jax.random.PRNGKey(0), obs)
    before = {
        role: harness.snapshot(net.encoder)
        for role, net in _agent_networks(agent).items()
    }
    return name, agent, steps, before


def test_every_agent_network_gets_its_own_encoder(pixel_agent):
    """Every actor and critic head the shipped config names — twin heads and
    target copies included — reaches the `vision` group."""
    name, agent, _, _ = pixel_agent
    for role, net in _agent_networks(agent).items():
        assert isinstance(net.encoder, PixelEncoder), f"{name}: {role} is blind"
        assert net.encoder.pixel_shape == PIXELS


def test_every_agent_learns_its_encoders_from_pixels(pixel_agent):
    """One learning pass: finite losses, and every ONLINE encoder moved. The
    targets are not asserted on: a hard-copied target is free to lag."""
    name, agent, steps, before = pixel_agent
    _, actor_loss, critic_loss = agent.learn(jax.random.PRNGKey(2))
    assert np.isfinite(float(actor_loss)), name
    assert np.isfinite(float(critic_loss)), name

    for role, net in _agent_networks(agent).items():
        if role.startswith("target"):
            continue
        assert harness.moved(before[role], net.encoder), (
            f"{name}: the {role} encoder did not train"
        )

    action = agent.eval_action(
        _obs(jax.random.PRNGKey(3), lead=(harness.ENVS,)),
        agent.state.actor, agent.state.obs_stats,
    )
    assert action.shape == (harness.ENVS, harness.ACT)
    assert jnp.all(jnp.isfinite(action))


# The configs


AGENT_CONFIGS = sorted(
    p
    for p in list((REPO / "roxie" / "configs" / "agent").glob("*.yaml"))
    + list(REPO.glob("experiments/*/agent/*.yaml"))
    if "actor_config" in OmegaConf.load(p)
)


def _network_blocks(node, path=""):
    """Every block in a resolved agent config that builds an MLP network."""
    if isinstance(node, dict):
        if "features" in node:
            yield path, node
        for key, value in node.items():
            yield from _network_blocks(value, f"{path}.{key}")


def test_agent_configs_were_found():
    """A bad glob would make the test below vacuously pass."""
    assert len(AGENT_CONFIGS) >= 14


@pytest.mark.parametrize(
    "path", AGENT_CONFIGS, ids=lambda p: str(p.relative_to(REPO))
)
def test_every_network_block_reads_the_vision_group(path):
    """A network block that does not read `vision` would leave that network
    blind under `/vision: cnn` while the rest of the agent sees — silently."""
    root = OmegaConf.create({
        "env": {"obs_size": OBS, "action_size": 3, "obs_action_size": OBS + 3,
                "parallel_envs": 2},
        "agent": OmegaConf.load(path),
    })
    blind = OmegaConf.to_container(root.agent, resolve=True)
    root.vision = VISION
    seeing = OmegaConf.to_container(root.agent, resolve=True)

    blocks = list(_network_blocks(seeing))
    assert blocks, f"{path} names no network"
    for where, block in blocks:
        assert block.get("vision") == VISION, f"{path}{where} ignores `vision`"
    for where, block in _network_blocks(blind):
        assert block.get("vision") is None, f"{path}{where} sees without `vision`"


def test_the_cnn_group_is_the_encoder_call():
    """`vision/cnn.yaml` IS the `PixelEncoder` call, the same contract
    `test_agent_configs.py` holds the agent configs to."""
    cfg = OmegaConf.load(REPO / "roxie" / "configs" / "vision" / "cnn.yaml")
    params = inspect.signature(PixelEncoder.__init__).parameters
    expected = {
        name for name, p in params.items()
        if name != "self" and p.kind is not p.KEYWORD_ONLY
    }
    assert set(cfg.keys()) == expected

    root = OmegaConf.create({"env": {"pixel_shape": [64, 64, 3]}, "vision": cfg})
    encoder = PixelEncoder(
        **OmegaConf.to_container(root.vision, resolve=True), rngs=nnx.Rngs(0)
    )
    assert encoder(jnp.zeros((2, 64 * 64 * 3))).shape == (2, cfg.latent_dim)


def test_the_pixel_launchable_composes():
    """`experiments/vision/cartpole_sac.yaml` must compose through Hydra — the
    `/vision` group reachable from a launchable in a subdirectory — and once
    the env has published its pixel shape, every network must resolve to it."""
    from hydra import compose, initialize_config_dir

    from roxie.utils import hydra_searchpath

    hydra_searchpath.register()
    with initialize_config_dir(
        config_dir=str(REPO / "roxie" / "configs"), version_base=None
    ):
        cfg = compose(
            config_name="vision/cartpole_sac", overrides=["vision.latent_dim=7"]
        )

    assert cfg.env.vision is True
    publish_env_shapes(cfg.env, OBS, 3, pixel_shape=PIXELS)
    resolved = OmegaConf.to_container(cfg.agent, resolve=True)
    blocks = list(_network_blocks(resolved))
    assert len(blocks) == 3
    for _, block in blocks:
        assert block["vision"]["pixel_shape"] == list(PIXELS)
        assert block["vision"]["latent_dim"] == 7


# The env side


class _DictObsEnv:
    """The surface of a mujoco_playground env whose observation is a dict, as a
    vision env's is."""

    observation_size = {"state": (3,), "pixels/view_0": (4, 4, 2)}
    mj_model = SimpleNamespace(actuator_ctrlrange=np.array([[-1.0, 1.0]]))


def test_a_dict_observation_flattens_with_the_image_first():
    func_env = PlaygroundFuncEnv(_DictObsEnv(), "warp")
    assert func_env.pixel_shape == (4, 4, 2)
    assert func_env.observation_space.shape == (4 * 4 * 2 + 3,)

    pixels = jnp.arange(32, dtype=jnp.float32).reshape(4, 4, 2)
    state = SimpleNamespace(obs={"state": jnp.array([-1.0, -2.0, -3.0]),
                                 "pixels/view_0": pixels})
    obs = func_env.observation(state, None)
    np.testing.assert_array_equal(obs[:32], pixels.reshape(-1))
    np.testing.assert_array_equal(obs[32:], [-1.0, -2.0, -3.0])

    assert JaxVectorEnv(func_env, 2).pixel_shape == (4, 4, 2)


class _RenderingEnv(_DictObsEnv):
    """A warp render context renders all its worlds on every call, so the
    unbatched trace behind `observation_size` sees every world's image at
    once; a reset vmapped over the worlds gives one each."""

    observation_size = {"state": (3,), "pixels/view_0": (3, 4, 4, 2)}

    @struct.dataclass
    class State:
        obs: dict

    def reset(self, rng):
        return self.State(obs={
            "state": jnp.zeros(3, jnp.float32),
            "pixels/view_0": jnp.zeros((4, 4, 2), jnp.float32),
        })


def test_a_rendering_env_is_sized_per_world():
    """Read off `observation_size`, the image would carry the world axis and
    the encoder would refuse its 4-D `pixel_shape`."""
    func_env = PlaygroundFuncEnv(_RenderingEnv(), "warp", nworld=3)
    assert func_env.pixel_shape == (4, 4, 2)
    assert func_env.observation_space.shape == (4 * 4 * 2 + 3,)


@pytest.mark.parametrize("extra, match", [
    ({"pixels/view_1": (4, 4, 2)}, "exactly one"),
    # An asymmetric task's critic-only group must not reach the actor.
    ({"privileged_state": (5,)}, "privileged_state"),
])
def test_a_dict_observation_beyond_one_camera_and_state_is_refused(extra, match):
    env = _DictObsEnv()
    env.observation_size = {**env.observation_size, **extra}
    with pytest.raises(ValueError, match=match):
        PlaygroundFuncEnv(env, "warp")


def test_envs_without_pixels_publish_none():
    assert functional.FuncEnv.pixel_shape is None
    cfg = OmegaConf.create({})
    publish_env_shapes(cfg, 8, 3)
    assert cfg.pixel_shape is None
    publish_env_shapes(cfg, OBS, 3, pixel_shape=PIXELS)
    assert cfg.pixel_shape == list(PIXELS)


def test_playground_vision_needs_the_warp_renderer():
    """mujoco_playground renders through mujoco_warp; on any other impl the
    failure would otherwise surface only when the first reset is traced."""
    with pytest.raises(ValueError, match="impl=warp"):
        load_playground_env("CartpoleBalance", impl="jax", vision=True)


def test_a_vision_flag_without_a_render_config_is_refused():
    with pytest.raises(ValueError, match="no vision mode"):
        load_playground_env("HumanoidWalk", impl="warp", vision=True)


# End to end


class _DotTrack(functional.FuncEnv):
    """A dot the agent steers to the centre of an 8x8 image, seen ONLY as
    pixels: the observation is the image and nothing else."""

    pixel_shape = (8, 8, 1)
    observation_space = functional.unbounded_box(64)
    action_space = functional.box(-1.0, 1.0, (harness.ACT,))

    def initial(self, rng, params=None):
        return jax.random.uniform(rng, (), jnp.float32, -1.0, 1.0)

    def transition(self, state, action, rng, params=None):
        return jnp.clip(state + 0.25 * action[0], -1.0, 1.0)

    def observation(self, state, rng, params=None):
        cols = jnp.arange(8, dtype=jnp.float32)
        row = jnp.exp(-jnp.square(cols - 3.5 * (state + 1.0)))
        return jnp.broadcast_to(row, (8, 8)).reshape(-1)

    def reward(self, state, action, next_state, rng, params=None):
        return -jnp.abs(next_state)

    def terminal(self, state, rng, params=None):
        return jnp.bool_(False)


DOT_VISION = {
    "pixel_shape": list(_DotTrack.pixel_shape),
    "channels": [4],
    "kernel_sizes": [3],
    "strides": [2],
    "latent_dim": 8,
}


# Between them these three build all five network classes, and run both the
# replay-driven and the on-policy update paths.
@pytest.mark.parametrize("name", ["td4", "sac", "ppo"])
def test_a_pixel_env_trains_saves_and_reloads(name, tmp_path):
    """Pixels through the real `Trainer`: fused collect, replay, observation
    statistics, learning passes, the held-out eval and a checkpoint, then a
    rebuild from that checkpoint that acts exactly like the trained agent."""
    from roxie.utils import logger
    from roxie.utils.checkpoint import find_checkpoint
    from roxie.utils.trainer import Trainer

    func_env = _DotTrack()
    env = JaxVectorEnv(func_env, harness.ENVS, max_episode_steps=8)
    test_env = JaxVectorEnv(func_env, 2, max_episode_steps=8)

    # Learning starts once the trajectory buffers TD4 and PPO use are deep
    # enough to sample, as `harness.WARMUP_ITERS` has it.
    sizes = dict(vision=DOT_VISION, obs_size=64,
                 memory_warmup=harness.WARMUP_ITERS * harness.ENVS)
    agent = harness.build(name, **sizes)
    before = harness.snapshot(agent.state.actor.encoder)

    logger.initialize(path=str(tmp_path), backends=[])
    trainer = Trainer(
        output_dir=str(tmp_path), steps=20 * harness.ENVS,
        epoch_steps=10 * harness.ENVS, save_steps=20 * harness.ENVS,
        test_episodes=2, show_progress=False,
    )
    try:
        trainer.run(agent, env, test_env, harness.ENVS, nnx.Rngs(envs=0, agent=3))
    finally:
        logger.close()

    assert harness.moved(before, agent.state.actor.encoder), (
        f"{name}: the trainer never trained the actor's encoder"
    )

    cfg = harness.agent_config(name, **sizes)
    config_blocks = {k: v for k, v in cfg.items() if k.endswith("_config")}
    if name in harness.DETERMINISTIC:
        config_blocks["noise_config"] = OmegaConf.load(
            harness.CONFIGS / "noise" / "gaussian.yaml"
        )
    reloaded = type(agent).load(
        find_checkpoint(tmp_path), 64, harness.ACT, **config_blocks
    )
    obs = jax.vmap(func_env.observation, in_axes=(0, None))(
        jnp.linspace(-1.0, 1.0, harness.ENVS, dtype=jnp.float32), None
    )
    np.testing.assert_allclose(
        reloaded.eval_action(obs, reloaded.state.actor, reloaded.state.obs_stats),
        agent.eval_action(obs, agent.state.actor, agent.state.obs_stats),
        rtol=1e-5, atol=1e-6,
    )
