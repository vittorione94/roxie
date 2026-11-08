"""A humanoid crossing generated terrain through a head-mounted RGB-D camera.

On flat ground a camera tells the policy nothing its joint readings do not, so
the floor here is a heightfield of rolling hills, rough patches, flat ground
and raised blocks (`TerrainConfig`). Every world shares the one field, so the
variety between episodes comes from starting each at a random spot and heading
on it. Head height, in the observation and the stand reward alike, is measured
from the ground under the head rather than from z=0, which would otherwise pay
for standing on a hill.

mujoco_playground's `Humanoid` refuses `vision=True`, so this subclass renders
its own pixels through mujoco_warp: the `egocentric` camera fixed to the head
body, colour and depth stacked into one `(H, W, 4)` image. The observation is a
dict, `pixels/view_0` plus the task's 67-entry proprioceptive `state`, which
`PlaygroundFuncEnv` flattens image-first into the layout
`roxie.models.vision.PixelEncoder` reads.

`build_humanoid_rgbd_env` is the builder an `env._target_` names; it lives here,
outside `roxie/`, the way a builder in another repo would.
`experiments/vision/humanoid_rgbd_sac.yaml` is the launchable.
"""

import copy
import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import humanoid
from scipy import ndimage

from roxie.environment.loader import (
    DEFAULT_WARP_GRAPH_MODE,
    EnvBundle,
    PlaygroundFuncEnv,
    put_warp_model,
)
from roxie.environment.vector import JaxVectorEnv
from roxie.utils import precision

_HIDDEN_GROUP = 5
# The field sits this far below the floor plane, so ground the terrain leaves
# at zero is the plane's alone: coplanar surfaces would double every contact
# there and fight over which one the viewer draws.
_TERRAIN_SINK = 0.005
# Half-width of the square under the feet that a start is lifted clear of.
_FOOTPRINT = 0.4

MOVE_SPEEDS = {
    "stand": 0.0,
    "walk": humanoid.WALK_SPEED,
    "run": humanoid.RUN_SPEED,
}


@dataclasses.dataclass(frozen=True)
class TerrainConfig:
    """The square field the humanoid walks on; lengths are in metres.

    Attributes:
        size: side of the field. Flat floor continues beyond it.
        cell: grid spacing of the heightfield, rounded so a whole number of
          cells spans `size`.
        hill_height: height of the rolling hills at their tallest.
        hill_length: feature size of the hills.
        roughness: height of the rough bumps at their tallest.
        bump_length: feature size of the bumps.
        region_length: feature size of the masks that split the field into
          flat, hilly and rough ground.
        block_height: how far a block's flat top stands above the highest
          ground it covers, at most.
        block_density: blocks per 100 square metres.
        spawn_margin: no episode starts closer than this to the edge.
        seed: the field's layout; every world shares one field.
    """

    size: float = 64.0
    # warp keeps the first 50 terrain triangles a geom touches and drops the
    # rest; at 0.1 m a falling humanoid overran that 5 times in 51,200
    # random-action world-steps, at 0.15 m never, at the same step time.
    cell: float = 0.15
    hill_height: float = 1.0
    hill_length: float = 8.0
    roughness: float = 0.12
    bump_length: float = 0.75
    region_length: float = 12.0
    block_height: float = 0.25
    block_density: float = 1.0
    spawn_margin: float = 8.0
    seed: int = 0


def generate_terrain(terrain: TerrainConfig) -> np.ndarray:
    """Heights of the field's grid, `[row (y), col (x)]`, all at least zero."""
    n = round(terrain.size / terrain.cell) + 1
    rng = np.random.default_rng(terrain.seed)

    def noise(length: float) -> np.ndarray:
        field = ndimage.gaussian_filter(
            rng.standard_normal((n, n)), length / terrain.cell / 4, mode="wrap",
        )
        return field / field.std()

    def unit(length: float) -> np.ndarray:
        return np.clip((noise(length) + 2) / 4, 0.0, 1.0)

    def region() -> np.ndarray:
        return np.clip(2 * noise(terrain.region_length), 0.0, 1.0)

    heights = (
        terrain.hill_height * region() * unit(terrain.hill_length)
        + terrain.roughness * region() * unit(terrain.bump_length)
    )

    n_blocks = round(terrain.block_density * terrain.size**2 / 100)
    for _ in range(n_blocks):
        w, h = (rng.uniform(0.6, 2.5, size=2) / terrain.cell).astype(int)
        r, c = rng.integers(0, n - max(w, h), size=2)
        top = heights[r:r + h, c:c + w].max()
        heights[r:r + h, c:c + w] = (
            top + terrain.block_height * rng.uniform(0.4, 1.0)
        )

    edge = np.clip(
        np.minimum(np.arange(n), np.arange(n)[::-1]) * terrain.cell / 2.0,
        0.0, 1.0,
    )
    return heights * np.outer(edge, edge)


def add_terrain(spec: mujoco.MjSpec, heights: np.ndarray, size: float) -> None:
    """Adds `heights` to `spec` as a heightfield centred on the origin, and a sun."""
    top = max(float(heights.max()), 1e-3)
    nrow, ncol = heights.shape
    spec.add_hfield(
        name="terrain",
        size=[size / 2, size / 2, top, 0.1],
        nrow=nrow,
        ncol=ncol,
        userdata=(heights / top).ravel().tolist(),
    )
    spec.worldbody.add_geom(
        name="terrain",
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname="terrain",
        pos=[0.0, 0.0, -_TERRAIN_SINK],
        material="terrain",
        conaffinity=1,
    )
    # Relief only reaches the colour channels as shading. dm_control's navy
    # floor is too dark to shade visibly, and the humanoid's own light is a
    # spot 2 m over it that leaves everything past arm's reach unlit, so the
    # field gets a light checker (1 m squares, for scale) and a low sun.
    spec.add_texture(
        name="terrain",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.46, 0.43, 0.35],
        rgb2=[0.56, 0.53, 0.43],
        mark=mujoco.mjtMark.mjMARK_EDGE,
        markrgb=[0.36, 0.34, 0.28],
        width=300,
        height=300,
    )
    material = spec.add_material(name="terrain", texrepeat=[1, 1], texuniform=True)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "terrain"
    spec.worldbody.add_light(
        name="sun",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        dir=[0.6, 0.4, -0.7],
        diffuse=[0.7, 0.7, 0.7],
        castshadow=False,
    )


def _rotate(vec: jax.Array, quat: jax.Array) -> jax.Array:
    """Rotates `vec` by the unit quaternion `quat`, MuJoCo's `(w, x, y, z)`."""
    w, u = quat[..., :1], quat[..., 1:]
    t = 2 * jnp.cross(u, vec)
    return vec + w * t + jnp.cross(u, t)


def default_config() -> config_dict.ConfigDict:
    """The playground humanoid's config plus the camera's."""
    config = humanoid.default_config()
    config.camera = "egocentric"
    config.hidden_geoms = ("head",)
    config.render_config = config_dict.create(
        nworld=1,
        cam_res=(64, 64),
        use_textures=True,
        use_shadows=False,
        enabled_geom_groups=[0, 1, 2],
    )
    config.depth_scale = 10.0
    config.proprio = True
    config.reset_qvel_scale = 0.01
    config.graph_mode = DEFAULT_WARP_GRAPH_MODE
    return config


class HumanoidRGBD(humanoid.Humanoid):
    """A playground humanoid observing RGB-D frames from one of its cameras."""

    def __init__(
        self,
        move_speed: float,
        config: config_dict.ConfigDict = default_config(),
        terrain: TerrainConfig | None = TerrainConfig(),
    ):
        super().__init__(move_speed, config)
        self._terrain = terrain
        if terrain is not None:
            heights = generate_terrain(terrain)
            spec = mujoco.MjSpec.from_file(humanoid._XML_PATH.as_posix())
            add_terrain(spec, heights, terrain.size)
            self._mj_model = spec.compile()
            self._mj_model.opt.timestep = self.sim_dt
            self._post_init()
            self._heights = jnp.asarray(heights, dtype=precision.FLOAT)
        # A camera with a resolution is one MuJoCo's viewer draws a frustum for
        # (`play.py --vision`), out to `vis.scale.frustum`; physics reads neither.
        cam_id = self._mj_model.camera(self._config.camera).id
        self._mj_model.cam_resolution[cam_id] = self._config.render_config.cam_res
        self._mj_model.vis.scale.frustum = self._config.depth_scale
        # mjx's default graph mode recaptures every step and leaks host memory;
        # see `loader.DEFAULT_WARP_GRAPH_MODE`.
        put_warp_model(self, self._config.graph_mode)

        tracking = (mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
                    mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM)
        if np.isin(self._mj_model.light_mode, tracking).any():
            raise NotImplementedError("lights that target a body are not posed")
        self._lights = {
            name: jnp.asarray(getattr(self._mj_model, f"light_{name}"), dtype=precision.FLOAT)
            for name in ("pos", "dir", "pos0", "poscom0", "dir0")
        }

        # The egocentric camera sits exactly on the head sphere's surface and
        # warp's ray tracer has no near plane, so whether a world sees out or
        # sees the inside of its own head is float rounding. The context picks
        # its geoms by group when it is built, so hiding them in a copy of the
        # model leaves the physics alone.
        render_model = copy.copy(self._mj_model)
        for name in self._config.hidden_geoms:
            render_model.geom_group[render_model.geom(name).id] = _HIDDEN_GROUP
        cam_active = [i == cam_id for i in range(render_model.ncam)]
        render_kwargs = self._config.render_config.to_dict()
        if _HIDDEN_GROUP in render_kwargs["enabled_geom_groups"]:
            raise ValueError(
                f"geom group {_HIDDEN_GROUP} holds the hidden geoms; leave it "
                f"out of enabled_geom_groups"
            )
        self._rc = mjx.create_render_context(
            mjm=render_model,
            cam_active=cam_active,
            render_rgb=True,
            render_depth=True,
            **render_kwargs,
        )
        self._rc_pytree = self._rc.pytree()

    def _ground_height(self, xy: jax.Array) -> jax.Array:
        """Height of the ground under each `(..., 2)` point, bilinear."""
        if self._terrain is None:
            return jnp.zeros(xy.shape[:-1], dtype=precision.FLOAT)
        n = self._heights.shape[0]
        spacing = self._terrain.size / (n - 1)
        grid = jnp.clip((xy + self._terrain.size / 2) / spacing, 0, n - 1)
        corner = jnp.clip(jnp.floor(grid).astype(jnp.int32), 0, n - 2)
        fx, fy = jnp.moveaxis(grid - corner, -1, 0)
        col, row = jnp.moveaxis(corner, -1, 0)
        h = self._heights
        return (
            h[row, col] * (1 - fx) * (1 - fy)
            + h[row, col + 1] * fx * (1 - fy)
            + h[row + 1, col] * (1 - fx) * fy
            + h[row + 1, col + 1] * fx * fy
        )

    def _head_height(self, data: mjx.Data) -> jax.Array:
        head = data.xpos[self._head_body_id]
        return head[2] - self._ground_height(head[:2])

    def _pose_lights(self, data: mjx.Data) -> mjx.Data:
        """Fills in the light poses warp's staged graph mode leaves at zero."""
        # Under `WARP_STAGED_EX` the step never copies `light_xpos`/`light_xdir`
        # back, so every light reaches the renderer pointing nowhere and frames
        # are lit by the ambient term alone (2026-11-01). `JAX` graph mode
        # keeps them but steps 12x slower: 117 ms against 9.7 ms, 256 worlds.
        # This is mujoco_warp's own `_light_local_to_global`, less the target
        # modes `__init__` refuses.
        lights = self._lights
        body = self._mj_model.light_bodyid
        mode = self._mj_model.light_mode[:, None]
        quat = data.xquat[body]
        xpos = jnp.select(
            [mode == mujoco.mjtCamLight.mjCAMLIGHT_TRACK,
             mode == mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM],
            [data.xpos[body] + lights["pos0"],
             data.subtree_com[body] + lights["poscom0"]],
            data.xpos[body] + _rotate(lights["pos"], quat),
        )
        xdir = jnp.where(
            mode == mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
            _rotate(lights["dir"], quat),
            lights["dir0"],
        )
        xdir = xdir / jnp.linalg.norm(xdir, axis=-1, keepdims=True)
        return data.tree_replace(
            {"_impl.light_xpos": xpos, "_impl.light_xdir": xdir}
        )

    def _render(self, data: mjx.Data) -> jnp.ndarray:
        data = self._pose_lights(data)
        render_data = mjx.refit_bvh(self.mjx_model, data, self._rc_pytree)
        rgb, depth, _ = mjx.render(self.mjx_model, render_data, self._rc_pytree)
        # Camera index 0 counts ACTIVE cameras, and only one is active.
        rgb = mjx.get_rgb(self._rc_pytree, 0, rgb)
        depth = mjx.get_depth(
            self._rc_pytree, 0, depth, self._config.depth_scale,
        )
        # A ray that hits nothing reads 0, which would put the sky right
        # against the lens; it is as far as depth goes.
        depth = jnp.where(depth > 0, depth, jnp.ones_like(depth))
        return jnp.concatenate([rgb, depth], axis=-1)

    def _observe(self, state: mjx_env.State) -> mjx_env.State:
        obs = {"pixels/view_0": self._render(state.data)}
        if self._config.proprio:
            obs["state"] = state.obs
        return state.replace(obs=obs)

    def reset(self, rng):
        rng, qvel_key, spawn_key, yaw_key = jax.random.split(rng, 4)
        state = super().reset(rng)
        # The parent's start pose ignores `rng`, so under vmap its data has no
        # batch axis, the render runs unbatched and hands every world all
        # `nworld` images. Drawing the start state from `rng` batches it.
        qvel = self._config.reset_qvel_scale * jax.random.normal(
            qvel_key, (self.mjx_model.nv,), dtype=precision.FLOAT,
        )
        yaw = jax.random.uniform(
            yaw_key, (), minval=0.0, maxval=2 * math.pi, dtype=precision.FLOAT,
        )
        qpos = state.data.qpos.at[3:7].set(
            jnp.stack([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)])
        )
        if self._terrain is not None:
            reach = self._terrain.size / 2 - self._terrain.spawn_margin
            xy = jax.random.uniform(
                spawn_key, (2,), minval=-reach, maxval=reach,
                dtype=precision.FLOAT,
            )
            offsets = jnp.linspace(-_FOOTPRINT, _FOOTPRINT, 5, dtype=precision.FLOAT)
            footprint = xy + jnp.stack(jnp.meshgrid(offsets, offsets), -1)
            ground = self._ground_height(footprint).max()
            qpos = qpos.at[:3].set(jnp.concatenate([xy, qpos[2:3] + ground]))
        data = mjx.forward(
            self.mjx_model, state.data.replace(qpos=qpos, qvel=qvel),
        )
        state = state.replace(data=data, obs=self._get_obs(data, state.info))
        return self._observe(state)

    def step(self, state, action):
        # The parent reads nothing from `state.obs`, so the dict passes through.
        return self._observe(super().step(state, action))


def build_humanoid_rgbd_env(
    task: str = "walk",
    *,
    mode: str = "train",
    num_envs: int = 1,
    test_episodes: int = 1,
    seed: int = 0,
    impl: str = "warp",
    camera: str = "egocentric",
    hidden_geoms: tuple[str, ...] = ("head",),
    cam_res: tuple[int, int] = (64, 64),
    depth_scale: float = 10.0,
    use_textures: bool = True,
    proprio: bool = True,
    terrain: TerrainConfig | Mapping[str, Any] | None = TerrainConfig(),
    naconmax: int | None = None,
    njmax: int | None = None,
    graph_mode: str | None = None,
    max_episode_steps: int | None = None,
) -> EnvBundle:
    """Instantiates training and test drivers for the RGB-D humanoid.

    `impl` is not a choice — rendering needs mujoco_warp — but `train.py` reads
    `env.impl` to leave warp its share of the GPU, and refusing anything else
    here is what stops a CPU replay from failing inside the renderer.
    `terrain` takes `TerrainConfig`'s fields; `None` is flat ground.
    """
    if impl != "warp":
        raise ValueError(
            f"the RGB-D humanoid renders through mujoco_warp, so it runs on "
            f"impl=warp and a CUDA GPU only (got impl={impl!r})"
        )
    if task not in MOVE_SPEEDS:
        raise ValueError(f"task must be one of {sorted(MOVE_SPEEDS)}, got {task!r}")
    config = default_config()
    config.camera = camera
    config.hidden_geoms = tuple(hidden_geoms)
    config.render_config.cam_res = tuple(int(d) for d in cam_res)
    config.render_config.use_textures = bool(use_textures)
    config.depth_scale = float(depth_scale)
    config.proprio = bool(proprio)
    if naconmax is not None:
        config.naconmax = naconmax
    if njmax is not None:
        config.njmax = njmax
    config.graph_mode = graph_mode or DEFAULT_WARP_GRAPH_MODE
    if isinstance(terrain, Mapping):
        terrain = TerrainConfig(**terrain)

    def load(nworld: int) -> PlaygroundFuncEnv:
        # `MjxEnv` locks the config it is handed, so each env takes a copy.
        world_config = config.copy_and_resolve_references()
        world_config.render_config.nworld = int(nworld)
        env = HumanoidRGBD(MOVE_SPEEDS[task], world_config, terrain)
        return PlaygroundFuncEnv(env, impl="warp", nworld=nworld)

    max_steps = int(max_episode_steps or config.episode_length)
    # A render context is allocated for a fixed `nworld` that JAX cannot see, so
    # the eval batch needs its own env rather than sharing the training one.
    return EnvBundle(
        env=JaxVectorEnv(load(num_envs), num_envs, max_episode_steps=max_steps),
        test_env=JaxVectorEnv(
            load(test_episodes), test_episodes, max_episode_steps=max_steps,
        ),
        env_cfg=config,
    )
