"""DeepMind Control "quadruped" Move task, ported to a playground ``MjxEnv``.

This is a worked example env (the brax/dm_control benchmark quadruped, often
referred to as the "ant"), kept under ``examples`` so the generic env
infrastructure in ``roxie/environment`` stays task-agnostic — the dependency is
one-way (examples import from roxie, never the reverse) and the core train/play
loops reach this lazily through the ``env.builder`` dotted path in the config.

The model, sensors and reward mirror
``dm_control.suite.quadruped`` (the ``walk``/``run`` ``Move`` task). Rather than
re-author the MJCF, we reuse dm_control's own ``make_model`` (with the
terrain/walls/ball/rangefinders stripped, exactly as the walk/run tasks do) so
the body, fixed-tendon coupling, filtered actuators and sensor suite match
upstream bit-for-bit. Everything else — reset, observations, reward — is
re-expressed against MJX/Warp data so it runs batched on GPU.

Reference: https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/quadruped.py
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
from dm_control.suite import common as dm_common
from dm_control.suite import quadruped as dm_quadruped
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env, reward

# Horizontal speeds (m/s) above which the move reward saturates to 1, straight
# from dm_control's quadruped: 0.5 == "walk", 5.0 == "run".
_WALK_SPEED = 0.5
_RUN_SPEED = 5.0
# dm_control sizes the (finite) floor to time_limit * speed; we keep the same
# default time limit so the arena is large enough for a full episode.
_DEFAULT_TIME_LIMIT = 20.0

# Toe force/torque sensors, in the XML's declaration order. arcsinh-compressed
# into the observation, matching dm_control's ``Physics.force_torque``.
_FORCE_TORQUE_SENSORS = (
    "force_toe_front_left",
    "force_toe_front_right",
    "force_toe_back_right",
    "force_toe_back_left",
    "torque_toe_front_left",
    "torque_toe_front_right",
    "torque_toe_back_right",
    "torque_toe_back_left",
)


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      # dm_control uses a 0.02s control step over a 0.005s sim step (4 substeps)
      # and a 20s (==1000 control step) episode.
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=1000,
      action_repeat=1,
      # Target horizontal speed; the move reward is maximised at/above it.
      # 0.5 == walk, 5.0 == run (see _WALK_SPEED / _RUN_SPEED).
      desired_speed=_WALK_SPEED,
      # Episode init mirrors dm_control: random torso orientation (hard
      # exploration) spawned above the floor and settled under gravity for
      # ``settle_steps`` sim steps before the clock is zeroed. Set
      # ``random_orientation=False`` for an upright start (much easier).
      random_orientation=True,
      spawn_height=0.8,
      settle_steps=200,
      # Physics backend: "warp" (mujoco_warp) or "jax" (MJX). The contact
      # (naconmax) / constraint (njmax) budgets are only consumed by the Warp
      # backend; the builder sizes them per parallel_envs (MJX ignores them).
      impl="warp",
      naconmax=20_000,
      njmax=128,
  )


class QuadrupedMove(mjx_env.MjxEnv):
  """dm_control quadruped ``Move`` task (walk/run) on MJX/Warp."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, float, bool]]] = None,
  ):
    super().__init__(config, config_overrides)

    floor_size = _DEFAULT_TIME_LIMIT * float(self._config.desired_speed)
    xml = dm_quadruped.make_model(floor_size=floor_size)
    if isinstance(xml, bytes):
      xml = xml.decode()
    self._xml_string = xml
    self._model_assets = dict(dm_common.ASSETS)
    self._mj_model = mujoco.MjModel.from_xml_string(xml, self._model_assets)
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()

  def _post_init(self) -> None:
    m = self._mj_model
    # The 16 leg hinges (4 per leg); the only free joint is the torso root,
    # which occupies the leading qpos[:7] / qvel[:6]. Egocentric obs uses just
    # the hinges (no global pose), matching dm_control's ``egocentric_state``.
    hinge = m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE
    self._hinge_qposadr = jp.asarray(m.jnt_qposadr[hinge])
    self._hinge_dofadr = jp.asarray(m.jnt_dofadr[hinge])
    self._torso_body_id = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_BODY, "torso"
    )
    self._desired_speed = float(self._config.desired_speed)
    self._qpos0 = jp.asarray(m.qpos0)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng_quat = jax.random.split(rng)

    qpos = self._qpos0
    if self._config.random_orientation:
      quat = jax.random.normal(rng_quat, (4,))
      quat = quat / jp.linalg.norm(quat)
    else:
      quat = jp.array([1.0, 0.0, 0.0, 0.0])
    qpos = qpos.at[2].set(self._config.spawn_height)
    qpos = qpos.at[3:7].set(quat)

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)
    # Let the (possibly tumbling) body settle onto the floor before t=0 — the
    # jittable analogue of dm_control's ``_find_non_contacting_height`` drop.
    data = mjx_env.step(
        self.mjx_model, data, jp.zeros(self.mjx_model.nu), self._config.settle_steps
    )
    data = data.replace(time=0.0)

    info = {"rng": rng}
    metrics = {"reward_move": jp.zeros(()), "reward_upright": jp.zeros(())}
    obs = self._get_obs(data)
    return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
    move_r, upright_r = self._reward_terms(data)
    rew = move_r * upright_r
    obs = self._get_obs(data)
    done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)
    metrics = {"reward_move": move_r, "reward_upright": upright_r}
    return mjx_env.State(data, obs, rew, done, metrics, state.info)

  def _torso_upright(self, data: mjx.Data) -> jax.Array:
    # zz element of the torso frame: projection of its z-axis on global z.
    return data.xmat[self._torso_body_id].reshape(3, 3)[2, 2]

  def _reward_terms(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
    # Forward speed in the torso's local frame (velocimeter x-axis).
    speed = mjx_env.get_sensor_data(self.mj_model, data, "velocimeter")[0]
    move = reward.tolerance(
        speed,
        bounds=(self._desired_speed, float("inf")),
        margin=self._desired_speed,
        value_at_margin=0.5,
        sigmoid="linear",
    )
    # 1 when fully upright, decaying linearly to 0 when upside-down.
    upright = reward.tolerance(
        self._torso_upright(data),
        bounds=(1.0, float("inf")),
        margin=2.0,
        value_at_margin=0.0,
        sigmoid="linear",
    )
    return move, upright

  def _get_obs(self, data: mjx.Data) -> jax.Array:
    egocentric = jp.concatenate([
        data.qpos[self._hinge_qposadr],
        data.qvel[self._hinge_dofadr],
        data.act,
    ])
    torso_velocity = mjx_env.get_sensor_data(self.mj_model, data, "velocimeter")
    torso_upright = self._torso_upright(data)
    imu = jp.concatenate([
        mjx_env.get_sensor_data(self.mj_model, data, "imu_accel"),
        mjx_env.get_sensor_data(self.mj_model, data, "imu_gyro"),
    ])
    force_torque = jp.arcsinh(
        jp.concatenate([
            mjx_env.get_sensor_data(self.mj_model, data, name)
            for name in _FORCE_TORQUE_SENSORS
        ])
    )
    return jp.concatenate([
        egocentric,
        torso_velocity,
        jp.array([torso_upright]),
        imu,
        force_torque,
    ])

  @property
  def xml_path(self) -> str:
    return ""

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
