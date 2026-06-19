from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.025,
        sim_dt=0.005,
        episode_length=1000,
        early_termination=True,
        action_repeat=1,
        action_scale=1.0,
        cyclic=True,
        random_start=True,
        reward_config=config_dict.create(
            w_pose=0.5,
            w_vel=0.1,
            w_ee=0.15,
            w_root=0.2,
            w_alive=0.05,
            sigma_pose=2.0,
            sigma_vel=0.1,
            sigma_ee=0.04,
            sigma_root=0.5,
        ),
        min_head_height=0.7,
    )


class MocapTrackingEnv(mjx_env.MjxEnv):

    def __init__(
        self,
        xml_path: str,
        clip_path: str,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)

        self._xml_path = xml_path
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)

        clip = np.load(clip_path)
        self._ref_qpos = jp.array(clip["qpos"], dtype=jp.float32)
        self._ref_qvel = jp.array(clip["qvel"], dtype=jp.float32)
        self._ref_body_pos = jp.array(clip["body_pos"], dtype=jp.float32)
        self._clip_len = self._ref_qpos.shape[0]

        self._post_init()

    def _post_init(self) -> None:
        self._head_body_id = self._mj_model.body("head").id
        self._torso_body_id = self._mj_model.body("torso").id

        ee_names = ["left_hand", "right_hand", "left_foot", "right_foot"]
        self._ee_body_ids = jp.array(
            [self._mj_model.body(name).id for name in ee_names]
        )

        self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
        self._uppers = self._mj_model.actuator_ctrlrange[:, 1]

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, start_rng = jax.random.split(rng)

        start_idx = jax.lax.cond(
            self._config.random_start,
            lambda r: jax.random.randint(r, (), 0, self._clip_len),
            lambda r: jp.int32(0),
            start_rng,
        )

        qpos = self._ref_qpos[start_idx]
        qvel = self._ref_qvel[start_idx]
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel)

        info = {
            "rng": rng,
            "phase_idx": start_idx,
            "last_act": jp.zeros(self.mjx_model.nu),
        }

        metrics = {
            "reward/pose": jp.zeros(()),
            "reward/vel": jp.zeros(()),
            "reward/ee": jp.zeros(()),
            "reward/root": jp.zeros(()),
        }

        reward_val, done = jp.zeros(2)
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, reward_val, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        rng, _ = jax.random.split(state.info["rng"])

        ctrl = jp.clip(
            action * self._config.action_scale, self._lowers, self._uppers
        )
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

        phase_idx = (state.info["phase_idx"] + 1) % self._clip_len

        reward_val = self._get_reward(data, phase_idx, state.metrics)

        done = self._get_termination(data, phase_idx)

        clip_ended = jp.where(
            self._config.cyclic,
            jp.float32(0),
            jp.float32(phase_idx >= self._clip_len - 1),
        )
        done = jp.maximum(done, clip_ended)

        info = {
            "rng": rng,
            "phase_idx": phase_idx,
            "last_act": action,
        }

        obs = self._get_obs(data, info)
        done = done.astype(jp.float32)
        return mjx_env.State(data, obs, reward_val, done, state.metrics, info)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        next_idx = (info["phase_idx"] + 1) % self._clip_len

        ref_qpos = self._ref_qpos[next_idx]
        ref_qvel = self._ref_qvel[next_idx]

        obs = jp.concatenate([
            data.qpos[7:],
            data.qvel[6:],
            data.qpos[3:7],
            data.qvel[3:6],
            data.qpos[2:3],
            data.qvel[0:3],
            info["last_act"],
            ref_qpos[7:],
            ref_qvel[6:],
            ref_qpos[3:7],
            ref_qpos[2:3],
        ])
        return obs

    def _get_reward(
        self,
        data: mjx.Data,
        phase_idx: jax.Array,
        metrics: dict[str, Any],
    ) -> jax.Array:
        cfg = self._config.reward_config
        ref_qpos = self._ref_qpos[phase_idx]
        ref_qvel = self._ref_qvel[phase_idx]
        ref_body_pos = self._ref_body_pos[phase_idx]

        pose_err = jp.sum(jp.square(data.qpos[7:] - ref_qpos[7:]))
        r_pose = jp.exp(-pose_err / cfg.sigma_pose)

        vel_err = jp.sum(jp.square(data.qvel[6:] - ref_qvel[6:]))
        r_vel = jp.exp(-vel_err / cfg.sigma_vel)

        ee_pos = data.xpos[self._ee_body_ids]
        ref_ee_pos = ref_body_pos[self._ee_body_ids]
        ee_err = jp.sum(jp.square(ee_pos - ref_ee_pos))
        r_ee = jp.exp(-ee_err / cfg.sigma_ee)

        root_pos_err = jp.sum(jp.square(data.qpos[:3] - ref_qpos[:3]))
        quat_dot = jp.dot(data.qpos[3:7], ref_qpos[3:7])
        root_quat_err = 1.0 - jp.square(quat_dot)
        root_err = root_pos_err + root_quat_err
        r_root = jp.exp(-root_err / cfg.sigma_root)

        metrics["reward/pose"] = r_pose
        metrics["reward/vel"] = r_vel
        metrics["reward/ee"] = r_ee
        metrics["reward/root"] = r_root

        return (
            cfg.w_pose * r_pose
            + cfg.w_vel * r_vel
            + cfg.w_ee * r_ee
            + cfg.w_root * r_root
            + cfg.w_alive
        )

    def _get_termination(self, data: mjx.Data, phase_idx: jax.Array) -> jax.Array:
        head_height = data.xpos[self._head_body_id, 2]
        fall = head_height < self._config.min_head_height
        nan_check = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        return jp.where(
            self._config.early_termination, fall | nan_check, nan_check
        )

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
