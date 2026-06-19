import copy
import os
import time
import xml.etree.ElementTree as ET

import click
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from omegaconf import OmegaConf

from roxie.agents import agents
from roxie.environment.loader import load_mocap_env, load_playground_env


def _build_ghost_model(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    asset = root.find("asset")
    ET.SubElement(asset, "material", {
        "name": "ghost",
        "rgba": "0.2 0.8 0.2 0.3",
    })

    worldbody = root.find("worldbody")
    torso = worldbody.find("body[@name='torso']")
    ghost = copy.deepcopy(torso)

    def _process(elem):
        if "name" in elem.attrib:
            elem.attrib["name"] = "ghost/" + elem.attrib["name"]
        if elem.tag == "geom":
            elem.attrib["contype"] = "0"
            elem.attrib["conaffinity"] = "0"
            elem.attrib["material"] = "ghost"
        to_remove = [c for c in elem if c.tag in ("site", "camera", "light")]
        for c in to_remove:
            elem.remove(c)
        for child in elem:
            _process(child)

    _process(ghost)
    worldbody.append(ghost)

    xml_string = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml_string)


@click.command()
@click.option("--checkpoint-path", type=str, help="Path to the checkpoint file.")
def main(checkpoint_path):
    cfg_path = os.path.join(checkpoint_path, "../../.hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)

    key = jax.random.PRNGKey(seed=0)

    env_type = cfg.env.get("env_type", "playground")
    if env_type == "mocap":
        env = load_mocap_env(cfg.env.xml_path, cfg.env.clip_path)
        env_cfg = None
    else:
        env, env_cfg = load_playground_env(cfg.env.env_name)

    agent_args = {}
    if "actor" in cfg.agent:
        agent_args["actor_config"] = cfg.agent.actor
    if "critic" in cfg.agent:
        agent_args["critic_config"] = cfg.agent.critic
    if "memory" in cfg.agent:
        agent_args["memory_config"] = cfg.agent.memory
    if "noise" in cfg:
        agent_args["noise_config"] = cfg.noise

    agent = agents[cfg.agent.name].load(
        path=checkpoint_path,
        env_obs_size=env.observation_size,
        env_act_size=env.action_size,
        **agent_args,
    )

    has_ghost = env_type == "mocap"
    if has_ghost:
        model = _build_ghost_model(cfg.env.xml_path)
        ref_clip = np.load(cfg.env.clip_path)
        ref_qpos = ref_clip["qpos"]
        ref_qvel = ref_clip["qvel"]
        nq = env.mj_model.nq
        nv = env.mj_model.nv
    else:
        model = env.mj_model

    data = mujoco.MjData(model)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        key, reset_key = jax.random.split(key)
        wrapped_state = jit_reset(key=reset_key)
        mujoco.mj_forward(model, data)

        score = 0.0
        actions = []

        while viewer.is_running():
            step_start = time.time()

            obs_b = jnp.expand_dims(wrapped_state.env_state.obs, axis=0)
            action = agent.step(obs_b, evaluate=True, key=key)

            wrapped_state = jit_step(wrapped_state, action[0])

            if has_ghost:
                phase_idx = int(wrapped_state.env_state.info["phase_idx"])
                data.qpos[:nq] = wrapped_state.env_state.data.qpos
                data.qvel[:nv] = wrapped_state.env_state.data.qvel
                data.qpos[nq:] = ref_qpos[phase_idx]
                data.qvel[nv:] = ref_qvel[phase_idx]
            else:
                data.qpos = wrapped_state.env_state.data.qpos
                data.qvel = wrapped_state.env_state.data.qvel

            data.ctrl = action
            mujoco.mj_forward(model, data)

            score += wrapped_state.env_state.reward
            actions.append(action)

            if wrapped_state.env_state.done:
                print(f"Total score: {score}")
                print(f"Actions mean: {jnp.mean(jnp.array(actions)):.2f}")
                print(f"Actions std: {jnp.std(jnp.array(actions)):.2f}")
                key, reset_key = jax.random.split(reset_key)
                wrapped_state = jit_reset(key=reset_key)
                mujoco.mj_resetData(model, data)
                score = 0.0
                actions = []

            viewer.sync()

            time_until_next_step = model.opt.timestep * 5 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
