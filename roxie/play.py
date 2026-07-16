import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
# The warp backend (impl=warp) allocates GPU memory outside JAX's pool. Cap JAX
# to a fraction of the device so warp has headroom (see train.py for rationale).
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import sys
import time

import click
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from omegaconf import OmegaConf

from hydra.utils import get_method

from roxie.agents import agents
from roxie.environment.loader import (
    DEFAULT_BUILDER,
    log_loaded_backend,
)
from roxie.utils import hydra_searchpath

# examples/ is not part of the installed roxie package; put the repo root on the
# path so the mocap example (imported lazily for mocap checkpoints) is importable.
sys.path.insert(0, str(hydra_searchpath.REPO_ROOT))


@click.command()
@click.option("--checkpoint-path", type=str, help="Path to the checkpoint file.")
def main(checkpoint_path):
    cfg_path = os.path.join(checkpoint_path, "../../.hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)

    key = jax.random.PRNGKey(seed=0)

    # Same builder protocol as train.py: ``env.builder`` names the env factory;
    # ``mode="play"`` lets it apply playback-specific tweaks (the mocap builder
    # shrinks its GPU clip pool here).
    build_env = get_method(cfg.env.get("builder", DEFAULT_BUILDER))
    env, _, env_cfg = build_env(cfg.env, mode="play")

    log_loaded_backend(env, requested_impl=cfg.env.get("impl", "jax"))

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

    # Optional per-env viewer overrides, injected via ``env.viewer`` (a dotted
    # path). The mocap example uses this to render a reference "ghost" alongside
    # the policy; envs that don't set it just render their own model.
    viewer_path = cfg.env.get("viewer", None)
    ghost = get_method(viewer_path)(env) if viewer_path else None
    model = ghost.model if ghost is not None else env.mj_model

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

            if ghost is not None:
                info = wrapped_state.env_state.info
                abs_idx = int(info["clip_start"]) + int(info["phase_idx"])
                data.qpos[:ghost.nq] = wrapped_state.env_state.data.qpos
                data.qvel[:ghost.nv] = wrapped_state.env_state.data.qvel
                data.qpos[ghost.nq:] = ghost.ref_qpos[abs_idx]
                data.qvel[ghost.nv:] = ghost.ref_qvel[abs_idx]
            else:
                data.qpos = wrapped_state.env_state.data.qpos
                data.qvel = wrapped_state.env_state.data.qvel

            data.ctrl = action
            mujoco.mj_forward(model, data)

            score += wrapped_state.env_state.reward
            actions.append(action)

            # Print the per-component rewards (envs expose these under
            # ``metrics["reward/*"]``) alongside the total step reward.
            metrics = wrapped_state.env_state.metrics
            components = " ".join(
                f"{k.split('/', 1)[1]}={float(v):+.3f}"
                for k, v in metrics.items()
                if k.startswith("reward/")
            )
            print(f"reward={float(wrapped_state.env_state.reward):+.3f} {components}")

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

            time_until_next_step = model.opt.timestep * 15 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
