import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
# The warp backend (impl=warp) allocates GPU memory outside JAX's pool. Rather
# than disable preallocation (which fragments and OOMs on large contiguous
# allocations like the replay buffer), cap JAX to a fraction of the device so
# warp has headroom for its solver/collision scratch.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import hydra
import jax
import jax.numpy as jnp
from flax import nnx
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from roxie.agents import agents
from roxie.environment.loader import (
    load_mocap_env,
    load_playground_env,
    log_loaded_backend,
)
from roxie.utils.trainer import Trainer


@hydra.main(version_base=None, config_path="configs", config_name="experiment/walker_ddpg")
def main(cfg: DictConfig):
    print(cfg.agent.name)

    print("JAX devices:", jax.devices())
    print("JAX platform:", jax.default_backend())

    env_type = cfg.env.get("env_type", "playground")
    # Physics backend selection is shared across env types: "warp" routes the
    # sim through mujoco_warp. naconmax is Warp's global contact arena (shared
    # across all vmapped worlds, so it scales with parallel_envs); njmax is the
    # per-world constraint budget.
    impl = cfg.env.get("impl", "jax")
    naconmax = cfg.env.get("naconmax", None)
    njmax = cfg.env.get("njmax", None)

    if env_type == "mocap":
        clip_ids = list(cfg.env.clip_ids) if cfg.env.get("clip_ids") else None
        gpu_clip_budget = cfg.env.get("gpu_clip_budget", 0)
        # The mocap env has no built-in Warp budgets, so auto-size when unset.
        if impl == "warp":
            if naconmax is None:
                naconmax = int(cfg.env.parallel_envs) * 16
            if njmax is None:
                njmax = 128
        env, test_env, _ = load_mocap_env(
            clip_ids,
            gpu_clip_budget=gpu_clip_budget,
            impl=impl,
            naconmax=naconmax,
            njmax=njmax,
        )
        env_cfg = None
    else:
        # Playground envs ship their own Warp budgets; pass through only what the
        # experiment overrides (None leaves the upstream default in place).
        env, env_cfg = load_playground_env(
            cfg.env.env_name, impl=impl, naconmax=naconmax, njmax=njmax,
        )
        test_env = None
    log_loaded_backend(env, requested_impl=impl)
    print("Environment configuration:", env_cfg)

    output_dir = HydraConfig.get().runtime.output_dir

    # Create RNGs for agent initialization
    training_rngs = nnx.Rngs(envs=cfg.env.seed, agent=3)  # Use your seed from cfg.seed

    # Agent handles all component instantiation internally
    ctrl_range = jnp.array(env.mj_model.actuator_ctrlrange)  # shape (action_dim, 2)
    action_low = ctrl_range[:, 0]
    action_high = ctrl_range[:, 1]

    agent_args = {
        "env_obs_size": env.observation_size,
        "env_action_size": env.action_size,
        "action_low": action_low,
        "action_high": action_high,
        **cfg.agent.args,
    }
    if "actor" in cfg.agent:
        agent_args["actor_config"] = cfg.agent.actor
    if "critic" in cfg.agent:
        agent_args["critic_config"] = cfg.agent.critic
    if "memory" in cfg.agent:
        agent_args["memory_config"] = cfg.agent.memory
    if "noise" in cfg:
        agent_args["noise_config"] = cfg.noise

    agent = agents[cfg.agent.name](**agent_args)

    trainer = Trainer(
        output_dir=output_dir,
        steps=int(cfg.trainer.steps),
        epoch_steps=int(cfg.trainer.epoch_steps),
        save_steps=int(cfg.trainer.save_steps),
        test_episodes=int(cfg.trainer.test_episodes),
        show_progress=cfg.trainer.show_progress,
        replace_checkpoint=cfg.trainer.replace_checkpoint,
    )
    test_environment = test_env if test_env is not None else env
    trainer.initialize(
        agent=agent, environment=env, test_environment=test_environment
    )
    trainer.run(cfg.env.parallel_envs, training_rngs)

    return


if __name__ == "__main__":
    main()
