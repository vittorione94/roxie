import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
# The warp backend (impl=warp) allocates GPU memory outside JAX's pool. Rather
# than disable preallocation (which fragments and OOMs on large contiguous
# allocations like the replay buffer), cap JAX to a fraction of the device so
# warp has headroom for its solver/collision scratch.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import sys

import hydra
import jax
import jax.numpy as jnp
from flax import nnx
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from hydra.utils import get_method

from roxie.agents import agents
from roxie.environment.loader import (
    DEFAULT_BUILDER,
    log_loaded_backend,
)
from roxie.utils import hydra_searchpath, logger
from roxie.utils.trainer import Trainer

# The launchable experiment configs live in top-level experiments/, grouped by
# env into ant/, walker/, mocap/ subfolders — register that dir on Hydra's search
# path so `--config-name <env>/<name>` resolves there while groups stay in
# roxie/configs.
hydra_searchpath.register()
# examples/ is not part of the installed roxie package; put the repo root on the
# path so the mocap example (imported lazily below) is importable from anywhere.
sys.path.insert(0, str(hydra_searchpath.REPO_ROOT))


@hydra.main(version_base=None, config_path="configs", config_name="walker/walker_ddpg")
def main(cfg: DictConfig):
    print(cfg.agent.name)

    print("JAX devices:", jax.devices())
    print("JAX platform:", jax.default_backend())

    # Each experiment names the callable that builds its env via ``env.builder``
    # (a dotted path); the default builds a mujoco_playground env. The builder
    # owns all env-specific setup (clip selection, Warp budget sizing, ...) and
    # returns a normalized EnvBundle, so this loop stays env-agnostic. ``impl``
    # selects the physics backend ("warp" routes through mujoco_warp) and is read
    # here only for the load banner — the builder reads it off cfg.env itself.
    impl = cfg.env.get("impl", "jax")
    build_env = get_method(cfg.env.get("builder", DEFAULT_BUILDER))
    env, test_env, env_cfg = build_env(cfg.env, mode="train")
    log_loaded_backend(env, requested_impl=impl)
    print("Environment configuration:", env_cfg)

    output_dir = HydraConfig.get().runtime.output_dir

    # Initialize the logger up front so trainer stats fan out to all backends.
    # Console + CSV (in output_dir) are always on; wandb is opt-in via the
    # `logging.wandb` config block so runs don't require the dependency.
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    backends = logger.default_backends(output_dir)
    wandb_cfg = (cfg.get("logging") or {}).get("wandb") if "logging" in cfg else None
    if wandb_cfg and wandb_cfg.get("enabled", False):
        backends.append(
            logger.WandbBackend(
                project=wandb_cfg.get("project"),
                entity=wandb_cfg.get("entity"),
                name=wandb_cfg.get("name"),
                group=wandb_cfg.get("group"),
                tags=wandb_cfg.get("tags"),
                mode=wandb_cfg.get("mode", "online"),
                relogin=wandb_cfg.get("relogin", True),
                config=cfg_dict,
                dir=output_dir,
            )
        )
    logger.initialize(path=output_dir, config=cfg_dict, backends=backends)

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
    try:
        trainer.run(cfg.env.parallel_envs, training_rngs)
    finally:
        logger.close()

    return


if __name__ == "__main__":
    main()
