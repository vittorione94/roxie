"""Main training entry point for Roxie RL algorithms."""

import os
import sys

# Extract CLI runtime flags before importing JAX.
_device = None
_resume = None
for _arg in list(sys.argv[1:]):
    _bare = _arg.lstrip("+")
    if _bare.startswith("device="):
        _device = _bare.split("=", 1)[1]
        sys.argv.remove(_arg)
    elif _bare.startswith("resume="):
        _resume = _bare.split("=", 1)[1]
        sys.argv.remove(_arg)
if _device:
    os.environ["JAX_PLATFORMS"] = _device


import hydra
import jax
import jax.numpy as jnp
from flax import nnx
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from roxie.agents.utils import build_agent
from roxie.environment import suites
from roxie.environment.functional import space_size
from roxie.environment.loader import (
    build_env,
    log_loaded_backend,
    pin_cpu_cores,
    publish_env_shapes,
    resolve_placement,
)
from roxie.utils import hydra_searchpath, logger, precision
from roxie.utils.checkpoint import checkpoint_steps, find_checkpoint
from roxie.utils.trainer import Trainer

hydra_searchpath.register()
suites.register_resolvers()
sys.path.insert(0, str(hydra_searchpath.REPO_ROOT))


@hydra.main(version_base=None, config_path="configs", config_name="dmc/bench_td3")
def main(cfg: DictConfig):
    """Parses environment/agent configuration and runs the training pipeline."""
    print("Agent:", cfg.agent._target_)

    if cfg.env.get("impl", None) == "warp":
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

    runtime_cfg = cfg.get("runtime") or {}
    device, platform = resolve_placement(cfg)
    if platform and not _device:
        jax.config.update("jax_platforms", str(platform))

    pinned = pin_cpu_cores(runtime_cfg.get("cpu_cores", None), _device or platform)
    if pinned:
        print(f"CPU affinity: {pinned} cores ({sorted(os.sched_getaffinity(0))})")

    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

    matmul_dtype = str(runtime_cfg.get("matmul_dtype", "") or "").lower()
    if matmul_dtype in ("bf16", "bfloat16"):
        os.environ["ROXIE_MATMUL_DTYPE"] = "bfloat16"
        if str(device) == "cpu":
            os.environ["XLA_FLAGS"] = (
                os.environ.get("XLA_FLAGS", "") + " " + precision.ONEDNN_BF16_FLAGS
            ).strip()
        print("Matmul dtype: bfloat16 (params/accumulation stay float32)")
    elif matmul_dtype not in ("", "float32", "f32"):
        raise ValueError(f"runtime.matmul_dtype={matmul_dtype!r} is invalid.")

    matmul_precision = runtime_cfg.get("matmul_precision", None)
    if matmul_precision:
        jax.config.update("jax_default_matmul_precision", matmul_precision)
        print(f"Matmul precision: {matmul_precision}")

    print("JAX devices:", jax.devices())
    print("JAX platform:", jax.default_backend())

    impl = cfg.env.get("impl", None)
    env, test_env, env_cfg = build_env(
        cfg.env,
        mode="train",
        num_envs=int(cfg.env.parallel_envs),
        test_episodes=int(cfg.trainer.test_episodes),
    )
    log_loaded_backend(
        env,
        requested_impl=impl,
        device=None if _device else device,
    )

    obs_space, act_space = env.single_observation_space, env.single_action_space
    publish_env_shapes(cfg.env, space_size(obs_space), space_size(act_space))

    output_dir = HydraConfig.get().runtime.output_dir

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
                job_type=wandb_cfg.get("job_type"),
                tags=wandb_cfg.get("tags"),
                mode=wandb_cfg.get("mode", "online"),
                relogin=wandb_cfg.get("relogin", True),
                config=cfg_dict,
                dir=output_dir,
            )
        )
    logger.initialize(path=output_dir, backends=backends)

    action_low = jnp.asarray(act_space.low, dtype=jnp.float32)
    action_high = jnp.asarray(act_space.high, dtype=jnp.float32)

    agent_kwargs = dict(
        env_obs_size=space_size(obs_space),
        env_action_size=space_size(act_space),
        action_low=action_low,
        action_high=action_high,
    )
    if "noise" in cfg:
        agent_kwargs["noise_config"] = cfg.noise

    agent = build_agent(cfg.agent, **agent_kwargs)

    resume_cfg = cfg.get("resume") or {}
    if isinstance(resume_cfg, str):
        resume_cfg = {"path": resume_cfg}
    resume_path = _resume or resume_cfg.get("path", None)
    resume_metadata = None
    if resume_path:
        checkpoint = find_checkpoint(resume_path)
        resume_metadata = agent.restore(checkpoint)
        if not resume_metadata.get("steps"):
            resume_metadata["steps"] = checkpoint_steps(checkpoint) or 0
        print(
            f"Resuming from {checkpoint} at {int(resume_metadata['steps']):,} "
            f"env steps (target {int(cfg.trainer.steps):,}).",
            flush=True,
        )

    resumed_steps = int((resume_metadata or {}).get("steps") or 0)
    training_rngs = nnx.Rngs(envs=cfg.env.seed + resumed_steps, agent=3)

    trainer = Trainer(
        output_dir=output_dir,
        steps=int(cfg.trainer.steps),
        epoch_steps=int(cfg.trainer.epoch_steps),
        save_steps=int(cfg.trainer.save_steps),
        test_episodes=int(cfg.trainer.test_episodes),
        show_progress=cfg.trainer.show_progress,
        replace_checkpoint=cfg.trainer.replace_checkpoint,
        save_buffer=bool(cfg.trainer.get("save_buffer", False)),
        resume=resume_metadata,
    )
    test_environment = test_env if test_env is not None else env
    trainer.initialize(
        agent=agent, environment=env, test_environment=test_environment
    )
    try:
        trainer.run(cfg.env.parallel_envs, training_rngs)
    finally:
        logger.close()


if __name__ == "__main__":
    main()