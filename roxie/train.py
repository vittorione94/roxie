import os
import sys

# `device=` must be parsed before JAX is imported (JAX_PLATFORMS is read at
# import time). Neither it nor `resume=` is a config key, so both are dropped
# from sys.argv or Hydra rejects the launch.
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
    resolve_placement,
)
from roxie.utils import hydra_searchpath, logger
from roxie.utils.checkpoint import checkpoint_steps, find_checkpoint
from roxie.utils.trainer import Trainer

# Lets `--config-name <env>/<name>` resolve against top-level experiments/ while
# config groups stay in roxie/configs.
hydra_searchpath.register()
# Must happen before Hydra composes: ${envpool_task:<Task>} is used by configs.
suites.register_resolvers()
# examples/ is not installed with roxie, but its env builders are referenced by
# dotted path.
sys.path.insert(0, str(hydra_searchpath.REPO_ROOT))


@hydra.main(version_base=None, config_path="configs", config_name="dmc/bench_td3")
def main(cfg: DictConfig):
    print("Agent:", cfg.agent._target_)

    # Still honoured after `import jax`: the CUDA client initializes lazily at
    # the first jax.* call below.
    if cfg.env.get("impl", None) == "warp":
        # Warp allocates GPU memory outside JAX's pool, so leave it headroom.
        # Disabling preallocation instead fragments and OOMs the replay buffer.
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
        # Do NOT set XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async here: with the env
        # step inside a `lax.scan` the async allocator fails to free pointers it
        # does not own and the process eventually SIGSEGVs.

    # `agent.device` and `env.device` are separate knobs — CPU physics with a
    # GPU learner is a real setup — so they are read, not reconciled. Must go
    # through `jax.config`: JAX_PLATFORMS was already parsed at `import jax`.
    runtime_cfg = cfg.get("runtime") or {}
    agent_device, env_device, platform = resolve_placement(cfg)
    if platform and not _device:
        jax.config.update("jax_platforms", str(platform))

    # XLA GPU autotuning hangs on Blackwell GPUs; harmless on CPU.
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

    # Global: reaches the agent's networks and MJX physics alike.
    matmul_precision = runtime_cfg.get("matmul_precision", None)
    if matmul_precision:
        jax.config.update("jax_default_matmul_precision", matmul_precision)
        print(f"Matmul precision: {matmul_precision}")

    print("JAX devices:", jax.devices())
    print("JAX platform:", jax.default_backend())
    if _device:
        print(f"JAX platform override: device={_device}")

    # `env:` is a `_target_` block like `agent:`: the builder it names owns all
    # env-specific setup and returns a normalized bundle.
    impl = cfg.env.get("impl", None)
    env, test_env, env_cfg = build_env(
        cfg.env, mode="train",
        num_envs=int(cfg.env.parallel_envs),
        test_episodes=int(cfg.trainer.test_episodes),
    )
    # `device=` deliberately overrides the config, so the declarations are not
    # checked against it.
    log_loaded_backend(
        env,
        requested_impl=impl,
        agent_device=None if _device else agent_device,
        env_device=None if _device else env_device,
    )
    print("Environment configuration:", env_cfg)

    output_dir = HydraConfig.get().runtime.output_dir

    # wandb is opt-in so runs don't require the dependency.
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

    obs_space, act_space = env.single_observation_space, env.single_action_space
    action_low = jnp.asarray(act_space.low, dtype=jnp.float32)
    action_high = jnp.asarray(act_space.high, dtype=jnp.float32)

    # Only what the env alone knows; the rest comes from the agent's yaml.
    agent_kwargs = dict(
        env_obs_size=space_size(obs_space),
        env_action_size=space_size(act_space),
        action_low=action_low,
        action_high=action_high,
    )
    # Agents that explore from their own policy carry no noise group.
    if "noise" in cfg:
        agent_kwargs["noise_config"] = cfg.noise

    agent = build_agent(cfg.agent, **agent_kwargs)

    # Only the agent's numbers come from the checkpoint — the yaml stays
    # authoritative, so a resume may raise `trainer.steps` or retune a knob.
    resume_cfg = cfg.get("resume") or {}
    if isinstance(resume_cfg, str):  # `resume: <path>` rather than `resume.path`
        resume_cfg = {"path": resume_cfg}
    resume_path = _resume or resume_cfg.get("path", None)
    resume_metadata = None
    if resume_path:
        checkpoint = find_checkpoint(resume_path)
        resume_metadata = agent.restore(checkpoint)
        # Older checkpoints only have the step count in the directory name.
        if not resume_metadata.get("steps"):
            resume_metadata["steps"] = checkpoint_steps(checkpoint) or 0
        print(
            f"Resuming from {checkpoint} at {int(resume_metadata['steps']):,} "
            f"env steps (target {int(cfg.trainer.steps):,}).",
            flush=True,
        )

    # A resume offsets the env stream by the steps already taken so it does not
    # revisit the first leg's start states. The agent stream stays fixed.
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
        async_learner=bool(cfg.trainer.get("async_learner", False)),
        learner_chunk=int(cfg.trainer.get("learner_chunk", 8)),
        # Exact off-policy resume (no warmup refill), at the cost of the
        # buffer's full size on disk per save.
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

    return


if __name__ == "__main__":
    main()
