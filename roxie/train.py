import os
import sys

# device=<cpu|gpu> overrides JAX platform selection. Must be parsed from
# sys.argv before JAX is imported — JAX_PLATFORMS is read at import time.
_device = next(
    (arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("device=")),
    None,
)
if _device:
    os.environ["JAX_PLATFORMS"] = _device


import hydra
import jax
import jax.numpy as jnp
from flax import nnx
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from hydra.utils import get_method

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
    print("Agent:", cfg.agent._target_)

    # Backend env vars are decided here, from the COMPOSED config, not at module
    # import from sys.argv: `env.impl: warp` set in an experiment yaml never
    # appears in argv, so an argv sniff misses it (JAX then preallocates its
    # default 75% of VRAM and warp/reset constants OOM). This works because
    # JAX's CUDA client initializes lazily at the first jax.* call below — the
    # vars just have to be set before that, not before `import jax`.
    if cfg.env.get("impl", None) == "warp":
        # Warp allocates GPU memory outside JAX's pool: cap JAX so Warp has
        # headroom for its solver/collision scratch. Don't disable preallocation
        # instead — that fragments and OOMs the large replay-buffer alloc.
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
        # Use CUDA's own async pool instead of XLA's BFC allocator. BFC caps out
        # on FRAGMENTATION, not on a leak: run 2026-08-13_23-03-23 died at epoch
        # 131 requesting a contiguous 2.02 GiB while GPU memory had been flat at
        # 12.5-14.4 GB and mem/live_arrays flat at ~1075 for the whole run (the
        # allocator dump showed the classic free/used checkerboard). What made it
        # bite there and not in the otherwise-identical 737-epoch run before it
        # is allocation CHURN: `ppo/steps_per_rollout` had climbed 5 -> 67 as the
        # KL early stop released, i.e. ~33x more alloc/free cycles per rollout.
        # `cuda_async` still pools (so it does not pay a cudaMalloc per
        # allocation the way "platform" does) but the driver pool tolerates that
        # churn. Overridable: setdefault, so XLA_PYTHON_CLIENT_ALLOCATOR=default
        # in the environment restores BFC if an agent regresses.
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
    elif cfg.env.get("impl", None) == "envpool":
        # CPU backend = CPU run. The pool's physics is pure native MuJoCo and
        # never touches the GPU, but the AGENT is plain JAX and would otherwise
        # still claim the card (and preallocate ~75% of it) — so "running on
        # CPU" would leave the GPU occupied, which is the opposite of the point.
        # Pin the whole process to CPU so the card is genuinely free.
        #
        # This is a real throughput trade, measured on this box (12-core 7900X,
        # PPO, parallel_envs=1000, obs 1069, nets [1024,512,256]):
        #   CPU physics + GPU learner   ~17.2k sps
        #   everything on CPU           ~9.1k sps
        # The gap is the dense actor/critic GEMMs, which is what the GPU is for.
        # Set `runtime.jax_platform: null` in the experiment to opt back into the
        # hybrid if you want the throughput and can spare ~1.4 GB of VRAM (the
        # measured peak for this arm — the 15 GB you see reported is XLA's
        # preallocated arena, not resident data).
        # NOTE: this one must go through `jax.config`, NOT an env var. The
        # XLA_PYTHON_CLIENT_* settings above work as `os.environ` writes because
        # the PJRT C++ client reads them when it lazily initializes. But
        # `JAX_PLATFORMS` is a JAX *Python* config option, parsed out of the
        # environment once at `import jax` — which already happened at the top of
        # this module — so setting the env var here is silently ignored and the
        # run still lands on the GPU.
        platform = (cfg.get("runtime") or {}).get("jax_platform", "cpu")
        if platform:
            jax.config.update("jax_platforms", str(platform))
    # XLA GPU autotuning hangs this machine's RTX 5080 (Blackwell) — required
    # for EVERY GPU run regardless of physics backend; unused/harmless on CPU.
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

    # Precision of every f32 matmul (see `runtime.matmul_precision` in the
    # experiment yaml). null leaves JAX's own default in place. This is a
    # GLOBAL setting — it reaches the agent's networks AND MJX physics — so
    # change it for a whole sweep at once, never for a single arm, or the
    # comparison stops being an A/B on the algorithm.
    matmul_precision = (cfg.get("runtime") or {}).get("matmul_precision", None)
    if matmul_precision:
        jax.config.update("jax_default_matmul_precision", matmul_precision)
        print(f"Matmul precision: {matmul_precision}")

    print("JAX devices:", jax.devices())
    print("JAX platform:", jax.default_backend())
    if _device:
        print(f"JAX platform override: device={_device}")

    # Each experiment names the callable that builds its env via ``env.builder``
    # (a dotted path); the default builds a mujoco_playground env. The builder
    # owns all env-specific setup (clip selection, Warp budget sizing, ...) and
    # returns a normalized EnvBundle, so this loop stays env-agnostic. ``impl``
    # selects the physics backend ("warp" routes through mujoco_warp) and is read
    # here only for the load banner — the builder reads it off cfg.env itself.
    impl = cfg.env.get("impl", None)
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
    logger.initialize(path=output_dir, backends=backends)

    # Create RNGs for agent initialization
    training_rngs = nnx.Rngs(envs=cfg.env.seed, agent=3)  # Use your seed from cfg.seed

    # Action bounds: MuJoCo/Playground envs expose mj_model.actuator_ctrlrange;
    # EnvPool and other non-MuJoCo envs provide action_low/action_high directly.
    if hasattr(env, "mj_model"):
        ctrl_range = jnp.array(env.mj_model.actuator_ctrlrange)  # shape (action_dim, 2)
        action_low = ctrl_range[:, 0]
        action_high = ctrl_range[:, 1]
    else:
        action_low = env.action_low
        action_high = env.action_high

    # The agent config IS the constructor call: `_target_` names the class and
    # every sibling key is one of its keywords, so a knob that exists in Python
    # but not in the yaml fails loudly here instead of silently taking its
    # default (see tests/test_agent_configs.py). Only the four env-derived
    # arguments are injected.
    #
    # `_recursive_=False` keeps the nested `*_config` blocks as DictConfigs: the
    # agent instantiates its own actor/critic/memory/optimizers, injecting shapes
    # (in_features, action_dim, rngs, num_atoms) that are unknown out here.
    agent_kwargs = dict(
        env_obs_size=env.observation_size,
        env_action_size=env.action_size,
        action_low=action_low,
        action_high=action_high,
    )
    # `noise` is its own top-level config group (agents that explore from their
    # own policy — SAC, MPO, PPO — carry no noise group and take no such arg).
    if "noise" in cfg:
        agent_kwargs["noise_config"] = cfg.noise

    agent = hydra.utils.instantiate(cfg.agent, _recursive_=False, **agent_kwargs)

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
