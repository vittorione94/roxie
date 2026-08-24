import os
import sys

# device=<cpu|gpu> overrides JAX platform selection. Must be parsed from
# sys.argv before JAX is imported — JAX_PLATFORMS is read at import time.
#
# `device` is not a config key, so the arg is consumed (dropped from sys.argv)
# rather than merely read: leaving it in argv makes Hydra reject the launch.
# Both `device=` and `+device=` are accepted and hidden from Hydra.
#
# `resume=<path>` is consumed the same way and for the same reason: it is not a
# config key either, so Hydra would reject it (and `+resume=` would only work
# where the config is not struct-locked). Handling both here keeps the two
# process-level switches — where to run, and what to continue from — spelled the
# same way on the command line.
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

from hydra.utils import get_method

from roxie.environment.functional import space_size
from roxie.environment.loader import (
    DEFAULT_BUILDER,
    log_loaded_backend,
)
from roxie.utils import hydra_searchpath, logger
from roxie.utils.checkpoint import checkpoint_steps, find_checkpoint
from roxie.utils.trainer import Trainer

# Launchable experiment configs live in top-level experiments/, grouped by env.
# Registering that dir lets `--config-name <env>/<name>` resolve there while
# config groups stay in roxie/configs.
hydra_searchpath.register()
# examples/ is not part of the installed roxie package; put the repo root on the
# path so example env builders are importable from anywhere.
sys.path.insert(0, str(hydra_searchpath.REPO_ROOT))


@hydra.main(version_base=None, config_path="configs", config_name="walker/bench_td3")
def main(cfg: DictConfig):
    print("Agent:", cfg.agent._target_)

    # Backend env vars are decided from the COMPOSED config, not from sys.argv at
    # module import: `env.impl: warp` set in an experiment yaml never appears in
    # argv. Setting them here still works because JAX's CUDA client initializes
    # lazily at the first jax.* call below.
    if cfg.env.get("impl", None) == "warp":
        # Warp allocates GPU memory outside JAX's pool: cap JAX so Warp has
        # headroom for its solver/collision scratch. Don't disable preallocation
        # instead — that fragments and OOMs the large replay-buffer alloc.
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
        # CUDA's async pool instead of XLA's BFC allocator: BFC fragments under
        # the alloc/free churn of a varying-size rollout and eventually fails a
        # large contiguous request. Set XLA_PYTHON_CLIENT_ALLOCATOR=default to
        # restore BFC.
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")

    # Where the AGENT (networks, optimizers, replay buffer) runs, applied for
    # every physics backend.
    #
    # `envpool` defaults to "cpu": its physics is native MuJoCo and never touches
    # the GPU, but the agent is plain JAX and would otherwise still claim (and
    # preallocate most of) the card. Set `runtime.jax_platform: null` to opt into
    # the hybrid — CPU physics with a GPU learner is faster, since the dense
    # actor/critic GEMMs dominate, at the cost of holding the card.
    #
    # Every other backend defaults to null = leave JAX's own choice. Setting it
    # explicitly is how an experiment records e.g. "MJX on the CPU" in its yaml
    # rather than relying on the caller to pass `device=cpu`, so a benchmark grid
    # stays reproducible from the config alone. An explicit `device=` on the CLI
    # still wins.
    #
    # NOTE: this must go through `jax.config`, NOT an env var. Unlike the
    # XLA_PYTHON_CLIENT_* settings above (read lazily by the PJRT client),
    # `JAX_PLATFORMS` is parsed once at `import jax` — already done above — so an
    # os.environ write here is silently ignored.
    platform = (cfg.get("runtime") or {}).get(
        "jax_platform", "cpu" if cfg.env.get("impl", None) == "envpool" else None
    )
    if platform and not _device:
        jax.config.update("jax_platforms", str(platform))

    # XLA GPU autotuning hangs on Blackwell GPUs; harmless on CPU.
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

    # Precision of every f32 matmul; null leaves JAX's own default. This is
    # GLOBAL — it reaches the agent's networks and MJX physics alike — so change
    # it for a whole sweep at once, never for a single arm.
    matmul_precision = (cfg.get("runtime") or {}).get("matmul_precision", None)
    if matmul_precision:
        jax.config.update("jax_default_matmul_precision", matmul_precision)
        print(f"Matmul precision: {matmul_precision}")

    print("JAX devices:", jax.devices())
    print("JAX platform:", jax.default_backend())
    if _device:
        print(f"JAX platform override: device={_device}")

    # ``env.builder`` is a dotted path to the callable that builds the env; the
    # default builds a mujoco_playground env. The builder owns all env-specific
    # setup (clip selection, Warp budget sizing, ...) and returns a normalized
    # bundle, so this stays env-agnostic. ``impl`` selects the physics backend and
    # is read here only for the load banner.
    #
    # ``num_envs``/``test_episodes`` are passed IN rather than read off
    # ``cfg.env`` because they are trainer quantities that size the two drivers
    # the builder returns: the same env definition is driven at
    # ``env.parallel_envs`` worlds for training and at ``trainer.test_episodes``
    # for evaluation. Deriving both from one place is what stops the eval batch
    # from silently disagreeing with the eval loop's expectations.
    impl = cfg.env.get("impl", None)
    build_env = get_method(cfg.env.get("builder", DEFAULT_BUILDER))
    env, test_env, env_cfg = build_env(
        cfg.env, mode="train",
        num_envs=int(cfg.env.parallel_envs),
        test_episodes=int(cfg.trainer.test_episodes),
    )
    log_loaded_backend(env, requested_impl=impl)
    print("Environment configuration:", env_cfg)

    output_dir = HydraConfig.get().runtime.output_dir

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
                job_type=wandb_cfg.get("job_type"),
                tags=wandb_cfg.get("tags"),
                mode=wandb_cfg.get("mode", "online"),
                relogin=wandb_cfg.get("relogin", True),
                config=cfg_dict,
                dir=output_dir,
            )
        )
    logger.initialize(path=output_dir, backends=backends)

    # Shapes and action bounds come from the env's gymnasium spaces — one
    # spelling for every backend, whether the bounds originate in
    # `mj_model.actuator_ctrlrange` or in a pool's declared action space.
    obs_space, act_space = env.single_observation_space, env.single_action_space
    action_low = jnp.asarray(act_space.low, dtype=jnp.float32)
    action_high = jnp.asarray(act_space.high, dtype=jnp.float32)

    # The agent config IS the constructor call: `_target_` names the class and
    # every sibling key is one of its keywords, so a knob that exists in Python
    # but not in the yaml fails loudly here instead of silently taking its
    # default. Only the four env-derived arguments are injected.
    #
    # `_recursive_=False` keeps the nested `*_config` blocks as DictConfigs: the
    # agent instantiates its own actor/critic/memory/optimizers, injecting shapes
    # (in_features, action_dim, rngs, num_atoms) that are unknown out here.
    agent_kwargs = dict(
        env_obs_size=space_size(obs_space),
        env_action_size=space_size(act_space),
        action_low=action_low,
        action_high=action_high,
    )
    # `noise` is its own top-level config group (agents that explore from their
    # own policy — SAC, MPO, PPO — carry no noise group and take no such arg).
    if "noise" in cfg:
        agent_kwargs["noise_config"] = cfg.noise

    agent = hydra.utils.instantiate(cfg.agent, _recursive_=False, **agent_kwargs)

    # Resume: the agent was just built from the CONFIG, and only its numbers come
    # from the checkpoint — so the yaml stays authoritative and a resume may
    # legitimately raise `trainer.steps` or retune a knob (unlike `play.py`,
    # which rebuilds the agent from the checkpoint's own hyperparameters).
    #
    # `resume=` accepts a run dir, its `checkpoints/` dir, or one `step_<N>` dir;
    # `resume.path` in a config does the same for a run that wants it recorded.
    # The returned metadata is the trainer's progress: env steps, epochs,
    # episodes, gradient steps, and whether a replay buffer came back with it.
    resume_cfg = cfg.get("resume") or {}
    if isinstance(resume_cfg, str):  # `resume: <path>` rather than `resume.path`
        resume_cfg = {"path": resume_cfg}
    resume_path = _resume or resume_cfg.get("path", None)
    resume_metadata = None
    if resume_path:
        checkpoint = find_checkpoint(resume_path)
        resume_metadata = agent.restore(checkpoint)
        # `Trainer.save` has recorded `steps` in the metadata since resume
        # existed; older checkpoints only have it in the directory name.
        if not resume_metadata.get("steps"):
            resume_metadata["steps"] = checkpoint_steps(checkpoint) or 0
        print(
            f"Resuming from {checkpoint} at {int(resume_metadata['steps']):,} "
            f"env steps (target {int(cfg.trainer.steps):,}).",
            flush=True,
        )

    # Seeded from the config, but a resume offsets the env stream by the steps
    # already taken: replaying the identical reset/exploration key sequence the
    # first leg consumed would make the resumed segment revisit exactly the start
    # states it has already trained on, which a run that never stopped would
    # never do. `agent` is untouched — its stream feeds gradient-step keys, whose
    # value comes from being reproducible.
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
        # Opt-in: checkpoints then carry the replay buffer, which makes a resumed
        # off-policy run exact (no warmup refill) at the cost of the buffer's
        # full size on disk per save.
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
