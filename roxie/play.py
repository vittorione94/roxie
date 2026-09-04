import os
# Playback is single-world, so GPU throughput buys nothing, and CPU/MJX
# playback is exactly reproducible — warp's atomic contact reductions make even
# identical rollouts diverge. JAX_PLATFORMS is read at jax import.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import inspect
import sys
import time

import click
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from omegaconf import OmegaConf

from hydra.utils import get_class, get_method

from roxie.environment import suites
from roxie.environment.functional import space_size
from roxie.environment.loader import (
    DEFAULT_BUILDER,
    TRAINER_ENV_KEYS,
    build_env,
    build_playground_env,
    log_loaded_backend,
    uses_envpool,
)
from roxie.utils import hydra_searchpath

# Derived from the signature so it cannot drift out of sync with the builder.
_PLAYGROUND_ENV_KEYS = (
    set(inspect.signature(build_playground_env).parameters)
    | set(TRAINER_ENV_KEYS)
    | {"_target_"}
)

# So an env builder named by a checkpoint's config stays importable, whether it
# lives in examples/ or an out-of-repo task tree.
sys.path.insert(0, str(hydra_searchpath.REPO_ROOT))
sys.path.insert(0, os.getcwd())
# A checkpoint's stored config can hold ${envpool_task:...}.
suites.register_resolvers()


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.option("--checkpoint-path", type=str, help="Path to the checkpoint file.")
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def main(checkpoint_path, overrides):
    cfg_path = os.path.join(checkpoint_path, "../../.hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)

    # This is a Click CLI, so this stands in for Hydra's own CLI overrides.
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    # Warp targets CUDA and JAX is forced onto CPU above. Checkpoints are
    # agent-side and replay identically either way.
    if cfg.env.get("impl", None) == "warp":
        cfg.env.impl = "jax"

    # EnvPool cannot be played back: the pool hands out no `MjModel` to open a
    # viewer on. Its checkpoints replay on the playground twin of the same task,
    # which shares the observation layout and action bounds.
    if uses_envpool(cfg.env):
        task = suites.playground_task(cfg.env.task_id)
        print(
            f"EnvPool has no viewer; replaying {cfg.env.task_id} on the "
            f"mujoco_playground env {task!r}.",
            flush=True,
        )
        # Edited in place, not replaced: the saved agent config interpolates
        # `${env.parallel_envs}`. A leftover pool key is a TypeError.
        cfg.env["_target_"] = DEFAULT_BUILDER
        cfg.env.env_name = task
        cfg.env.impl = "jax"
        cfg.env.pop("builder", None)  # pre-`_target_` spelling
        cfg.env.pop("task_id", None)
        for key in list(cfg.env.keys()):
            if key not in _PLAYGROUND_ENV_KEYS:
                cfg.env.pop(key, None)

    # So playback evaluates the policy the same way it was trained.
    matmul_precision = (cfg.get("runtime") or {}).get("matmul_precision", None)
    if matmul_precision:
        jax.config.update("jax_default_matmul_precision", matmul_precision)

    key = jax.random.PRNGKey(seed=0)

    env, _, env_cfg = build_env(cfg.env, mode="play", num_envs=1, test_episodes=1)

    log_loaded_backend(env, requested_impl=cfg.env.get("impl", "jax"))

    # Single-world, so it drives the `FuncEnv` directly rather than the batched
    # driver the trainer uses.
    func_env = env.func_env

    # Every construction block the saved config declares, plus the separate
    # `noise` group; the rest comes from the checkpoint.
    agent_args = {
        key: value
        for key, value in cfg.agent.items()
        if key.endswith("_config")
    }
    if "noise" in cfg:
        agent_args["noise_config"] = cfg.noise

    agent = get_class(cfg.agent._target_).load(
        path=checkpoint_path,
        env_obs_size=space_size(func_env.observation_space),
        env_act_size=space_size(func_env.action_space),
        **agent_args,
    )

    # Optional per-env override, used to render a reference "ghost" alongside
    # the policy.
    viewer_path = cfg.env.get("viewer", None)
    ghost = get_method(viewer_path)(func_env) if viewer_path else None
    model = ghost.model if ghost is not None else func_env.mj_model

    data = mujoco.MjData(model)

    # Wall-clock pacing: one iteration is one control step of ``timestep x
    # n_substeps``. Read off the env — assuming a substep count would run a
    # 1-substep dm_control task at 1/20 speed.
    substeps = int(
        getattr(func_env, "n_substeps", None)
        or getattr(func_env, "native_n_substeps", None)
        or 1
    )
    frame_dt = float(
        getattr(func_env, "dt", None) or model.opt.timestep * substeps
    )

    # MJX is GPU-first and very slow for a single world on CPU, so a native
    # MuJoCo stepper wins when the env provides one; the default factory returns
    # None without the ``native_*`` protocol.
    player_path = cfg.env.get("player", "roxie.utils.native_player.make_native_player")
    player = get_method(player_path)(func_env) if player_path else None
    if player is not None:
        print("Playback stepper: native MuJoCo (CPU)", flush=True)
        jit_reset, jit_step = player.reset, player.step
    else:
        # `transition` takes an rng every MuJoCo env here ignores (its own
        # stream rides in the state), so a fixed key is reproducible.
        _step = jax.jit(func_env.transition)
        jit_reset = jax.jit(func_env.initial)

        def jit_step(state, action):
            return _step(state, action, jax.random.PRNGKey(0))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        key, reset_key = jax.random.split(key)
        state = jit_reset(reset_key)
        mujoco.mj_forward(model, data)

        score = 0.0
        actions = []

        while viewer.is_running():
            step_start = time.time()

            obs_b = jnp.expand_dims(state.obs, axis=0)
            action = agent.step(obs_b, evaluate=True, key=key)

            state = jit_step(state, action[0])

            if ghost is not None:
                info = state.info
                abs_idx = int(info["clip_start"]) + int(info["phase_idx"])
                data.qpos[:ghost.nq] = state.data.qpos
                data.qvel[:ghost.nv] = state.data.qvel
                data.qpos[ghost.nq:] = ghost.ref_qpos[abs_idx]
                data.qvel[ghost.nv:] = ghost.ref_qvel[abs_idx]
            else:
                data.qpos = state.data.qpos
                data.qvel = state.data.qvel

            data.ctrl = action
            mujoco.mj_forward(model, data)

            score += state.reward
            actions.append(action)

            metrics = state.metrics
            components = " ".join(
                f"{k.split('/', 1)[1]}={float(v):+.3f}"
                for k, v in metrics.items()
                if k.startswith("reward/")
            )
            print(f"reward={float(state.reward):+.3f} {components}")

            if state.done:
                print(f"Total score: {score}")
                print(f"Actions mean: {jnp.mean(jnp.array(actions)):.2f}")
                print(f"Actions std: {jnp.std(jnp.array(actions)):.2f}")
                key, reset_key = jax.random.split(reset_key)
                state = jit_reset(reset_key)
                mujoco.mj_resetData(model, data)
                score = 0.0
                actions = []

            viewer.sync()

            time_until_next_step = frame_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
