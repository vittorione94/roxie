import os
# Always run playback on CPU. It drives a single world into the viewer, so GPU
# throughput buys nothing, and CPU/MJX playback is exactly reproducible — the warp
# GPU backend's atomic contact reductions make even identical rollouts diverge, a
# tiny per-step difference that the chaotic contact dynamics amplify.
# JAX_PLATFORMS is read at jax import, so this must be set before `import jax`.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

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
    log_loaded_backend,
)
from roxie.utils import hydra_searchpath

# examples/ and any out-of-repo task tree are not part of the installed roxie
# package; put the repo root and the working directory on the path so an env
# builder named by a checkpoint's config stays importable.
sys.path.insert(0, str(hydra_searchpath.REPO_ROOT))
sys.path.insert(0, os.getcwd())
# A checkpoint's stored config can hold ${envpool_task:...}, so the resolver has
# to exist before that config is loaded back.
suites.register_resolvers()


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.option("--checkpoint-path", type=str, help="Path to the checkpoint file.")
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def main(checkpoint_path, overrides):
    cfg_path = os.path.join(checkpoint_path, "../../.hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)

    # Trailing ``dotted.key=value`` args override the saved run config. This is a
    # Click CLI, not a Hydra entrypoint, so it stands in for Hydra's CLI overrides
    # — e.g. ``env.config.early_termination=false`` to watch a clip run to its end.
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    # JAX is forced onto CPU above, but the warp backend targets CUDA, so a
    # warp-trained checkpoint could not build its env here. Checkpoints are
    # agent-side and the policy replays identically on either backend, so the
    # physics backend is coerced to MJX; warp-only knobs are ignored by its builder.
    if cfg.env.get("impl", None) == "warp":
        cfg.env.impl = "jax"

    # EnvPool cannot be played back at all: the pool steps native MuJoCo inside
    # C++ and hands out nothing but observations — no `MjModel` to open a viewer
    # on, no per-world state to draw. So an envpool-trained checkpoint is
    # replayed on the playground twin of the same dm_control task. That is sound
    # for the same reason the release grid compares the two cells at all: they
    # are two implementations of one task definition, same observation layout
    # and same action bounds (see `roxie.environment.suites`).
    if cfg.env.get("impl", None) == "envpool":
        task = suites.playground_task(cfg.env.task_id)
        print(
            f"EnvPool has no viewer; replaying {cfg.env.task_id} on the "
            f"mujoco_playground env {task!r}.",
            flush=True,
        )
        # Edited in place rather than replaced: the saved agent config
        # interpolates `${env.parallel_envs}` (and a builder ignores every key
        # it does not read), so dropping the block would break instantiation.
        # Only the two keys that choose the env change, plus the pool-only ones
        # that would otherwise send this straight back to envpool.
        cfg.env.env_name = task
        cfg.env.impl = "jax"
        cfg.env.pop("builder", None)
        cfg.env.pop("task_id", None)

    # Match the training run's matmul precision so playback evaluates the policy
    # the same way it was trained.
    matmul_precision = (cfg.get("runtime") or {}).get("matmul_precision", None)
    if matmul_precision:
        jax.config.update("jax_default_matmul_precision", matmul_precision)

    key = jax.random.PRNGKey(seed=0)

    # Same builder protocol as train.py; ``mode="play"`` lets the builder apply
    # playback-specific tweaks, such as shrinking a GPU clip pool.
    build_env = get_method(cfg.env.get("builder", DEFAULT_BUILDER))
    env, _, env_cfg = build_env(cfg.env, mode="play", num_envs=1, test_episodes=1)

    log_loaded_backend(env, requested_impl=cfg.env.get("impl", "jax"))

    # Playback is SINGLE-WORLD, so it drives the `FuncEnv` directly rather than
    # the batched driver the trainer uses. This is the payoff of not mutating
    # the env with `transform(jax.vmap)` the way gymnasium's vector env does:
    # the same object is callable batched and unbatched.
    func_env = env.func_env

    # Same source of truth as train.py: the saved config's `_target_` names the
    # class. Every construction block it declares is forwarded, plus the separate
    # `noise` group; the remaining hyperparameters come from the checkpoint.
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

    # Optional per-env viewer override, a dotted path in ``env.viewer``. Used to
    # render a reference "ghost" alongside the policy; envs that don't set it just
    # render their own model.
    viewer_path = cfg.env.get("viewer", None)
    ghost = get_method(viewer_path)(func_env) if viewer_path else None
    model = ghost.model if ghost is not None else func_env.mj_model

    data = mujoco.MjData(model)

    # Wall-clock pacing: one loop iteration is one CONTROL step, which lasts
    # ``sim timestep x n_substeps``. Read that off the env — playground's
    # ``mjx_env`` publishes it as ``dt`` — rather than assuming a substep count,
    # which is how this used to run a 1-substep dm_control task at 1/20 speed.
    substeps = int(
        getattr(func_env, "n_substeps", None)
        or getattr(func_env, "native_n_substeps", None)
        or 1
    )
    frame_dt = float(
        getattr(func_env, "dt", None) or model.opt.timestep * substeps
    )

    # Prefer a native-MuJoCo CPU stepper when the env provides one. MJX is GPU-first
    # and pathologically slow for a single world on CPU, so the native player steps
    # ``mujoco.mj_step`` instead while reusing the env's own obs/reward.
    #
    # The default factory returns a player for any env implementing the ``native_*``
    # protocol and None for the rest, which fall back to the jitted-MJX path below.
    # ``env.player`` overrides it (dotted path, or null to force MJX).
    player_path = cfg.env.get("player", "roxie.utils.native_player.make_native_player")
    player = get_method(player_path)(func_env) if player_path else None
    if player is not None:
        print("Playback stepper: native MuJoCo (CPU)", flush=True)
        jit_reset, jit_step = player.reset, player.step
    else:
        # The FuncEnv entry points. `transition` takes an rng that every MuJoCo
        # env here ignores (its own stream rides in the state), so a fixed key
        # keeps playback reproducible.
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

            # Print the per-component rewards (envs expose these under
            # ``metrics["reward/*"]``) alongside the total step reward.
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
