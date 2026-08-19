import os
# Always run playback on CPU. Playback drives a single world into the viewer at
# ~15x realtime, so GPU throughput buys nothing here, and CPU/MJX playback is
# exactly reproducible — the warp GPU backend's atomic contact reductions make
# even identical rollouts diverge (a ~1e-6 per-step difference that the chaotic
# contact dynamics amplify). JAX_PLATFORMS is read at jax import, so this must be
# set before `import jax` below.
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

from roxie.environment.loader import (
    DEFAULT_BUILDER,
    log_loaded_backend,
)
from roxie.utils import hydra_searchpath

# examples/ is not part of the installed roxie package; put the repo root on the
# path so the mocap example (imported lazily for mocap checkpoints) is importable.
sys.path.insert(0, str(hydra_searchpath.REPO_ROOT))


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.option("--checkpoint-path", type=str, help="Path to the checkpoint file.")
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def main(checkpoint_path, overrides):
    cfg_path = os.path.join(checkpoint_path, "../../.hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)

    # Trailing ``dotted.key=value`` args override the saved run config (play.py
    # is a Click CLI, not a Hydra entrypoint, so this stands in for Hydra's CLI
    # overrides). E.g. ``env.config.early_termination=false`` to watch a clip run
    # to its end instead of resetting on tracking collapse.
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    # We force JAX onto CPU (top of file), but the warp backend targets CUDA, so
    # a warp-trained checkpoint would fail to build its env here. Coerce the
    # physics backend to MJX for playback: checkpoints are agent-side, so the
    # policy replays identically on either backend (and warp-only knobs like the
    # contact budgets / graph_mode are simply ignored by the MJX builder).
    if cfg.env.get("impl", None) == "warp":
        cfg.env.impl = "jax"

    # Match the training run's matmul precision so playback evaluates the
    # policy the same way it was trained (see `runtime.matmul_precision`).
    matmul_precision = (cfg.get("runtime") or {}).get("matmul_precision", None)
    if matmul_precision:
        jax.config.update("jax_default_matmul_precision", matmul_precision)

    key = jax.random.PRNGKey(seed=0)

    # Same builder protocol as train.py: ``env.builder`` names the env factory;
    # ``mode="play"`` lets it apply playback-specific tweaks (the mocap builder
    # shrinks its GPU clip pool here).
    build_env = get_method(cfg.env.get("builder", DEFAULT_BUILDER))
    env, _, env_cfg = build_env(cfg.env, mode="play")

    log_loaded_backend(env, requested_impl=cfg.env.get("impl", "jax"))

    # Same source of truth as train.py: the saved config's `_target_` names the
    # class. Forward every construction block it declares (actor/critic/memory,
    # the optimizer blocks) plus the separate `noise` group; the remaining
    # hyperparameters come from the checkpoint itself.
    agent_args = {
        key: value
        for key, value in cfg.agent.items()
        if key.endswith("_config")
    }
    if "noise" in cfg:
        agent_args["noise_config"] = cfg.noise

    agent = get_class(cfg.agent._target_).load(
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

    # Prefer a native-MuJoCo CPU stepper when the env provides one (the mocap
    # env does, via ``env.player``). The env is written in MJX, which is
    # GPU-first and pathologically slow for a single world on CPU; the native
    # player steps ``mujoco.mj_step`` instead while reusing the env's own
    # obs/reward, so a checkpoint can be watched on a laptop with the GPU busy
    # training. Falls back to the jitted MJX reset/step for envs without one.
    # Default to the generic native-CPU player factory; it returns a player for
    # any env implementing the ``native_*`` protocol and None for the rest (which
    # then fall back to the jitted-MJX path below). Envs can override or disable
    # it via the ``env.player`` config key (dotted path, or null to force MJX).
    player_path = cfg.env.get("player", "roxie.utils.native_player.make_native_player")
    player = get_method(player_path)(env) if player_path else None
    if player is not None:
        print("Playback stepper: native MuJoCo (CPU)", flush=True)
        jit_reset = player.reset
        jit_step = player.step
    else:
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

            time_until_next_step = model.opt.timestep * 20 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
