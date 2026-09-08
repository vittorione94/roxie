"""Loading envs: the MuJoCo adapters and the two built-in builders.

A builder turns the ``env:`` block into an ``EnvBundle`` of ready-to-drive vector
envs. Every builder in the repo returns the same thing, so ``train.py`` and
``play.py`` never branch on the env type.

**The env yaml IS the builder call**, exactly as the agent yaml is the
constructor call: ``env._target_`` names the builder and every sibling key is one
of its keyword arguments, so ``roxie/configs/env/`` is a Hydra config group like
``agent/`` and ``noise/`` (``env=playground env.env_name=CheetahRun``). ``build_env``
below is the one place that instantiates it; ``TRAINER_ENV_KEYS`` lists the keys
that live in the block but belong to the trainer rather than the builder.

``FuncEnv``'s eight methods split into the two that MAKE PHYSICS HAPPEN
(``initial``, ``transition``) and the six that ANSWER QUESTIONS ABOUT A STATE
(``observation``, ``reward``, ``terminal``, ``truncal``, ``state_info``,
``transition_info``). Every MuJoCo env here answers the six identically — read
the field off ``mjx_env.State`` — but produces that state differently. That split
is the class split::

    FuncEnv                  the interface; knows nothing about MuJoCo
    └── MuJoCoFuncEnv        + "my state is an mjx_env.State" -> the 6 accessors
        └── PlaygroundFuncEnv    + initial/transition by WRAPPING a playground env

The middle class deliberately carries no ``initial``/``transition``, so a bespoke
MuJoCo env in another repo inherits the six accessors instead of rewriting them.
That matters most for ``terminal`` versus ``truncal``, where a mistake is silent
and expensive: call a non-failure cutoff a termination and the critic zeroes its
bootstrap there. The accessors read fields back rather than recomputing them
because Playground produces physics, observation, reward and termination in one
``step``; recomputing any of them would pay for the physics twice.

EnvPool needs no adapter: its ``env_type="gymnasium"`` pools already return
``(obs, reward, terminated, truncated, info)`` and auto-reset and time-limit
internally, which is what ``roxie.environment.vector.EnvPoolVectorEnv`` presents,
so ``build_envpool_env`` below is a builder and nothing more.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from hydra.utils import get_method, instantiate
from omegaconf import DictConfig, OmegaConf
from mujoco_playground import registry

from roxie.environment import functional
from roxie.environment.functional import FuncEnv
from roxie.environment.vector import EnvPoolVectorEnv, JaxVectorEnv


class MuJoCoFuncEnv(FuncEnv):
    """Half a ``FuncEnv``: everything that follows from "my state is an
    ``mjx_env.State``".

    Deliberately no ``initial``/``transition`` — a subclass supplies those, by
    wrapping an env (``PlaygroundFuncEnv``) or by being one.
    """

    def observation(self, state, rng, params=None):
        return state.obs

    def reward(self, state, action, next_state, rng, params=None):
        return next_state.reward

    def terminal(self, state, rng, params=None):
        # MuJoCo envs report a float `done`; the driver wants a predicate.
        return state.done.astype(jnp.bool_)

    def truncal(self, state, rng, params=None):
        # `.get` runs at trace time on a plain Python dict, so an env without
        # the key costs nothing.
        return jnp.asarray(state.info.get("truncation", False), dtype=jnp.bool_)

    def transition_info(self, state, action, next_state, params=None):
        return {"metrics": next_state.metrics}


class PlaygroundFuncEnv(MuJoCoFuncEnv):
    """The other half, for a ``mujoco_playground`` env: wrap one and point
    ``initial``/``transition`` at its ``reset``/``step``.

    The state IS ``mjx_env.State``, unwrapped, which is what lets ``play.py``
    drive the same object single-world.
    """

    def __init__(self, env: Any, impl: str | None = None):
        self.env = env
        self.observation_space = functional.unbounded_box(env.observation_size)
        ctrl_range = env.mj_model.actuator_ctrlrange
        self.action_space = functional.box(ctrl_range[:, 0], ctrl_range[:, 1])
        self.metadata = {"jax": True, "impl": impl or _read_impl(env)}

    def initial(self, rng, params=None):
        return self.env.reset(rng)

    def transition(self, state, action, rng, params=None):
        return self.env.step(state, action)

    def __getattr__(self, name):
        """Forward anything else to the wrapped env.

        `play.py` and the viewer hooks want `mj_model`, `xml_path`, the native
        stepper protocol and so on; forwarding avoids enumerating them.
        """
        # Guard the dunder case so copy/pickle don't recurse through `self.env`.
        if name.startswith("__") or "env" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.env, name)


def _read_impl(env: Any) -> str:
    """The physics backend an env actually loaded.

    Read off the constructed env rather than echoed from config, so the startup
    banner reports what really happened. Playground envs keep it on
    ``_config.impl``; a bespoke env keeps it on ``_impl``.
    """
    impl = getattr(env, "_impl", None)
    if impl is None:
        cfg = getattr(env, "_config", None)
        if cfg is not None and "impl" in cfg:
            impl = cfg["impl"]
    return impl or "unknown"


def resolve_loaded_impl(env: Any) -> str:
    """Read the loaded physics backend off a driver or a func env."""
    func_env = getattr(env, "func_env", env)
    meta = getattr(func_env, "metadata", None) or {}
    return meta.get("impl") or _read_impl(func_env)


def resolve_agent_device() -> tuple[str, str]:
    """Where the agent's arrays live: ``("cpu"|"gpu", exact jax backend)``.

    The agent is pure JAX, so this is simply JAX's chosen backend. Anything not
    the CPU counts as "gpu" for the coarse comparison against the declared
    ``runtime.agent_device``; the exact backend string ("cuda", "rocm",
    "METAL", ...) rides alongside so the banner says which one answered.
    """
    backend = jax.default_backend()
    return ("cpu" if backend == "cpu" else "gpu", backend)


def resolve_env_device(env: Any) -> tuple[str, str]:
    """Where the physics runs: ``("cpu"|"gpu", how)``.

    Not the same question as the agent's device: an EnvPool pool steps native
    MuJoCo on C++ threads whatever JAX is doing, while an MJX/Warp env rides the
    agent's JAX device — there is one XLA program.
    """
    impl = resolve_loaded_impl(env)
    if impl == "envpool":
        return "cpu", "native MuJoCo, C++ thread pool"
    device, backend = resolve_agent_device()
    return device, f"{impl} kernels on the JAX device ({backend})"


def log_loaded_backend(
    env: Any,
    requested_impl: str | None = None,
    agent_device: str | None = None,
    env_device: str | None = None,
) -> None:
    """Print a loud banner reporting what loaded, and on which hardware.

    The two halves are placed independently — physics on CPU threads while the
    learner sits on the card — so the banner states both.
    ``agent_device``/``env_device`` are the ``runtime:`` declarations, checked
    here rather than trusted: a cell declaring ``gpu`` still runs on a machine
    with no card, and this is what says so.
    """
    actual = resolve_loaded_impl(env)
    agent_actual, agent_detail = resolve_agent_device()
    env_actual, env_detail = resolve_env_device(env)

    bar = "=" * 70
    lines = [
        "",
        bar,
        f"  PHYSICS BACKEND LOADED:  impl = {actual.upper()}",
        f"  AGENT (networks, optimizers, replay):  {agent_actual.upper():<4}"
        f"  [{agent_detail}]",
        f"  ENV   (physics):                       {env_actual.upper():<4}"
        f"  [{env_detail}]",
    ]
    for name, declared, got in (
        ("env.impl", requested_impl, actual),
        ("agent.device", agent_device, agent_actual),
        ("env.device", env_device, env_actual),
    ):
        if declared is None:
            continue
        if str(declared) != got:
            lines.append(
                f"  !!! MISMATCH !!!  {name} declares {str(declared)!r} "
                f"but the run is on {got!r}"
            )
        else:
            lines.append(f"  OK - matches {name} = {str(declared)!r}")
    lines += [bar, ""]
    print("\n".join(lines), flush=True)


# CUDA-graph capture mode for the Warp backend. mjx defaults to
# ``GraphMode.WARP``, which LEAKS HOST RAM under JAX -- see `_set_graph_mode`.
DEFAULT_WARP_GRAPH_MODE = "WARP_STAGED_EX"


def _set_graph_mode(env: Any, graph_mode: str) -> None:
    """Re-put a playground env's mjx model under an explicit CUDA-graph mode.

    Playground calls ``mjx.put_model(mj_model, impl=...)`` inside each env's
    ``__init__`` and exposes no way to pass ``graph_mode``, so mjx's own default
    applies: ``GraphMode.WARP``, whose capture cache is keyed on the step's
    input/output buffer ADDRESSES. Under JAX those addresses change every step,
    so a new CUDA graph is captured per step, and the cache's eviction only drops
    the Python reference -- the native host descriptors are never reclaimed.

    The result is a host-RAM leak of ~0.1 GB per million env steps that nothing
    in JAX accounts for (``jax.live_arrays()`` stays flat, GPU memory stays
    flat), which on a 500M-step release run reaches ~50 GB and gets the process
    OOM-killed several hours in. ``WARP_STAGED_EX`` captures the graph ONCE
    against fixed staging buffers and replays it, so it keeps graph-replay speed;
    the eager ``JAX``/``NONE`` modes avoid the leak too, but by launching the
    step's many small kernels one at a time, which is much slower.

    Re-putting after construction is safe: playground's ``_post_init`` reads
    ``mj_model``, never the mjx model, and every dm_control-suite env keeps its
    single copy on ``_mjx_model``. If that attribute ever disappears upstream,
    fail loudly here rather than leak silently for six hours.
    """
    from mujoco import mjx
    from mujoco.mjx.warp import types as mjxw_types

    if not hasattr(env, "_mjx_model"):
        raise RuntimeError(
            f"{type(env).__name__} has no `_mjx_model` to re-put; mujoco_playground "
            "changed shape. Without it the Warp backend runs under "
            "GraphMode.WARP, which leaks host RAM until the OOM killer fires."
        )
    env._mjx_model = mjx.put_model(
        env.mj_model,
        impl="warp",
        graph_mode=getattr(mjxw_types.GraphMode, graph_mode),
    )


def load_playground_env(
    env_name: str,
    impl: str | None = None,
    naconmax: int | None = None,
    njmax: int | None = None,
    graph_mode: str | None = None,
):
    """Load a mujoco_playground env, optionally forcing the physics backend.

    The playground env configs expose ``impl`` (``"jax"``/``"warp"``) plus the
    Warp contact/constraint budgets (``naconmax``/``njmax``). We override them
    only when explicitly provided so each env keeps its upstream-tuned default
    otherwise (e.g. Warp's per-env ``naconmax``).

    ``graph_mode`` is Warp-only and defaults to ``DEFAULT_WARP_GRAPH_MODE``
    rather than to mjx's own default, which leaks host RAM: see
    `_set_graph_mode`. An MJX (``impl="jax"``) env captures no graphs at all and
    is left untouched.
    """
    env_cfg = registry.get_default_config(env_name)
    if impl is not None and "impl" in env_cfg:
        env_cfg.impl = impl
    if naconmax is not None and "naconmax" in env_cfg:
        env_cfg.naconmax = naconmax
    if njmax is not None and "njmax" in env_cfg:
        env_cfg.njmax = njmax

    env = registry.load(env_name, config=env_cfg)
    if str(env_cfg.get("impl", "")) == "warp":
        _set_graph_mode(env, graph_mode or DEFAULT_WARP_GRAPH_MODE)
    return PlaygroundFuncEnv(env), env_cfg


class EnvBundle(NamedTuple):
    """Normalized result of an env builder.

    ``env`` and ``test_env`` are vector envs (see ``roxie.environment.vector``);
    ``test_env`` of None means "reuse the training env". ``env_cfg`` is the
    resolved env config when one exists (the playground registry returns one;
    bespoke example envs may not).
    """

    env: Any
    test_env: Any
    env_cfg: Any


# Default episode step limit, applied when neither the env config nor the
# experiment yaml names one.
DEFAULT_MAX_EPISODE_STEPS = 1000

# Keys that live under ``env:`` but are NOT builder arguments — roxie consumes
# them itself, so ``build_env`` strips them before instantiating. Everything not
# listed is forwarded, which makes a stale key a loud ``TypeError``.
TRAINER_ENV_KEYS = frozenset({
    "builder",         # pre-`_target_` spelling; still honoured by build_env
    "device",          # which hardware the physics runs on (see resolve_placement)
    "parallel_envs",   # -> num_envs
    "test_episodes",   # -> test_episodes, from trainer.test_episodes
    "viewer",          # play.py's ghost-renderer hook
    "player",          # play.py's stepper factory
})


def build_playground_env(
    env_name: str,
    *,
    mode: str = "train",
    num_envs: int = 1,
    test_episodes: int = 1,
    seed: int = 0,
    impl: str | None = "jax",
    naconmax: int | None = None,
    njmax: int | None = None,
    graph_mode: str | None = None,
    max_episode_steps: int | None = None,
) -> EnvBundle:
    """Builder for mujoco_playground envs.

    This is the default builder: experiments that don't set ``env._target_`` get
    a playground env loaded by ``env.env_name``. ``mode`` is part of the builder
    protocol (train.py passes "train", play.py "play") but playground envs load
    identically for both. Override semantics for ``naconmax``/``njmax`` match
    ``load_playground_env``: ``None`` leaves the env's upstream Warp budget.
    ``graph_mode`` is Warp-only and does NOT fall through to mjx's default,
    which leaks host RAM -- see `_set_graph_mode`.

    ``num_envs``/``test_episodes`` size the two drivers and come from the trainer:
    the same env definition is driven at 256 worlds for training and at
    ``trainer.test_episodes`` for evaluation.

    ``seed`` is accepted and unused — an MJX env takes its reset keys from the
    trainer's rng stream — but declared so the shared ``env.seed`` key can stay
    in the block for every backend (the envpool pools do seed from it).
    """
    func_env, env_cfg = load_playground_env(
        env_name, impl=impl, naconmax=naconmax, njmax=njmax,
        graph_mode=graph_mode,
    )
    max_steps = int(
        max_episode_steps
        or env_cfg.get("episode_length", None)
        or DEFAULT_MAX_EPISODE_STEPS
    )
    # One env object, two drivers: the heavy mjx model and its constants are
    # shared, only the batch size differs.
    return EnvBundle(
        env=JaxVectorEnv(func_env, num_envs, max_episode_steps=max_steps),
        test_env=JaxVectorEnv(func_env, test_episodes, max_episode_steps=max_steps),
        env_cfg=env_cfg,
    )


def build_envpool_env(
    task_id: str,
    *,
    mode: str = "train",
    num_envs: int = 1,
    test_episodes: int = 1,
    seed: int = 0,
    max_episode_steps: int | None = None,
    **task_kwargs: Any,
) -> EnvBundle:
    """Builder for envpool environments (CPU training path).

    Selected from an experiment yaml with::

        env:
          _target_: roxie.environment.loader.build_envpool_env
          task_id: HalfCheetah-v4
          parallel_envs: 32

    Arguments:
      task_id (str):            envpool task identifier, e.g. "HalfCheetah-v4".
      max_episode_steps (int):  episode time-limit (default 1000).
      seed (int):               RNG seed.
      **task_kwargs:            any other key under ``env:`` (minus
                                ``TRAINER_ENV_KEYS``) is forwarded to
                                ``envpool.make()`` as a task kwarg.

    ``num_envs``/``test_episodes`` come from the trainer, not the yaml: they size
    the two pools, and the eval pool MUST match ``trainer.test_episodes``.

    Two independent pools are created so eval rollouts never corrupt the
    training environment state.
    """
    import envpool  # lazy: envpool is an optional dependency

    max_steps = int(max_episode_steps or DEFAULT_MAX_EPISODE_STEPS)
    seed = int(seed)

    def make_pool(n: int, pool_seed: int):
        return envpool.make(
            task_id,
            env_type="gymnasium",
            num_envs=n,
            seed=pool_seed,
            max_episode_steps=max_steps,
            **task_kwargs,
        )

    train_env = EnvPoolVectorEnv(
        make_pool(num_envs, seed), num_envs=num_envs,
        max_episode_steps=max_steps,
    )
    # The eval pool carries a rebuild thunk so the rollout can pin its start
    # states to a fixed seed before every eval (see `EnvPoolVectorEnv.reseed`).
    test_env = EnvPoolVectorEnv(
        make_pool(test_episodes, seed + 1), num_envs=test_episodes,
        max_episode_steps=max_steps,
        rebuild=lambda s: make_pool(test_episodes, s),
    )

    return EnvBundle(env=train_env, test_env=test_env, env_cfg=None)


DEFAULT_BUILDER = "roxie.environment.loader.build_playground_env"


def builder_target(cfg_env: Any) -> str:
    """The dotted path of the builder an ``env:`` block names."""
    return (
        cfg_env.get("_target_", None)
        or cfg_env.get("builder", None)  # pre-`_target_` spelling
        or DEFAULT_BUILDER
    )


def uses_envpool(cfg_env: Any) -> bool:
    """Whether this env block builds EnvPool pools.

    Asked by ``train.py`` (an EnvPool run has no reason to claim the GPU, so the
    learner defaults onto the CPU with the physics) and by ``play.py`` (a pool
    hands out no ``MjModel``, so there is nothing to open a viewer on).

    It is the BUILDER that answers this, not an ``impl`` key: ``impl`` names the
    physics implementation *within* a builder — MJX or Warp kernels for the same
    playground env — and EnvPool is not one of those, it is the other builder.
    """
    try:
        return get_method(builder_target(cfg_env)) is build_envpool_env
    except Exception:
        # A stale dotted path is not this function's error to raise: `build_env`
        # is about to fail on it with the message that actually helps.
        return False


DEVICES = ("cpu", "gpu")


def resolve_placement(cfg: Any) -> tuple[str | None, str | None, str | None]:
    """``(agent_device, env_device, jax platform to force)`` for a run.

    This reads the two knobs rather than reconciling them; ``log_loaded_backend``
    then checks each against where that half actually ended up.

    * ``agent.device`` always decides: the agent is pure JAX, so it IS the JAX
      platform.
    * ``env.device`` decides nothing by itself. An **EnvPool** env steps native
      MuJoCo on C++ threads and is on the CPU whatever JAX does — the real
      hybrid, ``agent: gpu`` + ``env: cpu`` — while a **playground** env is
      traced into the agent's XLA program and lands on the agent's device.
      Declaring it is still what lets the banner say the run is not where the
      experiment says it is.

    An EnvPool run with no ``agent.device`` defaults to the CPU rather than
    preallocating VRAM the physics cannot use. ``"gpu"`` and an absent
    declaration both force nothing — forcing a platform name would turn a
    missing card into an import error instead of a loud banner line.
    """
    agent_device = (cfg.get("agent") or {}).get("device", None)
    env_device = (cfg.get("env") or {}).get("device", None)
    for name, value in (("agent.device", agent_device), ("env.device", env_device)):
        if value is not None and str(value) not in DEVICES:
            raise ValueError(
                f"{name}={value!r} is not one of {DEVICES} "
                f"(null means 'whatever JAX picks')."
            )

    if agent_device is None and uses_envpool(cfg.env):
        agent_device = "cpu"
    platform = None if agent_device in (None, "gpu") else agent_device
    return agent_device, env_device, platform


def build_env(
    cfg_env: Any,
    *,
    mode: str = "train",
    num_envs: int = 1,
    test_episodes: int = 1,
) -> EnvBundle:
    """Instantiate the ``env:`` block into an ``EnvBundle``.

    The single entry point both ``train.py`` and ``play.py`` use, so the two
    agree on what an env block means. It does three things and nothing else:

    1. strips ``TRAINER_ENV_KEYS`` — the keys roxie consumes itself;
    2. falls back to ``DEFAULT_BUILDER`` when the block names no ``_target_``
       (and still honours the older ``builder:`` spelling, so a checkpoint saved
       before the env group existed can still be replayed by ``play.py``);
    3. hands the rest to ``hydra.utils.instantiate`` with the three trainer
       quantities injected — those override any same-named key in the yaml.

    Everything else about an env — which builder, which task, the physics
    backend, a bespoke builder's nested blocks — is yaml, which is what lets a
    task living in another repo be launched through ``roxie.train`` unchanged.
    """
    # Resolve against the composed config FIRST: every interpolation in the
    # block is relative to the config root, and `instantiate` re-creates what it
    # is handed as a fresh, parentless node.
    if isinstance(cfg_env, DictConfig):
        resolved = OmegaConf.to_container(
            cfg_env, resolve=True, throw_on_missing=True,
        )
    else:
        resolved = dict(cfg_env)
    node = {k: v for k, v in resolved.items() if k not in TRAINER_ENV_KEYS}
    node["_target_"] = builder_target(cfg_env)
    return instantiate(
        node, mode=mode, num_envs=num_envs, test_episodes=test_episodes,
    )
