"""Loading envs: the MuJoCo adapters and the two built-in builders.

A builder (``env.builder`` in the experiment yaml, a dotted path) turns a config
block into an ``EnvBundle`` of ready-to-drive vector envs. Every builder in the
repo returns the same thing, so ``train.py`` and ``play.py`` never branch on the
env type.

``FuncEnv``'s eight methods fall into two groups: the two that MAKE PHYSICS
HAPPEN (``initial``, ``transition``) and the six that ANSWER QUESTIONS ABOUT A
STATE (``observation``, ``reward``, ``terminal``, ``truncal``, ``state_info``,
``transition_info``). Every MuJoCo env in this repo answers the six identically —
read the field off ``mjx_env.State``, which already carries them — but produces
that state in completely different ways. That split is the class split::

    FuncEnv                  the interface; knows nothing about MuJoCo
    └── MuJoCoFuncEnv        + "my state is an mjx_env.State" -> the 6 accessors
        ├── PlaygroundFuncEnv    + initial/transition by WRAPPING a playground env
        └── MocapTrackingEnv     + initial/transition of its OWN (examples/mocap)

The middle class exists for exactly one reason: playground and mocap must never
drift on what ``terminal`` versus ``truncal`` means. That is the rule where a
mistake is silent and expensive — call the clip-end cutoff a termination and the
critic zeroes its bootstrap there — so the two share one implementation of it
rather than each writing their own.

Why the accessors read fields back rather than computing them: Playground (like
Brax, and like Waymax) computes physics, observation, reward and termination in a
single ``step`` call, so recomputing any of them independently would mean paying
for the physics twice. That is a thin, honest adaptation and not a workaround —
``mjx_env.State`` already carries exactly the fields ``FuncEnv`` asks for.

EnvPool needs no adapter at all. Its ``env_type="gymnasium"`` pools already
return ``(obs, reward, terminated, truncated, info)`` and already auto-reset and
time-limit internally, which is exactly what
``roxie.environment.vector.EnvPoolVectorEnv`` presents — so ``build_envpool_env``
below is a builder and nothing more. (It used to live in its own module together
with two duck-typed stand-in classes whose only job was translating the
gymnasium 5-tuple INTO roxie's bespoke state; the classes and the module are
both gone.)
"""

from typing import Any, NamedTuple

import jax.numpy as jnp
from mujoco_playground import registry

from roxie.environment import functional
from roxie.environment.functional import FuncEnv
from roxie.environment.vector import EnvPoolVectorEnv, JaxVectorEnv


class MuJoCoFuncEnv(FuncEnv):
    """Half a ``FuncEnv``: everything that follows from "my state is an
    ``mjx_env.State``".

    Deliberately does NOT implement ``initial``/``transition`` — a subclass
    supplies those, either by wrapping an env (``PlaygroundFuncEnv``) or by being
    one (``MocapTrackingEnv``). See the module docstring for the hierarchy.
    """

    def observation(self, state, rng, params=None):
        return state.obs

    def reward(self, state, action, next_state, rng, params=None):
        return next_state.reward

    def terminal(self, state, rng, params=None):
        # MuJoCo envs report a float `done`; the driver wants a predicate. Envs
        # that fold their own non-failure cutoff into `done` declare it in
        # `truncal`, and the driver subtracts it — see `JaxVectorEnv.step`.
        return state.done.astype(jnp.bool_)

    def truncal(self, state, rng, params=None):
        # The mocap clip end is the case that matters. `.get` runs at TRACE time
        # on a plain Python dict, so an env without the key costs nothing and
        # simply never truncates for its own reasons.
        return jnp.asarray(state.info.get("truncation", False), dtype=jnp.bool_)

    def transition_info(self, state, action, next_state, params=None):
        return {"metrics": next_state.metrics}


class PlaygroundFuncEnv(MuJoCoFuncEnv):
    """The other half, for a ``mujoco_playground`` env: wrap one and point
    ``initial``/``transition`` at its ``reset``/``step``.

    The state IS ``mjx_env.State`` — unwrapped, no roxie-specific box around it.
    That is what lets ``play.py`` drive the same object single-world, and what
    removed the third state shape from the codebase.
    """

    def __init__(self, env: Any, impl: str | None = None):
        self.env = env
        self.observation_space = functional.unbounded_box(env.observation_size)
        # MuJoCo declares its action bounds on the model, so the space is exact
        # rather than a convention. This is the only place that reads
        # `actuator_ctrlrange`; the agents take their bounds from the space.
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
        stepper protocol and so on. Forwarding keeps this adapter from having to
        enumerate them.
        """
        # `__getattr__` only fires for names not found normally, but guard the
        # dunder case so copy/pickle don't recurse through `self.env`.
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


def log_loaded_backend(env: Any, requested_impl: str | None = None) -> None:
    """Print a loud banner reporting the loaded physics backend."""
    actual = resolve_loaded_impl(env)
    mismatch = requested_impl is not None and actual != requested_impl

    bar = "=" * 70
    lines = [
        "",
        bar,
        f"  PHYSICS BACKEND LOADED:  impl = {actual.upper()}",
    ]
    if requested_impl is not None:
        if mismatch:
            lines.append(
                f"  !!! MISMATCH !!!  requested {requested_impl!r} "
                f"but env loaded {actual!r}"
            )
        else:
            lines.append(f"  OK - matches requested impl = {requested_impl!r}")
    lines += [bar, ""]
    print("\n".join(lines), flush=True)


def load_playground_env(
    env_name: str,
    impl: str | None = None,
    naconmax: int | None = None,
    njmax: int | None = None,
):
    """Load a mujoco_playground env, optionally forcing the physics backend.

    The playground env configs expose ``impl`` (``"jax"``/``"warp"``) plus the
    Warp contact/constraint budgets (``naconmax``/``njmax``). We override them
    only when explicitly provided so each env keeps its upstream-tuned default
    otherwise (e.g. Warp's per-env ``naconmax``).
    """
    env_cfg = registry.get_default_config(env_name)
    if impl is not None and "impl" in env_cfg:
        env_cfg.impl = impl
    if naconmax is not None and "naconmax" in env_cfg:
        env_cfg.naconmax = naconmax
    if njmax is not None and "njmax" in env_cfg:
        env_cfg.njmax = njmax

    env = registry.load(env_name, config=env_cfg)
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


def build_playground_env(
    cfg_env: Any, mode: str = "train", num_envs: int = 1, test_episodes: int = 1,
) -> EnvBundle:
    """Builder for mujoco_playground envs.

    This is the default builder: experiments that don't set ``env.builder`` get
    a playground env loaded by ``env.env_name``. ``mode`` is part of the builder
    protocol (train.py passes "train", play.py "play") but playground envs load
    identically for both. Override semantics for ``naconmax``/``njmax`` match
    ``load_playground_env``: ``None`` leaves the env's upstream Warp budget.

    ``num_envs``/``test_episodes`` size the two drivers. They come from the
    trainer rather than from ``cfg_env`` because they are trainer quantities —
    the same env definition is driven at 256 worlds for training and at
    ``trainer.test_episodes`` for evaluation.
    """
    func_env, env_cfg = load_playground_env(
        cfg_env.env_name,
        impl=cfg_env.get("impl", "jax"),
        naconmax=cfg_env.get("naconmax", None),
        njmax=cfg_env.get("njmax", None),
    )
    # The step limit, most specific source first: an explicit experiment
    # override, then the env's own declared episode length, then the fallback.
    # The middle term is what `TerminationWrapper` used to ignore — it always
    # took the 1000 default — so this only changes behaviour for a playground
    # env whose `episode_length` is not 1000. WalkerWalk's is.
    max_steps = int(
        cfg_env.get("max_episode_steps", None)
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
    cfg_env: Any, mode: str = "train", num_envs: int = 1, test_episodes: int = 1,
) -> EnvBundle:
    """Builder for envpool environments (CPU training path).

    Selected from an experiment yaml with::

        env:
          builder: roxie.environment.loader.build_envpool_env
          task_id: HalfCheetah-v4
          parallel_envs: 32

    Reads from cfg_env:
      task_id (str):            envpool task identifier, e.g. "HalfCheetah-v4".
      max_episode_steps (int):  episode time-limit (default 1000).
      seed (int):               RNG seed (default 0).
      (any other non-reserved key is forwarded to ``envpool.make()``)

    ``num_envs``/``test_episodes`` come from the trainer, not from cfg_env: they
    size the two pools, and the eval pool MUST match ``trainer.test_episodes``.
    Taking them as arguments is what stops the two from drifting — they used to
    be separate config keys and a mismatch failed a broadcast mid-eval.

    Two independent pools are created — one for training, one for evaluation —
    so eval rollouts never corrupt the training environment state.
    """
    import envpool  # lazy: envpool is an optional dependency

    task_id: str = cfg_env.task_id
    seed: int = int(cfg_env.get("seed", 0))
    max_episode_steps: int = int(
        cfg_env.get("max_episode_steps", None) or DEFAULT_MAX_EPISODE_STEPS
    )

    _reserved = frozenset({
        "task_id", "parallel_envs", "seed", "max_episode_steps",
        "test_episodes", "builder", "impl", "viewer", "player",
    })
    extra = {k: v for k, v in cfg_env.items() if k not in _reserved}

    def make_pool(n: int, pool_seed: int):
        return envpool.make(
            task_id,
            env_type="gymnasium",
            num_envs=n,
            seed=pool_seed,
            max_episode_steps=max_episode_steps,
            **extra,
        )

    train_env = EnvPoolVectorEnv(
        make_pool(num_envs, seed), num_envs=num_envs,
        max_episode_steps=max_episode_steps,
    )
    # The eval pool carries a rebuild thunk so the rollout can pin its start
    # states to a fixed seed before every eval (see `EnvPoolVectorEnv.reseed`).
    test_env = EnvPoolVectorEnv(
        make_pool(test_episodes, seed + 1), num_envs=test_episodes,
        max_episode_steps=max_episode_steps,
        rebuild=lambda s: make_pool(test_episodes, s),
    )

    return EnvBundle(env=train_env, test_env=test_env, env_cfg=None)


# Dotted path of the default builder, used when ``env.builder`` is unset.
DEFAULT_BUILDER = "roxie.environment.loader.build_playground_env"
