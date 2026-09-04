"""Every env knob must be reachable from yaml — the agent-config rule, for envs.

`train.py` builds the env as `loader.build_env(cfg.env, ...)`, which is
`hydra.utils.instantiate` plus a strip of `TRAINER_ENV_KEYS`, so an env config IS
the builder call: `_target_` names the builder and every other key is one of its
keywords. A stale key is a `TypeError` at launch, after Hydra has composed and
the process has already paid for `import jax`; a missing one is a knob pinned to
its Python default, invisible in the run's config and in wandb.

Two shapes are checked, because env blocks come in two:

* the `roxie/configs/env/` group — a COMPLETE env block, so it must expose every
  builder keyword (and give the required ones a value or `???`);
* the `env:` block of an experiment config — a PARTIAL one, composed with other
  files, so only the "no unknown keys" direction can be checked there.
"""

import inspect
from pathlib import Path

import pytest
from hydra.utils import get_method
from omegaconf import OmegaConf

from roxie.environment.loader import TRAINER_ENV_KEYS

REPO = Path(__file__).resolve().parent.parent

# Injected by `build_env` from the trainer, never from the yaml.
INJECTED = {"mode", "num_envs", "test_episodes"}

GROUP_FILES = sorted((REPO / "roxie" / "configs" / "env").glob("*.yaml"))

# The rest of an experiment's `env:` block is checked through the group files,
# since a partial block cannot be resolved on its own.
EXPERIMENT_FILES = sorted(
    p
    for p in REPO.glob("experiments/**/*.yaml")
    if "_target_" in (OmegaConf.load(p).get("env") or {})
)


def _builder_kwargs(target: str) -> dict:
    """Every keyword the builder accepts, minus the injected ones."""
    fn = get_method(target)  # raises if the dotted path is stale
    return {
        name: p
        for name, p in inspect.signature(fn).parameters.items()
        if name not in INJECTED
        and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    }


def _takes_extra_kwargs(target: str) -> bool:
    """True for a builder with a `**kwargs` tail — `build_envpool_env` forwards
    unrecognised keys to `envpool.make()`, so unknown keys are legal there."""
    return any(
        p.kind is p.VAR_KEYWORD
        for p in inspect.signature(get_method(target)).parameters.values()
    )


def test_config_files_were_found():
    """A bad glob would make every test below vacuously pass."""
    assert GROUP_FILES, "no configs in roxie/configs/env/"
    assert EXPERIMENT_FILES, "no experiment config declares env._target_"


@pytest.mark.parametrize(
    "path", GROUP_FILES, ids=lambda p: str(p.relative_to(REPO))
)
def test_group_config_matches_builder(path):
    cfg = OmegaConf.load(path)
    target = cfg.get("_target_")
    assert target, f"{path}: no `_target_` — build_env cannot instantiate it"

    expected = _builder_kwargs(target)
    # `TRAINER_ENV_KEYS` are stripped before instantiate, so they are legal in
    # the block without being builder arguments.
    actual = set(cfg.keys()) - {"_target_"} - set(TRAINER_ENV_KEYS)

    missing = sorted(set(expected) - actual)
    assert not missing, (
        f"{path} does not expose {missing} — they would be pinned to their "
        f"Python defaults and could not be swept or logged. Add them with the "
        f"builder default."
    )

    unknown = sorted(actual - set(expected))
    assert not unknown or _takes_extra_kwargs(target), (
        f"{path} sets {unknown}, which {target} does not accept — this raises "
        f"TypeError at launch, after the config has composed."
    )


@pytest.mark.parametrize(
    "path", GROUP_FILES, ids=lambda p: str(p.relative_to(REPO))
)
def test_required_builder_args_are_set(path):
    """A keyword with no Python default must carry a value (or `???`)."""
    cfg = OmegaConf.load(path)
    # `keys()` rather than `in`: a key held at `???` is exactly how a required
    # argument is declared here, and `in` reads that as absent.
    keys = set(cfg.keys())
    for arg, param in _builder_kwargs(cfg.get("_target_")).items():
        if param.default is inspect.Parameter.empty:
            assert arg in keys, f"{path}: {arg} is required and has no default"


@pytest.mark.parametrize(
    "path", EXPERIMENT_FILES, ids=lambda p: str(p.relative_to(REPO))
)
def test_experiment_env_block_has_no_unknown_keys(path):
    """The half of the check a partial block can still answer."""
    cfg_env = OmegaConf.load(path).env
    target = cfg_env["_target_"]
    if _takes_extra_kwargs(target):
        return  # anything unrecognised is a forwarded task kwarg

    expected = set(_builder_kwargs(target)) | set(TRAINER_ENV_KEYS)
    unknown = sorted(set(cfg_env.keys()) - expected - {"_target_"})
    assert not unknown, (
        f"{path} sets env.{unknown}, which {target} does not accept — this "
        f"raises TypeError at launch, after the config has composed."
    )


# Instantiating a real env means loading MuJoCo physics, so `build_env`'s
# contract is pinned against a builder that only records its arguments.


def record_builder(**kwargs):
    """Stand-in builder: returns its own call as an `EnvBundle`-shaped tuple."""
    return build_env_result(kwargs)


def build_env_result(kwargs):
    from roxie.environment.loader import EnvBundle

    return EnvBundle(env=kwargs, test_env=None, env_cfg=None)


RECORD = f"{__name__}.record_builder"


def _call(env_block: dict, **injected) -> dict:
    from roxie.environment.loader import build_env

    injected = {"mode": "train", "num_envs": 4, "test_episodes": 2, **injected}
    return build_env(OmegaConf.create(env_block), **injected).env


def test_trainer_keys_never_reach_the_builder():
    """`parallel_envs`/`test_episodes` reach it as the injected sizes instead,
    and `viewer`/`player` are play.py's, so passing any of them on would be a
    TypeError against every builder in the repo."""
    kwargs = _call({
        "_target_": RECORD,
        "parallel_envs": 64,
        "test_episodes": 99,
        "viewer": "pkg.ghost",
        "player": "pkg.player",
        "seed": 7,
        "env_name": "CartpoleBalance",
    })
    assert set(kwargs) == {
        "mode", "num_envs", "test_episodes", "seed", "env_name"
    }
    assert kwargs["num_envs"] == 4
    assert kwargs["test_episodes"] == 2  # the trainer's, not the yaml's 99
    assert kwargs["seed"] == 7  # a genuine env key, forwarded


def test_interpolations_resolve_against_the_composed_config():
    """A backend group writes `env_name: ${release.task}`; the value has to be
    read while the node still has its parent."""
    cfg = OmegaConf.create({
        "release": {"task": "WalkerWalk"},
        "env": {"_target_": RECORD, "env_name": "${release.task}"},
    })
    from roxie.environment.loader import build_env

    kwargs = build_env(cfg.env, num_envs=1, test_episodes=1).env
    assert kwargs["env_name"] == "WalkerWalk"


def test_builder_defaults_to_playground():
    from roxie.environment.loader import DEFAULT_BUILDER

    fn = get_method(DEFAULT_BUILDER)
    assert fn.__name__ == "build_playground_env"
    # An env block that names no builder must still be instantiable.
    assert "env_name" in inspect.signature(fn).parameters


def test_legacy_builder_key_is_still_honoured():
    """Checkpoints written before the env group carry `builder:`, and `play.py`
    rebuilds the env from a checkpoint's own saved config."""
    kwargs = _call({"builder": RECORD, "env_name": "CartpoleBalance"})
    assert kwargs["env_name"] == "CartpoleBalance"
    assert "builder" not in kwargs


# EnvPool is not a value of `impl` — that picks MJX or Warp kernels *within*
# the playground builder — so both callers ask the builder instead.


def test_uses_envpool_reads_the_builder_not_an_impl_key():
    from roxie.environment.loader import uses_envpool

    envpool_cfg = OmegaConf.load(REPO / "roxie" / "configs" / "env" / "envpool.yaml")
    assert uses_envpool(envpool_cfg)
    assert "impl" not in set(envpool_cfg.keys()), (
        "`impl` names a physics implementation within the playground builder; "
        "EnvPool is the other builder, and `_target_` already says so."
    )

    playground_cfg = OmegaConf.load(
        REPO / "roxie" / "configs" / "env" / "playground.yaml"
    )
    assert not uses_envpool(playground_cfg)
    # An env block naming no builder at all gets the playground default.
    assert not uses_envpool(OmegaConf.create({"env_name": "WalkerWalk"}))
    # And a stale path is build_env's error to raise, not this function's.
    assert not uses_envpool(OmegaConf.create({"_target_": "no.such.module.builder"}))


def test_envpool_release_cell_is_recognised():
    """The cell that must stay GPU-free, checked through the composed block."""
    from roxie.environment.loader import uses_envpool

    cfg = OmegaConf.load(REPO / "experiments" / "dmc" / "backend" / "envpool_cpu.yaml")
    assert uses_envpool(cfg.env)
    assert (cfg.agent.device, cfg.env.device) == ("cpu", "cpu")


BACKEND_CELLS = sorted((REPO / "experiments" / "dmc" / "backend").glob("*.yaml"))


@pytest.mark.parametrize(
    "path", BACKEND_CELLS, ids=lambda p: str(p.relative_to(REPO))
)
def test_backend_cell_states_both_devices(path):
    """A cell IS a hardware choice, so it must say what that choice is for each
    half — the pair is what the startup banner then checks against reality."""
    from roxie.environment.loader import DEVICES

    cfg = OmegaConf.load(path)
    for block in ("agent", "env"):
        device = (cfg.get(block) or {}).get("device", None)
        assert device in DEVICES, (
            f"{path}: {block}.device={device!r}; a backend cell must state "
            f"which of {DEVICES} that half runs on."
        )


@pytest.mark.parametrize(
    "path", BACKEND_CELLS, ids=lambda p: str(p.relative_to(REPO))
)
def test_backend_cell_devices_are_achievable(path):
    """The declared pair has to be one the cell can actually deliver."""
    cfg = OmegaConf.load(path)
    agent_device, env_device = cfg.agent.device, cfg.env.device
    impl = (cfg.env or {}).get("impl", None)

    if impl == "warp":
        assert env_device == "gpu", f"{path}: Warp kernels are CUDA-only"
    if str(cfg.env["_target_"]).endswith("build_envpool_env"):
        assert env_device == "cpu", f"{path}: a pool steps on CPU threads"
    else:
        # One XLA program: a playground env lands on the agent's device, so a
        # cell declaring otherwise would run somewhere it does not claim.
        assert env_device == agent_device, (
            f"{path}: a mujoco_playground env shares the agent's device, so "
            f"env.device={env_device!r} with agent.device={agent_device!r} "
            f"cannot hold."
        )


def test_resolve_placement_reads_both_blocks():
    from roxie.environment.loader import resolve_placement

    playground = {"_target_": "roxie.environment.loader.build_playground_env"}
    envpool = {"_target_": "roxie.environment.loader.build_envpool_env"}

    # The agent's device is the JAX platform; `gpu` forces nothing, so a machine
    # with no card still runs (and the banner says so).
    cfg = OmegaConf.create({"agent": {"device": "cpu"}, "env": playground})
    assert resolve_placement(cfg) == ("cpu", None, "cpu")
    cfg = OmegaConf.create({"agent": {"device": "gpu"}, "env": playground})
    assert resolve_placement(cfg) == ("gpu", None, None)

    # The hybrid: pool on CPU threads, learner on the card. Both declarations
    # come back untouched — this reads them, it does not reconcile them.
    cfg = OmegaConf.create({
        "agent": {"device": "gpu"}, "env": {**envpool, "device": "cpu"},
    })
    assert resolve_placement(cfg) == ("gpu", "cpu", None)

    # An EnvPool run that says nothing stays off the card entirely.
    cfg = OmegaConf.create({"agent": {}, "env": envpool})
    assert resolve_placement(cfg) == ("cpu", None, "cpu")

    # ... unlike a playground run, which is left to JAX.
    cfg = OmegaConf.create({"agent": {}, "env": playground})
    assert resolve_placement(cfg) == (None, None, None)


def test_resolve_placement_rejects_a_device_it_cannot_mean():
    from roxie.environment.loader import resolve_placement

    cfg = OmegaConf.create({
        "agent": {"device": "cuda:0"},
        "env": {"_target_": "roxie.environment.loader.build_playground_env"},
    })
    with pytest.raises(ValueError, match="agent.device"):
        resolve_placement(cfg)
