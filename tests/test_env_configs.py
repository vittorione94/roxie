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

from roxie.environment.loader import (
    TRAINER_ENV_KEYS,
    _l3_domains,
    _parse_cpu_list,
    select_cpu_cores,
)

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
    assert cfg.runtime.device == "cpu"


BACKEND_CELLS = sorted((REPO / "experiments" / "dmc" / "backend").glob("*.yaml"))


@pytest.mark.parametrize(
    "path", BACKEND_CELLS, ids=lambda p: str(p.relative_to(REPO))
)
def test_backend_cell_states_its_device(path):
    """A cell IS a hardware choice, so it must say what that choice is — the
    declaration is what the startup banner then checks against reality."""
    from roxie.environment.loader import DEVICES

    cfg = OmegaConf.load(path)
    device = (cfg.get("runtime") or {}).get("device", None)
    assert device in DEVICES, (
        f"{path}: runtime.device={device!r}; a backend cell must state which "
        f"of {DEVICES} the run is on."
    )


@pytest.mark.parametrize(
    "path", BACKEND_CELLS, ids=lambda p: str(p.relative_to(REPO))
)
def test_backend_cell_device_is_achievable(path):
    """The declared device has to be one the cell can actually deliver."""
    cfg = OmegaConf.load(path)
    device = cfg.runtime.device
    impl = (cfg.env or {}).get("impl", None)

    if impl == "warp":
        assert device == "gpu", f"{path}: Warp kernels are CUDA-only"
    if str(cfg.env["_target_"]).endswith("build_envpool_env"):
        # A pool steps native MuJoCo on C++ threads, so the learner joins it
        # there — the split is what `resolve_placement` refuses outright.
        assert device == "cpu", f"{path}: a pool steps on CPU threads"


PLAYGROUND = {"_target_": "roxie.environment.loader.build_playground_env"}
ENVPOOL = {"_target_": "roxie.environment.loader.build_envpool_env"}


def test_resolve_placement_reads_the_one_knob():
    from roxie.environment.loader import resolve_placement

    # The device IS the JAX platform; `gpu` forces nothing, so a machine with
    # no card still runs (and the banner says so).
    cfg = OmegaConf.create({"runtime": {"device": "cpu"}, "env": PLAYGROUND})
    assert resolve_placement(cfg) == ("cpu", "cpu")
    cfg = OmegaConf.create({"runtime": {"device": "gpu"}, "env": PLAYGROUND})
    assert resolve_placement(cfg) == ("gpu", None)

    # An EnvPool run stays off the card whether or not it says so: the physics
    # is on C++ threads, and the learner is not allowed to leave it there alone.
    cfg = OmegaConf.create({"env": ENVPOOL})
    assert resolve_placement(cfg) == ("cpu", "cpu")
    cfg = OmegaConf.create({"runtime": {"device": "cpu"}, "env": ENVPOOL})
    assert resolve_placement(cfg) == ("cpu", "cpu")

    # ... unlike a playground run, which is left to JAX.
    cfg = OmegaConf.create({"env": PLAYGROUND})
    assert resolve_placement(cfg) == (None, None)


def test_resolve_placement_refuses_the_cpu_physics_gpu_agent_split():
    """The setup the async learner existed for. Nothing overlaps the two halves
    any more, so asking for it is an error rather than a slow serial run."""
    from roxie.environment.loader import resolve_placement

    cfg = OmegaConf.create({"runtime": {"device": "gpu"}, "env": ENVPOOL})
    with pytest.raises(ValueError, match="EnvPool"):
        resolve_placement(cfg)


def test_resolve_placement_rejects_a_device_it_cannot_mean():
    from roxie.environment.loader import resolve_placement

    cfg = OmegaConf.create({"runtime": {"device": "cuda:0"}, "env": PLAYGROUND})
    with pytest.raises(ValueError, match="runtime.device"):
        resolve_placement(cfg)


# ---- CPU core selection -----------------------------------------------------
#
# Which cpus the run gets, not just how many: a learning pass slows down once
# the mask spans two L3 domains, so it has to stay inside one while it fits.


def test_cpu_list_parsing_handles_ranges_and_singletons():
    assert _parse_cpu_list("0-5,12-17") == [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17]
    assert _parse_cpu_list("3") == [3]
    assert _parse_cpu_list("") == []


def test_core_selection_packs_into_one_l3_domain(monkeypatch):
    """A request that fits in one domain must not straddle two."""
    ccd0 = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17]
    ccd1 = [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23]
    monkeypatch.setattr(
        "roxie.environment.loader._l3_domains", lambda available: [ccd0, ccd1]
    )
    # The naive `available[:cores]` would return 0..11 here -- six cpus from
    # each CCD, which is the case this exists to avoid.
    assert select_cpu_cores(sorted(ccd0 + ccd1), 12) == sorted(ccd0)
    assert set(select_cpu_cores(sorted(ccd0 + ccd1), 6)) <= set(ccd0)


def test_core_selection_spills_to_a_second_domain_only_when_it_must():
    picked = select_cpu_cores(list(range(24)), 24)
    assert picked == list(range(24)), "everything available must still be usable"


def test_core_selection_stays_within_the_cpus_it_was_given():
    """An outer `taskset` chooses which cores a run gets; this only narrows."""
    available = [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23]
    for n in (1, 6, 12):
        assert set(select_cpu_cores(available, n)) <= set(available)
        assert len(select_cpu_cores(available, n)) == n


def test_core_selection_falls_back_when_topology_is_unreadable(monkeypatch):
    """No sysfs (container, non-Linux) must degrade to the flat behaviour, not
    raise."""
    def boom(*args, **kwargs):
        raise OSError("no sysfs here")

    monkeypatch.setattr("builtins.open", boom)
    assert _l3_domains([0, 1, 2, 3]) == [[0, 1, 2, 3]]
    assert select_cpu_cores([0, 1, 2, 3], 2) == [0, 1]

