"""Every agent knob must be reachable from yaml.

`train.py` builds an agent as ``agents.utils.build_agent(cfg.agent, ...)`` — one
``hydra.utils.instantiate`` — so the
agent config IS the constructor call: `_target_` names the class and every
sibling key is one of its keywords. A keyword absent from the yaml is silently
pinned to its Python default — invisible in the run's config, invisible in
wandb, and not sweepable. Conversely a stale key is a `TypeError` at launch,
after the env has already been built.

The same holds one level down, inside the `hyperparams:` block: it builds a
frozen dataclass (`agent.hyperparams_cls`) whose fields ARE the agent's knobs,
so the check recurses into it rather than stopping at the block.

This test pins both directions for every agent config in the repo, so adding a
constructor argument — or a hyperparameter field — without surfacing it fails
here rather than in a sweep.
"""

import dataclasses
import inspect
import re
from pathlib import Path

import pytest
from hydra.utils import get_class, get_method
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent

# Supplied by train.py from the env, plus `noise_config`, which comes from the
# separate top-level `noise` config group.
INJECTED = {
    "self",
    "env_obs_size",
    "env_action_size",
    "action_low",
    "action_high",
    "noise_config",
}

CONFIG_FILES = sorted(
    list((REPO / "roxie" / "configs" / "agent").glob("*.yaml"))
    + list(REPO.glob("experiments/*/agent/*.yaml"))
)


def _constructor_kwargs(cls) -> dict:
    """Every keyword `cls(**args)` accepts, walking the MRO — subclasses such as
    TD3 forward to their parent via ``*args, **kwargs``, so inspecting only
    ``cls.__init__`` would miss the inherited half."""
    params = {}
    for klass in reversed(cls.__mro__):
        init = klass.__dict__.get("__init__")
        if init is not None:
            params.update(inspect.signature(init).parameters)
    return {
        name: p
        for name, p in params.items()
        if name not in INJECTED
        and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    }


def test_config_files_were_found():
    """A bad glob would make every test below vacuously pass."""
    assert len(CONFIG_FILES) > 5


@pytest.mark.parametrize(
    "path", CONFIG_FILES, ids=lambda p: str(p.relative_to(REPO))
)
def test_agent_config_is_the_constructor_call(path):
    """One config, one `instantiate`, so the three ways it can disagree with
    the constructor are one check: a keyword the yaml never names (pinned to
    its Python default, unsweepable and unlogged), a key the class does not
    accept (a `TypeError` at launch, after the env is built), and a required
    keyword left without a value. The nested `*_config` blocks are instantiated
    the same way, so their `_target_`s have to resolve too.
    """
    cfg = OmegaConf.load(path)
    target = cfg.get("_target_")
    assert target, f"{path}: no `_target_` — train.py cannot instantiate it"
    cls = get_class(target)  # raises if the dotted path is stale

    expected = _constructor_kwargs(cls)
    actual = set(cfg.keys()) - {"_target_"}

    missing = sorted(set(expected) - actual)
    assert not missing, (
        f"{path} does not expose {missing} — they would be pinned to their "
        f"Python defaults and could not be swept or logged. Add them with the "
        f"constructor default."
    )

    unknown = sorted(actual - set(expected))
    assert not unknown, (
        f"{path} sets {unknown}, which {cls.__name__} does not accept — this "
        f"raises TypeError at launch, after the env is built."
    )

    for arg, param in expected.items():
        if param.default is inspect.Parameter.empty:
            assert arg in cfg, f"{path}: {arg} is required and has no default"

    for key, block in cfg.items():
        if not key.endswith("_config"):
            continue
        assert block.get("_target_"), f"{path}: `{key}` has no `_target_`"
        # These targets are functions, so they need get_method, not get_class.
        get_method(block["_target_"])


@pytest.mark.parametrize(
    "path", CONFIG_FILES, ids=lambda p: str(p.relative_to(REPO))
)
def test_hyperparams_block_matches_its_dataclass(path):
    """The `hyperparams:` block must name every field of `hyperparams_cls`.

    Same contract as above, one level down. `build_hyperparams` already raises
    on a key the dataclass does not have; what it cannot catch is a field the
    yaml never mentions, which then runs at its Python default.
    """
    cfg = OmegaConf.load(path)
    cls = get_class(cfg.get("_target_"))

    if cls.hyperparams_cls is None:
        assert "hyperparams" not in cfg, (
            f"{path} sets `hyperparams`, but {cls.__name__} has no "
            f"`hyperparams_cls` to build it into."
        )
        return

    block = cfg.get("hyperparams")
    assert block is not None, f"{path}: {cls.__name__} needs a `hyperparams` block"

    expected = {
        f.name for f in dataclasses.fields(cls.hyperparams_cls)
    } - cls.derived_hyperparams
    actual = set(block.keys()) - {"_target_"}

    missing = sorted(expected - actual)
    assert not missing, (
        f"{path} does not expose hyperparameter(s) {missing} — they would be "
        f"pinned to their dataclass defaults and could not be swept or logged."
    )

    derived = sorted(actual & cls.derived_hyperparams)
    assert not derived, (
        f"{path} sets {derived}, which {cls.__name__} computes at construction "
        f"and overwrites — the value here would have no effect."
    )

    unknown = sorted(actual - expected - cls.derived_hyperparams)
    assert not unknown, (
        f"{path} sets {unknown}, which "
        f"{cls.hyperparams_cls.__name__} does not accept."
    )


# The ones with a `defaults:` list naming an agent group, not the groups they
# pull in.
LAUNCHABLES = [
    p
    for p in sorted(REPO.glob("experiments/*/*.yaml"))
    if re.search(r"^\s*-\s*/?agent:", p.read_text(), re.M)
]


def _group_choice(text: str, group: str):
    m = re.search(rf"^\s*-\s*/?{group}:\s*(\S+)", text, re.M)
    return m.group(1) if m else None


def test_launchables_were_found():
    assert len(LAUNCHABLES) > 5


@pytest.mark.parametrize(
    "path", LAUNCHABLES, ids=lambda p: str(p.relative_to(REPO))
)
def test_launchable_noise_group_matches_its_agent(path):
    """A launchable's `noise` group and its agent must agree.

    `train.py` passes `noise_config` whenever the composed config has a `noise`
    group, so pairing one with SAC/MPO/PPO — which explore from their own policy
    and take no noise module — is a `TypeError` at launch, *after* the env is
    built. The reverse (DDPG-family with no noise group) is a missing required
    argument. This pins both statically, so neither needs a launch to surface.
    """
    text = path.read_text()
    agent_group = _group_choice(text, "agent")
    agent_path = path.parent / "agent" / f"{agent_group}.yaml"
    if not agent_path.exists():
        agent_path = REPO / "roxie" / "configs" / "agent" / f"{agent_group}.yaml"
    assert agent_path.exists(), f"{path}: agent group {agent_group!r} not found"

    cls = get_class(OmegaConf.load(agent_path).get("_target_"))
    params = _constructor_kwargs_including_injected(cls)
    has_noise_group = _group_choice(text, "noise") is not None

    if has_noise_group:
        assert "noise_config" in params, (
            f"{path} pulls in a noise group, but {cls.__name__} takes no "
            f"`noise_config` — this is a TypeError at launch."
        )
    elif "noise_config" in params:
        assert params["noise_config"].default is not inspect.Parameter.empty, (
            f"{path} declares no noise group, but {cls.__name__} requires "
            f"`noise_config`."
        )


def _constructor_kwargs_including_injected(cls) -> dict:
    """Like `_constructor_kwargs`, but keeps the injected names — the noise
    check is precisely about one of them."""
    params = {}
    for klass in reversed(cls.__mro__):
        init = klass.__dict__.get("__init__")
        if init is not None:
            params.update(inspect.signature(init).parameters)
    return params


