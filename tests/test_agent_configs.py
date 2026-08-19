"""Every agent knob must be reachable from yaml.

`train.py` builds an agent as ``hydra.utils.instantiate(cfg.agent, ...)``, so the
agent config IS the constructor call: `_target_` names the class and every
sibling key is one of its keywords. A keyword absent from the yaml is silently
pinned to its Python default — invisible in the run's config, invisible in
wandb, and not sweepable. Conversely a stale key is a `TypeError` at launch,
after the env has already been built.

This test pins both directions for every agent config in the repo, so adding a
constructor argument without surfacing it fails here rather than in a sweep.
"""

import inspect
import re
from pathlib import Path

import pytest
from hydra.utils import get_class
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent

# Supplied positionally by train.py from the env, plus `noise_config`, which
# comes from the separate top-level `noise` config group rather than from the
# agent's own yaml.
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
def test_agent_config_matches_constructor(path):
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


@pytest.mark.parametrize(
    "path", CONFIG_FILES, ids=lambda p: str(p.relative_to(REPO))
)
def test_required_constructor_args_are_set(path):
    """A keyword with no Python default must carry a value in the yaml."""
    cfg = OmegaConf.load(path)
    expected = _constructor_kwargs(get_class(cfg.get("_target_")))
    for arg, param in expected.items():
        if param.default is inspect.Parameter.empty:
            assert arg in cfg, f"{path}: {arg} is required and has no default"


# Launchable experiment configs: the ones with a `defaults:` list naming an
# agent group (`--config-name <env>/<name>`), not the groups they pull in.
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
    argument. Both used to be discoverable only by launching the run.
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


@pytest.mark.parametrize(
    "path", CONFIG_FILES, ids=lambda p: str(p.relative_to(REPO))
)
def test_nested_blocks_are_instantiable(path):
    """Every `*_config` block must itself name a `_target_`.

    The agent hands these straight to `hydra.utils.instantiate`, so a block that
    lost its `_target_` (or kept a stale dotted path) fails at construction time
    rather than at import.
    """
    cfg = OmegaConf.load(path)
    for key, block in cfg.items():
        if not key.endswith("_config"):
            continue
        target = block.get("_target_")
        assert target, f"{path}: `{key}` has no `_target_`"
        # `optax.adam` and `flashbax...make_flat_buffer` are functions, so
        # resolve as a generic callable rather than with get_class.
        from hydra.utils import get_method

        get_method(target)
