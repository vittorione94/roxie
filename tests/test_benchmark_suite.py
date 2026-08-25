"""The benchmark's task list, and the claim that both backends can run it.

`roxie.environment.suites` is a small module carrying a big claim: that the 25
tasks in the release grid exist on BOTH implementations, and that the two names
for each of them refer to the same task. Nothing else in the repo checks that —
a wrong envpool id composes fine, launches fine, and trains a policy on the
wrong env, which shows up as a score gap between cells that reads like a physics
bug.

These run without a device. `mujoco_playground` and `envpool` are both imported,
but only their registries are touched, not their physics.
"""

import re

import pytest

from roxie.environment import suites


def test_the_suite_is_the_playground_dm_control_registry():
    """DMC_TASKS is a literal, deliberately — a playground release that adds a
    task must not silently widen a published grid. This is the prompt to widen
    it on purpose, not a bug report."""
    registry = pytest.importorskip("mujoco_playground").registry
    upstream = set(registry.dm_control_suite.ALL_ENVS)
    ours = set(suites.DMC_TASKS)
    assert not ours - upstream, (
        f"benchmark tasks missing from mujoco_playground: {sorted(ours - upstream)}"
    )
    assert not upstream - ours, (
        "mujoco_playground registers dm_control tasks the benchmark does not "
        f"run: {sorted(upstream - ours)}. Add them to DMC_TASKS deliberately — "
        "it changes what the published grid means."
    )


def test_every_task_has_an_envpool_counterpart():
    """The cross-backend comparison only exists if both sides have the task."""
    envpool = pytest.importorskip("envpool")
    available = set(envpool.list_all_envs())
    missing = [
        task for task in suites.DMC_TASKS
        if suites.envpool_task_id(task) not in available
    ]
    assert not missing, (
        "no envpool task for: "
        + ", ".join(f"{t} -> {suites.envpool_task_id(t)}" for t in missing)
    )


def test_the_mapping_is_suffixing_plus_two_named_exceptions():
    """Guards the shape of the rule, so a future exception has to be added to
    the table rather than hidden in a regex."""
    exceptions = {
        task: suites.envpool_task_id(task)
        for task in suites.DMC_TASKS
        if suites.envpool_task_id(task) != f"{task}-v1"
    }
    assert exceptions == {
        "BallInCup": "BallInCupCatch-v1",
        "PointMass": "PointMassEasy-v1",
    }


def test_an_unknown_task_is_rejected_by_name():
    """`${envpool_task:...}` runs at config-compose time, so a typo in
    `release.task=` has to fail there — loudly, listing the suite — rather than
    resolving to a plausible-looking id that envpool then rejects from C++."""
    with pytest.raises(KeyError, match="CheetahWalk"):
        suites.envpool_task_id("CheetahWalk")


def test_playground_task_inverts_the_mapping():
    """`play.py` replays an envpool checkpoint on the playground twin, so the
    mapping has to be readable in both directions — including for the two
    exceptions, which are the only ones a naive de-suffixing would get wrong."""
    for task in suites.DMC_TASKS:
        assert suites.playground_task(suites.envpool_task_id(task)) == task


def test_an_unknown_envpool_id_is_rejected_by_name():
    """A checkpoint trained on a task outside the suite (envpool's gym-MuJoCo
    ids, say) has no twin to render, and must say so rather than resolve to a
    neighbouring task."""
    with pytest.raises(KeyError, match="HalfCheetah-v4"):
        suites.playground_task("HalfCheetah-v4")


def test_resolver_is_registered_for_hydra():
    from omegaconf import OmegaConf

    suites.register_resolvers()
    cfg = OmegaConf.create({"task": "BallInCup",
                            "id": "${envpool_task:${task}}"})
    assert cfg.id == "BallInCupCatch-v1"


def test_task_names_are_usable_as_wandb_projects():
    """The grid names a project `roxie-<Task>` per env. W&B accepts letters,
    digits, dashes, underscores and dots; a task name outside that would create
    a mangled project instead of failing."""
    for task in suites.DMC_TASKS:
        assert re.fullmatch(r"[A-Za-z0-9._-]+", task), task
