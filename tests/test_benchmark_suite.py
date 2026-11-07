"""The benchmark's task list."""

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


def test_task_names_are_usable_as_wandb_projects():
    """The grid names a project `roxie-<Task>` per env. W&B accepts letters,
    digits, dashes, underscores and dots; a task name outside that would create
    a mangled project instead of failing."""
    for task in suites.DMC_TASKS:
        assert re.fullmatch(r"[A-Za-z0-9._-]+", task), task
