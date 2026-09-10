"""The benchmark's task list, and the one place the two backends are matched up.

The v1 release benchmark runs the dm_control suite twice: through
``mujoco_playground`` (GPU physics) and through ``envpool`` (native MuJoCo on
CPU threads). Two independent implementations of the same 25 tasks, from the
same definitions and on the same 0-1000 return scale, which is what makes a
per-task GPU-vs-CPU comparison meaningful.

Their naming does not line up: envpool suffixes a version (``-v1``) and
disambiguates two tasks that playground names without a difficulty. That mapping
is the whole content of this module, stated here once rather than in each
experiment yaml, and checked by a test.

``register_resolvers()`` exposes it to Hydra as ``${envpool_task:<Task>}``, so a
backend group can write::

    task_id: ${envpool_task:${release.task}}

and an interactive override (``release.task=BallInCup``) reaches the right pool
without the caller knowing about the two exceptions.
"""

from __future__ import annotations

# A literal rather than an import from the registry: this list defines the
# benchmark, and a playground release that adds a task should not silently widen
# a published grid. `tests/test_benchmark_suite.py` fails when the two drift.
DMC_TASKS = (
    "AcrobotSwingup",
    "AcrobotSwingupSparse",
    "BallInCup",
    "CartpoleBalance",
    "CartpoleBalanceSparse",
    "CartpoleSwingup",
    "CartpoleSwingupSparse",
    "CheetahRun",
    "FingerSpin",
    "FingerTurnEasy",
    "FingerTurnHard",
    "FishSwim",
    "HopperHop",
    "HopperStand",
    "HumanoidStand",
    "HumanoidWalk",
    "HumanoidRun",
    "PendulumSwingup",
    "PointMass",
    "ReacherEasy",
    "ReacherHard",
    "SwimmerSwimmer6",
    "WalkerRun",
    "WalkerStand",
    "WalkerWalk",
)

# envpool's id is the playground name plus "-v1" for 23 of the 25. These two
# are naming differences, not task differences.
_ENVPOOL_EXCEPTIONS = {
    "BallInCup": "BallInCupCatch-v1",
    "PointMass": "PointMassEasy-v1",
}


def envpool_task_id(task: str) -> str:
    """The envpool task id for a playground dm_control-suite task name."""
    if task not in DMC_TASKS:
        raise KeyError(
            f"{task!r} is not in the roxie benchmark suite. "
            f"Known tasks: {', '.join(DMC_TASKS)}"
        )
    return _ENVPOOL_EXCEPTIONS.get(task, f"{task}-v1")


def playground_task(envpool_id: str) -> str:
    """The playground task name for an envpool task id — ``envpool_task_id``
    read backwards.

    Playback is what needs this. An EnvPool pool steps its physics in C++ and
    exposes no ``MjModel``, so there is nothing for the MuJoCo viewer to render;
    ``play.py`` swaps an envpool-trained checkpoint onto the playground twin of
    the same task, which the mapping above says is the same task.
    """
    for task in DMC_TASKS:
        if envpool_task_id(task) == envpool_id:
            return task
    raise KeyError(
        f"{envpool_id!r} is not an envpool id in the roxie benchmark suite. "
        f"Known ids: {', '.join(envpool_task_id(t) for t in DMC_TASKS)}"
    )


def register_resolvers() -> None:
    """Register ``${envpool_task:<Task>}`` with OmegaConf. Idempotent."""
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver(
        "envpool_task", envpool_task_id, replace=True
    )
