"""The benchmark's task list, and the one place the two backends are matched up.

The v1 release benchmark runs the **dm_control suite** twice: once through
``mujoco_playground`` (GPU physics, MJX or Warp kernels) and once through
``envpool`` (native MuJoCo stepped on CPU threads). Those are two independent
implementations of the same 25 tasks, from the same dm_control definitions, with
the same 0-1000 episode-return scale — which is what makes a per-task
GPU-vs-CPU score comparison meaningful rather than a coincidence.

Their naming does not quite line up: envpool suffixes a version (``-v1``) and
disambiguates two tasks that playground names without a difficulty. That is the
whole content of this module, and it lives here rather than in each experiment
yaml so the mapping is stated once and checked by a test.

``register_resolvers()`` exposes it to Hydra as ``${envpool_task:<Task>}``, so a
backend group can write::

    task_id: ${envpool_task:${release.task}}

and an interactive override (``release.task=BallInCup``) reaches the right pool
without the caller knowing about the two exceptions.
"""

from __future__ import annotations

# The 25 dm_control-suite tasks mujoco_playground registers, in its own order
# (`mujoco_playground.registry.dm_control_suite.ALL_ENVS`). Kept as a literal
# rather than imported from the registry: this list defines the BENCHMARK, and a
# playground release that adds a task should not silently widen a published
# grid. `tests/test_benchmark_suite.py` fails when the two drift, which is the
# prompt to update this deliberately.
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

# envpool's id is the playground name plus "-v1" for 23 of the 25. The two
# exceptions are naming, not task differences: playground's `BallInCup` is
# dm_control's ball_in_cup/catch, and its `PointMass` is point_mass/easy — the
# only variants either package ships for those domains.
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


def register_resolvers() -> None:
    """Register ``${envpool_task:<Task>}`` with OmegaConf. Idempotent."""
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver(
        "envpool_task", envpool_task_id, replace=True
    )
