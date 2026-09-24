"""Task definitions and backend alignment for the benchmark suite.

Maps task names and observation layouts between `mujoco_playground` (GPU)
and `envpool` (CPU) to ensure identical benchmark conditions across backends.
"""

from __future__ import annotations

# Defined as a literal to prevent silent upstream drift from the registry.
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

_ENVPOOL_EXCEPTIONS = {
    "BallInCup": "BallInCupCatch-v1",
    "PointMass": "PointMassEasy-v1",
}


def envpool_task_id(task: str) -> str:
    """Resolves the envpool task ID for a given playground task name.

    Args:
        task: The playground task name (e.g., "BallInCup").

    Returns:
        The corresponding envpool task ID.

    Raises:
        KeyError: If the task is not in the recognized benchmark suite.
    """
    if task not in DMC_TASKS:
        raise KeyError(
            f"{task!r} is not in the roxie benchmark suite. "
            f"Known tasks: {', '.join(DMC_TASKS)}"
        )
    return _ENVPOOL_EXCEPTIONS.get(task, f"{task}-v1")


_HUMANOID_OBS_KEYS = (
    "joint_angles", "head_height", "extremities",
    "torso_vertical", "com_velocity", "velocity",
)

# Overrides for tasks where EnvPool and Playground differ in observation subsets
# or concatenation order.
_ENVPOOL_OBS_KEYS = {
    "HumanoidStand": _HUMANOID_OBS_KEYS,
    "HumanoidWalk": _HUMANOID_OBS_KEYS,
    "HumanoidRun": _HUMANOID_OBS_KEYS,
    "FingerSpin": ("position", "velocity", "touch"),
    "FishSwim": ("upright", "joint_angles", "target", "velocity"),
}


def envpool_obs_keys(envpool_id: str) -> tuple[str, ...] | None:
    """Retrieves the observation group order for an envpool task.

    Args:
        envpool_id: The envpool task ID.

    Returns:
        A tuple of observation keys to concatenate, or None if the pool's 
        default dictionary order already matches the playground layout.
    """
    try:
        task = playground_task(envpool_id)
    except KeyError:
        return None
    return _ENVPOOL_OBS_KEYS.get(task)


def playground_task(envpool_id: str) -> str:
    """Resolves the playground task name for a given envpool task ID.

    Args:
        envpool_id: The envpool task ID (e.g., "BallInCupCatch-v1").

    Returns:
        The corresponding playground task name.

    Raises:
        KeyError: If the ID is not in the recognized benchmark suite.
    """
    for task in DMC_TASKS:
        if envpool_task_id(task) == envpool_id:
            return task
    raise KeyError(
        f"{envpool_id!r} is not an envpool id in the roxie benchmark suite. "
        f"Known ids: {', '.join(envpool_task_id(t) for t in DMC_TASKS)}"
    )


def register_resolvers() -> None:
    """Registers the `${envpool_task:<Task>}` resolver with OmegaConf.

    This allows backend configuration files to dynamically map playground 
    task names to their envpool equivalents. Idempotent.
    """
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver(
        "envpool_task", envpool_task_id, replace=True
    )