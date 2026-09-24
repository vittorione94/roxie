"""The benchmark's task list, and the claim that both backends can run it.

`roxie.environment.suites` is a small module carrying a big claim: that the 25
tasks in the release grid exist on BOTH implementations, that the two names for
each of them refer to the same task, and that both hand the agent the same
observation. Nothing else in the repo checks any of it — a wrong envpool id, or
an observation assembled from the wrong groups, composes fine, launches fine,
and trains a policy on something that is not the task, which shows up as a score
gap between cells that reads like a physics bug.

These run without a device: envs and pools are CONSTRUCTED (to read the shapes
off them, which is the only honest way to compare the two) but never stepped.
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


@pytest.fixture(scope="module")
def backends():
    """Per task: the observation width each backend hands the agent, and the
    observation groups the pool declares.

    One pass over the suite rather than one per test — the 25 playground envs
    and the 25 pools are the whole cost of this file, and both tests below need
    the same construction. No physics is stepped either side.
    """
    registry = pytest.importorskip("mujoco_playground").registry
    pytest.importorskip("envpool")
    from roxie.environment.loader import build_envpool_env

    out = {}
    for task in suites.DMC_TASKS:
        bundle = build_envpool_env(
            suites.envpool_task_id(task), num_envs=1, test_episodes=1,
        )
        # The driver's own pool rather than a third `envpool.make` per task:
        # what the groups test needs is the Dict the pool DECLARES, before
        # `obs_keys` narrows it, and this is where that survives.
        out[task] = (
            int(bundle.env.single_observation_space.shape[0]),
            int(registry.load(task).observation_size),
            set(bundle.env._pool.observation_space.spaces),
        )
    return out


def test_both_backends_hand_the_agent_the_same_observation(backends):
    """The comparison's other half, and the one that fails silently.

    A wrong envpool id at least trains on a visibly different task; an envpool
    observation that is WIDER than playground's trains fine, scores fine, and
    reports a cell that was never running the same benchmark — dm_control's
    humanoid observation is 67 numbers, and envpool's Dict declares 95 because
    `run_pure_state` needs `position`. Nothing else in the repo compares the
    two widths.
    """
    mismatched = {
        task: (pool_width, playground_width)
        for task, (pool_width, playground_width, _groups) in backends.items()
        if pool_width != playground_width
    }
    assert not mismatched, (
        "envpool and playground observations differ in width (envpool, "
        "playground): "
        + ", ".join(f"{t}: {w}" for t, w in sorted(mismatched.items()))
        + ". State the task's groups in `suites._ENVPOOL_OBS_KEYS`."
    )


def test_the_declared_groups_are_the_ones_the_pool_hands_back(backends):
    """`envpool_obs_keys` names groups by hand, so a renamed or dropped one has
    to fail here rather than as a KeyError on the first step of a run."""
    for task, (_pool_width, _playground_width, declared) in backends.items():
        keys = suites.envpool_obs_keys(suites.envpool_task_id(task))
        if keys is None:
            continue
        assert not set(keys) - declared, (
            f"{task}: no such observation group(s) "
            f"{sorted(set(keys) - declared)}; the pool declares {sorted(declared)}"
        )


def test_only_the_five_known_tasks_restate_their_groups():
    """Guards the shape of the rule, as the naming test above does: the default
    is the pool's own Dict order, and a sixth entry means a new disagreement
    between the backends that someone has to have looked at."""
    restated = {
        task for task in suites.DMC_TASKS
        if suites.envpool_obs_keys(suites.envpool_task_id(task)) is not None
    }
    assert restated == {
        "HumanoidStand", "HumanoidWalk", "HumanoidRun", "FingerSpin", "FishSwim",
    }


def test_a_task_outside_the_suite_has_no_layout_to_correct():
    """`build_envpool_env` also builds envpool's gym-MuJoCo tasks, which have no
    playground twin to agree with — and a flat Box observation regardless."""
    assert suites.envpool_obs_keys("HalfCheetah-v4") is None


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
