"""Loader for the CMU mocap-tracking example environment.

This lives in ``examples`` rather than the core ``roxie`` package: the mocap
task is a worked example, so the env code and its loader are kept out of
``roxie/environment`` (which holds the generic env infrastructure). The
dependency direction is one-way — examples import from ``roxie``, never the
reverse. ``train.py``/``play.py`` import this lazily, only when
``env.env_type == "mocap"``.
"""

from roxie.data.cmu_mocap_data import build_cmu_humanoid, load_cmu_clips
from roxie.environment.loader import TerminationWrapper

from examples.mocap.mocap_tracking import MocapTrackingEnv


def load_mocap_env(
    clip_ids: list[str] | None = None,
    ctrl_dt: float = 0.025,
    gpu_clip_budget: int = 0,
    impl: str = "jax",
    naconmax: int | None = None,
    njmax: int | None = None,
):
    mj_model, xml_path = build_cmu_humanoid()
    dataset = load_cmu_clips(mj_model, clip_ids=clip_ids, ctrl_dt=ctrl_dt)
    env = MocapTrackingEnv(
        mj_model=mj_model, dataset=dataset, gpu_clip_budget=gpu_clip_budget,
        impl=impl, naconmax=naconmax, njmax=njmax,
    )
    env._xml_path = xml_path
    train_wrapper = TerminationWrapper(env)
    test_wrapper = TerminationWrapper(env)
    return train_wrapper, test_wrapper, xml_path
