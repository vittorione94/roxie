"""The update schedule must not depend on stride/offset alignment.

Regression test for the v1 release grid, where all six off-policy arms of the walker suite
ran 5M env steps at exactly zero gradient steps. The warmup offset was
30_000 and the trainer advances `steps` in strides of `parallel_envs` = 256;
30_000 % 256 == 48, so `(steps - 30_000) % 2_048` cycled 208, 464, ... 2_000 and
never once reached 0. Nothing crashed, nothing logged a warning, and every run
completed and published a curve that was pure exploration noise.

These tests drive `Agent.due_for_update` directly rather than a real agent: the
schedule is plain Python arithmetic, and the bug was in the arithmetic. Testing
it here means the check does not need a card, an env or a replay buffer.
"""

import pytest

from roxie.agents.agent import Agent


class _Schedule(Agent):
    """Bare carrier for the two attributes `due_for_update` reads."""

    def __init__(self, before, between):
        self.memory_warmup = before
        self.steps_between_updates = between

    def _export_hyperparams(self):
        return {}


def _fire_count(before, between, stride, total, start=None):
    """Bursts fired by a trainer loop stepping `stride` at a time up to `total`."""
    sched = _Schedule(before, between)
    steps = (before // stride) * stride if start is None else start
    fired = 0
    while steps < total:
        fired += sched.due_for_update(steps)
        steps += stride
    return fired


class TestUnalignedOffset:
    def test_release_v1_grid_config_now_fires(self):
        """The exact numbers that produced the zero-gradient-step release grid."""
        fired = _fire_count(before=30_000, between=2_048, stride=256, total=5_000_000)
        assert fired > 0, (
            "memory_warmup=30_000 with parallel_envs=256 fired no updates "
            "— this is the v1 release-grid bug, the schedule is disarmed again"
        )
        # ~(5M - 30k) / 2048 boundaries, allow one either side for edge effects.
        assert fired == pytest.approx(2_427, abs=2)

    @pytest.mark.parametrize("before", [0, 1, 47, 30_000, 30_720, 99_999])
    @pytest.mark.parametrize("stride", [1, 100, 256, 400, 1_000])
    def test_fires_for_every_offset_stride_pair(self, before, stride):
        """No (offset, stride) combination may silently disarm the schedule."""
        total = before + 200 * 2_048
        assert _fire_count(before, 2_048, stride, total) > 0


class TestRate:
    def test_one_burst_per_window(self):
        """A stride narrower than the window fires exactly once per window."""
        assert _fire_count(before=30_720, between=2_048, stride=256,
                           total=30_720 + 50 * 2_048) == 50

    def test_wide_stride_collapses_to_one_per_iteration(self):
        """Pre-existing behaviour: the schedule cannot outrun how often it is
        called, so a stride wider than the window fires once per call rather
        than queueing a backlog."""
        assert _fire_count(before=0, between=100, stride=1_000, total=20_000) == 20


class TestNoBacklog:
    def test_before_warmup_never_fires(self):
        sched = _Schedule(30_720, 2_048)
        assert not any(sched.due_for_update(s) for s in range(0, 30_720, 256))

    def test_resumed_run_does_not_storm(self):
        """A fresh instance resuming deep into training fires once, not once per
        boundary it 'missed' while the counter was unset."""
        sched = _Schedule(30_720, 2_048)
        assert sched.due_for_update(4_000_000) is True
        assert sched.due_for_update(4_000_256) is False
        assert sched.due_for_update(4_000_000 + 2_048) is True

    def test_same_step_twice_fires_once(self):
        sched = _Schedule(0, 2_048)
        assert sched.due_for_update(10_240) is True
        assert sched.due_for_update(10_240) is False


class TestBenchConfigsAligned:
    """The configs no longer *need* alignment, but drift back toward a round
    30_000 is a smell worth catching — it was the original defect."""

    def test_bench_offsets_are_multiples_of_parallel_envs(self):
        import glob
        import re

        from omegaconf import OmegaConf

        parallel_envs = OmegaConf.load(
            "experiments/dmc/bench/dmc.yaml"
        ).env.parallel_envs

        checked = 0
        for path in glob.glob("experiments/dmc/agent/*_bench.yaml"):
            text = open(path).read()
            for key in ("memory_warmup", "steps_between_updates"):
                m = re.search(rf"^{key}: ([\d_]+)", text, re.M)
                if m is None:
                    continue  # PPO carries none of these
                value = int(m.group(1).replace("_", ""))
                assert value % parallel_envs == 0, (
                    f"{path}: {key}={value} is not a multiple of "
                    f"parallel_envs={parallel_envs}"
                )
                checked += 1
        assert checked >= 12, f"only checked {checked} keys — did the configs move?"
