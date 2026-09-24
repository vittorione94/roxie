"""Which policy gets published, and the config couplings the step budget creates.

Two things are pinned here.

`scripts/export_release_weights.py` decides what ships with the release: the
BEST checkpoint of a COMPLETED run. Both rules are silent when they go wrong — a
bundle built from the final checkpoint of a crashed pilot loads perfectly and is
simply the wrong policy. The logic is pure CSV/path arithmetic, so it tests
without a device.

The benchmark's exploration anneal is measured in env steps, so it is tied to
`trainer.steps` — at 40% of the budget by its own design note. Raising the
budget without moving it turns the anneal into "no exploration for most of the
run", which no test would otherwise catch and no run would visibly fail on.
"""

import csv
import importlib.util
from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent


def _load_export_module():
    """Import the export script by path — `scripts/` is not a package."""
    path = REPO / "scripts" / "export_release_weights.py"
    spec = importlib.util.spec_from_file_location("export_release_weights", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export = _load_export_module()


def _write_run(root: Path, rows, checkpoint_steps):
    """A run dir with a log.csv and empty `<N>` step checkpoint dirs.

    `rows` is a list of (steps, test/score). The checkpoints are directories
    only: nothing under test reads their contents.
    """
    root.mkdir(parents=True, exist_ok=True)
    with (root / "log.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "steps", "test/score"])
        for epoch, (steps, score) in enumerate(rows, start=1):
            writer.writerow([epoch, steps, score])
    for steps in checkpoint_steps:
        (root / "checkpoints" / str(steps)).mkdir(parents=True)
    return root


def test_picks_the_best_checkpoint_not_the_last(tmp_path):
    """The reason the script exists. A saturating arm peaks mid-run, and the
    final checkpoint is whatever it had decayed to when the budget ran out."""
    run = _write_run(
        tmp_path / "run",
        rows=[(100, 10.0), (200, 90.0), (300, 40.0)],
        checkpoint_steps=[100, 200, 300],
    )
    steps, path, row = export.pick_checkpoint(run, "test/score")
    assert steps == 200
    assert path.name == "200"
    assert row["test/score"] == "90.0"


def test_only_checkpointed_steps_are_candidates(tmp_path):
    """Evals run every epoch, saves every `save_steps`, so most scored epochs
    have no checkpoint. Selecting one of those would name a policy that is not
    on disk — here the 95.0 peak at step 250 must not win."""
    run = _write_run(
        tmp_path / "run",
        rows=[(100, 10.0), (200, 50.0), (250, 95.0), (300, 40.0)],
        checkpoint_steps=[100, 200, 300],
    )
    steps, _path, _row = export.pick_checkpoint(run, "test/score")
    assert steps == 200


def test_ties_go_to_the_later_checkpoint(tmp_path):
    run = _write_run(
        tmp_path / "run",
        rows=[(100, 50.0), (200, 50.0)],
        checkpoint_steps=[100, 200],
    )
    steps, _path, _row = export.pick_checkpoint(run, "test/score")
    assert steps == 200


def test_missing_metric_values_are_skipped_not_read_as_zero(tmp_path):
    """The CSV backend writes the string "None" for a metric an epoch did not
    produce. Coercing that to 0.0 would make a gap compete as a real score."""
    run = _write_run(tmp_path / "run", rows=[(100, 30.0)], checkpoint_steps=[100, 200])
    with (run / "log.csv").open("a", newline="") as handle:
        csv.writer(handle).writerow([2, 200, "None"])
    steps, _path, _row = export.pick_checkpoint(run, "test/score")
    assert steps == 100


def test_run_without_checkpoints_is_not_publishable(tmp_path):
    run = _write_run(tmp_path / "run", rows=[(100, 30.0)], checkpoint_steps=[])
    assert export.pick_checkpoint(run, "test/score") is None


def test_run_without_a_logged_score_is_not_publishable(tmp_path):
    """Killed before its first epoch boundary: checkpoints exist, nothing was
    ever evaluated, so there is no basis on which to publish one."""
    run = _write_run(tmp_path / "run", rows=[], checkpoint_steps=[100])
    assert export.pick_checkpoint(run, "test/score") is None


# task, cell, agent, stamp, steps reached, marked done
GRID_RUNS = [
    ("CheetahRun", "warp_gpu", "td3", "2026-01-01", 50_000_000, True),
    ("CheetahRun", "warp_gpu", "sac", "2026-01-01", 20_000_000, False),
    ("CheetahRun", "warp_gpu", "ppo", "2026-01-01", 5_000_000, True),
    ("CheetahRun", "envpool_cpu", "td3", "2026-01-01", 50_000_000, True),
    ("WalkerWalk", "warp_gpu", "td3", "2026-01-01", 50_000_000, True),
]


def _write_grid(tmp_path, extra=()):
    """A release tree plus the `.done` markers, as `discover_runs` reads them.

    The marker is per ARM — `<task>_<agent>_<cell>`, no stamp — which is why
    these fixtures carry the budget each run actually reached: it is the only
    thing that can separate two attempts at the same arm.
    """
    root, done = tmp_path / "outputs", tmp_path / "done"
    done.mkdir(parents=True, exist_ok=True)
    for task, cell, agent, stamp, steps, finished in [*GRID_RUNS, *extra]:
        _write_run(
            root / task / cell / agent / stamp,
            rows=[(steps // 2, 10.0), (steps, 50.0)],
            checkpoint_steps=[steps // 2, steps],
        )
        if finished:
            (done / f"{task}_{agent}_{cell}").touch()
    return export.discover_runs(root, done)


def test_unmarked_runs_are_never_published(tmp_path):
    """A crashed run leaves loadable checkpoints on disk. The `.done` marker is
    the only signal that the arm actually completed."""
    chosen = export.release_runs(_write_grid(tmp_path), "CheetahRun",
                                 "warp_gpu", None)
    assert "sac" not in chosen
    assert set(chosen) == {"td3", "ppo"}


def test_budget_filter_excludes_a_shorter_pilot(tmp_path):
    """PPO's arm is a 5M pilot; asking for 50M must not publish it as one."""
    chosen = export.release_runs(_write_grid(tmp_path), "CheetahRun",
                                 "warp_gpu", 50_000_000)
    assert set(chosen) == {"td3"}


def test_budget_filter_tolerates_a_different_step_increment(tmp_path):
    """Agents step the env at different per-iteration increments, so one budget
    lands on slightly different totals — 500,000,768 for six of the seven and
    500,006,912 for PPO. An exact match would publish only PPO."""
    runs = _write_grid(tmp_path, extra=[
        ("CheetahRun", "warp_gpu", "d4pg", "2026-01-01", 50_006_912, True),
    ])
    budget = export.budget_for(runs, "CheetahRun", "warp_gpu")
    assert budget == 50_006_912
    chosen = export.release_runs(runs, "CheetahRun", "warp_gpu", budget)
    assert set(chosen) == {"td3", "d4pg"}


def test_default_budget_is_the_longest_completed_one(tmp_path):
    """A smoke pass touches the same marker as a release run. Defaulting to the
    newest finished run would publish a smoke policy the moment someone
    validated the grid after training it — so the newer 100k run must lose."""
    runs = _write_grid(tmp_path, extra=[
        ("CheetahRun", "warp_gpu", "td3", "2026-06-01", 100_000, True),
    ])
    budget = export.budget_for(runs, "CheetahRun", "warp_gpu")
    assert budget == 50_000_000
    chosen = export.release_runs(runs, "CheetahRun", "warp_gpu", budget)
    assert chosen["td3"]["stamp"] == "2026-01-01"


def test_default_budget_is_none_when_nothing_finished(tmp_path):
    """No finished run on the cell: the caller must report "nothing to export"
    rather than silently fall through to an unfiltered export."""
    assert export.budget_for(_write_grid(tmp_path), "CheetahRun",
                             "mjx_cpu") is None


def test_other_cells_and_tasks_are_excluded(tmp_path):
    chosen = export.release_runs(_write_grid(tmp_path), "CheetahRun",
                                 "warp_gpu", None)
    assert all(run["cell"] == "warp_gpu" for run in chosen.values())
    assert all(run["task"] == "CheetahRun" for run in chosen.values())


def test_the_newest_run_of_an_agent_wins(tmp_path):
    """A re-run of an arm supersedes its first pass at the same budget, which is
    what publishes the second sweep of an agent rather than the first."""
    runs = _write_grid(tmp_path, extra=[
        ("CheetahRun", "warp_gpu", "td3", "2026-06-01", 50_000_000, True),
    ])
    chosen = export.release_runs(runs, "CheetahRun", "warp_gpu", None)
    assert chosen["td3"]["stamp"] == "2026-06-01"


def test_final_mean_reports_where_the_run_ended(tmp_path):
    """Published beside the selected checkpoint's own score, because on an arm
    that peaked and then collapsed the two disagree by hundreds of points and it
    is the collapse the release figures plot."""
    run = _write_run(
        tmp_path / "run",
        rows=[(100, 10.0), (200, 90.0), (300, 20.0), (400, 0.0)],
        checkpoint_steps=[100, 200, 300, 400],
    )
    assert export.pick_checkpoint(run, "test/score")[0] == 200
    assert export.final_mean(run, "test/score", n=2) == pytest.approx(10.0)


BENCH = OmegaConf.load(REPO / "experiments" / "dmc" / "bench" / "dmc.yaml")
NOISE = OmegaConf.load(REPO / "experiments" / "dmc" / "noise" / "bench_gaussian.yaml")


def test_noise_anneal_tracks_the_step_budget():
    """`decay_steps` is 40% of `trainer.steps` by design (see the header of
    experiments/dmc/noise/bench_gaussian.yaml). A decay measured in env steps
    becomes "no anneal" at a shorter budget and "no exploration" at a longer
    one, and neither shows up as a failure — only as a worse curve."""
    budget = BENCH.trainer.steps
    decay = NOISE.decay_schedule.decay_steps
    assert decay == pytest.approx(0.4 * budget), (
        f"noise decay_steps={decay:,} is {decay / budget:.0%} of the "
        f"{budget:,}-step budget; the benchmark pins it at 40%."
    )


def test_save_cadence_divides_the_budget():
    """Every save must land on a step the eval also lands on, or the exported
    checkpoint has no score of its own and `pick_checkpoint` has to fall back on
    a neighbouring epoch's number."""
    trainer = BENCH.trainer
    assert trainer.steps % trainer.save_steps == 0
    assert trainer.save_steps % trainer.epoch_steps == 0
    assert trainer.steps % trainer.epoch_steps == 0


def test_a_run_produces_enough_checkpoints_to_choose_from():
    """Publishing the best of N is only meaningful for N well above 1."""
    trainer = BENCH.trainer
    assert trainer.steps // trainer.save_steps >= 10
