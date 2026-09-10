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


MANIFEST_ROWS = [
    # status, task, cell, agent, steps
    ("ok", "CheetahRun", "warp_gpu", "td3", "50000000"),
    ("fail", "CheetahRun", "warp_gpu", "sac", "50000000"),
    ("ok", "CheetahRun", "warp_gpu", "ppo", "5000000"),
    ("ok", "CheetahRun", "envpool_cpu", "td3", "50000000"),
    ("ok", "WalkerWalk", "warp_gpu", "td3", "50000000"),
]


def _manifest(tmp_path):
    path = tmp_path / "manifest.tsv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["status", "task", "cell", "agent", "steps",
                         "seconds", "sps", "run_dir"])
        for status, task, cell, agent, steps in MANIFEST_ROWS:
            writer.writerow([status, task, cell, agent, steps, "1", "1",
                             f"outputs/{task}/{cell}/{agent}/x"])
    return export.read_manifest(path)


def test_failed_runs_are_never_published(tmp_path):
    """A crashed run leaves loadable checkpoints on disk. `ok` is the only
    signal that the arm actually completed its budget."""
    runs = export.latest_ok_runs(_manifest(tmp_path), "CheetahRun",
                                 "warp_gpu", steps=None)
    assert "sac" not in runs
    assert set(runs) == {"td3", "ppo"}


def test_steps_filter_excludes_a_shorter_pilot(tmp_path):
    """PPO's `ok` row is a 5M pilot; asking for 50M must not publish it as one."""
    runs = export.latest_ok_runs(_manifest(tmp_path), "CheetahRun",
                                 "warp_gpu", steps=50_000_000)
    assert set(runs) == {"td3"}


def test_default_budget_is_the_longest_completed_one(tmp_path):
    """`--smoke` records its 100k runs as `ok` in the same ledger. Defaulting to
    "newest ok run" would publish a smoke policy the moment someone validated
    the grid after training it — the run that finished the longest budget is the
    release run by construction."""
    rows = _manifest(tmp_path)
    rows.append({"status": "ok", "task": "CheetahRun", "cell": "warp_gpu",
                 "agent": "td3", "steps": "100000", "seconds": "1", "sps": "1",
                 "run_dir": "outputs/smoke"})
    assert export.budget_for(rows, "CheetahRun", "warp_gpu") == 50_000_000
    runs = export.latest_ok_runs(
        rows, "CheetahRun", "warp_gpu",
        steps=export.budget_for(rows, "CheetahRun", "warp_gpu"),
    )
    assert runs["td3"]["run_dir"] != "outputs/smoke"


def test_default_budget_is_none_when_nothing_completed(tmp_path):
    """No `ok` row for the cell: the caller must report "nothing to export"
    rather than silently fall through to an unfiltered export."""
    assert export.budget_for(_manifest(tmp_path), "CheetahRun",
                             "mjx_cpu") is None


def test_other_cells_and_tasks_are_excluded(tmp_path):
    runs = export.latest_ok_runs(_manifest(tmp_path), "CheetahRun",
                                 "warp_gpu", steps=None)
    assert all(row["cell"] == "warp_gpu" for row in runs.values())
    assert all(row["task"] == "CheetahRun" for row in runs.values())


def test_the_newest_run_of_an_agent_wins(tmp_path):
    """The manifest is append-only, so a later line is a later run — a re-run
    with `--force` must supersede the row it replaced."""
    path = tmp_path / "manifest.tsv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["status", "task", "cell", "agent", "steps",
                         "seconds", "sps", "run_dir"])
        for stamp in ("first", "second"):
            writer.writerow(["ok", "CheetahRun", "warp_gpu", "td3",
                             "50000000", "1", "1", f"outputs/{stamp}"])
    runs = export.latest_ok_runs(export.read_manifest(path), "CheetahRun",
                                 "warp_gpu", steps=None)
    assert runs["td3"]["run_dir"] == "outputs/second"


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
