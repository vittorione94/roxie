"""Dump a CSV of every CMU mocap clip's duration and applied foot correction.

Loads each clip through the exact training pipeline (``_load_single_clip`` — same
resample to ``ctrl_dt`` and same ground-correction offset), so the reported
duration and vertical shift match what the env actually trains/plays on. Useful
for picking clips by length and for auditing how much each clip is lifted onto
the floor.

Columns:
  - ``clip_id``          the CMU clip id.
  - ``frames``           frame count after resampling to ``ctrl_dt``.
  - ``duration_s``       frames * ctrl_dt.
  - ``ground_offset_m``  the vertical shift subtracted to ground the clip (the
                         "foot correction"; 0 means it was already on/under the
                         floor and left untouched).

Run from the repo root::

    python examples/mocap/check_cmu_mocap_data.py
    python examples/mocap/check_cmu_mocap_data.py --clip-ids CMU_016_22,CMU_002_01
    python examples/mocap/check_cmu_mocap_data.py --out /tmp/clips.csv --ctrl-dt 0.025
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import csv
import sys
import warnings

import click
import numpy as np
from tqdm import tqdm

from roxie.utils import hydra_searchpath

sys.path.insert(0, str(hydra_searchpath.REPO_ROOT))

from examples.mocap.cmu_mocap_data import (  # noqa: E402
    _cmu_data,
    _foot_geom_ids,
    _load_single_clip,
    build_cmu_humanoid,
    mocap_loader,
)


@click.command()
@click.option(
    "--clip-ids",
    type=str,
    default=None,
    help="Comma-separated clip ids. Omit to process every clip in the HDF5 file.",
)
@click.option("--ctrl-dt", type=float, default=0.025, help="Resample timestep (s).")
@click.option(
    "--out",
    type=str,
    default="cmu_mocap_clips.csv",
    help="Output CSV path.",
)
def main(clip_ids, ctrl_dt, out):
    mj_model, _ = build_cmu_humanoid()
    foot_geom_ids = _foot_geom_ids(mj_model)

    h5_path = _cmu_data.get_path_for_cmu(version="2020")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="label\\(\\) is deprecated")
        loader = mocap_loader.HDF5TrajectoryLoader(h5_path)

    ids = [c.strip() for c in clip_ids.split(",")] if clip_ids else list(loader.keys())

    rows = []
    for cid in tqdm(ids, desc="Scanning clips"):
        clip = _load_single_clip(cid, loader, mj_model, ctrl_dt, foot_geom_ids)
        frames = int(clip["qpos"].shape[0])
        rows.append(
            {
                "clip_id": cid,
                "frames": frames,
                "duration_s": round(frames * ctrl_dt, 4),
                "ground_offset_m": round(float(clip["ground_offset"]), 5),
            }
        )

    out = os.path.abspath(out)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["clip_id", "frames", "duration_s", "ground_offset_m"]
        )
        writer.writeheader()
        writer.writerows(rows)

    frames = np.array([r["frames"] for r in rows])
    durations = np.array([r["duration_s"] for r in rows])
    offsets = np.array([r["ground_offset_m"] for r in rows])
    print(f"\nWrote {len(rows)} clips -> {out}")
    print(
        f"  frames         : total={frames.sum()}  "
        f"min={frames.min()}  median={int(np.median(frames))}  max={frames.max()}"
    )
    print(
        f"  duration_s     : total={durations.sum():.1f}  "
        f"min={durations.min():.2f}  median={np.median(durations):.2f}  "
        f"max={durations.max():.2f}"
    )
    print(
        f"  ground_offset_m: min={offsets.min():.4f}  "
        f"median={np.median(offsets):.4f}  max={offsets.max():.4f}"
    )


if __name__ == "__main__":
    main()
