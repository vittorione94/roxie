#!/usr/bin/env python
"""One-shot check that the mujoco_warp (impl='warp') backend is usable.

Roxie's environments dispatch the physics backend through mjx's pluggable
``impl`` argument. The "warp" path
only works when three pieces line up:

  1. ``mujoco`` ships its warp bridge (``mujoco.mjx.warp``),
  2. ``warp-lang`` is the version that bridge is built against, which
     ``mujoco-mjx`` pins in its own ``warp`` extra, and
  3. a real ``impl='warp'`` step runs and stays finite.

Run this script after any dependency bump to tell, in one shot, whether the
backend is usable. Exit code 0 == warp works; non-zero == blocked (with the
reason printed).

Usage:
    uv run python scripts/check_warp.py
"""

import os
import sys

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body pos="0 0 1">
      <freejoint/>
      <geom type="capsule" size=".05 .1" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _report_versions() -> None:
    import importlib.metadata as md

    for pkg in ("mujoco", "mujoco-mjx", "playground", "warp-lang"):
        try:
            print(f"  {pkg:12s} {md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"  {pkg:12s} <not installed>")


def _diagnose_bridge() -> str | None:
    """Return a human-readable reason if the warp bridge can't load, else None."""
    import importlib.util as u

    if u.find_spec("warp") is None:
        return "warp-lang is not installed"
    try:
        from mujoco.mjx.warp import types  # noqa: F401
    except Exception as e:  # mjx raises TypeError, not ImportError, on an old warp
        return f"mujoco's warp bridge failed to import: {type(e).__name__}: {e}"
    return None


def main() -> int:
    print("Backend dependency versions:")
    _report_versions()
    print()

    reason = _diagnose_bridge()
    if reason is not None:
        print(f"BLOCKED (pre-flight): {reason}")
        return 1

    import jax
    import jax.numpy as jp
    import mujoco
    from mujoco import mjx

    mj = mujoco.MjModel.from_xml_string(_XML)
    try:
        model = mjx.put_model(mj, impl="warp")
        # Warp's make_data requires the raw MjModel (not the mjx Model).
        data = mjx.make_data(mj, impl="warp")
        data = mjx.forward(model, data)

        # Exercise the path roxie actually uses: vmapped step over a batch.
        def one(qpos):
            d = data.replace(qpos=qpos)
            d = mjx.step(model, d)
            return d.qpos

        out = jax.vmap(one)(jp.tile(data.qpos, (4, 1)))
    except Exception as e:  # noqa: BLE001 - surface any backend failure verbatim
        print(f"BLOCKED: impl='warp' failed: {type(e).__name__}: {e}")
        return 1

    if not bool(jp.isfinite(out).all()):
        print("BLOCKED: warp step produced non-finite output")
        return 1

    print(f"OK: impl='warp' works (batched step output {tuple(out.shape)}).")
    print("You can flip `impl: warp` in the experiment configs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
