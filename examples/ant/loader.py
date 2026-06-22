"""Loader/builder for the dm_control quadruped ("ant") example env.

Like the mocap example, this lives in ``examples`` rather than ``roxie`` and is
reached only through the ``env.builder`` dotted path set in the experiment
config (resolved by train.py/play.py via ``hydra.utils.get_method``), so the
core loops never name "quadruped". It pulls everything off ``cfg_env`` and
returns the normalized ``EnvBundle`` the loops expect.
"""

from typing import Any

from examples.ant.quadruped import QuadrupedMove
from roxie.environment.loader import EnvBundle, TerminationWrapper

# Per-world Warp budgets. The quadruped only ever touches the floor (4 toes +
# the occasional body geom on a fall), so the contact count per world is small;
# 24 leaves generous headroom over the handful seen at peak. njmax is the
# per-world constraint budget (joint/tendon limits + the 4 coupling equalities +
# contact friction rows) — 128 clears it comfortably.
_PER_WORLD_CONTACTS = 24
_NJMAX = 128
# Single-world playback: size above the one-world peak; memory is irrelevant.
_PLAY_NACONMAX = 256


def build_ant_env(cfg_env: Any, mode: str = "train") -> EnvBundle:
  """Builder for the dm_control quadruped Move task (see ``env.builder``).

  The env ships fixed default Warp budgets; on the Warp backend we override
  ``naconmax`` to scale with the number of parallel worlds (linear in
  ``parallel_envs``) so memory tracks the world count rather than a fixed cap.
  The MJX backend ignores both budgets. ``desired_speed`` selects walk (0.5) vs
  run (5.0) and is forwarded so the reward and floor size match.
  """
  impl = cfg_env.get("impl", "warp")
  naconmax = cfg_env.get("naconmax", None)
  njmax = cfg_env.get("njmax", None)

  if impl == "warp":
    if naconmax is None:
      naconmax = (
          int(cfg_env.parallel_envs) * _PER_WORLD_CONTACTS
          if mode == "train"
          else _PLAY_NACONMAX
      )
    if njmax is None:
      njmax = _NJMAX

  overrides: dict[str, Any] = {"impl": impl}
  if naconmax is not None:
    overrides["naconmax"] = int(naconmax)
  if njmax is not None:
    overrides["njmax"] = int(njmax)
  # Optional per-experiment env knobs; only forward those actually set so the
  # env's default_config supplies the rest.
  for key in ("desired_speed", "spawn_height", "settle_steps"):
    val = cfg_env.get(key, None)
    if val is not None:
      overrides[key] = float(val) if key != "settle_steps" else int(val)
  if cfg_env.get("random_orientation", None) is not None:
    overrides["random_orientation"] = bool(cfg_env.random_orientation)

  env = TerminationWrapper(QuadrupedMove(config_overrides=overrides))
  test_env = TerminationWrapper(QuadrupedMove(config_overrides=overrides))
  return EnvBundle(env=env, test_env=test_env, env_cfg=None)
