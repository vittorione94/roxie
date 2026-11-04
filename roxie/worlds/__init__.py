"""Learned world models, the agent's peer for model-based training.

Built from yaml like agents: a world config names its class with `_target_`
(e.g. `_target_: roxie.worlds.DeterministicWorld`), and `train.py` builds one
whenever the composed config has a `world:` block.
"""

from .deterministic import DeterministicWorld
from .hyperparams import WorldHyperparams
from .world import WorldModel, WorldOutput, WorldState, build_world

__all__ = [
    "DeterministicWorld",
    "WorldHyperparams",
    "WorldModel",
    "WorldOutput",
    "WorldState",
    "build_world",
]
