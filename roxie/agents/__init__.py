"""Agent implementations.

Agents are built from yaml by `hydra.utils.instantiate` — each agent config
names its class with `_target_` (e.g. ``_target_: roxie.agents.TD3``) and every
constructor keyword alongside it. There is deliberately no name->class registry
any more: a registry meant a second place to keep in sync, and the config's
`_target_` is already the complete, unambiguous answer.
"""

from .basic import Constant, NormalRandom, OrnsteinUhlenbeck, UniformRandom
from .d4pg import D4PG
from .ddpg import DDPG
from .mpo import MPO
from .ppo import PPO
from .sac import SAC
from .td3 import TD3
from .td4 import TD4

__all__ = [
    "Constant",
    "NormalRandom",
    "OrnsteinUhlenbeck",
    "UniformRandom",
    "D4PG",
    "DDPG",
    "MPO",
    "PPO",
    "SAC",
    "TD3",
    "TD4",
]
