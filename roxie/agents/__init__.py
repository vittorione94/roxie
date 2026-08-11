from .basic import Constant, NormalRandom, OrnsteinUhlenbeck, UniformRandom
from .d4pg import D4PG
from .ddpg import DDPG
from .mpo import MPO
from .ppo import PPO
from .sac import SAC
from .td3 import TD3
from .td4 import TD4
from .tdmpc import TDMPC

__all__ = [
    Constant,
    NormalRandom,
    OrnsteinUhlenbeck,
    UniformRandom,
    D4PG,
    DDPG,
    MPO,
    PPO,
    SAC,
    TD3,
    TD4,
    TDMPC,
]

agents = {
    "constant": Constant,
    "normal_random": NormalRandom,
    "uniform_random": UniformRandom,
    "ou": OrnsteinUhlenbeck,
    "d4pg": D4PG,
    "ddpg": DDPG,
    "mpo": MPO,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    "td4": TD4,
    "tdmpc": TDMPC,
}
