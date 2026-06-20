from .basic import Constant, NormalRandom, OrnsteinUhlenbeck, UniformRandom
from .ddpg import DDPG
from .ppo import PPO
from .sac import SAC
from .td3 import TD3

__all__ = [
    Constant,
    NormalRandom,
    OrnsteinUhlenbeck,
    UniformRandom,
    DDPG,
    PPO,
    SAC,
    TD3,
]

agents = {
    "constant": Constant,
    "normal_random": NormalRandom,
    "uniform_random": UniformRandom,
    "ou": OrnsteinUhlenbeck,
    "ddpg": DDPG,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}
