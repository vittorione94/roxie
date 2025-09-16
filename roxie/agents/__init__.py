from .basic import Constant, NormalRandom, OrnsteinUhlenbeck, UniformRandom
from .ddpg import DDPG
from .ppo import PPO

__all__ = [Constant, NormalRandom, OrnsteinUhlenbeck, UniformRandom, DDPG, PPO]

agents = {
    "constant": Constant,
    "normal_random": NormalRandom,
    "uniform_random": UniformRandom,
    "ou": OrnsteinUhlenbeck,
    "ddpg": DDPG,
    "ppo": PPO,
}
