from .basic import Constant, NormalRandom, OrnsteinUhlenbeck, UniformRandom
from .ddpg import DDPG

__all__ = [Constant, NormalRandom, OrnsteinUhlenbeck, UniformRandom]

agents = {
    'constant': Constant,
    'normal_random': NormalRandom,
    'uniform_random': UniformRandom,
    'ou': OrnsteinUhlenbeck,
    'ddpg': DDPG,
}