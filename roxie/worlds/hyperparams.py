"""World-model hyperparameter dataclasses."""

import dataclasses

from roxie.agents.hyperparams import coerce_fields


@dataclasses.dataclass(frozen=True)
class WorldHyperparams:
    """Hyperparameters shared by every learned world model.

    Attributes:
        seed: Random seed recorded for checkpointing.
        learning_rate: Learning rate of the model's optimizer.
        max_grad_norm: Global-norm gradient clip; 0 disables it.
        learning_steps: Gradient steps per learning pass.
        window: Steps of the autoregressive fit, `k`; samples are `k + 1` long.
        reward_coef: Weight of the reward term against the state term.
        termination_coef: Weight of the termination term.
        normalize_observations: Whether the model reads and predicts in the
            space of its own running observation statistics.
        obs_norm_eps: Variance floor of those statistics. Looser than the
            agents' default: a constant observation dimension would otherwise
            get a std of 1e-4, and the state loss would score float noise in it
            ten thousand times over.
    """

    seed: int = 0
    learning_rate: float = 2e-4
    max_grad_norm: float = 1.0
    learning_steps: int = 10
    window: int = 8
    reward_coef: float = 1.0
    termination_coef: float = 1.0
    normalize_observations: bool = True
    obs_norm_eps: float = 1e-6

    def __post_init__(self):
        """Coerces primitive fields and refuses a window too short to roll."""
        coerce_fields(self)
        if self.window < 2:
            raise ValueError(
                f"window must be >= 2 for an autoregressive fit, got "
                f"{self.window}"
            )
