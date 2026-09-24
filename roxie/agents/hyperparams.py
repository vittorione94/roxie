"""Agent hyperparameter dataclasses and instantiation utilities."""

import dataclasses
import typing
from typing import Optional

from omegaconf import OmegaConf

_COERCIBLE = (bool, int, float)


@dataclasses.dataclass(frozen=True)
class AgentHyperparams:
    """Base hyperparameters shared across all learning agents.

    Attributes:
        seed: Random seed recorded for checkpointing.
        gamma: Discount factor for future rewards.
        actor_learning_rate: Learning rate for actor optimizer.
        critic_learning_rate: Learning rate for critic optimizer.
        max_grad_norm: Maximum gradient norm for clipping.
        learning_steps: Number of gradient steps per learning pass.
        steps_between_updates: Environment steps between learning passes.
        normalize_observations: Whether to normalize observations.
        obs_norm_clip: Bound on the normalized observation; None leaves it
            unbounded. Written to disk as a number, so non-positive means None.
        obs_norm_eps: Small constant for observation normalization stability.
    """

    seed: int = 0
    gamma: float = 0.99
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    learning_steps: int = 5
    steps_between_updates: int = 0
    normalize_observations: bool = True
    obs_norm_clip: Optional[float] = 5.0
    obs_norm_eps: float = 1e-8

    def __post_init__(self):
        """Coerces primitive fields to their declared type hints."""
        hints = typing.get_type_hints(type(self))
        for field in dataclasses.fields(self):
            hint = hints.get(field.name)
            if hint in _COERCIBLE:
                object.__setattr__(self, field.name, hint(getattr(self, field.name)))
        if self.obs_norm_clip is not None:
            object.__setattr__(self, "obs_norm_clip", float(self.obs_norm_clip))
            if self.obs_norm_clip <= 0:
                object.__setattr__(self, "obs_norm_clip", None)


@dataclasses.dataclass(frozen=True)
class OffPolicyHyperparams(AgentHyperparams):
    """Base hyperparameters for replay-driven off-policy algorithms.

    Attributes:
        tau: Target network soft-update coefficient.
        memory_warmup: Environment steps before learning begins.
    """

    tau: float = 0.005
    memory_warmup: int = 100


@dataclasses.dataclass(frozen=True)
class DDPGHyperparams(OffPolicyHyperparams):
    """Hyperparameters for DDPG and D4PG algorithms.

    Attributes:
        n_step: TD return step horizon for replay buffering.
        target_policy_noise: Scale of target policy smoothing noise.
        target_noise_clip: Clip threshold for target policy noise.
        pre_activation_coef: Penalty coefficient for pre-tanh saturation.
    """

    n_step: int = 1
    target_policy_noise: float = 0.1
    target_noise_clip: float = 0.1
    pre_activation_coef: float = 0.0


@dataclasses.dataclass(frozen=True)
class TD3Hyperparams(DDPGHyperparams):
    """Hyperparameters for TD3 and TD4 algorithms.

    Attributes:
        policy_delay: Gradient step frequency for delayed policy updates.
        twin_q_weight: Blend of the twin bootstrap between `min` (1.0), their
            mean (0.5) and `max` (0.0).
    """

    policy_delay: int = 2
    twin_q_weight: float = 0.5


@dataclasses.dataclass(frozen=True)
class SACHyperparams(OffPolicyHyperparams):
    """Hyperparameters for Soft Actor-Critic (SAC).

    Attributes:
        alpha_learning_rate: Learning rate for entropy temperature optimizer.
        n_step: TD return step horizon for replay buffering.
        policy_delay: Delay frequency for policy updates.
        init_log_alpha: Initial log temperature value.
        auto_alpha: Whether to automatically tune the temperature.
        target_entropy: Target entropy value; None defaults to heuristic.
        target_entropy_scale: Scale factor applied to heuristic target entropy.
        twin_q_weight: Blend of the twin bootstrap between `min` (1.0), their
            mean (0.5) and `max` (0.0).
    """

    alpha_learning_rate: float = 3e-4
    n_step: int = 1
    policy_delay: int = 1
    init_log_alpha: float = 0.0
    auto_alpha: bool = True
    target_entropy: Optional[float] = None
    target_entropy_scale: float = 1.0
    twin_q_weight: float = 0.5


@dataclasses.dataclass(frozen=True)
class MPOHyperparams(OffPolicyHyperparams):
    """Hyperparameters for Maximum A Posteriori Policy Optimization (MPO).

    Attributes:
        n_step: TD return step horizon for replay buffering.
        dual_learning_rate: Learning rate for dual variable optimizers.
        num_action_samples: Number of action samples per state.
        epsilon: KL bound constraint for policy change.
        epsilon_mean: Mean KL bound constraint for Gaussian policy.
        epsilon_stddev: Symmetrized standard deviation KL bound for Gaussian policy.
        init_temperature: Initial temperature value for action weighting.
        init_alpha_mean: Initial Lagrange multiplier for mean constraint.
        init_alpha_stddev: Initial Lagrange multiplier for stddev constraint.
    """

    n_step: int = 1
    dual_learning_rate: float = 1e-2
    num_action_samples: int = 20
    epsilon: float = 0.1
    epsilon_mean: float = 1e-3
    epsilon_stddev: float = 1e-5
    init_temperature: float = 1.0
    init_alpha_mean: float = 1.0
    init_alpha_stddev: float = 1.0


@dataclasses.dataclass(frozen=True)
class PPOHyperparams(AgentHyperparams):
    """Hyperparameters for Proximal Policy Optimization (PPO).

    Attributes:
        gae_lambda: Exponential weight for Generalized Advantage Estimation.
        clip_eps: Surrogate objective clipping threshold.
        entropy_coef: Coefficient for entropy regularization term.
        target_kl: Target KL threshold for early stopping; None disables.
        num_minibatches: Number of minibatches per rollout epoch.
        adv_norm_decay: Decay of the running advantage scale the surrogate
            normalizes by; 0.0 falls back to this rollout's own spread.
    """

    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    target_kl: Optional[float] = 0.02
    num_minibatches: int = 1
    adv_norm_decay: float = 0.99

    def __post_init__(self):
        """Coerces target_kl to float or None for negative disk representations."""
        super().__post_init__()
        if self.target_kl is not None:
            object.__setattr__(self, "target_kl", float(self.target_kl))
            if self.target_kl < 0:
                object.__setattr__(self, "target_kl", None)


def build_hyperparams(cls, config):
    """Instantiates a hyperparameter dataclass from a dict, OmegaConf, or instance.

    Args:
        cls: Target `AgentHyperparams` subclass to instantiate.
        config: Raw configuration object (dict, OmegaConf DictConfig, or `cls`).

    Returns:
        An instance of `cls`.

    Raises:
        TypeError: If `config` contains keys not declared on `cls`.
    """
    if config is None:
        return cls()
    if isinstance(config, cls):
        return config
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    config = dict(config)
    config.pop("_target_", None)

    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(config) - known)
    if unknown:
        raise TypeError(
            f"{cls.__name__} has no hyperparameter(s) {unknown}. "
            f"Known: {sorted(known)}"
        )
    return cls(**config)