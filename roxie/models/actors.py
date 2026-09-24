"""Actor neural network models and step execution functions."""

import functools
import math
from typing import Callable, Optional, Sequence

import distrax
import jax
import jax.numpy as jnp
from flax import nnx

from roxie.utils import precision
from roxie.utils.math import finite_or_zero, inv_softplus


def torch_linear_init(fan_in: int, gain: float = 1.0):
    """`(kernel_init, bias_init)` matching `torch.nn.Linear`'s own defaults.

    Both are U(-b, b) with b = 1/sqrt(fan_in); only the kernel takes `gain`, the
    way the mjbatch reference's `mlp()` does (`head.weight.data *= gain`, bias
    untouched).
    """
    bound = 1.0 / math.sqrt(fan_in)

    def kernel(key, shape, dtype=precision.FLOAT):
        return gain * jax.random.uniform(key, shape, dtype, -bound, bound)

    def bias(key, shape, dtype=precision.FLOAT):
        return jax.random.uniform(key, shape, dtype, -bound, bound)

    return kernel, bias


def _log_one_minus_tanh_sq(u: jnp.ndarray) -> jnp.ndarray:
    """Computes log(1 - tanh(u)^2) in a numerically stable manner."""
    return 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))


class TanhNormal:
    """Diagonal Gaussian distribution squashed through a tanh transformation."""

    def __init__(self, loc: jnp.ndarray, scale: jnp.ndarray):
        self._base = distrax.MultivariateNormalDiag(loc, scale)

    @property
    def loc(self) -> jnp.ndarray:
        """The pre-squash mean array."""
        return self._base.loc

    @property
    def scale_diag(self) -> jnp.ndarray:
        """The pre-squash diagonal scale array."""
        return self._base.scale_diag

    def log_prob_from_pre(self, u: jnp.ndarray) -> jnp.ndarray:
        """Evaluates the log-density of tanh(u) using the pre-activation array u."""
        return self._base.log_prob(u) - jnp.sum(_log_one_minus_tanh_sq(u), axis=-1)

    def sample(self, seed, sample_shape=()):
        """Draws squashed action samples in (-1, 1)."""
        return jnp.tanh(self._base.sample(seed=seed, sample_shape=sample_shape))

    def sample_from_pre(self, seed, sample_shape=()):
        """Draws a sample, returning `(action, pre_activation)`."""
        u = self._base.sample(seed=seed, sample_shape=sample_shape)
        return jnp.tanh(u), u

    def mean(self) -> jnp.ndarray:
        """Returns the mode of the squashed density (tanh of base mean)."""
        return jnp.tanh(self._base.mean())

    def stddev(self) -> jnp.ndarray:
        """Returns the pre-squash diagonal scale array."""
        return self._base.stddev()

    def entropy(self, seed=None) -> jnp.ndarray:
        """Estimates squashed distribution entropy using a single sample draw."""
        if seed is None:
            seed = jax.random.PRNGKey(0)
        u = self._base.sample(seed=seed)
        base_entropy = self._base.entropy().astype(self._base.loc.dtype)
        return base_entropy + jnp.sum(_log_one_minus_tanh_sq(u), axis=-1)


class ClippedNormal:
    """Diagonal Gaussian whose actions are clipped, not squashed, into [-1, 1]."""

    def __init__(self, loc: jnp.ndarray, scale: jnp.ndarray):
        self._base = distrax.MultivariateNormalDiag(loc, scale)

    @property
    def loc(self) -> jnp.ndarray:
        """The pre-clip mean array."""
        return self._base.loc

    @property
    def scale_diag(self) -> jnp.ndarray:
        """The diagonal scale array."""
        return self._base.scale_diag

    def log_prob_from_pre(self, u: jnp.ndarray) -> jnp.ndarray:
        """Evaluates the Gaussian log-density of the unclipped draw `u`."""
        return self._base.log_prob(u)

    def sample(self, seed, sample_shape=()):
        """Draws clipped action samples in [-1, 1]."""
        return jnp.clip(
            self._base.sample(seed=seed, sample_shape=sample_shape), -1.0, 1.0
        )

    def sample_from_pre(self, seed, sample_shape=()):
        """Draws a sample, returning `(clipped_action, unclipped_draw)`."""
        u = self._base.sample(seed=seed, sample_shape=sample_shape)
        return jnp.clip(u, -1.0, 1.0), u

    def mean(self) -> jnp.ndarray:
        """Returns the clipped mean."""
        return jnp.clip(self._base.mean(), -1.0, 1.0)

    def stddev(self) -> jnp.ndarray:
        """Returns the diagonal scale array."""
        return self._base.stddev()

    def entropy(self, seed=None) -> jnp.ndarray:
        """The exact Gaussian entropy; no draw is needed, so `seed` is ignored."""
        del seed
        # distrax returns float64 here even for float32 params, whatever
        # `jax_enable_x64` says; pinned so an f64 never reaches the loss.
        return self._base.entropy().astype(self._base.loc.dtype)


def deterministic_action(output) -> jnp.ndarray:
    """Extracts deterministic evaluation actions from a policy output."""
    return output if isinstance(output, jax.Array) else output.mean()


class DeterministicActor(nnx.Module):
    """MLP deterministic policy squashed into [-1, 1] via tanh."""

    def __init__(
        self,
        in_features: int,
        features: Sequence[int],
        action_dim: int,
        *,
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.relu,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        output_init_scale: float = 0.01,
    ):
        self.use_layer_norm = use_layer_norm
        self.activation_fn = activation_fn
        self.action_dim = action_dim

        hidden_layers = []
        norm_layers = []

        current_features = in_features
        for feat in features:
            hidden_layers.append(nnx.Linear(current_features, feat, rngs=rngs))
            if self.use_layer_norm:
                norm_layers.append(nnx.LayerNorm(feat, rngs=rngs))
            current_features = feat

        self.hidden_layers = nnx.List(hidden_layers)
        if self.use_layer_norm:
            self.norm_layers = nnx.List(norm_layers)

        self.output_layer = nnx.Linear(
            current_features,
            action_dim,
            rngs=rngs,
            kernel_init=nnx.initializers.variance_scaling(
                output_init_scale, "fan_in", "truncated_normal"
            ),
            bias_init=nnx.initializers.zeros_init(),
        )

    def forward(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Computes a single forward pass, returning `(action, pre_activation)`."""
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.norm_layers[i](x)
            x = self.activation_fn(x)

        pre_activation = self.output_layer(x)
        return nnx.tanh(pre_activation), pre_activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(x)[0]


class StochasticActor(nnx.Module):
    """MLP stochastic policy outputting a `TanhNormal` or a `ClippedNormal`."""

    def __init__(
        self,
        in_features: int,
        features: Sequence[int],
        action_dim: int,
        *,
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.relu,
        use_layer_norm: bool = False,
        std_min: float = 1e-4,
        std_max: float = 1.0,
        output_init_scale: float = 0.01,
        init_std: float = 0.5,
        state_dependent_std: bool = True,
        squash: bool = True,
        log_std_param: bool = False,
        torch_init: bool = False,
    ):
        if log_std_param and state_dependent_std:
            raise ValueError(
                "log_std_param parameterizes a single shared log sigma; it "
                "cannot be combined with state_dependent_std"
            )
        self.use_layer_norm = use_layer_norm
        self.activation_fn = activation_fn
        self.action_dim = action_dim
        self.std_min = std_min
        self.std_max = std_max
        self.state_dependent_std = state_dependent_std
        self.squash = squash
        self.log_std_param = log_std_param

        hidden_layers = []
        norm_layers = []

        current_features = in_features
        for feat in features:
            extra = {}
            if torch_init:
                k, b = torch_linear_init(current_features)
                extra = {"kernel_init": k, "bias_init": b}
            hidden_layers.append(
                nnx.Linear(current_features, feat, rngs=rngs, **extra)
            )
            if self.use_layer_norm:
                norm_layers.append(nnx.LayerNorm(feat, rngs=rngs))
            current_features = feat

        self.hidden_layers = nnx.List(hidden_layers)
        if self.use_layer_norm:
            self.norm_layers = nnx.List(norm_layers)

        if torch_init:
            head_kernel, head_bias = torch_linear_init(
                current_features, gain=output_init_scale
            )
        else:
            head_kernel = nnx.initializers.variance_scaling(
                output_init_scale, "fan_in", "truncated_normal"
            )
            head_bias = nnx.initializers.zeros_init()
        small_init = head_kernel
        self.output_layer = nnx.Linear(
            current_features, action_dim, rngs=rngs,
            kernel_init=head_kernel, bias_init=head_bias,
        )

        std_bias = (
            math.log(init_std) if log_std_param else inv_softplus(init_std)
        )
        if state_dependent_std:
            self.log_std_layer = nnx.Linear(
                current_features, action_dim, rngs=rngs,
                kernel_init=small_init,
                bias_init=nnx.initializers.constant(std_bias),
            )
        else:
            self.log_std = nnx.Param(
                jnp.full((action_dim,), std_bias, dtype=precision.FLOAT)
            )

    def __call__(self, x: jnp.ndarray) -> TanhNormal:
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.norm_layers[i](x)
            x = self.activation_fn(x)

        mean = self.output_layer(x)
        if self.log_std_param:
            # The parameter IS log sigma, exponentiated and left unclipped, as
            # the mjbatch reference has it. That makes d(log sigma)/d(param)
            # exactly 1, so Adam's per-parameter step moves the spread by the
            # learning rate. Under the softplus branch below that derivative is
            # sigmoid(raw)/sigma instead — 0.82 at sigma=0.4, and it drifts as
            # sigma moves, so the entropy bonus and the surrogate pull on the
            # spread in a ratio that depends on where the spread already is.
            std = jnp.exp(jnp.broadcast_to(self.log_std[...], mean.shape))
        else:
            raw_std = (
                self.log_std_layer(x)
                if self.state_dependent_std
                else jnp.broadcast_to(self.log_std[...], mean.shape)
            )
            std = jnp.clip(
                nnx.softplus(raw_std) + 1e-5, self.std_min, self.std_max
            )

        return TanhNormal(mean, std) if self.squash else ClippedNormal(mean, std)


@functools.partial(nnx.jit, static_argnames=("evaluate",))
def deterministic_step_fn(
    actor_model: nnx.Module,
    observation: jnp.ndarray,
    key: jax.Array,
    noise_module: nnx.Module,
    evaluate: bool = False,
):
    """Executes action selection for a deterministic policy with exploration noise."""
    action = finite_or_zero(actor_model(observation))
    noisy_action = noise_module.add_noise(action, key, evaluate)
    noisy_action = jnp.clip(noisy_action, -1.0, 1.0)
    return noisy_action, action - noisy_action


@functools.partial(nnx.jit, static_argnames=("evaluate",))
def stochastic_step_fn(
    actor_model: nnx.Module,
    observation: jnp.ndarray,
    evaluate: bool,
    key: jax.Array,
    critic_model: nnx.Module = None,
):
    """Executes action selection for a stochastic policy in a single dispatch."""
    distribution = actor_model(observation)
    mode = finite_or_zero(deterministic_action(distribution))

    if evaluate:
        action, pre_activation = mode, distribution.loc
    else:
        action, pre_activation = distribution.sample_from_pre(seed=key)

    action = finite_or_zero(action)

    if critic_model is None:
        return action, mode - action, None, None, None
    return (
        action,
        mode - action,
        pre_activation,
        distribution.log_prob_from_pre(pre_activation),
        critic_model(observation),
    )