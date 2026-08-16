from typing import Callable, Optional, Sequence

import distrax
import jax.numpy as jnp
from flax import nnx


class DeterministicActor(nnx.Module):
    """MLP policy whose output is squashed into [-1, 1] by a final tanh.

    NOTE ON SATURATION: the deterministic policy gradient (``-Q(s, pi(s))``)
    pushes each action dimension monotonically outward and nothing in the DPG
    objective prices the *pre-tanh* magnitude, so the logits drift until tanh
    saturates. Past that point ``d(tanh u)/du = 1 - tanh^2 u`` underflows to
    zero, the actor gradient dies, and the policy is frozen as a bang-bang
    controller. Two defences live here and in the actor losses:

    * ``output_init_scale`` starts the final layer deep inside tanh's linear
      region (the original DDPG paper's small-final-layer trick), so the
      logits have to be *driven* out rather than starting near the knee.
    * ``forward`` also returns the pre-activation, so the actor loss can add a
      one-sided penalty on it (``pre_activation_coef`` on the agents). Without
      that penalty a small init only delays the collapse, it does not prevent
      it.
    """

    def __init__(
        self,
        in_features: int,
        features: Sequence[int],
        action_dim: int,
        *,  # Make rngs a keyword-only argument
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.relu,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        output_init_scale: float = 0.01,
    ):
        # --- Store static configuration ---
        self.use_layer_norm = use_layer_norm
        self.activation_fn = activation_fn
        self.action_dim = action_dim

        # --- Define stateful layers ---
        hidden_layers = []
        norm_layers = []

        # Create hidden layers dynamically
        current_features = in_features
        for feat in features:
            hidden_layers.append(nnx.Linear(current_features, feat, rngs=rngs))
            if self.use_layer_norm:
                norm_layers.append(nnx.LayerNorm(feat, rngs=rngs))
            current_features = feat

        self.hidden_layers = nnx.List(hidden_layers)
        if self.use_layer_norm:
            self.norm_layers = nnx.List(norm_layers)

        # Dropout and output layers
        # self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        # `output_init_scale` scales the VARIANCE of the default lecun_normal
        # init (1.0), so 0.01 means 10x smaller weights and pre-tanh logits
        # starting at ~0.1 instead of ~1 -- squarely in tanh's linear region.
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
        """Return ``(action, pre_activation)`` from a single forward pass.

        The pre-activation is what the actor losses regularize; returning it
        here (rather than recovering it with arctanh, which is meaningless once
        the action has saturated to exactly +-1 in float32) keeps the penalty
        differentiable at the only point where it matters.
        """
        # Use the layers defined in __init__
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.norm_layers[i](x)
            x = self.activation_fn(x)
            # x = self.dropout(x, deterministic=not training)

        pre_activation = self.output_layer(x)
        # Scale output to action space range, e.g., [-1, 1]
        return nnx.tanh(pre_activation), pre_activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(x)[0]


# --- Actor for SAC/PPO ---
class StochasticActor(nnx.Module):
    def __init__(
        self,
        in_features: int,
        features: Sequence[int],
        action_dim: int,
        *,  # Make rngs a keyword-only argument
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.relu,
        use_layer_norm: bool = False,
        std_min: float = 1e-4,
        std_max: float = 1.0,
    ):
        # --- Store static configuration ---
        self.use_layer_norm = use_layer_norm
        self.activation_fn = activation_fn
        self.action_dim = action_dim
        self.std_min = std_min
        self.std_max = std_max

        # --- Define stateful layers ---
        hidden_layers = []
        norm_layers = []

        # Create hidden layers dynamically
        current_features = in_features
        for feat in features:
            hidden_layers.append(nnx.Linear(current_features, feat, rngs=rngs))
            if self.use_layer_norm:
                norm_layers.append(nnx.LayerNorm(feat, rngs=rngs))
            current_features = feat

        self.hidden_layers = nnx.List(hidden_layers)
        if self.use_layer_norm:
            self.norm_layers = nnx.List(norm_layers)

        # Output layers for mean and log_std
        self.output_layer = nnx.Linear(current_features, action_dim, rngs=rngs)
        self.log_std_layer = nnx.Linear(current_features, action_dim, rngs=rngs)

        self.distribution = distrax.MultivariateNormalDiag

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Use the layers defined in __init__
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.norm_layers[i](x)
            x = self.activation_fn(x)

        mean = self.output_layer(x)
        std = nnx.softplus(self.log_std_layer(x)) + 1e-5  # Ensure std is positive

        # clamp std to avoid numerical issues
        std = jnp.clip(std, a_min=self.std_min, a_max=self.std_max)

        return self.distribution(mean, std)
