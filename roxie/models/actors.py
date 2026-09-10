from typing import Callable, Optional, Sequence

import distrax
import jax
import jax.numpy as jnp
from flax import nnx



def _log_one_minus_tanh_sq(u: jnp.ndarray) -> jnp.ndarray:
    """log(1 - tanh(u)^2), numerically stable for large |u|.

    The naive form underflows to log(0) = -inf once tanh saturates. Uses the
    identity log(1 - tanh^2 u) = 2 * (log 2 - u - softplus(-2u)).
    """
    return 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))


class TanhNormal:
    """Diagonal Normal pushed through tanh, so samples live in (-1, 1).

    With an unsquashed Normal nothing bounds the policy mean: once it drifts
    past the actuator range the env clips it, the control effect saturates and
    the gradient pulling it back vanishes, while any action-magnitude reward term
    keeps charging for output the simulator discarded. Its unbounded entropy
    (const + sum log sigma) also lets an entropy bonus inflate sigma for free.
    Squashing removes both.

    Only the surface the agents use is implemented — deliberately not a full
    ``distrax.Distribution``. In particular there is NO `log_prob(action)`: the
    squash is not invertible in float32, so every density is scored from the
    pre-tanh `u`. See `log_prob_from_pre`.
    """

    def __init__(self, loc: jnp.ndarray, scale: jnp.ndarray):
        # The base log_prob is already summed over the action dimension, so the
        # tanh correction must be summed to match.
        self._base = distrax.MultivariateNormalDiag(loc, scale)

    @property
    def loc(self) -> jnp.ndarray:
        """Pre-squash mean, under distrax's name for it.

        Exposed because tanh is a bijection, so the KL between two of these
        distributions is exactly the KL between their bases: MPO's decoupled
        trust region keeps working verbatim on `loc` / `scale_diag` and never
        has to account for the squash.
        """
        return self._base.loc

    @property
    def scale_diag(self) -> jnp.ndarray:
        return self._base.scale_diag

    def log_prob_from_pre(self, u: jnp.ndarray) -> jnp.ndarray:
        """Log-density of ``tanh(u)``, scored from the pre-squash `u` itself.

        y = tanh(u)  =>  log p_Y(y) = log p_U(u) - sum_i log(1 - tanh^2 u_i).

        The only path there is, and the reason there is no action-keyed
        overload: recovering `u` from `tanh(u)` needs an arctanh clipped short
        of 1.0, and `tanh` rounds to exactly 1.0 for |u| >= 8, so every
        saturated draw comes back as the same rail (~7.25 in float32). That
        kills SAC's pathwise gradient exactly where the policy rails, drags
        MPO's weighted maximum-likelihood fit back toward the rail, and makes
        PPO's stored log-prob stop tracking its policy — which is why PPO keeps
        `u` in its buffer (`transition_prototype`).
        """
        return self._base.log_prob(u) - jnp.sum(_log_one_minus_tanh_sq(u), axis=-1)

    def sample(self, seed, sample_shape=()):
        return jnp.tanh(self._base.sample(seed=seed, sample_shape=sample_shape))

    def sample_from_pre(self, seed, sample_shape=()):
        """Reparameterized draw as ``(action, pre_activation)``.

        Pairs with `log_prob_from_pre`, where the reason to keep `u` is spelled
        out. `u` is also what the saturation diagnostics are read off.
        """
        u = self._base.sample(seed=seed, sample_shape=sample_shape)
        return jnp.tanh(u), u

    def mean(self) -> jnp.ndarray:
        """tanh of the base mean -- the mode of the squashed density, not its
        true mean (which has no closed form). This is the standard deterministic
        action for a squashed policy and is what evaluation should use."""
        return jnp.tanh(self._base.mean())

    def stddev(self) -> jnp.ndarray:
        """Pre-squash scale. Reported for diagnostics only."""
        return self._base.stddev()

    def entropy(self, seed=None) -> jnp.ndarray:
        """Single-sample estimate of the squashed entropy.

        H[tanh(U)] = H[U] + E_u[sum_i log(1 - tanh^2 u_i)], and the expectation
        has no closed form, so it is estimated with one reparameterized draw
        (the same approach Brax takes). The correction is <= 0 and grows more
        negative as sigma grows, which is exactly the saturation that stops an
        entropy bonus from paying to inflate sigma forever.

        `seed=None` falls back to a fixed key: fine for logging, but callers that
        put entropy in a loss must pass a real key.
        """
        if seed is None:
            seed = jax.random.PRNGKey(0)
        u = self._base.sample(seed=seed)
        return self._base.entropy() + jnp.sum(_log_one_minus_tanh_sq(u), axis=-1)


def deterministic_action(output) -> jnp.ndarray:
    """The evaluation action for either actor family, from one forward pass.

    ``DeterministicActor.__call__`` already returns the action; `StochasticActor`
    returns a `TanhNormal`, whose deterministic action is its mean (the mode of
    the squashed density — see `TanhNormal.mean`). Callers that must not sample
    (evaluation rollouts) go through this instead of assuming one shape of
    output, so the same eval path serves every agent. Dispatch is a
    Python-level check resolved at trace time.
    """
    return output if isinstance(output, jax.Array) else output.mean()


class DeterministicActor(nnx.Module):
    """MLP policy whose output is squashed into [-1, 1] by a final tanh.

    SATURATION: the deterministic policy gradient (``-Q(s, pi(s))``) pushes each
    action dimension monotonically outward and nothing in the DPG objective
    prices the *pre-tanh* magnitude, so the logits drift until tanh saturates,
    ``d(tanh u)/du`` underflows and the policy freezes as a bang-bang
    controller. Two defences: ``output_init_scale`` starts the final layer deep
    inside tanh's linear region, so the logits have to be driven out; and
    ``forward`` returns the pre-activation, so the actor loss can charge a
    one-sided penalty on it (``pre_activation_coef``). A small init alone only
    delays the collapse.
    """

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

        # Scales the *variance* of the default lecun_normal init, so 0.01 gives
        # 10x smaller weights and pre-tanh logits inside tanh's linear region.
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
    """MLP policy emitting a `TanhNormal` over actions in (-1, 1).

    The squash is unconditional: PPO needs a policy mean the actuator range can
    bound and an entropy bonus that cannot pay to inflate sigma forever, and SAC
    and MPO additionally need the stable tanh log-prob correction their losses
    read off `TanhNormal.log_prob_from_pre`.
    """

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
    ):
        self.use_layer_norm = use_layer_norm
        self.activation_fn = activation_fn
        self.action_dim = action_dim
        self.std_min = std_min
        self.std_max = std_max

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

        self.output_layer = nnx.Linear(current_features, action_dim, rngs=rngs)
        self.log_std_layer = nnx.Linear(current_features, action_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> TanhNormal:
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.norm_layers[i](x)
            x = self.activation_fn(x)

        mean = self.output_layer(x)
        # softplus keeps std positive; the clip keeps it numerically sane.
        std = nnx.softplus(self.log_std_layer(x)) + 1e-5
        std = jnp.clip(std, self.std_min, self.std_max)

        return TanhNormal(mean, std)
