from typing import Callable, Optional, Sequence

import distrax
import jax
import jax.numpy as jnp
from flax import nnx


# Largest magnitude fed to arctanh: tanh saturates in float32 well before 1.0,
# so the inverse must be clipped or it returns inf.
_TANH_CLIP = 1.0 - 1e-6


def _log_one_minus_tanh_sq(u: jnp.ndarray) -> jnp.ndarray:
    """log(1 - tanh(u)^2), numerically stable for large |u|.

    The naive form underflows to log(0) = -inf once tanh saturates. Uses the
    identity log(1 - tanh^2 u) = 2 * (log 2 - u - softplus(-2u)).
    """
    return 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))


class TanhNormal:
    """Diagonal Normal pushed through tanh, so samples live in (-1, 1).

    Why this exists: with an unsquashed Normal nothing bounds the policy mean.
    Once it drifts past the actuator range the environment clips it, the control
    effect saturates and the gradient pulling it back vanishes -- while any
    action-magnitude or action-rate reward term keeps charging for output the
    simulator discarded. The unbounded entropy (const + sum log sigma) also lets
    an entropy bonus inflate sigma for free, since the extra spread is clipped
    away before it can cost anything. Squashing removes both: the mean is bounded
    by construction and the entropy saturates instead of growing without limit.

    Only the methods the agents actually call are implemented (`sample`,
    `sample_and_log_prob`, `log_prob`, `mean`, `entropy`, `stddev`); this is
    deliberately not a full ``distrax.Distribution``.
    """

    def __init__(self, loc: jnp.ndarray, scale: jnp.ndarray):
        # The base log_prob is already summed over the action dimension, so the
        # tanh correction must be summed to match.
        self._base = distrax.MultivariateNormalDiag(loc, scale)

    def _log_prob_from_pre(self, u: jnp.ndarray) -> jnp.ndarray:
        # y = tanh(u)  =>  log p_Y(y) = log p_U(u) - sum_i log(1 - tanh^2 u_i)
        return self._base.log_prob(u) - jnp.sum(_log_one_minus_tanh_sq(u), axis=-1)

    def sample(self, seed):
        return jnp.tanh(self._base.sample(seed=seed))

    def sample_and_log_prob(self, seed):
        """Sample an action and score it the way `log_prob` will score it later.

        The density is deliberately NOT taken from the `u` that was drawn. tanh
        saturates in float32 long before the sampler stops producing large |u|
        (with std_max=5, |u| > arctanh(1 - 1e-6) ~ 7.25 is routine), so the
        returned action no longer identifies the `u` behind it: `log_prob` can
        only recover the clipped value. Scoring the draw with the original `u`
        would hand PPO an `old_log_probs` that its own recomputation cannot
        reproduce -- at the first epoch, with the policy still untouched, the
        ratio would differ from 1, reporting KL and clipping that never happened.
        Going through `log_prob` makes the pair consistent by construction.

        This costs one arctanh, and makes the returned log-prob flat w.r.t. the
        sample once tanh saturates. That is fine here: `TanhNormal` is used by
        PPO, whose only caller of this method is act-time action selection (see
        `Agent.stochastic_step_fn`) and never backprops through it. A
        reparameterized objective (SAC-style) must not route its pathwise
        gradient through this method.
        """
        action = jnp.tanh(self._base.sample(seed=seed))
        return action, self.log_prob(action)

    def log_prob(self, actions: jnp.ndarray) -> jnp.ndarray:
        """Log-density of an already-squashed action.

        Recovers the pre-squash `u` with arctanh, clipping first because tanh
        saturates in float32 and the exact inverse would return inf. This is the
        one path that defines `u` for a stored action -- `sample_and_log_prob`
        routes through it too, so behaviour and current policies agree on `u`.
        Given that, the correction term is a function of `u` alone, is identical
        for both policies and CANCELS in PPO's importance ratio, which then
        depends only on the base log-probs.
        """
        u = jnp.arctanh(jnp.clip(actions, -_TANH_CLIP, _TANH_CLIP))
        return self._log_prob_from_pre(u)

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

    ``DeterministicActor.__call__`` already returns the action; a stochastic
    actor returns a distribution, whose deterministic action is its mean (for
    `TanhNormal`, the mode of the squashed density — see `TanhNormal.mean`).
    Callers that must not sample (evaluation rollouts) go through this instead
    of assuming one shape of output, so the same eval path serves PPO and the
    deterministic agents. Dispatch is a Python-level check resolved at trace
    time; distrax exposes ``mean`` as a method on some versions and a property
    on others, hence the `callable` test.
    """
    if isinstance(output, jax.Array):
        return output
    mean = output.mean
    return mean() if callable(mean) else mean


def distribution_entropy(distribution, key=None) -> jnp.ndarray:
    """Entropy for either a plain distrax distribution or a `TanhNormal`.

    The squashed entropy needs a sample (see `TanhNormal.entropy`); the plain
    Normal's is analytic and takes no key. Dispatch is a Python-level isinstance
    check, resolved at trace time.
    """
    if isinstance(distribution, TanhNormal):
        return distribution.entropy(seed=key)
    return distribution.entropy()


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
        squash: bool = False,
    ):
        self.use_layer_norm = use_layer_norm
        self.activation_fn = activation_fn
        self.action_dim = action_dim
        self.std_min = std_min
        self.std_max = std_max
        # False so SAC/MPO, which share this class, are untouched.
        self.squash = squash

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

        self.distribution = TanhNormal if squash else distrax.MultivariateNormalDiag

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.norm_layers[i](x)
            x = self.activation_fn(x)

        mean = self.output_layer(x)
        # softplus keeps std positive; the clip keeps it numerically sane.
        std = nnx.softplus(self.log_std_layer(x)) + 1e-5
        std = jnp.clip(std, self.std_min, self.std_max)

        return self.distribution(mean, std)
