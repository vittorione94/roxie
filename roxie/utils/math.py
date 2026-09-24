"""Mathematical utilities, quaternion transformations, and running statistics."""

import functools

import flax.struct as struct
import jax
import jax.numpy as jnp
import math


@jax.jit
def scale_to_env(x: jnp.ndarray, low: jnp.ndarray, high: jnp.ndarray) -> jnp.ndarray:
    """Scales actions from normalized [-1, 1] range to environment bounds [low, high]."""
    return low + 0.5 * (x + 1.0) * (high - low)


def finite_or_zero(x: jnp.ndarray) -> jnp.ndarray:
    """Replaces non-finite array entries (NaN, Inf) with zeros."""
    return jnp.where(jnp.isfinite(x), x, 0.0)


# `clip` is static so `None` can switch the bound off in the trace rather than
# at runtime. It comes off the frozen hyperparams and never moves within a run,
# so this costs no retrace.
@functools.partial(jax.jit, static_argnums=(3,))
def normalize_obs(
    x: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray, clip: float | None
) -> jnp.ndarray:
    """Standardizes observations using running moments, clipping if asked.

    `clip=None` leaves the z-score unbounded, which a task whose informative
    states are far in the tail of its own running distribution needs: on
    HumanoidWalk the 92% of every episode a fallen humanoid spends motionless
    sets the moments, so standing sits at z ~ 8.5 and a bound of 5 maps the
    whole upright-to-falling range onto one number (measured 2026-09-20).
    """
    z = (x - mean) / std
    return z if clip is None else jnp.clip(z, -clip, clip)


@struct.dataclass
class ObsStats:
    """Running moment accumulators for observation normalization."""

    count: jnp.ndarray
    sum: jnp.ndarray
    sumsq: jnp.ndarray


def init_obs_stats(obs_shape) -> ObsStats:
    """Initializes zero-filled observation statistics for a given shape."""
    return ObsStats(
        count=jnp.array(0.0, dtype=jnp.float32),
        sum=jnp.zeros(obs_shape, dtype=jnp.float32),
        sumsq=jnp.zeros(obs_shape, dtype=jnp.float32),
    )

def inv_softplus(y: float) -> float:
    """Computes the inverse of softplus for parameter initialization."""
    return float(math.log(math.expm1(y)))

@jax.jit
def update_obs_stats(stats: ObsStats, batch_obs: jnp.ndarray) -> ObsStats:
    """Updates running sum accumulators with a batch of observations."""
    batch_obs = finite_or_zero(batch_obs)
    b = batch_obs.shape[0]
    batch_sum = jnp.sum(batch_obs, axis=0)
    batch_sumsq = jnp.sum(jnp.square(batch_obs), axis=0)
    return stats.replace(
        count=stats.count + b,
        sum=stats.sum + batch_sum,
        sumsq=stats.sumsq + batch_sumsq,
    )


def obs_mean_std(stats: ObsStats, eps: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes mean and standard deviation from running statistics."""
    count = jnp.maximum(stats.count, 1.0)
    mean = stats.sum / count
    var = jnp.maximum(stats.sumsq / count - jnp.square(mean), 0.0)
    std = jnp.where(stats.count > 1.0, jnp.sqrt(var + eps), 1.0)
    return mean, std


def normalize_samples(
    samples: dict,
    mean: jnp.ndarray,
    std: jnp.ndarray,
    clip: float,
    enabled: bool = True,
) -> dict:
    """Normalizes observation and next_observation entries in a sample dictionary."""
    if not enabled:
        return samples
    return {
        **samples,
        "observations": normalize_obs(samples["observations"], mean, std, clip),
        "next_observations": normalize_obs(
            samples["next_observations"], mean, std, clip
        ),
    }


def quat_conjugate(q: jnp.ndarray) -> jnp.ndarray:
    """Computes the conjugate of a quaternion."""
    return jnp.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def quat_norm(q: jnp.ndarray) -> jnp.ndarray:
    """Computes the Euclidean norm of a quaternion."""
    return jnp.linalg.norm(q, axis=-1, keepdims=True)


def quat_inverse(q: jnp.ndarray) -> jnp.ndarray:
    """Computes the multiplicative inverse of a quaternion."""
    return quat_conjugate(q) / jnp.square(quat_norm(q))


def quat_multiply(q: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    """Multiplies two quaternions q and p."""
    w1, x1, y1, z1 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w2, x2, y2, z2 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return jnp.stack([w, x, y, z], axis=-1)


def batched_quat_diff(q_from: jnp.ndarray, q_to: jnp.ndarray) -> jnp.ndarray:
    """Computes the relative rotation quaternion from q_from to q_to."""
    q_from_inv = quat_inverse(q_from)
    return quat_multiply(q_from_inv, q_to)


def quaternion_distance(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Angular geodesic distance between two unit quaternions in radians."""
    dot_product = jnp.abs(jnp.vdot(q1, q2))
    dot_product = jnp.clip(dot_product, -1.0, 1.0)
    return 2.0 * jnp.arccos(dot_product)


def quat_to_rot6d(q: jnp.ndarray) -> jnp.ndarray:
    """Converts a (w, x, y, z) quaternion to a continuous 6D rotation representation."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    c0x = 1.0 - 2.0 * (y * y + z * z)
    c0y = 2.0 * (x * y + w * z)
    c0z = 2.0 * (x * z - w * y)
    c1x = 2.0 * (x * y - w * z)
    c1y = 1.0 - 2.0 * (x * x + z * z)
    c1z = 2.0 * (y * z + w * x)
    return jnp.stack([c0x, c0y, c0z, c1x, c1y, c1z], axis=-1)


def mat_to_rot6d(mat: jnp.ndarray) -> jnp.ndarray:
    """Converts a 3x3 rotation matrix (or flattened 9D array) to 6D rotation format."""
    mat = jnp.asarray(mat)
    if mat.shape[-1] == 9:
        mat = mat.reshape(mat.shape[:-1] + (3, 3))
    return jnp.concatenate([mat[..., :, 0], mat[..., :, 1]], axis=-1)