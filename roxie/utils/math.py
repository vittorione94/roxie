import jax
import jax.numpy as jnp
from jax import vmap


@jax.jit
def scale_to_env(x: jnp.ndarray, low: jnp.ndarray, high: jnp.ndarray):
    # x in [-1, 1] -> [low, high]
    return low + 0.5 * (x + 1.0) * (high - low)


def finite_or_zero(x: jnp.ndarray) -> jnp.ndarray:
    """Replace every non-finite entry with zero.

    Used at the three boundaries where a NaN stops being one bad number and
    becomes permanent: the action leaving the actor (integrated into qpos/qvel,
    and the world is dead), the observation entering the running statistics
    (summed, so one NaN poisons the mean/std for the rest of the run), and
    anything entering the replay buffer (resampled until the run ends).

    Infinities are zeroed alongside NaN rather than clipped to a rail: both mean
    something upstream has diverged, and an inf that survives is the same NaN one
    `0 * inf` later.
    """
    return jnp.where(jnp.isfinite(x), x, 0.0)


@jax.jit
def normalize_obs(x: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray, clip: float):
    return jnp.clip((x - mean) / std, -clip, clip)

def quat_conjugate(q):
    return jnp.concatenate([q[..., :1], -q[..., 1:]], axis=-1)

def quat_norm(q):
    return jnp.linalg.norm(q, axis=-1, keepdims=True)

def quat_inverse(q):
    return quat_conjugate(q) / jnp.square(quat_norm(q))

def quat_multiply(q, p):
    w1, x1, y1, z1 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w2, x2, y2, z2 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return jnp.stack([w, x, y, z], axis=-1)

def batched_quat_diff(q_from, q_to):
    q_from_inv = quat_inverse(q_from)
    return quat_multiply(q_from_inv, q_to)


def quaternion_distance(q1, q2):
    """Angular geodesic distance between two unit quaternions, in radians.

    Accepts either (x, y, z, w) or (w, x, y, z) ordering.
    """
    # abs() folds the double cover (q and -q are the same rotation); the clip
    # guards arccos against NaN from float round-off just outside [-1, 1].
    dot_product = jnp.abs(jnp.vdot(q1, q2))
    dot_product = jnp.clip(dot_product, -1.0, 1.0)
    return 2.0 * jnp.arccos(dot_product)

def quat_to_rot6d(q):
    """MuJoCo quaternion (w, x, y, z) -> 6D continuous rotation representation.

    The 6D rep (Zhou et al., "On the Continuity of Rotation Representations in
    Neural Networks", CVPR 2019) is the first two columns of the 3x3 rotation
    matrix, flattened. Unlike a raw quaternion it is *continuous* in Euclidean
    space (no double-cover q == -q discontinuity), so it is the representation
    to feed a network. For a network INPUT the two columns are already
    orthonormal, so no Gram-Schmidt is needed (that step is only for mapping a
    network's 6D *output* back to SO(3)). Assumes q is unit norm.

    Batches over leading dims: q shape (..., 4) -> (..., 6).
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # Column 0 of R(q).
    c0x = 1.0 - 2.0 * (y * y + z * z)
    c0y = 2.0 * (x * y + w * z)
    c0z = 2.0 * (x * z - w * y)
    # Column 1 of R(q).
    c1x = 2.0 * (x * y - w * z)
    c1y = 1.0 - 2.0 * (x * x + z * z)
    c1z = 2.0 * (y * z + w * x)
    return jnp.stack([c0x, c0y, c0z, c1x, c1y, c1z], axis=-1)


def mat_to_rot6d(mat):
    """3x3 rotation matrix -> 6D continuous rotation rep.

    Same representation and rationale as `quat_to_rot6d`, and the same output
    ordering, so the two are interchangeable network inputs. Feeds directly off
    MuJoCo `xmat`, whose layout differs by backend: native MjData stores each
    body frame row-major as 9 contiguous floats, mjx as an explicit (3, 3). Both
    are accepted. Assumes `mat` is a proper rotation.

    Batches over leading dims: (..., 9) or (..., 3, 3) -> (..., 6).
    """
    mat = jnp.asarray(mat)
    if mat.shape[-1] == 9:
        mat = mat.reshape(mat.shape[:-1] + (3, 3))
    return jnp.concatenate([mat[..., :, 0], mat[..., :, 1]], axis=-1)