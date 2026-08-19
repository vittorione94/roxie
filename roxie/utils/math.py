import jax.numpy as jnp
from jax import vmap

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
    """
    Computes the angular geodesic distance between two unit quaternions in radians.
    Expects quaternions in the format: (x, y, z, w) or (w, x, y, z)
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

    Same representation and rationale as `quat_to_rot6d` (Zhou et al.), but taken
    straight from a rotation MATRIX instead of a quaternion. Feeds directly off
    MuJoCo `xmat`, whose layout differs by backend: native MjData stores each
    body frame row-major as 9 contiguous floats [r00, r01, r02, ...], while mjx
    keeps it as an explicit (3, 3). Both are accepted — a trailing 9 is folded to
    (3, 3). The 6D rep is the matrix's first two columns, flattened as
    [c0x, c0y, c0z, c1x, c1y, c1z] — identical ordering to `quat_to_rot6d`, so
    both are interchangeable network inputs. Assumes `mat` is a proper rotation
    (orthonormal columns).

    Batches over leading dims: (..., 9) or (..., 3, 3) -> (..., 6).
    """
    mat = jnp.asarray(mat)
    if mat.shape[-1] == 9:
        mat = mat.reshape(mat.shape[:-1] + (3, 3))
    return jnp.concatenate([mat[..., :, 0], mat[..., :, 1]], axis=-1)