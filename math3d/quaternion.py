import math

import numpy as np

from core.assertions import engine_assert


# =========================================================
# Quaternions
# =========================================================
#
# Unit quaternions as numpy float64 arrays (x, y, z, w).
#
# Conventions match Transform:
#
#   * column vectors: v' = R v
#   * Euler angles in degrees, applied X first, then Y,
#     then Z:  R = Rz @ Ry @ Rx,  q = qz * qy * qx
#   * local forward is -Z, up +Y, right +X
#
# Orientations are stored as quaternions because Euler
# angles break down for arbitrary orientations (gimbal
# lock, e.g. a camera orbiting a planet).

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def identity() -> np.ndarray:

    return IDENTITY.copy()


def normalize(
    q
) -> np.ndarray:

    q = np.asarray(q, dtype=np.float64)

    length = float(np.sqrt(np.dot(q, q)))

    engine_assert(
        length > 0.0,
        "Cannot normalize a zero quaternion."
    )

    return q / length


def multiply(
    a,
    b
) -> np.ndarray:
    """a * b: applies b first, then a."""

    ax, ay, az, aw = a
    bx, by, bz, bw = b

    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def conjugate(
    q
) -> np.ndarray:

    x, y, z, w = q

    return np.array([-x, -y, -z, w])


def from_axis_angle(
    axis,
    degrees: float
) -> np.ndarray:

    axis = np.asarray(axis, dtype=np.float64)

    length = float(np.linalg.norm(axis))

    engine_assert(
        length > 0.0,
        "Rotation axis cannot be zero."
    )

    half = math.radians(degrees) * 0.5

    s = math.sin(half) / length

    return np.array(
        [
            axis[0] * s,
            axis[1] * s,
            axis[2] * s,
            math.cos(half),
        ]
    )


def from_euler(
    degrees
) -> np.ndarray:
    """(X, Y, Z) degrees -> quaternion (X applied first)."""

    x, y, z = (math.radians(float(a)) * 0.5 for a in degrees)

    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)

    # qz * qy * qx, expanded.
    return np.array(
        [
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz,
        ]
    )


def to_matrix3(
    q
) -> np.ndarray:

    x, y, z, w = q

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    )


def from_matrix3(
    m
) -> np.ndarray:
    """Pure rotation matrix -> quaternion (Shepperd's method)."""

    m = np.asarray(m, dtype=np.float64)

    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0.0:

        s = math.sqrt(trace + 1.0) * 2.0

        q = [
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
            0.25 * s,
        ]

    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:

        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0

        q = [
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[2, 1] - m[1, 2]) / s,
        ]

    elif m[1, 1] > m[2, 2]:

        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0

        q = [
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
            (m[0, 2] - m[2, 0]) / s,
        ]

    else:

        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0

        q = [
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
            (m[1, 0] - m[0, 1]) / s,
        ]

    return normalize(q)


def to_euler(
    q
) -> np.ndarray:
    """Quaternion -> (X, Y, Z) degrees (one of the valid solutions)."""

    # Deferred: math3d.matrices imports this module.
    from math3d.matrices import euler_from_rotation_matrix

    return euler_from_rotation_matrix(
        to_matrix3(q)
    )


def rotate_vector(
    q,
    v
) -> np.ndarray:

    return to_matrix3(q) @ np.asarray(v, dtype=np.float64)


def look_rotation(
    forward,
    up
) -> np.ndarray:
    """
    Orientation whose local -Z points along `forward` and
    whose local +Y is as close to `up` as possible.
    """

    forward = np.asarray(forward, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    f = forward / np.linalg.norm(forward)

    right = np.cross(f, up)

    right_length = float(np.linalg.norm(right))

    if right_length < 1e-9:

        # Looking straight along `up`: pick any right vector.
        fallback = np.array([1.0, 0.0, 0.0]) if abs(f[0]) < 0.9 else np.array([0.0, 0.0, 1.0])

        right = np.cross(f, fallback)
        right_length = float(np.linalg.norm(right))

    right /= right_length

    true_up = np.cross(right, f)

    # Columns: local X, Y, Z axes in world space (Z = -forward).
    return from_matrix3(
        np.column_stack((right, true_up, -f))
    )


def slerp(
    a,
    b,
    t: float
) -> np.ndarray:

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    dot = float(np.dot(a, b))

    # Take the short way around.
    if dot < 0.0:
        b = -b
        dot = -dot

    if dot > 0.9995:
        return normalize(a + t * (b - a))

    theta = math.acos(min(dot, 1.0))

    return (
        a * math.sin((1.0 - t) * theta)
        + b * math.sin(t * theta)
    ) / math.sin(theta)


def angle_between(
    a,
    b
) -> float:
    """Rotation angle (degrees) taking orientation a to b."""

    dot = abs(float(np.dot(normalize(a), normalize(b))))

    return math.degrees(2.0 * math.acos(min(dot, 1.0)))
