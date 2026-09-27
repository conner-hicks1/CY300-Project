import numpy as np

from core.assertions import engine_assert


# =========================================================
# Matrix Helpers
# =========================================================
#
# All matrices are row-major numpy arrays that transform
# column vectors (M @ v), matching Transform and Camera.
# Shader.set_mat4 transposes on upload.


# =========================================================
# Vectors
# =========================================================

def normalize(
    vector,
    message: str = "Cannot normalize a zero-length vector."
) -> np.ndarray:

    vector = np.asarray(
        vector,
        dtype=np.float32
    )

    length = float(
        np.linalg.norm(vector)
    )

    engine_assert(
        length > 0.0,
        message
    )

    return (
        vector
        / length
    ).astype(
        np.float32
    )


# =========================================================
# View
# =========================================================

# ---------------------------------------------------------
# Scalar 3-vector helpers
# ---------------------------------------------------------
#
# Called every frame on single 3-vectors. np.cross/np.dot
# spend far more time on argument handling (axis
# normalization, moveaxis) than on the math itself, so
# per-frame paths use plain floats and build one array at
# the end. (Batched geometry work, e.g. in mesh_data.py,
# should keep using numpy.)

def _cross(
    a,
    b
) -> tuple[float, float, float]:

    ax, ay, az = a
    bx, by, bz = b

    return (
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx
    )


def _dot(
    a,
    b
) -> float:

    return (
        a[0] * b[0]
        + a[1] * b[1]
        + a[2] * b[2]
    )


def _normalized(
    v,
    message: str
) -> tuple[float, float, float]:

    length = _dot(v, v) ** 0.5

    engine_assert(
        length > 0.0,
        message
    )

    return (
        v[0] / length,
        v[1] / length,
        v[2] / length
    )


def look_at(
    eye,
    target,
    up
) -> np.ndarray:

    ex, ey, ez = (float(c) for c in eye)
    tx, ty, tz = (float(c) for c in target)

    eye = (ex, ey, ez)

    forward = _normalized(
        (tx - ex, ty - ey, tz - ez),
        "look_at eye and target cannot be identical."
    )

    right = _normalized(
        _cross(forward, tuple(float(c) for c in up)),
        "look_at up vector cannot be parallel to the view direction."
    )

    camera_up = _cross(
        right,
        forward
    )

    return np.array(
        [
            [
                right[0],
                right[1],
                right[2],
                -_dot(right, eye)
            ],
            [
                camera_up[0],
                camera_up[1],
                camera_up[2],
                -_dot(camera_up, eye)
            ],
            [
                -forward[0],
                -forward[1],
                -forward[2],
                _dot(forward, eye)
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0
            ]
        ],
        dtype=np.float32
    )


# =========================================================
# Projection
# =========================================================

def perspective(
    fov_degrees: float,
    aspect_ratio: float,
    near: float,
    far: float
) -> np.ndarray:

    engine_assert(
        0.0 < fov_degrees < 180.0,
        "Perspective FOV must be between 0 and 180 degrees."
    )

    engine_assert(
        aspect_ratio > 0.0,
        "Perspective aspect ratio must be positive."
    )

    engine_assert(
        0.0 < near < far,
        "Perspective planes must satisfy 0 < near < far."
    )

    f = 1.0 / np.tan(
        np.radians(fov_degrees) / 2.0
    )

    return np.array(
        [
            [
                f / aspect_ratio,
                0.0,
                0.0,
                0.0
            ],
            [
                0.0,
                f,
                0.0,
                0.0
            ],
            [
                0.0,
                0.0,
                (far + near) / (near - far),
                (2.0 * far * near) / (near - far)
            ],
            [
                0.0,
                0.0,
                -1.0,
                0.0
            ]
        ],
        dtype=np.float32
    )


def orthographic(
    left: float,
    right: float,
    bottom: float,
    top: float,
    near: float,
    far: float
) -> np.ndarray:

    engine_assert(
        right != left
        and top != bottom
        and far != near,
        "Orthographic bounds cannot be zero-sized."
    )

    return np.array(
        [
            [
                2.0 / (right - left),
                0.0,
                0.0,
                -(right + left) / (right - left)
            ],
            [
                0.0,
                2.0 / (top - bottom),
                0.0,
                -(top + bottom) / (top - bottom)
            ],
            [
                0.0,
                0.0,
                -2.0 / (far - near),
                -(far + near) / (far - near)
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0
            ]
        ],
        dtype=np.float32
    )


# =========================================================
# Normal Matrix
# =========================================================

def normal_matrix(
    model: np.ndarray
) -> np.ndarray:
    """
    Matrix that transforms normals for `model`.

    Uses the cofactor matrix rather than inverse-transpose:
    cofactor(M) == det(M) * inverse(M)^T, but it is still
    defined when M is singular (e.g. a scale of 0 on one
    axis flattens a cube into a quad that still needs
    valid normals). Shaders normalize the result, so the
    missing 1/det only matters for its sign, which is
    restored for mirrored (negative determinant) matrices.
    """

    # Per-draw hot path: plain floats (see _cross).

    (m00, m01, m02, _), (m10, m11, m12, _), (m20, m21, m22, _) = (
        np.asarray(model, dtype=np.float64)[:3].tolist()
    )

    c0 = (m00, m10, m20)
    c1 = (m01, m11, m21)
    c2 = (m02, m12, m22)

    # Columns of the cofactor matrix.

    k0 = _cross(c1, c2)
    k1 = _cross(c2, c0)
    k2 = _cross(c0, c1)

    sign = (
        -1.0
        if _dot(c0, k0) < 0.0
        else 1.0
    )

    return np.array(
        [
            [k0[0] * sign, k1[0] * sign, k2[0] * sign],
            [k0[1] * sign, k1[1] * sign, k2[1] * sign],
            [k0[2] * sign, k1[2] * sign, k2[2] * sign],
        ],
        dtype=np.float32
    )
