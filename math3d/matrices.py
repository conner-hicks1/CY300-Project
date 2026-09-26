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

def look_at(
    eye,
    target,
    up
) -> np.ndarray:

    eye = np.asarray(
        eye,
        dtype=np.float32
    )

    forward = normalize(
        np.asarray(target, dtype=np.float32) - eye,
        "look_at eye and target cannot be identical."
    )

    right = np.cross(
        forward,
        np.asarray(up, dtype=np.float32)
    )

    right = normalize(
        right,
        "look_at up vector cannot be parallel to the view direction."
    )

    camera_up = np.cross(
        right,
        forward
    )

    return np.array(
        [
            [
                right[0],
                right[1],
                right[2],
                -np.dot(right, eye)
            ],
            [
                camera_up[0],
                camera_up[1],
                camera_up[2],
                -np.dot(camera_up, eye)
            ],
            [
                -forward[0],
                -forward[1],
                -forward[2],
                np.dot(forward, eye)
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

    linear = np.asarray(
        model,
        dtype=np.float64
    )[:3, :3]

    c0 = linear[:, 0]
    c1 = linear[:, 1]
    c2 = linear[:, 2]

    cofactor = np.column_stack(
        (
            np.cross(c1, c2),
            np.cross(c2, c0),
            np.cross(c0, c1)
        )
    )

    determinant = float(
        np.dot(c0, np.cross(c1, c2))
    )

    if determinant < 0.0:
        cofactor = -cofactor

    return cofactor.astype(
        np.float32
    )
