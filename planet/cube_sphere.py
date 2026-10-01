import math

from dataclasses import dataclass

import numpy as np


# =========================================================
# Cube-Sphere
# =========================================================
#
# The planet surface is a cube projected onto a sphere.
# Each of the 6 faces is the root of a quadtree; a node
# (face, depth, x, y) covers the square
#
#     a in [-1 + 2x / 2^depth, -1 + 2(x+1) / 2^depth]
#     b in [-1 + 2y / 2^depth, -1 + 2(y+1) / 2^depth]
#
# of face coordinates. A face point maps to a direction
# with the equal-angle ("tangent") projection:
#
#     direction = normalize(n + tan(a pi/4) u + tan(b pi/4) v)
#
# which keeps cells within ~1.4x of each other in size (a
# plain normalized cube varies by ~5x).
#
# Every face has u x v = n, so grid quads in increasing
# (a, b) order wind counter-clockwise seen from outside.

FACE_NORMALS = np.array([
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
])

FACE_U = np.array([
    (0.0, 0.0, -1.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
])

FACE_V = np.array([
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
])

FACE_COUNT = 6

_QUARTER_PI = math.pi / 4.0


@dataclass(frozen=True, slots=True)
class ChunkKey:

    face: int
    depth: int
    x: int
    y: int

    @property
    def cells(
        self
    ) -> int:
        """Nodes per face edge at this depth."""

        return 1 << self.depth

    def bounds(
        self
    ) -> tuple[float, float, float, float]:
        """(a0, a1, b0, b1) face-coordinate square."""

        size = 2.0 / self.cells

        a0 = -1.0 + self.x * size
        b0 = -1.0 + self.y * size

        return a0, a0 + size, b0, b0 + size

    def children(
        self
    ) -> tuple["ChunkKey", "ChunkKey", "ChunkKey", "ChunkKey"]:

        depth = self.depth + 1
        x = self.x * 2
        y = self.y * 2

        return (
            ChunkKey(self.face, depth, x, y),
            ChunkKey(self.face, depth, x + 1, y),
            ChunkKey(self.face, depth, x, y + 1),
            ChunkKey(self.face, depth, x + 1, y + 1),
        )

    @property
    def parent(
        self
    ) -> "ChunkKey | None":

        if self.depth == 0:
            return None

        return ChunkKey(self.face, self.depth - 1, self.x // 2, self.y // 2)


ROOT_KEYS = tuple(
    ChunkKey(face, 0, 0, 0)
    for face in range(FACE_COUNT)
)


def edge_length(
    radius: float,
    depth: int
) -> float:
    """Approximate surface length of a node's edge."""

    return (math.pi * 0.5 * radius) / (1 << depth)


def face_directions(
    face,
    a,
    b
) -> np.ndarray:
    """
    Unit directions for face coordinates. `face`, `a`, `b`
    broadcast together (scalars or arrays); the result has
    shape broadcast_shape + (3,).
    """

    face = np.asarray(face)

    ta = np.tan(np.asarray(a, dtype=np.float64) * _QUARTER_PI)[..., None]
    tb = np.tan(np.asarray(b, dtype=np.float64) * _QUARTER_PI)[..., None]

    cube = FACE_NORMALS[face] + ta * FACE_U[face] + tb * FACE_V[face]

    return cube / np.linalg.norm(cube, axis=-1, keepdims=True)


def direction_to_face(
    direction
) -> tuple[int, float, float]:
    """Face and (a, b) coordinates of a unit direction."""

    direction = np.asarray(direction, dtype=np.float64)

    face = int(np.argmax(FACE_NORMALS @ direction))

    a, b = face_coordinates(face, direction)

    return face, a, b


def face_coordinates(
    face: int,
    direction
):
    """
    (a, b) of `direction` projected onto `face`. Only
    meaningful where the direction points into the face's
    hemisphere; values beyond [-1, 1] lie past its edges.
    """

    direction = np.asarray(direction, dtype=np.float64)

    along = direction @ FACE_NORMALS[face]

    a = np.arctan((direction @ FACE_U[face]) / along) / _QUARTER_PI
    b = np.arctan((direction @ FACE_V[face]) / along) / _QUARTER_PI

    return a, b
