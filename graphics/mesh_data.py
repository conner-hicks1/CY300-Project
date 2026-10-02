from dataclasses import dataclass

import numpy as np

from core.assertions import engine_assert


# =========================================================
# Standard Vertex Format
# =========================================================
#
#     location 0  position  vec3
#     location 1  normal    vec3
#     location 2  color     vec3
#     location 3  uv        vec2
#     location 4  tangent   vec4  (xyz tangent, w = bitangent sign)
#
# The tangent's w stores handedness so mirrored UVs still
# produce correct normal mapping:
#
#     bitangent = cross(normal, tangent.xyz) * tangent.w

POSITION_SIZE = 3
NORMAL_SIZE = 3
COLOR_SIZE = 3
UV_SIZE = 2
TANGENT_SIZE = 4

FLOATS_PER_VERTEX = (
    POSITION_SIZE
    + NORMAL_SIZE
    + COLOR_SIZE
    + UV_SIZE
    + TANGENT_SIZE
)


# =========================================================
# Mesh Data
# =========================================================
#
# CPU-side geometry, independent of OpenGL. Produced by
# MeshFactory and the model loaders, uploaded by
# Mesh.from_data().

@dataclass(slots=True)
class MeshData:

    # (vertex_count, FLOATS_PER_VERTEX) float32
    vertices: np.ndarray

    # (index_count,) uint32, triangles
    indices: np.ndarray

    # -----------------------------------------------------
    # Construction
    # -----------------------------------------------------

    @classmethod
    def from_attributes(
        cls,
        positions,
        indices,
        normals=None,
        uvs=None,
        colors=None,
        tangents=None
    ) -> "MeshData":
        """
        Build interleaved vertex data. Missing normals are
        computed (smooth), missing UVs default to 0,
        missing colors to white, and missing tangents are
        derived from positions/UVs. (Planet terrain passes
        `tangents` to carry per-vertex data instead; its
        shading uses no normal map.)
        """

        positions = np.asarray(
            positions,
            dtype=np.float32
        ).reshape(-1, 3)

        indices = np.asarray(
            indices,
            dtype=np.uint32
        ).reshape(-1)

        count = len(
            positions
        )

        engine_assert(
            count > 0,
            "MeshData needs at least one vertex."
        )

        engine_assert(
            len(indices) > 0
            and len(indices) % 3 == 0,
            "MeshData indices must describe whole triangles."
        )

        engine_assert(
            int(indices.max()) < count,
            "MeshData index references a nonexistent vertex."
        )

        if normals is None:

            normals = compute_smooth_normals(
                positions,
                indices
            )

        normals = np.asarray(
            normals,
            dtype=np.float32
        ).reshape(-1, 3)

        if uvs is None:

            uvs = np.zeros(
                (count, 2),
                dtype=np.float32
            )

        uvs = np.asarray(
            uvs,
            dtype=np.float32
        ).reshape(-1, 2)

        if colors is None:

            colors = np.ones(
                (count, 3),
                dtype=np.float32
            )

        colors = np.asarray(
            colors,
            dtype=np.float32
        ).reshape(-1, 3)

        engine_assert(
            len(normals) == count
            and len(uvs) == count
            and len(colors) == count,
            "MeshData attribute arrays must have one entry per vertex."
        )

        if tangents is None:

            tangents = compute_tangents(
                positions,
                normals,
                uvs,
                indices
            )

        tangents = np.asarray(tangents, dtype=np.float32).reshape(-1, 4)

        engine_assert(
            len(tangents) == count,
            "MeshData tangents must have one entry per vertex."
        )

        vertices = np.hstack(
            (
                positions,
                normals,
                colors,
                uvs,
                tangents
            )
        ).astype(
            np.float32
        )

        return cls(
            vertices=vertices,
            indices=indices
        )

    # -----------------------------------------------------
    # Accessors
    # -----------------------------------------------------

    @property
    def vertex_count(
        self
    ) -> int:

        return len(
            self.vertices
        )

    @property
    def triangle_count(
        self
    ) -> int:

        return len(
            self.indices
        ) // 3

    @property
    def positions(
        self
    ) -> np.ndarray:

        return self.vertices[:, 0:3]

    @property
    def normals(
        self
    ) -> np.ndarray:

        return self.vertices[:, 3:6]

    @property
    def uvs(
        self
    ) -> np.ndarray:

        return self.vertices[:, 9:11]

    @property
    def tangents(
        self
    ) -> np.ndarray:

        return self.vertices[:, 11:15]

    @property
    def bounds(
        self
    ) -> tuple[np.ndarray, np.ndarray]:
        """Local-space axis-aligned bounds (min, max)."""

        positions = self.positions

        return (
            positions.min(axis=0),
            positions.max(axis=0)
        )


# =========================================================
# Normals
# =========================================================

def compute_smooth_normals(
    positions: np.ndarray,
    indices: np.ndarray
) -> np.ndarray:
    """
    Area-weighted per-vertex normals from counter-clockwise
    triangles.
    """

    positions = np.asarray(
        positions,
        dtype=np.float64
    )

    triangles = np.asarray(
        indices,
        dtype=np.int64
    ).reshape(-1, 3)

    p0 = positions[triangles[:, 0]]
    p1 = positions[triangles[:, 1]]
    p2 = positions[triangles[:, 2]]

    # Unnormalized cross product = 2 * area * normal.

    face_normals = np.cross(
        p1 - p0,
        p2 - p0
    )

    normals = np.zeros_like(
        positions
    )

    for corner in range(3):

        np.add.at(
            normals,
            triangles[:, corner],
            face_normals
        )

    return _normalize_rows(
        normals,
        fallback=(0.0, 1.0, 0.0)
    ).astype(
        np.float32
    )


# =========================================================
# Tangents
# =========================================================

def compute_tangents(
    positions: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    indices: np.ndarray
) -> np.ndarray:
    """
    Per-vertex tangents (vec4) from UV gradients, using
    Lengyel's method with Gram-Schmidt orthogonalization.
    w holds the bitangent sign.
    """

    positions = np.asarray(positions, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    uvs = np.asarray(uvs, dtype=np.float64)

    triangles = np.asarray(
        indices,
        dtype=np.int64
    ).reshape(-1, 3)

    count = len(
        positions
    )

    tangent_sum = np.zeros(
        (count, 3)
    )

    bitangent_sum = np.zeros(
        (count, 3)
    )

    p0 = positions[triangles[:, 0]]
    p1 = positions[triangles[:, 1]]
    p2 = positions[triangles[:, 2]]

    uv0 = uvs[triangles[:, 0]]
    uv1 = uvs[triangles[:, 1]]
    uv2 = uvs[triangles[:, 2]]

    edge1 = p1 - p0
    edge2 = p2 - p0

    du1 = (uv1 - uv0)[:, 0:1]
    dv1 = (uv1 - uv0)[:, 1:2]
    du2 = (uv2 - uv0)[:, 0:1]
    dv2 = (uv2 - uv0)[:, 1:2]

    determinant = (
        du1 * dv2
        - du2 * dv1
    )

    # Triangles without a usable UV mapping contribute
    # nothing; their vertices fall back below.

    valid = np.abs(
        determinant
    ) > 1e-12

    inverse = np.zeros_like(
        determinant
    )

    inverse[valid] = 1.0 / determinant[valid]

    face_tangents = (
        edge1 * dv2
        - edge2 * dv1
    ) * inverse

    face_bitangents = (
        edge2 * du1
        - edge1 * du2
    ) * inverse

    for corner in range(3):

        np.add.at(
            tangent_sum,
            triangles[:, corner],
            face_tangents
        )

        np.add.at(
            bitangent_sum,
            triangles[:, corner],
            face_bitangents
        )

    # Gram-Schmidt: remove the normal component.

    normal_dot = np.sum(
        normals * tangent_sum,
        axis=1,
        keepdims=True
    )

    tangents = tangent_sum - normals * normal_dot

    lengths = np.linalg.norm(
        tangents,
        axis=1
    )

    # Vertices with no usable UV gradient get any vector
    # perpendicular to the normal.

    degenerate = lengths <= 1e-12

    if np.any(degenerate):

        tangents[degenerate] = _any_perpendicular(
            normals[degenerate]
        )

    tangents = _normalize_rows(
        tangents,
        fallback=(1.0, 0.0, 0.0)
    )

    handedness = np.where(
        np.sum(
            np.cross(normals, tangents) * bitangent_sum,
            axis=1
        ) < 0.0,
        -1.0,
        1.0
    )

    return np.hstack(
        (
            tangents,
            handedness[:, None]
        )
    ).astype(
        np.float32
    )


# =========================================================
# Helpers
# =========================================================

def _normalize_rows(
    vectors: np.ndarray,
    fallback
) -> np.ndarray:

    lengths = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    result = np.empty_like(
        vectors,
        dtype=np.float64
    )

    zero = (
        lengths[:, 0]
        <= 1e-12
    )

    result[~zero] = (
        vectors[~zero]
        / lengths[~zero]
    )

    result[zero] = fallback

    return result


def _any_perpendicular(
    normals: np.ndarray
) -> np.ndarray:

    # Cross with whichever axis is least parallel.

    axis = np.where(
        np.abs(normals[:, 0:1]) < 0.9,
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0]])
    )

    return np.cross(
        normals,
        axis
    )
