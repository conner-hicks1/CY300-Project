from dataclasses import dataclass

import numpy as np


# =========================================================
# Draw Lists
# =========================================================
#
# Pure numpy preparation for batched rendering:
#
#   * per-draw data (camera-relative model matrix + normal
#     matrix) for every object, computed for all objects at
#     once instead of per draw call in Python;
#   * frustum culling of bounding spheres against any clip
#     matrix (the camera, each shadow cascade, each spot);
#   * indirect draw commands, grouped by material.
#
# The Renderer uploads the arrays and issues one
# glMultiDrawElementsIndirect per group.

# std430 per-draw record: mat4 model, mat4 normal (a mat3
# padded to mat4 keeps the layout trivial). 32 floats.
DRAW_RECORD_FLOATS = 32

# DrawElementsIndirectCommand: count, instanceCount,
# firstIndex, baseVertex, baseInstance.
COMMAND_UINTS = 5


@dataclass(slots=True)
class DrawItem:

    mesh: object                 # graphics.mesh.Mesh
    material: object             # graphics.material.Material
    world_matrix: np.ndarray     # float64 4x4
    casts_shadows: bool = True


@dataclass(slots=True)
class PreparedDraws:

    items: list[DrawItem]

    # (N, DRAW_RECORD_FLOATS) float32, ready for the GPU.
    records: np.ndarray

    # Render-space (camera-relative) bounding spheres.
    centers: np.ndarray          # (N, 3) float64
    radii: np.ndarray            # (N,)   float64

    # Pool ranges per item.
    first_index: np.ndarray      # (N,) uint32
    index_count: np.ndarray      # (N,) uint32
    base_vertex: np.ndarray      # (N,) int32

    casts_shadows: np.ndarray    # (N,) bool


def prepare(
    items: list[DrawItem],
    origin
) -> PreparedDraws:
    """
    Build per-draw records for `items`, relative to the
    render origin (camera position).
    """

    count = len(items)

    origin = np.asarray(origin, dtype=np.float64)

    if count == 0:

        empty = np.zeros(0)

        return PreparedDraws(
            items=[],
            records=np.zeros((0, DRAW_RECORD_FLOATS), dtype=np.float32),
            centers=np.zeros((0, 3)),
            radii=empty,
            first_index=np.zeros(0, dtype=np.uint32),
            index_count=np.zeros(0, dtype=np.uint32),
            base_vertex=np.zeros(0, dtype=np.int32),
            casts_shadows=np.zeros(0, dtype=bool)
        )

    world = np.stack(
        [np.asarray(item.world_matrix, dtype=np.float64) for item in items]
    )

    # Camera-relative in float64, then narrowed.
    relative = world.copy()
    relative[:, :3, 3] -= origin

    linear = relative[:, :3, :3]

    normals = np.zeros((count, 4, 4))
    normals[:, :3, :3] = cofactor_matrices(linear)
    normals[:, 3, 3] = 1.0

    records = np.empty((count, DRAW_RECORD_FLOATS), dtype=np.float32)

    # std430 / GLSL matrices are column-major.
    records[:, :16] = np.transpose(relative, (0, 2, 1)).reshape(count, 16)
    records[:, 16:] = np.transpose(normals, (0, 2, 1)).reshape(count, 16)

    # Bounding spheres in render space.

    local_centers = np.stack(
        [item.mesh.bounding_center for item in items]
    )

    local_radii = np.array(
        [item.mesh.bounding_radius for item in items]
    )

    centers = (
        np.einsum("nij,nj->ni", linear, local_centers)
        + relative[:, :3, 3]
    )

    # Largest axis scale bounds how far the sphere grows.
    scale = np.linalg.norm(linear, axis=1).max(axis=1)

    radii = local_radii * scale

    allocations = [item.mesh.allocation for item in items]

    return PreparedDraws(
        items=list(items),
        records=records,
        centers=centers,
        radii=radii,
        first_index=np.array([a.first_index for a in allocations], dtype=np.uint32),
        index_count=np.array([a.index_count for a in allocations], dtype=np.uint32),
        base_vertex=np.array([a.base_vertex for a in allocations], dtype=np.int32),
        casts_shadows=np.array([item.casts_shadows for item in items], dtype=bool)
    )


def cofactor_matrices(
    linear: np.ndarray
) -> np.ndarray:
    """
    Batched version of math3d.matrices.normal_matrix:
    (N, 3, 3) -> cofactor matrices with determinant sign
    restored. Valid for singular (zero-scale) matrices.
    """

    c0 = linear[:, :, 0]
    c1 = linear[:, :, 1]
    c2 = linear[:, :, 2]

    k0 = np.cross(c1, c2)
    k1 = np.cross(c2, c0)
    k2 = np.cross(c0, c1)

    cofactor = np.stack((k0, k1, k2), axis=2)

    determinant = np.einsum("ni,ni->n", c0, k0)

    sign = np.where(determinant < 0.0, -1.0, 1.0)

    return cofactor * sign[:, None, None]


# =========================================================
# Frustum Culling
# =========================================================

def frustum_planes(
    clip_matrix
) -> np.ndarray:
    """
    Planes (a, b, c, d), normalized, with inside where
    a*x + b*y + c*z + d >= 0, for a render-space -> clip
    matrix with [0, 1] clip depth (glClipControl
    ZERO_TO_ONE; Gribb & Hartmann).

    Works for the reversed-Z infinite camera projection:
    its "far" plane degenerates (zero normal) and is
    dropped, and its near plane comes out of the w - z row.
    """

    m = np.asarray(clip_matrix, dtype=np.float64)

    rows = m

    candidates = np.array(
        [
            rows[3] + rows[0],   # left
            rows[3] - rows[0],   # right
            rows[3] + rows[1],   # bottom
            rows[3] - rows[1],   # top
            rows[2],             # z >= 0
            rows[3] - rows[2],   # z <= w
        ]
    )

    lengths = np.linalg.norm(candidates[:, :3], axis=1)

    keep = lengths > 1e-12

    return candidates[keep] / lengths[keep, None]


def spheres_in_frustum(
    planes: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray
) -> np.ndarray:
    """(N,) bool: sphere intersects or is inside the frustum."""

    if len(centers) == 0:
        return np.zeros(0, dtype=bool)

    distances = centers @ planes[:, :3].T + planes[:, 3]

    return np.all(
        distances >= -radii[:, None],
        axis=1
    )


# =========================================================
# Commands
# =========================================================

def build_commands(
    prepared: PreparedDraws,
    indices: np.ndarray
) -> np.ndarray:
    """
    (k, 5) uint32 DrawElementsIndirectCommands for the
    given item indices. baseInstance = item index, which the
    shaders use to find the item's record.
    """

    indices = np.asarray(indices, dtype=np.int64)

    commands = np.empty((len(indices), COMMAND_UINTS), dtype=np.uint32)

    commands[:, 0] = prepared.index_count[indices]
    commands[:, 1] = 1
    commands[:, 2] = prepared.first_index[indices]

    # baseVertex is a signed int in the command; ours are
    # never negative, so the uint32 view is identical.
    commands[:, 3] = prepared.base_vertex[indices].astype(np.uint32)
    commands[:, 4] = indices.astype(np.uint32)

    return commands


def group_by_material(
    prepared: PreparedDraws,
    indices: np.ndarray
) -> list[tuple[object, np.ndarray]]:
    """
    Split item indices into (material, indices) groups, in
    first-seen order, so each group is one material setup
    plus one multi-draw.
    """

    groups: dict[int, tuple[object, list[int]]] = {}

    for index in np.asarray(indices, dtype=np.int64):

        material = prepared.items[int(index)].material

        key = id(material)

        if key not in groups:
            groups[key] = (material, [])

        groups[key][1].append(int(index))

    return [
        (material, np.array(members, dtype=np.int64))
        for material, members in groups.values()
    ]
