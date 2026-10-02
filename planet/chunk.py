from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from graphics.mesh_data import MeshData

from planet.cube_sphere import (
    ChunkKey,
    edge_length,
    face_directions
)
from planet.terrain import Terrain


# =========================================================
# Chunk Data
# =========================================================
#
# CPU result of building one quadtree node, produced on a
# worker thread. The GPU upload happens later on the main
# thread (PlanetSystem).
#
# Vertices are stored relative to `center` (float64,
# planet space), so float32 positions stay precise however
# far the chunk is from the planet's center.

@dataclass(slots=True)
class ChunkData:

    key: ChunkKey
    mesh: MeshData

    # Planet-space origin of the vertex positions.
    center: np.ndarray

    min_elevation: float
    max_elevation: float


def build_chunk(
    key: ChunkKey,
    terrain: Terrain,
    resolution: int
) -> ChunkData:
    """
    Build the mesh for `key`: a resolution x resolution
    vertex grid displaced by the terrain, plus skirts.

    Normals come from the height field itself, sampled one
    vertex beyond each edge, so adjacent chunks at the same
    depth have identical edge normals (no lighting seams).

    Skirts: a strip hanging down from every edge. Where a
    chunk meets a coarser neighbour their edges do not
    match exactly; the skirt fills the crack.
    """

    radius = terrain.settings.radius

    n = resolution

    a0, a1, b0, b1 = key.bounds()

    step_a = (a1 - a0) / (n - 1)
    step_b = (b1 - b0) / (n - 1)

    # Grid with a one-sample border: indices -1 .. n.
    steps = np.arange(-1, n + 1, dtype=np.float64)

    grid_a, grid_b = np.meshgrid(
        a0 + steps * step_a,
        b0 + steps * step_b,
        indexing="xy"
    )

    # Row = b (v), column = a (u).
    directions = face_directions(key.face, grid_a, grid_b).reshape(-1, 3)

    spacing = edge_length(radius, key.depth) / (n - 1)

    elevation = terrain.elevation(directions, spacing)

    # Oceans: the visible surface is the water at sea level.
    surface = np.maximum(elevation, 0.0)

    positions = directions * (radius + surface)[:, None]

    size = n + 2

    positions = positions.reshape(size, size, 3)

    # -----------------------------------------------------
    # Normals (central differences)
    # -----------------------------------------------------

    along_u = positions[1:-1, 2:] - positions[1:-1, :-2]
    along_v = positions[2:, 1:-1] - positions[:-2, 1:-1]

    normals = np.cross(along_u, along_v).reshape(-1, 3)

    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    # -----------------------------------------------------
    # Interior samples
    # -----------------------------------------------------

    def interior(values, width):

        return values.reshape(size, size, width)[1:-1, 1:-1].reshape(-1, width)

    positions = positions[1:-1, 1:-1].reshape(-1, 3)
    directions = interior(directions, 3)
    elevation = interior(elevation[:, None], 1)[:, 0]

    slope = np.einsum("ij,ij->i", normals, directions)

    # Terrain inputs for the per-pixel biome shading
    # (assets/shaders/include/terrain.glsl): color = (elevation,
    # slope, precipitation), uv.x = temperature,
    # uv.y = depth / 20.
    temperature, precipitation = terrain.surface_climate(directions, elevation)

    colors = np.stack(
        (elevation, slope, precipitation),
        axis=1
    )

    # -----------------------------------------------------
    # Skirts
    # -----------------------------------------------------

    edge = _edge_vertices(n)

    skirt_depth = 3.0 * spacing + 10.0

    skirt_positions = (
        positions[edge]
        - directions[edge] * skirt_depth
    )

    center = face_directions(
        key.face,
        (a0 + a1) * 0.5,
        (b0 + b1) * 0.5
    ) * radius

    all_positions = np.vstack((positions, skirt_positions)) - center

    # uv.y: quadtree depth, for the "Detail level" view.
    uv = np.stack((temperature, np.full(n * n, key.depth / 20.0)), axis=1)

    # Tectonic data for the plate / crust views rides in
    # the tangent slot (terrain shading uses no normal map).
    tectonic = terrain.tectonic_data(directions)

    mesh = MeshData.from_attributes(
        positions=all_positions,
        indices=_indices(n),
        normals=np.vstack((normals, normals[edge])),
        uvs=np.vstack((uv, uv[edge])),
        colors=np.vstack((colors, colors[edge])),
        tangents=np.vstack((tectonic, tectonic[edge]))
    )

    return ChunkData(
        key=key,
        mesh=mesh,
        center=center,
        min_elevation=float(elevation.min()),
        max_elevation=float(elevation.max())
    )


# =========================================================
# Topology (shared by every chunk of one resolution)
# =========================================================

@lru_cache(maxsize=8)
def _edge_vertices(
    n: int
) -> np.ndarray:
    """
    Grid indices around the border, in order: bottom row
    (left to right), right column (up), top row (right to
    left), left column (down). Closed loop, corners once.
    """

    bottom = [i for i in range(n)]
    right = [(j * n) + n - 1 for j in range(1, n)]
    top = [(n - 1) * n + i for i in range(n - 2, -1, -1)]
    left = [j * n for j in range(n - 2, 0, -1)]

    loop = np.array(bottom + right + top + left, dtype=np.int64)

    loop.setflags(write=False)

    return loop


@lru_cache(maxsize=8)
def _indices(
    n: int
) -> np.ndarray:

    # Grid quads, counter-clockwise from outside.

    rows, cols = np.meshgrid(
        np.arange(n - 1),
        np.arange(n - 1),
        indexing="ij"
    )

    i0 = (rows * n + cols).ravel()
    i1 = i0 + 1
    i2 = i0 + n + 1
    i3 = i0 + n

    grid = np.stack(
        (i0, i1, i2, i0, i2, i3),
        axis=1
    ).ravel()

    # Skirt quads between the edge loop and the skirt loop
    # (vertices n*n onward, same order). Both windings,
    # so skirts show whichever side faces the camera.

    edge = _edge_vertices(n)

    count = len(edge)

    top_a = edge
    top_b = np.roll(edge, -1)

    low_a = n * n + np.arange(count)
    low_b = np.roll(low_a, -1)

    skirt = np.stack(
        (
            top_a, low_a, low_b, top_a, low_b, top_b,
            top_a, low_b, low_a, top_a, top_b, low_b,
        ),
        axis=1
    ).ravel()

    indices = np.concatenate((grid, skirt)).astype(np.uint32)

    indices.setflags(write=False)

    return indices
