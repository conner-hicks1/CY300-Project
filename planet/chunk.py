from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from core.disk_cache import cache_key, cached

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


def build_cached_chunk(
    key: ChunkKey,
    terrain: Terrain,
    resolution: int
) -> ChunkData:
    """build_chunk through the disk cache (core/disk_cache.py)."""

    terrain_key = terrain.cache_key

    return cached(
        "chunks",
        None if terrain_key is None else cache_key("chunk", terrain_key, key, resolution),
        lambda: build_chunk(key, terrain, resolution)
    )


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

    # Grid with a one-sample border: indices -1 .. n, counted
    # across the whole face (exact in float, and the same
    # numbers in neighboring chunks: shared edges come out
    # bit for bit alike).
    steps = np.arange(-1, n + 1, dtype=np.float64)

    unit = 2.0 / (key.cells * (n - 1))

    grid_a, grid_b = np.meshgrid(
        -1.0 + (key.x * (n - 1) + steps) * unit,
        -1.0 + (key.y * (n - 1) + steps) * unit,
        indexing="xy"
    )

    # Row = b (v), column = a (u).
    directions = face_directions(key.face, grid_a, grid_b).reshape(-1, 3)

    spacing = edge_length(radius, key.depth) / (n - 1)

    elevation, water = terrain.elevation_and_water(directions, spacing)

    # Oceans: the visible surface is the liquid at sea level
    # (dry worlds keep their basins).
    surface = (
        np.maximum(elevation, 0.0)
        if terrain.settings.has_liquid
        else elevation
    )

    # Relief stands on the body's shape (a flattened or
    # irregular body's base height; 0 on a sphere).
    base = terrain.base_height(directions)

    positions = directions * (radius + base + surface)[:, None]

    size = n + 2

    positions = positions.reshape(size, size, 3)

    # -----------------------------------------------------
    # Normals (central differences)
    # -----------------------------------------------------

    along_u = positions[1:-1, 2:] - positions[1:-1, :-2]
    along_v = positions[2:, 1:-1] - positions[:-2, 1:-1]

    normals = np.cross(along_u, along_v).reshape(-1, 3)

    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    # "Up" for slopes: away from the center on a sphere; on
    # a shaped body, the normal of the shape itself (a flat
    # plain on the flank of a potato is not a cliff).
    if terrain.shape is not None:

        ground = (directions * (radius + base)[:, None]).reshape(size, size, 3)

        up = np.cross(
            ground[1:-1, 2:] - ground[1:-1, :-2],
            ground[2:, 1:-1] - ground[:-2, 1:-1]
        ).reshape(-1, 3)

        up /= np.linalg.norm(up, axis=1, keepdims=True)

    else:

        up = None

    # -----------------------------------------------------
    # Interior samples
    # -----------------------------------------------------

    def interior(values, width):

        return values.reshape(size, size, width)[1:-1, 1:-1].reshape(-1, width)

    # Edges as a coarser neighbor draws them: every other
    # edge vertex at the height midway between its two
    # neighbors. A chunk next to one a level coarser then
    # meets it without a crack (no T-junction slivers, which
    # show as dashed lines up close); same-level neighbors
    # snap alike. The normals (above) stay the true surface's.
    snapped = _snap_edges(surface.reshape(size, size)[1:-1, 1:-1], n).reshape(-1)

    positions = interior(directions, 3) * (radius + interior(base[:, None], 1)[:, 0] + snapped)[:, None]

    directions = interior(directions, 3)
    elevation = interior(elevation[:, None], 1)[:, 0]

    # (Where the surface is the ground, its height too.)
    if not terrain.settings.has_liquid:
        elevation = snapped
    else:
        elevation = np.where(elevation > 0.0, snapped, elevation)
    water = interior(water[:, None], 1)[:, 0]

    slope = np.einsum("ij,ij->i", normals, directions if up is None else up)

    # Terrain inputs for the per-pixel biome shading
    # (assets/shaders/include/terrain.glsl): color = (elevation,
    # slope, precipitation), uv.x = temperature,
    # uv.y = quadtree depth + 0.99 * surface water (the
    # depth is the same across a chunk, so the shader splits
    # them with floor / fract after interpolation).
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
    uv = np.stack((temperature, key.depth + 0.99 * np.clip(water, 0.0, 1.0)), axis=1)

    # Tectonic data for the plate / crust views rides in
    # the tangent slot (terrain shading uses no normal map).
    tectonic = terrain.tectonic_data(directions)

    # Skirts are marked by a slope of 2 (no real slope is
    # above 1): seen only through pinholes along the seams,
    # they shade as the flat ground beside them
    # (assets/shaders/lit.frag.glsl), not as dark cliffs.
    skirt_colors = colors[edge].copy()
    skirt_colors[:, 1] = 2.0

    mesh = MeshData.from_attributes(
        positions=all_positions,
        indices=_indices(n),
        normals=np.vstack((normals, normals[edge])),
        uvs=np.vstack((uv, uv[edge])),
        colors=np.vstack((colors, skirt_colors)),
        tangents=np.vstack((tectonic, tectonic[edge]))
    )

    return ChunkData(
        key=key,
        mesh=mesh,
        center=center,
        min_elevation=float(elevation.min()),
        max_elevation=float(elevation.max())
    )


def _snap_edges(
    heights: np.ndarray,
    n: int
) -> np.ndarray:
    """An (n, n) height grid with odd edge samples at their neighbors' mean."""

    grid = np.array(heights, dtype=np.float64)

    if (n - 1) % 2 != 0:
        return grid

    odd = np.arange(1, n - 1, 2)

    for row in (0, n - 1):
        grid[row, odd] = 0.5 * (grid[row, odd - 1] + grid[row, odd + 1])

    for column in (0, n - 1):
        grid[odd, column] = 0.5 * (grid[odd - 1, column] + grid[odd + 1, column])

    return grid


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


def _grid_indices(
    n: int
) -> np.ndarray:
    """
    The grid's triangles, counter-clockwise from outside.

    Along the four edges only every other vertex is used
    (the ones a chunk a level coarser also has): the outer
    ring of quads is fanned from those to the row inside.
    Neighbors then share their edges vertex for vertex,
    whichever level they are, so the surface is watertight
    (triangles meeting mid-edge, T-junctions, leave pixel
    gaps however exactly the vertex lies on the edge).
    """

    if n < 5 or (n - 1) % 2 != 0:

        rows, cols = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")

        i0 = (rows * n + cols).ravel()

        return np.stack((i0, i0 + 1, i0 + n + 1, i0, i0 + n + 1, i0 + n), axis=1).ravel()

    triangles = []

    # The inside: plain quads between rows / columns 1 .. n-2.
    for row in range(1, n - 2):
        for col in range(1, n - 2):

            i0 = row * n + col

            triangles += [(i0, i0 + 1, i0 + n + 1), (i0, i0 + n + 1, i0 + n)]

    def vertex(x, y):
        return y * n + x

    inner = (1, n - 2)

    def clip(value):
        return min(max(value, inner[0]), inner[1])

    # Each edge, as (edge vertex at t, inner vertex at t).
    edges = (
        (lambda t: (t, 0), lambda t: (clip(t), 1)),             # bottom
        (lambda t: (n - 1, t), lambda t: (n - 2, clip(t))),     # right
        (lambda t: (t, n - 1), lambda t: (clip(t), n - 2)),     # top
        (lambda t: (0, t), lambda t: (1, clip(t))),             # left
    )

    for on_edge, inside in edges:

        for t0 in range(0, n - 1, 2):

            t1, t2 = t0 + 1, t0 + 2

            e0, e2 = on_edge(t0), on_edge(t2)

            # The inner vertices this pair of edge vertices
            # faces (clipped at the corners).
            fan = []

            for t in (t0, t1, t2):

                v = inside(t)

                if v not in fan:
                    fan.append(v)

            # A triangle from the edge pair to the middle of
            # the fan, then the rest of the fan to its ends.
            middle = fan[len(fan) // 2]

            triangles.append((e0, e2, middle))

            # (Winding is fixed below.)
            for a, b in zip(fan[:-1], fan[1:]):

                end = e0 if fan.index(b) <= len(fan) // 2 else e2

                triangles.append((end, b, a))

    # Counter-clockwise in grid coordinates (x = column, y =
    # row), as the grid quads are; degenerate ones dropped.
    result = []

    for triangle in triangles:

        points = [(index % n, index // n) if isinstance(index, int) else index for index in triangle]

        (x0, y0), (x1, y1), (x2, y2) = points

        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)

        if area == 0:
            continue

        indices = [vertex(x, y) for x, y in points]

        if area < 0:
            indices = [indices[0], indices[2], indices[1]]

        result.append(indices)

    return np.array(result, dtype=np.int64).ravel()


@lru_cache(maxsize=8)
def _indices(
    n: int
) -> np.ndarray:

    grid = _grid_indices(n)

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
