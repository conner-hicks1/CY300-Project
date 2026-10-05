import math
import zlib

from dataclasses import dataclass

import numpy as np

from planet.cube_sphere import ChunkKey, direction_to_face, edge_length, face_directions


# =========================================================
# Scattered Things: Rocks, Trees, Grass
# =========================================================
#
# Up close, ground is not a smooth surface: it has boulders
# and stones (more of them under cliffs, where the rock falls
# to), trees where the climate grows forest, grass tufts on
# meadows. They are placed per cell of the cube-sphere grid,
# each layer at its own cell size, around the camera only,
# from the same terrain and climate the ground is shaded by
# (assets/shaders/include/terrain.glsl), and drawn instanced
# (graphics/scatter.py).
#
# Placement is deterministic per cell (its own random seed),
# so a cell rebuilt later looks the same. Heights come from a
# small grid of terrain samples over the cell, interpolated:
# the ground under them is the drawn mesh's, which is no
# finer.

@dataclass(frozen=True, slots=True)
class ScatterLayer:

    name: str
    cell: float             # target cell edge (m)
    reach: float            # drawn within this distance (m)
    max_density: float      # candidates per m^2


LAYERS = (
    ScatterLayer("trees", 600.0, 900.0, 1.0 / 80.0),
    ScatterLayer("rocks", 150.0, 320.0, 1.0 / 12.0),
    ScatterLayer("grass", 32.0, 40.0, 1.2),
)

# Kinds (mesh variants, graphics/scatter.py MESHES).
CONIFER, BROADLEAF, RAINFOREST, ACACIA = 0, 1, 2, 3
ROCK_KINDS = (4, 5, 6, 7)
GRASS_KINDS = (8, 9)

TREE_KINDS = (CONIFER, BROADLEAF, RAINFOREST, ACACIA)

# Height range (m) of each tree kind.
TREE_HEIGHTS = {
    CONIFER: (10.0, 26.0),
    BROADLEAF: (9.0, 22.0),
    RAINFOREST: (20.0, 38.0),
    ACACIA: (5.0, 9.0),
}

# The shader's rock color on the biomes palette.
BIOME_ROCK = (0.16, 0.14, 0.12)

GRID = 17       # height samples per cell edge


@dataclass(frozen=True, slots=True)
class SurfaceSettings:
    """What the scatter needs of a planet's surface material."""

    life: bool
    has_liquid: bool
    mineral: bool                       # mineral palette (bare worlds)
    frost_point: float                  # C
    rock_color: tuple[float, float, float]
    seed: int = 1


@dataclass(slots=True)
class ScatterCell:

    layer: str
    key: ChunkKey

    positions: np.ndarray       # (n, 3) float64, planet frame (m)
    up: np.ndarray              # (n, 3) float32: which way the thing stands
    yaw: np.ndarray             # (n,) radians about `up`
    scale: np.ndarray           # (n,) m
    kind: np.ndarray            # (n,) int
    tint: np.ndarray            # (n, 3) float32

    def __len__(
        self
    ) -> int:

        return len(self.kind)


def cell_depth(
    radius: float,
    target: float
) -> int:
    """Quadtree depth whose cells are about `target` m across."""

    return max(0, int(round(math.log2(max(edge_length(radius, 0) / target, 1.0)))))


def cells_around(
    radius: float,
    direction: np.ndarray,
    reach: float,
    depth: int
) -> set[ChunkKey]:
    """Cells of `depth` within `reach` m of the surface point under `direction`."""

    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)

    edge = edge_length(radius, depth)

    reference = np.array([0.0, 1.0, 0.0]) if abs(direction[1]) < 0.9 else np.array([1.0, 0.0, 0.0])

    east = np.cross(reference, direction)
    east /= np.linalg.norm(east)

    north = np.cross(direction, east)

    extent = reach + edge

    steps = np.arange(-extent, extent + 1e-6, 0.4 * edge)

    u, v = np.meshgrid(steps, steps)

    inside = u * u + v * v <= extent * extent

    points = direction * radius + (u[inside][:, None] * east + v[inside][:, None] * north)

    points /= np.linalg.norm(points, axis=1, keepdims=True)

    cells = 1 << depth

    keys = set()

    for point in points:

        face, a, b = direction_to_face(point)

        x = min(max(int((a + 1.0) * 0.5 * cells), 0), cells - 1)
        y = min(max(int((b + 1.0) * 0.5 * cells), 0), cells - 1)

        keys.add(ChunkKey(face, depth, x, y))

    return keys


def _smoothstep(
    edge0: float,
    edge1: float,
    x
):

    t = np.clip((np.asarray(x, dtype=np.float64) - edge0) / (edge1 - edge0), 0.0, 1.0)

    return t * t * (3.0 - 2.0 * t)


def _seed(
    surface: SurfaceSettings,
    layer: ScatterLayer,
    key: ChunkKey
) -> int:

    return zlib.crc32(f"{surface.seed}/{layer.name}/{key.face}/{key.depth}/{key.x}/{key.y}".encode())


def _bilinear(
    grid: np.ndarray,
    s: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """Sample a (G, G, ...) grid at fractional (column s, row t)."""

    g = grid.shape[0]

    s = np.clip(s, 0.0, g - 1.000001)
    t = np.clip(t, 0.0, g - 1.000001)

    i = s.astype(np.int64)
    j = t.astype(np.int64)

    fs = s - i
    ft = t - j

    if grid.ndim == 3:
        fs = fs[:, None]
        ft = ft[:, None]

    return (
        (grid[j, i] * (1.0 - fs) + grid[j, i + 1] * fs) * (1.0 - ft)
        + (grid[j + 1, i] * (1.0 - fs) + grid[j + 1, i + 1] * fs) * ft
    )


def generate_cell(
    terrain,
    surface: SurfaceSettings,
    layer: ScatterLayer,
    key: ChunkKey
) -> ScatterCell:
    """
    terrain: planet.terrain.Terrain. The things standing in
    one cell of a layer.
    """

    radius = terrain.settings.radius

    a0, a1, b0, b1 = key.bounds()

    edge = edge_length(radius, key.depth)

    # -----------------------------------------------------
    # The ground over the cell: a small grid of samples
    # -----------------------------------------------------

    t = np.linspace(0.0, 1.0, GRID)

    ga, gb = np.meshgrid(a0 + t * (a1 - a0), b0 + t * (b1 - b0), indexing="xy")

    directions = face_directions(key.face, ga, gb).reshape(-1, 3)

    spacing = edge / (GRID - 1)

    elevation, water = terrain.elevation_and_water(directions, spacing)

    base = terrain.base_height(directions)

    ground = np.maximum(elevation, 0.0) if terrain.settings.has_liquid else elevation

    points = directions * (radius + base + ground)[:, None]

    temperature, precipitation = terrain.surface_climate(directions, elevation)

    grid_points = points.reshape(GRID, GRID, 3)

    # Normals and slope (1 = flat).
    du = np.gradient(grid_points, axis=1)
    dv = np.gradient(grid_points, axis=0)

    normals = np.cross(du, dv)
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)

    up_grid = directions.reshape(GRID, GRID, 3)

    slope = np.einsum("ijk,ijk->ij", normals, up_grid)

    # The steepest slope nearby (two samples around): scree
    # gathers below cliffs.
    steepest = 1.0 - slope

    for _ in range(2):

        padded = np.pad(steepest, 1, mode="edge")

        steepest = np.maximum.reduce([
            padded[1:-1, 1:-1], padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]
        ])

    fields = np.stack(
        (
            elevation.reshape(GRID, GRID),
            water.reshape(GRID, GRID),
            temperature.reshape(GRID, GRID),
            precipitation.reshape(GRID, GRID),
            slope,
            steepest,
        ),
        axis=2
    )

    # -----------------------------------------------------
    # Candidates: stratified random points
    # -----------------------------------------------------

    rng = np.random.default_rng(_seed(surface, layer, key))

    count = int(min(layer.max_density * edge * edge, 200_000))

    side = max(int(math.sqrt(count)), 1)

    count = side * side

    ii, jj = np.meshgrid(np.arange(side), np.arange(side), indexing="xy")

    u = (ii.ravel() + rng.random(count)) / side
    v = (jj.ravel() + rng.random(count)) / side

    s = u * (GRID - 1)
    r = v * (GRID - 1)

    sampled = _bilinear(fields, s, r)

    elevation_c, water_c, temperature_c, precipitation_c, slope_c, steepest_c = sampled.T

    normal_c = _bilinear(normals, s, r)
    normal_c /= np.linalg.norm(normal_c, axis=1, keepdims=True)

    point_c = _bilinear(grid_points, s, r)

    up_c = point_c / np.linalg.norm(point_c, axis=1, keepdims=True)

    # (The interpolated point lies on the chord between
    # samples: lift it to the ground's radius there.)
    radius_c = _bilinear(np.linalg.norm(grid_points, axis=2), s, r)

    point_c = up_c * radius_c[:, None]

    # -----------------------------------------------------
    # The ground's material (as the shader shades it)
    # -----------------------------------------------------

    wetness = precipitation_c / (300.0 + 30.0 * np.maximum(temperature_c, 0.0))

    snow = _smoothstep(surface.frost_point - 0.1, surface.frost_point - 5.1, temperature_c) * _smoothstep(0.7, 0.82, slope_c)

    wet = water_c > 0.3

    if surface.has_liquid:
        wet |= elevation_c < 3.0

    cliff = 1.0 - _smoothstep(0.75, 0.9, slope_c)

    alive = surface.life and not surface.mineral

    plants = _smoothstep(0.15, 0.45, wetness) * _smoothstep(-4.0, -1.0, temperature_c) if alive else np.zeros(count)

    forest = (
        _smoothstep(0.9, 1.6, wetness) * _smoothstep(0.9, 0.96, slope_c) * _smoothstep(-4.0, -1.0, temperature_c)
        if alive else np.zeros(count)
    )

    roll = rng.random(count)

    keep = np.zeros(count, dtype=bool)

    kind = np.zeros(count, dtype=np.int64)
    scale = np.ones(count)
    tint = np.ones((count, 3))

    up = up_c.copy()

    if layer.name == "trees":

        savanna = (
            _smoothstep(0.35, 0.8, wetness) * (1.0 - forest) * 0.1 * _smoothstep(10.0, 18.0, temperature_c)
            if alive else np.zeros(count)
        )

        density = (forest + savanna) * (1.0 - snow) * (slope_c > 0.8) / 80.0

        keep = (roll < density / layer.max_density) & ~wet

        pick = rng.random(count)

        kind = np.where(
            temperature_c < 4.0,
            CONIFER,
            np.where(
                (temperature_c < 12.0) & (pick < 0.4),
                CONIFER,
                np.where(
                    (temperature_c > 20.0) & (wetness > 1.5),
                    RAINFOREST,
                    np.where((forest < 0.3) & (temperature_c > 15.0), ACACIA, BROADLEAF)
                )
            )
        )

        low = np.vectorize(lambda k: TREE_HEIGHTS[int(k)][0])(kind) if count else np.zeros(0)
        high = np.vectorize(lambda k: TREE_HEIGHTS[int(k)][1])(kind) if count else np.zeros(0)

        scale = low + (high - low) * rng.random(count) ** 1.5

        # Leaves a little different tree to tree; drier,
        # yellower.
        shade = 0.75 + 0.5 * rng.random(count)

        tint = np.stack(
            (
                shade * (1.0 + 0.25 * (1.0 - _smoothstep(0.6, 1.4, wetness))),
                shade,
                shade * (0.85 + 0.15 * rng.random(count)),
            ),
            axis=1
        )

    elif layer.name == "rocks":

        bare = 1.0 - plants

        # Scree: moderate slopes below steep ground nearby.
        scree = _smoothstep(0.12, 0.35, steepest_c) * _smoothstep(0.6, 0.85, slope_c)

        density = (
            bare * (0.06 if surface.mineral else 0.15)
            + cliff * 0.3 * (slope_c > 0.55)
            + scree * 1.0
            + (1.0 - bare) * 0.01
        ) * (1.0 - 0.8 * snow) / 12.0

        keep = (roll < density / layer.max_density) & ~wet

        kind = rng.choice(np.array(ROCK_KINDS), size=count)

        # Many small, few large (a power law); bigger in scree.
        size = 0.25 * (1.0 - rng.random(count)) ** (-1.0 / 2.2)

        scale = np.clip(size * (1.0 + 1.5 * scree), 0.2, 6.0)

        # Rocks settle into the slope.
        up = normal_c * 0.6 + up_c * 0.4
        up /= np.linalg.norm(up, axis=1, keepdims=True)

        tint = np.asarray(surface.rock_color)[None, :] * (0.7 + 0.6 * rng.random(count))[:, None]

    elif layer.name == "grass":

        grass = (
            _smoothstep(0.3, 0.6, wetness)
            * _smoothstep(-1.0, 4.0, temperature_c)
            * (1.0 - 0.6 * forest)
            * (1.0 - cliff)
            * (1.0 - snow)
            if alive else np.zeros(count)
        )

        keep = (roll < grass * (layer.max_density / 1.2)) & ~wet

        kind = rng.choice(np.array(GRASS_KINDS), size=count)

        scale = 0.25 + 0.45 * rng.random(count)

        # Green where wet, straw where dry.
        dry = 1.0 - _smoothstep(0.5, 1.1, wetness)

        green = np.array([0.07, 0.16, 0.04])
        straw = np.array([0.28, 0.24, 0.10])

        tint = (green[None, :] * (1.0 - dry[:, None]) + straw[None, :] * dry[:, None]) * (0.7 + 0.6 * rng.random(count))[:, None]

        up = normal_c * 0.3 + up_c * 0.7
        up /= np.linalg.norm(up, axis=1, keepdims=True)

    positions = point_c[keep]

    # Sunk in a little: the drawn ground between its
    # vertices can sit lower than these samples.
    sink = np.where(layer.name == "rocks", 0.3, 0.05) * scale[keep]

    if layer.name == "trees":
        sink = np.full(int(keep.sum()), 1.0)

    positions = positions - up_c[keep] * sink[:, None]

    return ScatterCell(
        layer=layer.name,
        key=key,
        positions=positions,
        up=up[keep].astype(np.float32),
        yaw=(rng.random(count) * 2.0 * math.pi)[keep],
        scale=scale[keep],
        kind=kind[keep].astype(np.int32),
        tint=tint[keep].astype(np.float32)
    )
