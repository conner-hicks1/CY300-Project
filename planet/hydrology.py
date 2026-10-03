import heapq
import math
import time

from dataclasses import dataclass

import numpy as np

from planet.noise import Perlin, fbm
from planet.sphere_grid import SphereGrid, sphere_grid


# =========================================================
# Rivers, Lakes, Deltas and Glaciers
# =========================================================
#
# Where rain (or Titan's methane rain) goes once it falls:
#
#   1. Flow routing on a global grid (~80 km cells at
#      Earth's size): a priority flood from the coasts
#      (Barnes et al. 2014) visits every land cell from the
#      sea inland, lowest first, so each cell drains to the
#      neighbor it was reached from and every depression
#      is filled up to its spill point: a lake.
#   2. Discharge: each cell's runoff (rainfall times a
#      runoff fraction, less where the climate is dry) is
#      added up downstream.
#   3. Rivers are the cells with the most water: polylines
#      from cell to cell down to the sea, as wide as their
#      discharge. Where the ground is frozen they are
#      glaciers instead.
#   4. Deltas: sediment fans at the mouths of the largest
#      rivers, built out into the sea with distributary
#      channels.
#
# The terrain then carves them in per sample (any level of
# detail): the ground falls toward each river's water level
# across a valley (V-shaped for rivers, U-shaped and wider
# for glaciers), with the channel itself inside; sample
# points are warped by noise first, so rivers meander. A
# water mask goes to the shader (water, methane or ice).
#
# Units: meters, m^3 / s.

_RUNOFF_FRACTION = 0.5
_SECONDS_PER_YEAR = 3.156e7


@dataclass(frozen=True, slots=True)
class HydrologySettings:

    seed: int = 1

    # Grid cells per cube-face edge.
    resolution: int = 128

    # Share of land cells that carry rivers.
    river_fraction: float = 0.035

    # Rivers freeze into glaciers below this annual mean
    # temperature (C); None = never (methane).
    glacier_temperature: float | None = 0.0


@dataclass(frozen=True, slots=True, eq=False)
class HydrologyField:

    grid: SphereGrid
    radius: float

    # Per river segment (cell -> downstream cell).
    segment_start: np.ndarray       # (k, 3) unit
    segment_end: np.ndarray         # (k, 3) unit
    segment_levels: np.ndarray      # (k, 2) water level at each end (m)
    segment_width: np.ndarray       # (k,) half width (m)
    segment_glacier: np.ndarray     # (k,) bool
    segment_discharge: np.ndarray   # (k,) m^3 / s

    # Cell -> its segment (-1: none).
    segment_of_cell: np.ndarray

    # Lakes (padded, for sampling): 1 in lake cells, and
    # the water level there and next to them (elsewhere far
    # below anything).
    lake: np.ndarray
    lake_level: np.ndarray

    # Deltas.
    delta_center: np.ndarray        # (d, 3) unit
    delta_axis: np.ndarray          # (d, 3) unit tangent, seaward
    delta_radius: np.ndarray        # (d,) m

    # Summary.
    river_count: int = 0
    glacier_count: int = 0
    lake_fraction: float = 0.0
    largest_discharge: float = 0.0
    compute_seconds: float = 0.0

    # -----------------------------------------------------
    # Carving (per sample)
    # -----------------------------------------------------

    def near_water(
        self,
        directions: np.ndarray
    ) -> np.ndarray:
        """
        Points a river or lake might reach (within two
        routing cells): the rest need no meander noise.
        """

        grid = self.grid

        cell = grid.cell_of(directions)
        near = grid.neighbors[cell]

        has_river = self.segment_of_cell >= 0

        reach = (
            has_river[cell]
            | has_river[near].any(axis=1)
            | has_river[grid.neighbors[near].reshape(len(cell), -1)].any(axis=1)
        )

        return reach | (grid.sample(self.lake, directions) > 0.0)

    def apply(
        self,
        directions: np.ndarray,
        elevation: np.ndarray,
        spacing: float = 0.0,
        meander=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        (elevation with valleys, channels, lakes and deltas;
        water 0..1) at unit directions.
        meander: (n, 3) tangent offsets (unit-sphere units)
            that make rivers wander, or None.
        """

        elevation = np.array(elevation, dtype=np.float64)
        water = np.zeros(len(directions))

        if len(directions) == 0:
            return elevation, water

        self._rivers(directions, elevation, water, spacing, meander)
        self._lakes(directions if meander is None else directions + meander, elevation, water)
        self._deltas(directions, elevation, water, spacing)

        return elevation, water

    def _rivers(
        self,
        directions: np.ndarray,
        elevation: np.ndarray,
        water: np.ndarray,
        spacing: float,
        meander
    ):

        if len(self.segment_width) == 0:
            return

        grid = self.grid
        n = len(directions)

        points = directions if meander is None else directions + meander
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        # Segments that can reach a point start in its cell,
        # a neighbor or a neighbor's neighbor.
        cell = grid.cell_of(points)
        near = grid.neighbors[cell]
        far = grid.neighbors[near].reshape(n, -1)

        candidates = np.concatenate([cell[:, None], near, far], axis=1)

        segment = self.segment_of_cell[candidates]

        point, slot = np.nonzero(segment >= 0)

        if len(point) == 0:
            return

        segment = segment[point, slot]

        a = self.segment_start[segment]
        b = self.segment_end[segment]
        p = points[point]

        ab = b - a

        t = np.clip(np.sum((p - a) * ab, axis=1) / np.maximum(np.sum(ab * ab, axis=1), 1e-18), 0.0, 1.0)

        distance = np.linalg.norm(p - (a + t[:, None] * ab), axis=1) * self.radius

        levels = self.segment_levels[segment]

        level = levels[:, 0] + t * (levels[:, 1] - levels[:, 0])

        width = self.segment_width[segment]
        glacier = self.segment_glacier[segment]

        cell_size = grid.cell_angle * self.radius

        # Valleys: V-shaped for rivers, wide and U-shaped
        # (flat floored) for glaciers.
        valley = np.where(
            glacier,
            np.clip(60.0 * width, 6_000.0, 0.45 * cell_size),
            np.clip(30.0 * width, 3_000.0, 0.45 * cell_size)
        )

        reach = distance < valley

        if not np.any(reach):
            return

        point, distance, level, width, glacier, valley = (
            point[reach], distance[reach], level[reach], width[reach], glacier[reach], valley[reach]
        )

        ground = elevation[point]

        factor = np.where(
            glacier,
            _smoothstep(0.35 * valley, valley, distance),
            _smoothstep(0.0, valley, distance) ** 0.8
        )

        carved = np.where(ground > level, level + (ground - level) * factor, ground)

        # The channel itself (glaciers fill theirs with ice).
        depth = np.clip(3.0 + 10.0 * np.sqrt(width / 250.0), 3.0, 40.0)

        inside = (distance < width) & ~glacier

        carved = np.where(
            inside,
            np.minimum(carved, level - depth * (1.0 - (distance / np.maximum(width, 1.0)) ** 2)),
            carved
        )

        carved = np.maximum(carved, ground - 1_500.0)

        np.minimum.at(elevation, point, carved)

        # Water (or ice) mask: at least ~a sample wide, so
        # big rivers stay visible from far away.
        shown = np.where(glacier, 3.0 * width, width)
        shown = np.maximum(shown, np.where(width > 300.0, 1.8 * spacing, 0.0))

        # Soft edges: the shader sharpens the interpolated
        # mask (straight-edged contours instead of a
        # triangle-by-triangle zigzag).
        mask = 1.0 - _smoothstep(0.5 * shown, shown, distance)

        np.maximum.at(water, point, mask)

    def _lakes(
        self,
        directions: np.ndarray,
        elevation: np.ndarray,
        water: np.ndarray
    ):

        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

        # Shores from the (warped, so irregular) lake cells;
        # the ground inside flattens to the water level.
        inside = _smoothstep(0.35, 0.55, self.grid.sample(self.lake, directions))

        lake = np.nonzero(inside > 0.0)[0]

        if len(lake) == 0:
            return

        level = self.grid.sample(self.lake_level, directions[lake])

        share = inside[lake] * (level > -1e5)

        elevation[lake] = elevation[lake] + (level - elevation[lake]) * share

        water[lake] = np.maximum(water[lake], share)

    def _deltas(
        self,
        directions: np.ndarray,
        elevation: np.ndarray,
        water: np.ndarray,
        spacing: float
    ):

        for center, axis, radius in zip(self.delta_center, self.delta_axis, self.delta_radius):

            offset = (directions - center) * self.radius

            x = np.linalg.norm(offset, axis=1) / radius

            near = x < 1.0

            if not np.any(near):
                continue

            offset = offset[near]
            x = x[near]

            # Seaward coordinate (-1 inland .. 1 out at sea)
            # and across.
            along = offset @ axis / radius
            side = np.cross(center, axis)
            across = offset @ side / radius

            lobe = _smoothstep(1.0, 0.6, x) * _smoothstep(-0.7, -0.2, along)

            # A fan surface: a few meters above the sea at its
            # apex, gently below it at its front.
            fan = 3.0 - 9.0 * np.clip(along + 0.3, 0.0, 1.3)

            ground = elevation[near]

            elevation[near] = ground + (fan - ground) * lobe * 0.9

            # Distributary channels splaying out from the apex.
            angle = np.arctan2(across, along + 0.7)

            channels = (0.5 + 0.5 * np.cos(9.0 * angle + 1.3 * fbm(_DELTA_NOISE, directions[near] * 900.0, 3))) ** 24

            channels *= lobe * (fan > -1.0)

            width = max(1.0, 0.5 * spacing / radius)

            water[near] = np.maximum(water[near], np.clip(channels * (1.0 + width), 0.0, 1.0))


_DELTA_NOISE = Perlin(4_242)


# =========================================================
# Computation
# =========================================================

def compute_hydrology(
    terrain,
    climate,
    settings: HydrologySettings
) -> HydrologyField | None:
    """
    Rivers, lakes and deltas for a terrain (planet/terrain.py,
    without hydrology) under a climate (planet/climate.py
    ClimateField). None without seas to drain into.
    """

    started = time.perf_counter()

    grid = sphere_grid(settings.resolution)

    radius = terrain.settings.radius

    directions = grid.directions

    cell_size = grid.cell_angle * radius

    elevation = terrain.elevation(directions, cell_size)

    ocean = elevation < 0.0

    land = ~ocean

    if not np.any(ocean) or not np.any(land):
        return None

    neighbors = grid.neighbors

    # -----------------------------------------------------
    # Priority flood from the coasts
    # -----------------------------------------------------

    filled = elevation.copy()
    receiver = np.full(grid.cell_count, -1, dtype=np.int64)
    closed = ocean.copy()

    coast = ocean & np.any(land[neighbors], axis=1)

    heap = [(float(elevation[c]), int(c)) for c in np.nonzero(coast)[0]]

    heapq.heapify(heap)

    order = []

    neighbor_lists = neighbors.tolist()
    elevation_list = elevation.tolist()
    filled_list = filled.tolist()
    closed_list = closed.tolist()
    receiver_list = receiver.tolist()

    while heap:

        level, cell = heapq.heappop(heap)

        order.append(cell)

        for neighbor in neighbor_lists[cell]:

            if closed_list[neighbor]:
                continue

            closed_list[neighbor] = True

            # Strictly downhill toward the sea (lakes fill to
            # just above their spill point).
            value = max(elevation_list[neighbor], level + 0.01)

            filled_list[neighbor] = value
            receiver_list[neighbor] = cell

            heapq.heappush(heap, (value, neighbor))

    filled = np.asarray(filled_list)
    receiver = np.asarray(receiver_list, dtype=np.int64)
    order = np.asarray(order, dtype=np.int64)

    # -----------------------------------------------------
    # Discharge
    # -----------------------------------------------------

    temperature, precipitation = climate.surface(directions, np.maximum(elevation, 0.0))

    mean_rain = float(np.mean(precipitation[land])) if np.any(land) else 0.0

    if mean_rain <= 0.0:
        return None

    # Dry climates lose most rain to evaporation (relative
    # to the planet's own rainfall: Titan's is tiny).
    fraction = _RUNOFF_FRACTION * _smoothstep(0.15 * mean_rain, 0.8 * mean_rain, precipitation)

    cell_area = 4.0 * math.pi * radius ** 2 / grid.cell_count

    discharge = np.where(land, fraction * precipitation * 1e-3 * cell_area / _SECONDS_PER_YEAR, 0.0)

    # Downstream accumulation: the flood visited each cell
    # after its receiver, so reverse order is upstream
    # first.
    discharge_list = discharge.tolist()

    for cell in order[::-1].tolist():

        down = receiver_list[cell]

        if down >= 0:
            discharge_list[down] += discharge_list[cell]

    discharge = np.asarray(discharge_list)

    # -----------------------------------------------------
    # Rivers
    # -----------------------------------------------------

    drained = land & (receiver >= 0)

    threshold = float(np.percentile(discharge[drained], 100.0 * (1.0 - settings.river_fraction)))

    threshold = max(threshold, 1e-6)

    river = drained & (discharge >= threshold)

    cells = np.nonzero(river)[0]
    down = receiver[cells]

    # Water level along each segment; at the sea, sea level.
    start_level = filled[cells]
    end_level = np.where(ocean[down], 0.0, filled[down])

    width = np.clip(250.0 * np.sqrt(discharge[cells] / threshold), 30.0, 3_000.0)

    glacier = (
        temperature[cells] < settings.glacier_temperature
        if settings.glacier_temperature is not None
        else np.zeros(len(cells), dtype=bool)
    )

    segment_of_cell = np.full(grid.cell_count, -1, dtype=np.int64)
    segment_of_cell[cells] = np.arange(len(cells))

    # Smooth courses: the routing grid only steps between
    # its 4 neighbors (a staircase), so each river node is
    # pulled toward its main tributary and its downstream
    # node a few times (corner cutting). Mouths stay put.
    upstream = np.full(grid.cell_count, -1, dtype=np.int64)

    by_flow = cells[np.argsort(discharge[cells])]

    # Ascending discharge: the largest tributary is written
    # last and wins.
    upstream[receiver[by_flow]] = by_flow

    nodes = directions.copy()

    inner = cells[~ocean[receiver[cells]]]

    for _ in range(3):

        up = upstream[inner]
        before = np.where((up >= 0)[:, None], nodes[np.maximum(up, 0)], nodes[inner])

        moved = 0.25 * before + 0.5 * nodes[inner] + 0.25 * nodes[receiver[inner]]

        nodes[inner] = moved / np.linalg.norm(moved, axis=1, keepdims=True)

    # -----------------------------------------------------
    # Lakes: depressions filled in humid climates
    # -----------------------------------------------------

    # Deep enough to hold water, with rivers feeding them.
    lake = (
        land
        & (filled - elevation > 50.0)
        & (precipitation > 0.5 * mean_rain)
        & (discharge >= 0.3 * threshold)
    )

    # Levels in the lake and its neighbors (so shores fall
    # where the ground rises above the water).
    lake_level = np.where(lake, filled, -1e6)

    neighbor_level = np.where(lake[neighbors], filled[neighbors], -1e6).max(axis=1)

    lake_level = np.where(lake, lake_level, neighbor_level)

    # -----------------------------------------------------
    # Deltas at the largest mouths
    # -----------------------------------------------------

    mouths = cells[ocean[down] & (discharge[cells] >= np.percentile(discharge[cells], 90.0))]

    centers, axes, radii = [], [], []

    for cell in mouths:

        sea = receiver[cell]

        center = directions[cell] + directions[sea]
        center /= np.linalg.norm(center)

        axis = directions[sea] - directions[cell]
        axis -= center * (axis @ center)
        axis /= max(np.linalg.norm(axis), 1e-12)

        centers.append(center)
        axes.append(axis)
        radii.append(float(np.clip(
            0.35 * cell_size * (discharge[cell] / threshold) ** 0.25,
            0.25 * cell_size,
            0.7 * cell_size
        )))

    return HydrologyField(
        grid=grid,
        radius=radius,
        segment_start=nodes[cells],
        segment_end=nodes[down],
        segment_levels=np.stack([start_level, end_level], axis=1),
        segment_width=width,
        segment_glacier=glacier,
        segment_discharge=discharge[cells],
        segment_of_cell=segment_of_cell,
        lake=grid.pad(lake.astype(np.float32)),
        lake_level=grid.pad(lake_level.astype(np.float32)),
        delta_center=np.asarray(centers).reshape(-1, 3),
        delta_axis=np.asarray(axes).reshape(-1, 3),
        delta_radius=np.asarray(radii),
        river_count=int(len(cells)),
        glacier_count=int(np.count_nonzero(glacier)),
        lake_fraction=float(np.count_nonzero(lake) / max(np.count_nonzero(land), 1)),
        largest_discharge=float(discharge[cells].max()) if len(cells) else 0.0,
        compute_seconds=time.perf_counter() - started
    )


def meander_offsets(
    directions: np.ndarray,
    cell_angle: float,
    seed: int = 1
) -> np.ndarray:
    """
    Tangent offsets (unit-sphere units) that bend river
    courses: smooth noise, typically about a third of a
    routing cell.
    """

    frequency = 1.0 / max(cell_angle * 0.9, 1e-6)

    offsets = np.stack(
        [
            fbm(Perlin(seed * 53 + axis), directions * frequency, 4)
            for axis in range(3)
        ],
        axis=1
    ) * (1.6 * cell_angle)

    # Keep them tangent.
    return offsets - directions * np.sum(offsets * directions, axis=1, keepdims=True)


def _smoothstep(
    edge0,
    edge1,
    x
) -> np.ndarray:

    edge0 = np.asarray(edge0, dtype=np.float64)
    edge1 = np.asarray(edge1, dtype=np.float64)

    t = np.clip((np.asarray(x) - edge0) / np.where(edge1 == edge0, 1e-9, edge1 - edge0), 0.0, 1.0)

    return t * t * (3.0 - 2.0 * t)
