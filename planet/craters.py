import math

from dataclasses import dataclass

import numpy as np


# =========================================================
# Impact Craters
# =========================================================
#
# Procedural craters at every size, evaluated per sample
# point like noise (so they appear at any level of detail,
# with no seams):
#
#   * Space is divided into a 3D lattice per size octave
#     (cell size ~3x the octave's largest crater). Each
#     cell may hold one crater, centered at a random point
#     of the cell projected onto the planet; a sample only
#     checks the 8 cells around it, so craters are kept
#     within a quarter cell of their center (with their
#     ejecta) and only centers near the surface count.
#   * Sizes follow a power law, N(>D) ~ D^-2 (each halving
#     of size brings 4x as many craters), so every octave
#     fills the same fraction of its cells.
#   * How many there are follows the surface's age (from
#     the tectonic regime): the lunar cratering chronology
#     (Neukum), with the late heavy bombardment before
#     ~3.8 billion years ago. The Moon's highlands are
#     saturated, its maria much sparser, Europa and Io
#     nearly blank.
#   * Shapes: simple bowls (depth 1/5 of the diameter)
#     below a transition diameter that shrinks with gravity
#     (Moon ~15 km, Earth ~3 km); larger ones are complex:
#     shallower, flat floored, with a central peak. A raised
#     rim and an ejecta blanket around both. Older craters
#     are worn shallower.
#   * An atmosphere burns up small impactors (Venus has no
#     craters under ~3 km) and weather erases old craters
#     (Earth, Titan).
#   * A few young craters on airless bodies throw bright
#     ejecta rays (Tycho, Copernicus): returned separately
#     as surface brightness.
#
# Units: meters, Myr.


# Craters (and their ejecta, out to EJECTA radii) stay
# within a quarter lattice cell of their center...
EJECTA = 1.6

# ...so the largest radius is this fraction of a cell, and
# each octave spans a factor of 2 in diameter.
_MAX_RADIUS_IN_CELLS = 0.25 / EJECTA
_D_HIGH = 2.0 * _MAX_RADIUS_IN_CELLS          # in cells
_D_LOW = _D_HIGH * 0.5

# Craters counted per km^2 per unit of N1 (craters > 1 km
# per km^2) in one octave, relative to cells per km^2 near
# the surface (a band half a cell thick): the chance that a
# cell holds a crater.
_CELL_CHANCE = 2.0 * (_D_LOW ** -2 - _D_HIGH ** -2)

LATTICE_CORNERS = np.array(
    [[(c >> 0) & 1, (c >> 1) & 1, (c >> 2) & 1] for c in range(8)],
    dtype=np.int64
)

# Most craters per lattice cell (saturated surfaces).
SLOTS = 4

# Smallest craters drawn (m): below this they add nothing
# visible at any distance the camera reaches.
MIN_DIAMETER = 30.0

# Largest basin, as a fraction of the body's radius.
BASIN_LIMIT = 0.5


def basin_depth(
    transition: float
) -> float:
    """
    How deep the largest basins get (m): ~7 km on the Moon,
    less where gravity is stronger (the crust relaxes, the
    floor rebounds): South Pole-Aitken, Hellas.
    """

    return 7_000.0 * (transition / 15_000.0) ** 0.25

# Ray craters: at most this many diameters across, their
# rays out to RAY_REACH radii.
RAY_REACH = 20.0
RAY_DIAMETER = (8_000.0, 50_000.0)

# Ray craters per (large) lattice cell at most.
RAY_SLOTS = 6


@dataclass(frozen=True, slots=True)
class CraterSettings:

    seed: int = 1

    # Impact rate relative to the Moon's (0 = no craters).
    density: float = 1.0

    # Surface age (Myr) where no tectonic field gives one.
    surface_age: float = 4_000.0

    # Craters older than about this are erased (weather,
    # rain, ice flow; Myr). 0 = never.
    erosion_time: float = 0.0

    # Smaller impactors burn up in the air (m).
    min_diameter: float = 0.0

    # Largest crater (m; also at most BASIN_LIMIT radii).
    max_diameter: float = 1.0e9

    # Simple -> complex craters above this diameter (m).
    transition_diameter: float = 15_000.0

    # Bright ejecta rays around young craters (airless).
    rays: bool = True


def cumulative_density(
    age_myr
) -> np.ndarray:
    """
    Craters larger than 1 km per km^2 on a surface of this
    age (Neukum's lunar chronology): steady impacts plus the
    heavy bombardment, which rises steeply before ~3.8 Gyr.
    """

    t = np.clip(np.asarray(age_myr, dtype=np.float64), 0.0, 4_500.0) / 1_000.0

    return 5.44e-14 * (np.exp(6.93 * t) - 1.0) + 8.38e-4 * t


class Craters:

    def __init__(
        self,
        settings: CraterSettings,
        radius: float
    ):

        self.settings = settings
        self.radius = radius

        s = settings

        self.enabled = s.density > 0.0

        # Up to giant basins half the body's size (the Moon's
        # South Pole-Aitken, Mars's Hellas), rare as the power
        # law makes them.
        top = min(s.max_diameter, BASIN_LIMIT * radius)

        # Octave cell sizes (m), largest first, down to the
        # smallest crater drawn.
        smallest = max(MIN_DIAMETER, s.min_diameter)

        self.cells: list[float] = []

        cell = top / _D_HIGH

        while self.enabled and cell * _D_HIGH >= smallest and len(self.cells) < 20:

            self.cells.append(cell)

            cell *= 0.5

        # Deepest bowl and highest rim (m), for culling
        # bounds: the largest three octaves stacked, crater in
        # crater (smaller ones inside add little; basins are
        # capped at basin_depth, so several octaves are as
        # deep).
        stacked = sum(self._depth(cell * _D_HIGH) for cell in self.cells[:3])

        self.max_depth = stacked
        self.max_rim = 0.25 * stacked

        self._ray_cell = RAY_DIAMETER[1] * 0.5 * RAY_REACH / 0.25

    # -----------------------------------------------------
    # Height
    # -----------------------------------------------------

    def _depth(
        self,
        diameter: float
    ) -> float:
        """Fresh crater depth (m) for a diameter."""

        transition = self.settings.transition_diameter

        if diameter < transition:
            return 0.2 * diameter

        return min(0.2 * transition * (diameter / transition) ** 0.3, basin_depth(transition))

    def chance(
        self,
        age_myr
    ) -> np.ndarray:
        """
        Expected craters per lattice cell, per point (up to
        SLOTS: old surfaces like the lunar highlands are
        saturated, crater upon crater).
        """

        s = self.settings

        age = np.asarray(age_myr, dtype=np.float64)

        if s.erosion_time > 0.0:
            age = np.minimum(age, s.erosion_time)

        return np.clip(_CELL_CHANCE * s.density * cumulative_density(age), 0.0, SLOTS)

    def height(
        self,
        directions: np.ndarray,
        spacing: float = 0.0,
        age_myr=None
    ) -> np.ndarray:
        """
        Crater relief (m) at unit directions. spacing: sample
        spacing (m); craters much smaller are skipped.
        age_myr: surface age per point (None = the setting).
        """

        n = len(directions)

        out = np.zeros(n)

        if not self.enabled or n == 0:
            return out

        if age_myr is None:
            age_myr = self.settings.surface_age

        chance = np.broadcast_to(self.chance(age_myr), (n,))

        if not np.any(chance > 0.0):
            return out

        points = directions * self.radius

        slots = int(math.ceil(float(np.max(chance))))

        for octave, cell in enumerate(self.cells):

            if cell * _D_HIGH < 1.5 * spacing:
                break

            # Each cell holds up to SLOTS craters, the k-th
            # with chance (expected - k).
            lattice = lattice_cells(points, cell)

            for slot in range(slots):

                self._octave(
                    lattice,
                    directions,
                    cell,
                    octave * 8 + slot,
                    np.clip(chance - slot, 0.0, 1.0),
                    out
                )

        return out

    def _octave(
        self,
        lattice: tuple,
        directions: np.ndarray,
        cell: float,
        octave: int,
        chance: np.ndarray,
        out: np.ndarray
    ):

        s = self.settings

        keys, cell_of, point = lattice

        # Per candidate crater (each cell once).
        h = lattice_hash(keys, s.seed * 1_000 + octave)

        roll = lattice_uniform(h, 0)

        center = (keys + np.stack([lattice_uniform(h, k) for k in (1, 2, 3)], axis=1)) * cell

        length = np.linalg.norm(center, axis=1)

        # Only centers near the surface (so the 8 cells
        # checked always contain every crater in reach).
        near = np.abs(length - self.radius) < 0.25 * cell

        pairs = np.nonzero(near[cell_of])[0]

        if len(pairs) == 0:
            return

        cell_of, point = cell_of[pairs], point[pairs]

        # Fade in rather than cut where the age (and so the
        # chance) changes across a crater.
        weight = np.clip((chance[point] - roll[cell_of]) / 0.03, 0.0, 1.0)

        live = np.nonzero(weight > 0.0)[0]

        if len(live) == 0:
            return

        cell_of, point, weight = cell_of[live], point[live], weight[live]

        unit = center / length[:, None]

        # Power-law size within the octave.
        diameter = _D_LOW * cell / np.sqrt(1.0 - 0.75 * lattice_uniform(h, 4))

        x = np.linalg.norm(
            directions[point] - unit[cell_of],
            axis=1
        ) * self.radius / (0.5 * diameter[cell_of])

        hit = x < EJECTA

        if not np.any(hit):
            return

        cell_of, point, weight, x = cell_of[hit], point[hit], weight[hit], x[hit]

        # Old craters are worn shallower.
        fresh = 0.3 + 0.7 * np.sqrt(lattice_uniform(h, 5))

        profile = crater_profile(x, diameter[cell_of], s.transition_diameter)

        np.add.at(out, point, profile * fresh[cell_of] * weight)

    # -----------------------------------------------------
    # Rays (brightness)
    # -----------------------------------------------------

    def brightness(
        self,
        directions: np.ndarray,
        age_myr=None
    ) -> np.ndarray:
        """
        Surface brightening (0..1) from young craters: bright
        floors and ejecta, and rays streaking outward.
        """

        n = len(directions)

        out = np.zeros(n)

        s = self.settings

        if not self.enabled or not s.rays or n == 0:
            return out

        if age_myr is None:
            age_myr = s.surface_age

        # Rays fade within ~1 Gyr, so they need surfaces at
        # least that old to show on (and impacts to make
        # them): a dozen or so on the Moon.
        chance = np.broadcast_to(
            0.6 * min(s.density, 1.0) * np.clip(np.asarray(age_myr, dtype=np.float64) / 1_000.0, 0.0, 1.0),
            (n,)
        )

        cell = self._ray_cell

        keys, cell_of, point = lattice_cells(directions * self.radius, cell)

        most = float(np.max(chance))

        for slot in range(RAY_SLOTS):

            # Per candidate ray crater (each cell once).
            h = lattice_hash(keys, s.seed * 7_919 + 99 + slot * 131)

            roll = lattice_uniform(h, 0)

            center = (keys + np.stack([lattice_uniform(h, k) for k in (1, 2, 3)], axis=1)) * cell

            length = np.linalg.norm(center, axis=1)

            candidate = (roll < most) & (np.abs(length - self.radius) < 0.25 * cell)

            if not np.any(candidate):
                continue

            pairs = np.nonzero(candidate[cell_of])[0]

            pairs = pairs[roll[cell_of[pairs]] < chance[point[pairs]]]

            if len(pairs) == 0:
                continue

            owner = cell_of[pairs]
            live = point[pairs]

            c = center[owner] / length[owner, None]

            r = 0.5 * (RAY_DIAMETER[0] + (RAY_DIAMETER[1] - RAY_DIAMETER[0]) * lattice_uniform(h[owner], 4))

            d = directions[live]

            x = np.linalg.norm(d - c, axis=1) * self.radius / r

            hit = x < RAY_REACH

            if not np.any(hit):
                continue

            # (h becomes per hit: the slot's next pass rehashes.)
            c, d, x, h = c[hit], d[hit], x[hit], h[owner[hit]]

            # Angle around the crater.
            helper = np.where(np.abs(c[:, 1:2]) < 0.9, [[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]])

            t1 = np.cross(c, helper)
            t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
            t2 = np.cross(c, t1)

            theta = np.arctan2(np.sum(d * t2, axis=1), np.sum(d * t1, axis=1))

            # Streaks: a few sharp lobes at unrelated
            # frequencies (integers, so they close around the
            # crater), each reaching its own distance, so the
            # pattern is lopsided like Tycho's.
            streaks = np.zeros(len(x))
            reach = np.zeros(len(x))

            for lobe in range(3):

                k = np.floor(5.0 + 22.0 * lattice_uniform(h, 5 + lobe))
                phase = 6.283 * lattice_uniform(h, 8 + lobe)

                lobes = (0.5 + 0.5 * np.cos(k * theta + phase)) ** (6 + 4 * lobe)

                streaks = np.maximum(streaks, lobes * (1.0 - 0.25 * lobe))

                reach += lobes * (0.3 + 0.7 * lattice_uniform(h, 11 + lobe))

            # Some directions throw rays much farther.
            reach = RAY_REACH * np.clip(0.25 + reach, 0.25, 1.0) * (
                0.6 + 0.4 * (0.5 + 0.5 * np.cos(np.floor(1.0 + 3.0 * lattice_uniform(h, 14)) * theta + 6.283 * lattice_uniform(h, 15)))
            )

            rays = streaks * (1.0 - _smoothstep(0.3, 1.0, x / reach)) * _smoothstep(0.8, 1.2, x)

            blanket = 1.0 - _smoothstep(1.0, 2.5, x)

            np.maximum.at(out, live[hit], np.clip(0.7 * blanket + 0.45 * rays, 0.0, 1.0))

        return out


def crater_profile(
    x: np.ndarray,
    diameter: np.ndarray,
    transition: float
) -> np.ndarray:
    """
    Fresh crater height (m) at x = distance / radius:
    bowl (or flat floor and central peak when complex)
    inside, rim at x = 1, ejecta thinning outside to 0 at
    EJECTA.
    """

    complex_ = diameter >= transition

    depth = np.where(
        complex_,
        np.minimum(
            0.2 * transition * (np.maximum(diameter, transition) / transition) ** 0.3,
            basin_depth(transition)
        ),
        0.2 * diameter
    )

    rim = 0.18 * depth

    inside = x < 1.0

    # Simple: a parabolic bowl. Complex: flat floor, steep
    # walls, central peak.
    bowl = -depth + (depth + rim) * x * x

    floor = _smoothstep(0.3, 1.0, x)

    walled = -depth + (depth + rim) * floor * np.sqrt(floor)

    # Central peaks; the largest basins have rings instead.
    peak = 0.5 * depth * np.exp(-(x / 0.15) ** 2) * (diameter > 1.5 * transition) * (diameter < 12.0 * transition)

    interior = np.where(complex_, walled + peak, bowl)

    tail = EJECTA ** -3.0

    ejecta = rim * (np.maximum(x, 1.0) ** -3.0 - tail) / (1.0 - tail)

    return np.where(inside, interior, np.where(x < EJECTA, ejecta, 0.0))


# =========================================================
# Hashing
# =========================================================

_M1 = np.uint64(0x9E3779B97F4A7C15)
_M2 = np.uint64(0xC2B2AE3D27D4EB4F)
_M3 = np.uint64(0x165667B19E3779F9)
_F1 = np.uint64(0xBF58476D1CE4E5B9)
_F2 = np.uint64(0x94D049BB133111EB)


def _mix(
    h: np.ndarray
) -> np.ndarray:
    """splitmix64 finalizer."""

    h = h ^ (h >> np.uint64(30))
    h = h * _F1
    h = h ^ (h >> np.uint64(27))
    h = h * _F2

    return h ^ (h >> np.uint64(31))


def lattice_cells(
    points: np.ndarray,
    cell: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The 8 lattice cells (of this size) around every point,
    each distinct cell once: (cells (m, 3), the cell index
    of each point-cell pair (8n,), the point of each pair
    (8n,)). Neighboring points share cells, so per-cell work
    (hashing, placing the feature) runs m << 8n times.
    """

    n = len(points)

    base = np.floor(points / cell - 0.5).astype(np.int64)

    keys = (base[:, None, :] + LATTICE_CORNERS[None, :, :]).reshape(-1, 3)

    point = np.repeat(np.arange(n), 8)

    if n == 0:
        return keys, np.zeros(0, dtype=np.int64), point

    # The points of one chunk span only a few cells: index a
    # dense box of them (no sort). Spread-out points (whole-
    # planet sampling) fall back to a sorted unique.
    low = base.min(axis=0)
    extent = base.max(axis=0) - low + 2

    volume = int(extent[0]) * int(extent[1]) * int(extent[2])

    if volume <= 4 * len(keys) + 4_096:

        local = keys - low

        linear = (local[:, 0] * extent[1] + local[:, 1]) * extent[2] + local[:, 2]

        present = np.zeros(volume, dtype=bool)
        present[linear] = True

        cells = np.flatnonzero(present)

        index = np.cumsum(present) - 1

        unique = np.stack(
            (
                cells // (extent[1] * extent[2]),
                (cells // extent[2]) % extent[1],
                cells % extent[2]
            ),
            axis=1
        ) + low

        return unique, index[linear], point

    offset = np.int64(1 << 20)

    packed = (
        ((keys[:, 0] + offset) << np.int64(42))
        | ((keys[:, 1] + offset) << np.int64(21))
        | (keys[:, 2] + offset)
    )

    _, first, cell_of = np.unique(packed, return_index=True, return_inverse=True)

    return keys[first], cell_of.reshape(-1), point


def lattice_hash(
    key: np.ndarray,
    salt: int
) -> np.ndarray:
    """uint64 hash of integer lattice cells (n, 3)."""

    k = key.astype(np.uint64)

    with np.errstate(over="ignore"):
        return _mix(
            (k[:, 0] * _M1)
            ^ (k[:, 1] * _M2)
            ^ (k[:, 2] * _M3)
            ^ np.uint64(salt & 0xFFFFFFFF)
        )


def lattice_uniform(
    h: np.ndarray,
    stream: int
) -> np.ndarray:
    """Uniform [0, 1) numbers from a hash, one stream per use."""

    with np.errstate(over="ignore"):
        mixed = _mix(h + np.uint64(stream + 1) * _M1)

    return (mixed >> np.uint64(11)).astype(np.float64) * (1.0 / 9_007_199_254_740_992.0)


def _smoothstep(
    edge0: float,
    edge1: float,
    x
) -> np.ndarray:

    t = np.clip((np.asarray(x) - edge0) / (edge1 - edge0), 0.0, 1.0)

    return t * t * (3.0 - 2.0 * t)
