import math

from dataclasses import dataclass

import numpy as np

from planet.craters import lattice_cells, lattice_hash, lattice_uniform


# =========================================================
# Volcanoes
# =========================================================
#
# Volcanic edifices built where the tectonic simulation
# says magma reaches the surface (its activity field):
#
#   shield volcanoes   broad and gentle (a few degrees),
#                      basaltic, over mantle plumes and
#                      hot spots: Olympus Mons, Mauna Loa,
#                      Maat Mons. Summit calderas (nested
#                      collapse pits), radial lava-flow
#                      ridges on the flanks, and a basal
#                      cliff on the giants (Olympus's
#                      6 km escarpment).
#   stratovolcanoes    steep cones with concave flanks and a
#                      small summit crater, in chains above
#                      subduction zones (the Andes,
#                      Cascades, Japan).
#   small shields      low domes a few km across in fields
#                      on volcanic plains (Venus has
#                      hundreds of thousands; Mars's
#                      provinces, the Moon's few domes).
#
# Gravity sets the ceiling: a volcano grows until its base
# can no longer bear it, so the tallest scale with 1/g
# (Earth ~10 km from the sea floor, Mars ~25 km).
#
# Like craters (planet/craters.py), they are placed on a
# 3D lattice per size octave and evaluated per sample, so
# they appear at every level of detail with no seams; but a
# volcano exists only if the activity at its *center* is
# high enough, so none is cut off where the activity
# changes. Overlapping volcanoes merge (the higher surface
# wins).

# Kinds.
SHIELD = 0
STRATO = 1
SMALL_SHIELD = 2


@dataclass(frozen=True, slots=True)
class VolcanoSettings:

    seed: int = 1

    # 0..1: how much the body builds volcanoes. Large
    # edifices need >= 0.5 (sustained plumes); below that
    # only small domes, rarer.
    volcanism: float = 1.0

    # Tallest volcano (m), from gravity.
    max_height: float = 10_000.0


@dataclass(frozen=True, slots=True)
class _Octave:

    kind: int
    radius: float           # largest radius in the octave (m)
    rate: float             # chance per candidate at full activity
    height_ratio: tuple[float, float]   # radius / height range

    # Candidates per lattice cell: the giants' cells are so
    # large that a volcanic province would otherwise often
    # get none.
    slots: int = 1


# Volcano kinds each tectonic regime builds.
_REGIME_KINDS = {
    "plate_tectonics": (SHIELD, STRATO, SMALL_SHIELD),
    "stagnant_lid": (SHIELD, SMALL_SHIELD),
    "episodic_resurfacing": (SHIELD, SMALL_SHIELD),
    "heat_pipe": (SMALL_SHIELD,),
}

# Shield radius over height (Olympus ~14, Earth's from the
# sea floor ~10-20) and stratovolcanoes (~4).
_SHIELD_RATIO = (11.0, 20.0)
_STRATO_RATIO = (3.0, 5.0)
_SMALL_RATIO = (12.0, 30.0)


def max_volcano_height(
    surface_gravity: float
) -> float:
    """Tallest volcano (m) a body's gravity allows."""

    return float(min(10_000.0 * 9.81 / max(surface_gravity, 0.1), 27_000.0))


class Volcanoes:

    def __init__(
        self,
        settings: VolcanoSettings,
        radius: float,
        field
    ):
        """
        field: planet.tectonics.TectonicField (where the
        activity is), or None (no volcanoes).
        """

        self.settings = settings
        self.radius = radius
        self.field = field

        s = settings

        self.octaves: list[_Octave] = []

        if field is None or s.volcanism <= 0.0:
            return

        # Large shields: the biggest octave holds the giants.
        if s.volcanism >= 0.5:

            top = min(s.max_height * 15.0, 0.15 * radius)

            for index, (rate, slots) in enumerate(((0.8, 3), (0.7, 2), (0.8, 1))):

                self.octaves.append(
                    _Octave(SHIELD, top * 0.5 ** index, rate * s.volcanism, _SHIELD_RATIO, slots)
                )

            # Stratovolcano chains (subduction arcs).
            for index in range(2):

                self.octaves.append(
                    _Octave(STRATO, 14_000.0 * 0.5 ** index, 0.8 * s.volcanism, _STRATO_RATIO)
                )

        # Fields of small shields and domes.
        for index in range(2):

            self.octaves.append(
                _Octave(SMALL_SHIELD, 12_000.0 * 0.5 ** index, 0.9 * s.volcanism, _SMALL_RATIO)
            )

        # Kinds this regime never builds cost nothing.
        regime = getattr(field, "regime", "plate_tectonics")

        self.octaves = [
            o for o in self.octaves
            if o.kind in _REGIME_KINDS.get(regime, ())
        ]

        self.max_height = max(
            (min(o.radius / o.height_ratio[0], s.max_height) for o in self.octaves),
            default=0.0
        )

    @property
    def enabled(
        self
    ) -> bool:

        return bool(self.octaves)

    # -----------------------------------------------------
    # Where magma reaches the surface
    # -----------------------------------------------------

    def potential(
        self,
        kind: int,
        directions: np.ndarray
    ) -> np.ndarray:
        """0..1: how volcanic the ground is for this kind of volcano."""

        field = self.field

        regime = getattr(field, "regime", "plate_tectonics")

        activity = field.sample("activity", directions)

        if regime == "plate_tectonics":

            # Arcs above subducting plates (converging
            # boundaries); rare hot spots anywhere.
            arcs = _smoothstep(1.0, 4.0, activity)

            if kind == SHIELD:
                return np.full(len(directions), 0.03)

            if kind == STRATO:
                return arcs

            return 0.3 * arcs

        if regime == "stagnant_lid":

            # Volcanic provinces over mantle plumes (Tharsis).
            provinces = _smoothstep(0.03, 0.25, activity)

            if kind == STRATO:
                return np.zeros(len(directions))

            return provinces

        if regime == "episodic_resurfacing":

            # Large shields on volcanic rises and coronae;
            # small shields over the young plains.
            if kind == SHIELD:
                return _smoothstep(0.15, 0.7, activity)

            if kind == STRATO:
                return np.zeros(len(directions))

            age = field.sample("age", directions)

            return np.maximum(
                _smoothstep(0.05, 0.4, activity),
                0.35 * _smoothstep(900.0, 300.0, age)
            )

        if regime == "heat_pipe":

            # Io: low shields around active vents.
            if kind == SMALL_SHIELD:
                return 0.5 * _smoothstep(0.2, 1.0, activity)

            return np.zeros(len(directions))

        return np.zeros(len(directions))

    # -----------------------------------------------------
    # Height
    # -----------------------------------------------------

    def height(
        self,
        directions: np.ndarray,
        spacing: float = 0.0
    ) -> np.ndarray:
        """Volcanic relief (m, >= 0) at unit directions."""

        out = np.zeros(len(directions))

        if not self.enabled or len(directions) == 0:
            return out

        points = directions * self.radius

        for index, octave in enumerate(self.octaves):

            # Smaller than the sample spacing: invisible.
            if octave.radius < spacing:
                continue

            for slot in range(octave.slots):
                self._octave(points, directions, octave, index * 8 + slot, out)

        return out

    def _octave(
        self,
        points: np.ndarray,
        directions: np.ndarray,
        octave: _Octave,
        index: int,
        out: np.ndarray
    ):

        s = self.settings

        # A volcano and its flanks stay within a quarter cell
        # of its center (so the 8 cells around a point hold
        # every volcano that reaches it).
        cell = octave.radius / 0.25

        keys, cell_of, point = lattice_cells(points, cell)

        # Per candidate volcano (each cell once).
        h = lattice_hash(keys, s.seed * 3_571 + index * 17 + 5)

        roll = lattice_uniform(h, 0)

        center = (keys + np.stack([lattice_uniform(h, k) for k in (1, 2, 3)], axis=1)) * cell

        length = np.linalg.norm(center, axis=1)

        # Possible at full activity, and centered near the
        # surface.
        candidate = (roll < octave.rate) & (np.abs(length - self.radius) < 0.25 * cell)

        if not np.any(candidate):
            return

        # Exists if the ground at its center is volcanic
        # enough (one field lookup per candidate).
        exists = np.zeros(len(keys), dtype=bool)

        chosen = np.nonzero(candidate)[0]

        unit = center / length[:, None]

        exists[chosen] = roll[chosen] < octave.rate * self.potential(octave.kind, unit[chosen])

        pairs = np.nonzero(exists[cell_of])[0]

        if len(pairs) == 0:
            return

        cell_of, point = cell_of[pairs], point[pairs]

        radius_of = octave.radius * (0.5 + 0.5 * lattice_uniform(h, 4))

        x = np.linalg.norm(directions[point] - unit[cell_of], axis=1) * self.radius / radius_of[cell_of]

        hit = x < 1.0

        if not np.any(hit):
            return

        cell_of, point, x = cell_of[hit], point[hit], x[hit]

        h, unit, radius = h[cell_of], unit[cell_of], radius_of[cell_of]

        low, high = octave.height_ratio

        height = np.minimum(
            radius / (low + (high - low) * lattice_uniform(h, 5)),
            s.max_height
        )

        profile = volcano_profile(
            octave.kind,
            x,
            height,
            lattice_uniform(h, 6),
            _azimuth(directions[point], unit),
            h,
            height / s.max_height
        )

        np.maximum.at(out, point, profile)


def volcano_profile(
    kind: int,
    x: np.ndarray,
    height: np.ndarray,
    caldera: np.ndarray,
    azimuth: np.ndarray,
    h: np.ndarray,
    stature: np.ndarray
) -> np.ndarray:
    """
    Height (m) at x = distance / radius (0 at the center,
    1 at the foot).
    caldera: 0..1 caldera size draw; azimuth: angle around
    the summit (radians, for flank ridges); h: per-volcano
    hash (more draws); stature: height / the tallest
    possible (giants get a basal cliff).
    """

    x = np.clip(x, 0.0, 1.0)

    if kind == STRATO:

        # Concave flanks steepening to the summit, a small
        # crater on top.
        surface = height * (1.0 - x) ** 2.2

        # The crater: a bowl sunk below its rim.
        crater = 0.04 + 0.04 * caldera

        rim = height * (1.0 - crater) ** 2.2

        bowl = rim - 0.05 * height * (1.0 - (x / crater) ** 2)

        surface = np.where(x < crater, bowl, surface)

        return np.maximum(surface, 0.0)

    # Shields: a convex dome with long, gentle flanks.
    surface = height * (1.0 - x ** 1.4) ** 1.6

    if kind == SHIELD:

        # Giants: a cliff around the base (Olympus Mons).
        cliff = _smoothstep(0.55, 0.85, stature)

        surface = surface * (1.0 - 0.25 * cliff) + 0.25 * cliff * height * _smoothstep(1.0, 0.93, x)

        # Lava-flow ridges radiating down the flanks.
        ridges = 0.5 + 0.5 * np.cos(
            np.floor(18.0 + 30.0 * lattice_uniform(h, 7)) * azimuth + 6.283 * lattice_uniform(h, 8)
        )

        surface *= 1.0 + 0.03 * (ridges - 0.5) * _smoothstep(0.2, 0.5, x) * (1.0 - x)

        # Summit caldera: nested collapse pits, flat floors.
        rim = 0.06 + 0.1 * caldera

        surface -= 0.12 * height * _smoothstep(rim, rim * 0.85, x)

        inner = rim * (0.35 + 0.3 * lattice_uniform(h, 9))

        surface -= 0.05 * height * _smoothstep(inner, inner * 0.85, x)

    else:

        # Small shields: a summit pit.
        surface -= 0.15 * height * _smoothstep(0.12, 0.08, x)

    return np.maximum(surface, 0.0)


def _azimuth(
    directions: np.ndarray,
    centers: np.ndarray
) -> np.ndarray:
    """Angle of each point around its volcano's summit."""

    helper = np.where(np.abs(centers[:, 1:2]) < 0.9, [[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]])

    t1 = np.cross(centers, helper)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(centers, t1)

    return np.arctan2(np.sum(directions * t2, axis=1), np.sum(directions * t1, axis=1))


def _smoothstep(
    edge0: float,
    edge1: float,
    x
) -> np.ndarray:

    t = np.clip((np.asarray(x) - edge0) / (edge1 - edge0), 0.0, 1.0)

    return t * t * (3.0 - 2.0 * t)
