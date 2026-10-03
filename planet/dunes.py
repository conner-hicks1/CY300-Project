import math

from dataclasses import dataclass

import numpy as np

from planet.climate import prevailing_wind
from planet.craters import lattice_cells
from planet.noise import Perlin, fbm


# =========================================================
# Wind-Blown Dunes
# =========================================================
#
# Sand piles into dunes wherever there is air enough to
# move it, sand to move, and too little rain to hold it
# down: Earth's ergs (the Sahara, Arabia), Mars's dark
# basaltic dune fields (crater floors, the north polar
# erg), Titan's equatorial sand seas of organic grains.
#
#   transverse   crests across the wind, a gentle windward
#                (stoss) side and a steep lee (slip face):
#                Earth, Mars, Venus
#   linear       long parallel ridges along the wind,
#                kilometers apart, merging in Y-junctions:
#                Titan's belt within ~30 degrees of the
#                equator
#
# Dune fields cover the ground that is dry, sandy (patchy
# noise; Titan's belt nearly all) and within the latitude
# band. Their crests follow the prevailing wind, sampled at
# coarse lattice frames and blended, with phases from
# absolute position (so the crests run on unbroken
# between frames).
# From afar, where the dunes themselves are smaller than a
# sample, only the fields' (often darker) sand shows.

TRANSVERSE = "transverse"
LINEAR = "linear"


@dataclass(frozen=True, slots=True)
class DuneSettings:

    seed: int = 1

    # 0 = none; ~1 = dry worlds (Mars), less where it rains.
    density: float = 0.0

    # Crest height (m) and spacing (m).
    amplitude: float = 50.0
    wavelength: float = 1_500.0

    linear: bool = False

    # Dune fields only within this latitude (degrees).
    max_latitude: float = 90.0

    # How much darker dune sand is than the ground (0..1).
    darkening: float = 0.0


# Field patch radii (m), largest first.
_PATCHES = (180_000.0, 80_000.0)


class Dunes:

    def __init__(
        self,
        settings: DuneSettings,
        radius: float,
        climate=None
    ):
        """climate: ClimateField for rainfall (None: everywhere dry)."""

        self.settings = settings
        self.radius = radius
        self.climate = climate

        self.patches = [
            r for r in _PATCHES
            if r < 0.15 * radius
        ] if settings.density > 0.0 else []

        self._crests = Perlin(settings.seed * 61 + 7)
        self._sand = Perlin(settings.seed * 61 + 8)

    @property
    def enabled(
        self
    ) -> bool:

        return bool(self.patches)

    @property
    def max_height(
        self
    ) -> float:

        return self.settings.amplitude if self.enabled else 0.0

    # -----------------------------------------------------
    # Where
    # -----------------------------------------------------

    def potential(
        self,
        directions: np.ndarray
    ) -> np.ndarray:
        """0..1: dry, sandy ground within the band."""

        s = self.settings

        latitude = np.degrees(np.arcsin(np.clip(np.abs(directions[:, 1]), 0.0, 1.0)))

        band = _smoothstep(s.max_latitude, s.max_latitude - 5.0, latitude)

        if self.climate is not None and not s.linear:

            rain = self.climate.grid.sample(self.climate.precipitation, directions)

            dry = _smoothstep(400.0, 150.0, rain)

        else:

            # (Titan's sand seas lie in its dry equatorial
            # belt, which the Earth-like circulation misses.)
            dry = np.ones(len(directions))

        # Sand gathers in patches (sand seas between rockier
        # ground); Titan's belt is nearly continuous.
        floor = -0.45 if s.linear else 0.0

        sand = _smoothstep(floor, floor + 0.3, fbm(self._sand, directions * 3.0, 4))

        return np.clip(s.density * band * dry * sand, 0.0, 1.0)

    # -----------------------------------------------------
    # Height and coverage
    # -----------------------------------------------------

    def apply(
        self,
        directions: np.ndarray,
        spacing: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """(dune height m, dune-field coverage 0..1) at unit directions."""

        n = len(directions)

        height = np.zeros(n)

        if not self.enabled or n == 0:
            return height, np.zeros(n)

        s = self.settings

        # Sand seas: where the ground is dry and sandy enough.
        cover = _smoothstep(0.3, 0.7, self.potential(directions))

        # Dunes smaller than the sample spacing would alias:
        # only the field shows.
        if spacing >= 0.5 * s.wavelength or not np.any(cover > 0.0):
            return height, cover

        live = np.nonzero(cover > 0.0)[0]

        d = directions[live]

        points = d * self.radius

        # Crests follow the prevailing wind, sampled at lattice
        # frames (cells ~4 patch radii): each frame lays a
        # plane wave from its own origin (on a sphere, phases
        # from different frames cannot agree), and the frames'
        # dune patterns are blended with narrow weights, so
        # one frame rules nearly everywhere and the seams are
        # short stretches of crossing crests (as in real
        # complex dunes).
        cell = self.patches[0] / 0.25

        keys, cell_of, point = lattice_cells(points, cell)

        centers = (keys + 0.5) * cell
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)

        wind = prevailing_wind(centers)
        wind -= centers * np.sum(wind * centers, axis=1, keepdims=True)
        wind /= np.maximum(np.linalg.norm(wind, axis=1, keepdims=True), 1e-9)

        # The wave travels across the crests: along the wind
        # for transverse dunes, across it for linear ones.
        axis = np.cross(centers, wind) if s.linear else wind

        along = np.cross(centers, axis)

        offset = (d[point] - centers[cell_of]) * self.radius

        distance = np.linalg.norm(offset, axis=1) / cell

        weight = np.exp(-(distance / 0.22) ** 2)

        # Sinuous crests: phase warped by noise.
        warp = fbm(self._crests, d * (self.radius / (4.0 * s.wavelength)), 3)

        phase = np.sum(offset * axis[cell_of], axis=1) / s.wavelength + 0.35 * warp[point]

        f = phase - np.floor(phase)

        field = cover[live]

        if s.linear:

            # Symmetric ridges fading in and out along their
            # length (Y-junctions where they merge).
            pattern = (0.5 + 0.5 * np.cos(2.0 * math.pi * f)) ** 1.5

            pattern *= 0.55 + 0.45 * np.cos(
                np.sum(offset * along[cell_of], axis=1) / (6.0 * s.wavelength) + 2.0 * warp[point]
            )

        else:

            # Long gentle stoss, short steep lee.
            pattern = np.where(f < 0.78, f / 0.78, (1.0 - f) / 0.22)

            pattern = 0.75 * pattern + 0.25 * (0.5 - 0.5 * np.cos(2.0 * math.pi * f))

        blended = np.zeros(len(live))
        total = np.zeros(len(live))

        np.add.at(blended, point, pattern * weight)
        np.add.at(total, point, weight)

        profile = blended / np.maximum(total, 1e-12)

        if not s.linear:

            # Broken into crescents (barchans) where the sand
            # thins at the field's edges.
            profile *= np.clip(0.5 + warp + field, 0.0, 1.0)

        height[live] = s.amplitude * profile * field

        return height, cover


def _smoothstep(
    edge0: float,
    edge1: float,
    x
) -> np.ndarray:

    t = np.clip((np.asarray(x) - edge0) / (edge1 - edge0), 0.0, 1.0)

    return t * t * (3.0 - 2.0 * t)
