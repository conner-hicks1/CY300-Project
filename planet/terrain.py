from dataclasses import dataclass

import numpy as np

from planet.noise import (
    Perlin,
    fbm,
    octaves_for_spacing,
    ridged
)


# =========================================================
# Terrain Settings
# =========================================================
#
# Everything that determines the planet's shape. Chunks
# built with equal settings are identical, so a change here
# means rebuilding the planet.

@dataclass(frozen=True, slots=True)
class TerrainSettings:

    seed: int = 1
    radius: float = 6_371_000.0

    # Continents: low-frequency noise. land_bias shifts the
    # coastline (higher = more land).
    continent_frequency: float = 1.2
    continent_height: float = 2_500.0
    land_bias: float = 0.05

    # Mountains: ridged noise (~40 km base wavelength at
    # Earth size), placed in ranges by a low-frequency mask
    # and kept to land.
    mountain_frequency: float = 150.0
    mountain_height: float = 5_000.0

    # Hills: fractal noise from ~8 km wavelength down to
    # the vertex spacing.
    detail_height: float = 300.0

    @property
    def max_elevation(
        self
    ) -> float:
        """Upper bound of surface height above sea level."""

        # Continents reach ~2x continent_height, squared
        # ridges 1.6x mountain_height (see elevation()).
        return (
            2.0 * self.continent_height
            + _RIDGE_GAIN * self.mountain_height
            + self.detail_height
        )


# Above this many octaves the extra detail is far below
# the finest LOD's vertex spacing.
_MAX_OCTAVES = 14

_DETAIL_FREQUENCY_SCALE = 5.0

# Mountain ranges are placed at this multiple of the
# continent frequency.
_RANGE_FREQUENCY_SCALE = 5.0

# Scale of squared ridged noise (whose typical peaks are
# well under 1) so peaks approach mountain_height.
_RIDGE_GAIN = 1.6


class Terrain:

    # =====================================================
    # Height Field
    # =====================================================
    #
    # Elevation (meters above sea level; negative = ocean
    # floor) as a function of direction on the unit sphere.
    #
    # Octave counts follow the sample spacing, so a coarse
    # chunk seen from orbit does not pay for pebble-scale
    # noise it cannot show.

    def __init__(
        self,
        settings: TerrainSettings
    ):

        self.settings = settings

        seed = settings.seed * 7919

        self._continents = Perlin(seed)
        self._mountain_mask = Perlin(seed + 1)
        self._mountains = Perlin(seed + 2)
        self._detail = Perlin(seed + 3)
        self._moisture = Perlin(seed + 4)

    def elevation(
        self,
        directions: np.ndarray,
        spacing: float = 0.0
    ) -> np.ndarray:
        """
        directions: (n, 3) unit vectors.
        spacing: distance between samples in meters (0 =
            full detail).
        """

        s = self.settings

        radius = s.radius

        def octaves(frequency):

            return octaves_for_spacing(
                radius / frequency,
                spacing,
                _MAX_OCTAVES
            )

        # -------------------------------------------------
        # Continents
        # -------------------------------------------------

        continent_points = directions * s.continent_frequency

        continents = fbm(
            self._continents,
            continent_points,
            min(octaves(s.continent_frequency), 7)
        ) + s.land_bias

        # 0 in the ocean, rising to 1 just inland.
        land = _smoothstep(-0.02, 0.06, continents)

        elevation = continents * (2.0 * s.continent_height)

        # -------------------------------------------------
        # Mountains
        # -------------------------------------------------

        mask = _smoothstep(
            0.0,
            0.35,
            fbm(
                self._mountain_mask,
                directions * (s.continent_frequency * _RANGE_FREQUENCY_SCALE),
                3
            ) + 0.1
        ) * land

        if np.any(mask > 0.0):

            mountain_frequency = s.mountain_frequency

            ridges = ridged(
                self._mountains,
                directions * mountain_frequency,
                octaves(mountain_frequency)
            )

            # Squaring keeps valleys low and peaks sharp
            # (raw ridged noise lifts whole ranges into a
            # plateau).
            elevation += (
                ridges * ridges * _RIDGE_GAIN
                * mask
                * s.mountain_height
            )

        # -------------------------------------------------
        # Detail
        # -------------------------------------------------

        detail_frequency = s.mountain_frequency * _DETAIL_FREQUENCY_SCALE

        detail_octaves = octaves(detail_frequency)

        if s.detail_height > 0.0 and spacing < radius / detail_frequency:

            elevation += (
                fbm(
                    self._detail,
                    directions * detail_frequency,
                    detail_octaves
                )
                * s.detail_height
                * (0.3 + 0.7 * land)
            )

        return elevation

    # =====================================================
    # Surface Color
    # =====================================================

    def colors(
        self,
        directions: np.ndarray,
        elevation: np.ndarray,
        slope: np.ndarray
    ) -> np.ndarray:
        """
        Linear RGB per sample.

        elevation: meters (negative under water).
        slope: dot(surface normal, radial up), 1 = flat.
        """

        latitude = np.abs(directions[:, 1])

        moisture = fbm(
            self._moisture,
            directions * 3.0,
            3
        ) * 0.5 + 0.5

        # -------------------------------------------------
        # Land
        # -------------------------------------------------

        grass = _mix(_DRY_GRASS, _GRASS, _smoothstep(0.35, 0.65, moisture)[:, None])

        lowland = _mix(_SAND, grass, _smoothstep(5.0, 60.0, elevation)[:, None])

        # Forest on moist, low, gentle ground.
        forest = (
            _smoothstep(0.5, 0.75, moisture)
            * (1.0 - _smoothstep(1200.0, 2500.0, elevation))
            * _smoothstep(0.92, 0.97, slope)
            * _smoothstep(60.0, 200.0, elevation)
        )

        color = _mix(lowland, _FOREST, forest[:, None])

        color = _mix(color, _ROCK, _smoothstep(1500.0, 3000.0, elevation)[:, None])

        # Cliffs: steep ground is bare rock.
        color = _mix(color, _ROCK, (1.0 - _smoothstep(0.75, 0.88, slope))[:, None])

        # Snow line falls toward the poles; snow does not
        # stick to cliffs.
        snow_line = 4200.0 * (1.0 - latitude ** 3) - 300.0

        snow = (
            _smoothstep(snow_line - 200.0, snow_line + 200.0, elevation)
            * _smoothstep(0.7, 0.82, slope)
        )

        color = _mix(color, _SNOW, snow[:, None])

        # -------------------------------------------------
        # Water
        # -------------------------------------------------

        depth = np.clip(-elevation / 3000.0, 0.0, 1.0)

        water = _mix(_SHALLOW_WATER, _DEEP_WATER, np.sqrt(depth)[:, None])

        # Polar sea ice.
        water = _mix(water, _ICE, _smoothstep(0.95, 0.97, latitude)[:, None])

        underwater = (elevation < 0.0)[:, None]

        return np.where(underwater, water, color)


# Linear-space albedos.
_SAND = np.array([0.42, 0.36, 0.22])
_DRY_GRASS = np.array([0.28, 0.24, 0.10])
_GRASS = np.array([0.07, 0.16, 0.04])
_FOREST = np.array([0.025, 0.07, 0.025])
_ROCK = np.array([0.16, 0.14, 0.12])
_SNOW = np.array([0.80, 0.82, 0.86])
_SHALLOW_WATER = np.array([0.02, 0.10, 0.14])
_DEEP_WATER = np.array([0.004, 0.015, 0.05])
_ICE = np.array([0.65, 0.72, 0.78])


def _smoothstep(
    edge0: float,
    edge1: float,
    x
) -> np.ndarray:

    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)

    return t * t * (3.0 - 2.0 * t)


def _mix(
    a,
    b,
    t
) -> np.ndarray:

    return a + (b - a) * t
