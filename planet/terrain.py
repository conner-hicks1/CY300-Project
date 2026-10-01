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
    # Surface Inputs
    # =====================================================
    #
    # Biome colors themselves are applied per pixel on the
    # GPU (assets/shaders/include/terrain.glsl) from
    # elevation, slope, latitude and this moisture value.

    def moisture(
        self,
        directions: np.ndarray
    ) -> np.ndarray:
        """Wetness in [0, 1] (grass vs dry grass, forests)."""

        return np.clip(
            fbm(self._moisture, directions * 3.0, 3) * 0.5 + 0.5,
            0.0,
            1.0
        )


def _smoothstep(
    edge0: float,
    edge1: float,
    x
) -> np.ndarray:

    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)

    return t * t * (3.0 - 2.0 * t)
