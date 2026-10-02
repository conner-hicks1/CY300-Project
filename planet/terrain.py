from dataclasses import dataclass

import numpy as np

from planet.climate import fallback_surface
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

    # A liquid fills everything below sea level (the mesh
    # stops at its surface). Without one, basins stay dry.
    has_liquid: bool = True

    # Gas giants: cloud bands instead of a solid surface
    # (no relief). 0 = solid surface.
    bands: int = 0

    @property
    def min_elevation(
        self
    ) -> float:
        """Lower bound of the visible surface (0 under a liquid)."""

        if self.has_liquid or self.bands:
            return 0.0

        return -(2.0 * self.continent_height + self.detail_height + 7_000.0)

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

# With a tectonic field: the share of the noise mountains
# added on top (the field already raises the belts), and the
# coastline roughening.
_TECTONIC_RIDGE_SHARE = 0.85
_TECTONIC_COAST_FREQUENCY = 14.0
_TECTONIC_COAST_NOISE = 450.0


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
        settings: TerrainSettings,
        field=None,
        climate=None
    ):
        """
        field: optional planet.tectonics.TectonicField. With
            one, continents, ocean basins and mountain belts
            come from the plate simulation; noise adds the
            detail it cannot resolve (~80 km cells).
        climate: optional planet.climate.ClimateField:
            temperature and rainfall for the biomes (without
            one: latitude and noise).
        """

        self.settings = settings
        self.field = field
        self.climate = climate

        # The field's height range (culling bounds), and the
        # coastline roughening (plate worlds only: elsewhere
        # it would swamp low relief such as Europa's).
        if field is not None:

            self._field_max = float(field.elevation.max())
            self._field_min = float(field.elevation.min())

            self._coast_noise = (
                _TECTONIC_COAST_NOISE
                if getattr(field, "regime", "plate_tectonics") == "plate_tectonics"
                else 0.0
            )

        seed = settings.seed * 7919

        self._continents = Perlin(seed)
        self._mountain_mask = Perlin(seed + 1)
        self._mountains = Perlin(seed + 2)
        self._detail = Perlin(seed + 3)
        self._moisture = Perlin(seed + 4)

    @property
    def max_elevation(
        self
    ) -> float:
        """Upper bound of the surface height (horizon culling)."""

        if self.field is None:
            return self.settings.max_elevation

        s = self.settings

        return (
            max(self._field_max, 0.0)
            + _RIDGE_GAIN * s.mountain_height * _TECTONIC_RIDGE_SHARE
            + s.detail_height
            + self._coast_noise
        )

    @property
    def min_elevation(
        self
    ) -> float:
        """Lower bound of the visible surface (0 under a liquid)."""

        s = self.settings

        if self.field is None or s.has_liquid or s.bands:
            return s.min_elevation

        return min(self._field_min, 0.0) - s.detail_height - self._coast_noise - 200.0

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

        if s.bands:

            # Cloud tops of a giant: a smooth sphere.
            return np.zeros(len(directions))

        radius = s.radius

        def octaves(frequency):

            return octaves_for_spacing(
                radius / frequency,
                spacing,
                _MAX_OCTAVES
            )

        # -------------------------------------------------
        # Continents and mountain placement
        # -------------------------------------------------

        if self.field is not None:

            elevation, land, mask = self._tectonic_base(directions, octaves)

            ridge_share = _TECTONIC_RIDGE_SHARE

        else:

            continent_points = directions * s.continent_frequency

            continents = fbm(
                self._continents,
                continent_points,
                min(octaves(s.continent_frequency), 7)
            ) + s.land_bias

            # 0 in the ocean, rising to 1 just inland.
            land = _smoothstep(-0.02, 0.06, continents)

            elevation = continents * (2.0 * s.continent_height)

            mask = _smoothstep(
                0.0,
                0.35,
                fbm(
                    self._mountain_mask,
                    directions * (s.continent_frequency * _RANGE_FREQUENCY_SCALE),
                    3
                ) + 0.1
            ) * land

            ridge_share = 1.0

        # -------------------------------------------------
        # Mountains
        # -------------------------------------------------

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
                * ridge_share
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

    def _tectonic_base(
        self,
        directions: np.ndarray,
        octaves
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        (elevation, land 0..1, mountain mask) from the plate
        simulation, roughened below its resolution.
        """

        field = self.field

        elevation = field.sample("elevation", directions)

        # Coastlines and shelves finer than the grid.
        if self._coast_noise > 0.0:

            elevation = elevation + fbm(
                self._continents,
                directions * _TECTONIC_COAST_FREQUENCY,
                min(octaves(_TECTONIC_COAST_FREQUENCY), 6)
            ) * self._coast_noise

        land = _smoothstep(-100.0, 200.0, elevation)

        # Peaks where crust was recently uplifted (young
        # belts); rugged foothills wherever land stands high.
        orogeny = field.sample("orogeny", directions)

        mask = np.maximum(
            _smoothstep(200.0, 2_000.0, orogeny),
            _smoothstep(400.0, 3_000.0, elevation)
        ) * land

        return elevation, land, mask

    def tectonic_data(
        self,
        directions: np.ndarray
    ) -> np.ndarray:
        """
        (n, 4) per-vertex data for the tectonic views:
        plate index, crust age / 400 Myr, continental
        fraction, boundary closing speed (cm/yr). Plate -1
        without a simulation.
        """

        data = np.zeros((len(directions), 4), dtype=np.float32)

        field = self.field

        if field is None:

            data[:, 0] = -1.0

            return data

        data[:, 0] = field.plate_at(directions)
        data[:, 1] = field.sample("age", directions) / getattr(field, "age_scale", 400.0)
        data[:, 2] = field.sample("continental", directions)
        data[:, 3] = field.sample("activity", directions)

        return data

    # =====================================================
    # Surface Inputs
    # =====================================================
    #
    # Biome colors themselves are applied per pixel on the
    # GPU (assets/shaders/include/terrain.glsl) from
    # elevation, slope, temperature and rainfall.

    def surface_climate(
        self,
        directions: np.ndarray,
        elevation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        (annual mean temperature C at each point's own
        height, precipitation mm / year).
        """

        if self.settings.bands:
            return self._bands(directions)

        if self.climate is not None:
            return self.climate.surface(directions, elevation)

        return fallback_surface(directions, elevation, self.moisture(directions))

    def _bands(
        self,
        directions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Giant planet cloud belts: the "precipitation" slot
        carries a band coordinate in [0, 1] (belt vs zone),
        from latitude, wavy with stretched turbulence.
        """

        latitude = np.arcsin(np.clip(directions[:, 1], -1.0, 1.0))

        # Turbulence stretched along the bands (east-west).
        stretched = directions * np.array([2.0, 14.0, 2.0])

        turbulence = fbm(self._detail, stretched, 5)

        band = 0.5 + 0.5 * np.sin(latitude * self.settings.bands + 1.2 * turbulence)

        temperature = np.full(len(directions), -120.0)

        return temperature, band

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
