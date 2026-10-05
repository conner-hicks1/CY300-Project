from dataclasses import dataclass, replace

import numpy as np

from planet.climate import fallback_surface
from planet.craters import CraterSettings, Craters
from planet.dunes import DuneSettings, Dunes
from planet.erosion import WAVELENGTHS as GULLY_WAVELENGTHS, gullies, relief_gradient
from planet.hydrology import meander_offsets
from core.disk_cache import cache_key

from planet.maps import load_elevation_map
from planet.shape import BodyShape, ShapeSettings
from planet.volcanoes import VolcanoSettings, Volcanoes

# Crust brightness of fresh ejecta and rays (1 = the
# brightest highlands).
RAY_BRIGHTNESS = 1.6
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

    # The surface the relief stands on (planet/shape.py):
    # flattening by the spin ((equatorial - polar) /
    # equatorial radius; an ellipsoid of revolution about the
    # local y axis), and for irregular bodies a triaxial,
    # lumpy or two-lobed shape with giant basins. Relief and
    # sea level are measured from it.
    oblateness: float = 0.0
    shape: ShapeSettings | None = None

    # Measured heights (planet/maps.py dataset id; "" =
    # generated terrain): Mars, the Moon, Earth as they are.
    # They replace continents, plates, mountains and
    # volcanoes; hills and craters smaller than the map can
    # show are added on top.
    elevation_map: str = ""

    # Impact craters (planet/craters.py): rate relative to
    # the Moon's (0 = none), surface age where no tectonic
    # field gives one (Myr), erosion time (Myr; 0 = never),
    # smallest crater the air lets through and simple ->
    # complex transition (m), bright rays.
    crater_density: float = 0.0
    surface_age: float = 4_000.0
    crater_erosion: float = 0.0
    crater_min_diameter: float = 0.0
    crater_transition: float = 15_000.0
    crater_rays: bool = False

    # Volcanoes (planet/volcanoes.py, with a tectonic
    # field): 0..1 how much the body builds them, and the
    # tallest its gravity allows (m).
    volcanism: float = 0.0
    volcano_max_height: float = 10_000.0

    # Erosion: rivers, lakes, deltas and glaciers from the
    # climate (planet/hydrology.py; needs a liquid), and
    # wind-blown dunes (planet/dunes.py).
    rivers: bool = False

    # Slopes carved into gullies and valleys by running water
    # (planet/erosion.py): 0 none (airless worlds), 1 rainy
    # (Earth, Titan); Mars's ancient valleys about half.
    gullies: float = 0.0

    dune_density: float = 0.0
    dune_amplitude: float = 50.0
    dune_wavelength: float = 1_500.0
    dune_linear: bool = False
    dune_max_latitude: float = 90.0
    dune_darkening: float = 0.0

    @property
    def min_elevation(
        self
    ) -> float:
        """Lower bound of the visible surface (0 under a liquid)."""

        if self.bands or self.has_liquid:
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
        self.climate = climate

        # Computed on first use (cache_key).
        self._cache_key = False

        self.shape = body_shape(settings)

        self.elevation_map = (
            load_elevation_map(settings.elevation_map)
            if settings.elevation_map and not settings.bands
            else None
        )

        # A measured surface needs no simulated one.
        self.field = field if self.elevation_map is None else None

        field = self.field

        self.craters = None

        # Rivers come with the climate they drain.
        self.hydrology = (
            getattr(climate, "hydrology", None)
            if settings.rivers and settings.has_liquid and not settings.bands
            else None
        )

        self.dunes = None

        if settings.dune_density > 0.0 and not settings.bands:

            self.dunes = Dunes(
                DuneSettings(
                    seed=settings.seed,
                    density=settings.dune_density,
                    amplitude=settings.dune_amplitude,
                    wavelength=max(settings.dune_wavelength, 50.0),
                    linear=settings.dune_linear,
                    max_latitude=settings.dune_max_latitude,
                    darkening=settings.dune_darkening
                ),
                settings.radius,
                climate
            )

            if not self.dunes.enabled:
                self.dunes = None

        self.volcanoes = None

        if field is not None and settings.volcanism > 0.0 and not settings.bands:

            volcanoes = Volcanoes(
                VolcanoSettings(
                    seed=settings.seed,
                    volcanism=settings.volcanism,
                    max_height=settings.volcano_max_height
                ),
                settings.radius,
                field
            )

            if volcanoes.enabled:
                self.volcanoes = volcanoes

        if settings.crater_density > 0.0 and not settings.bands:

            # Over a map, only craters too small for it to show.
            largest = (
                4.0 * self.elevation_map.resolution(settings.radius)
                if self.elevation_map is not None
                else CraterSettings().max_diameter
            )

            self.craters = Craters(
                CraterSettings(
                    seed=settings.seed,
                    density=settings.crater_density,
                    surface_age=settings.surface_age,
                    erosion_time=settings.crater_erosion,
                    min_diameter=settings.crater_min_diameter,
                    transition_diameter=settings.crater_transition,
                    rays=settings.crater_rays,
                    max_diameter=largest
                ),
                settings.radius
            )

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
        self._mottle = Perlin(seed + 5)

    @property
    def max_elevation(
        self
    ) -> float:
        """Upper bound of the surface height (horizon culling)."""

        rim = self.craters.max_rim if self.craters is not None else 0.0

        if self.volcanoes is not None:
            rim += self.volcanoes.max_height

        if self.dunes is not None:
            rim += self.dunes.max_height

        if self.elevation_map is not None:
            return self.elevation_map.max_height + self.settings.detail_height + rim

        if self.field is None:
            return self.settings.max_elevation + rim

        s = self.settings

        return (
            rim
            + max(self._field_max, 0.0)
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

        if s.has_liquid or s.bands:
            return s.min_elevation

        depth = self.craters.max_depth if self.craters is not None else 0.0

        if self.elevation_map is not None:
            return self.elevation_map.min_height - s.detail_height - depth

        if self.field is None:
            return s.min_elevation - depth

        return min(self._field_min, 0.0) - s.detail_height - self._coast_noise - 200.0 - depth

    @property
    def cache_key(
        self
    ) -> str | None:
        """
        Identifies everything the terrain is built from
        (settings, tectonic state, climate) for the disk
        cache; None when an input has no key.
        """

        if self._cache_key is not False:
            return self._cache_key

        field_key = getattr(self.field, "content_key", "") if self.field is not None else "-"
        climate_key = getattr(self.climate, "content_key", "") if self.climate is not None else "-"

        self._cache_key = (
            cache_key("terrain", self.settings, field_key, climate_key)
            if field_key and climate_key
            else None
        )

        return self._cache_key

    # -----------------------------------------------------
    # Shape
    # -----------------------------------------------------

    def base_height(
        self,
        directions: np.ndarray
    ) -> np.ndarray:
        """
        Height (m) of the body's shape above the reference
        radius (0 for a sphere); relief stands on it.
        """

        if self.shape is None:
            return np.zeros(len(directions))

        return self.shape.height(directions)

    @property
    def base_bounds(
        self
    ) -> tuple[float, float]:
        """Lowest and highest base heights (m)."""

        if self.shape is None:
            return 0.0, 0.0

        return self.shape.min_height, self.shape.max_height

    def elevation(
        self,
        directions: np.ndarray,
        spacing: float = 0.0
    ) -> np.ndarray:
        """
        Relief (m) above the body's shape (sea level).

        directions: (n, 3) unit vectors.
        spacing: distance between samples in meters (0 =
            full detail).
        """

        return self.elevation_and_water(directions, spacing)[0]

    def elevation_and_water(
        self,
        directions: np.ndarray,
        spacing: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        (elevation m, surface water 0..1: rivers, lakes and
        delta channels, or glaciers where frozen).
        """

        elevation = self._shape(directions, spacing)

        water = np.zeros(len(directions))

        # Active dune fields bury the craters under them.
        dunes = cover = None

        if self.dunes is not None:
            dunes, cover = self.dunes.apply(directions, spacing)

        # -------------------------------------------------
        # Craters, as many as the surface is old
        # -------------------------------------------------

        if self.craters is not None:

            craters = self.craters.height(
                directions,
                spacing,
                self.surface_age(directions)
            )

            if cover is not None:
                craters = craters * (1.0 - 0.9 * cover)

            elevation = elevation + craters

        # -------------------------------------------------
        # Erosion: rivers carve valleys, lakes fill basins,
        # deltas build out to sea; wind piles up dunes.
        # -------------------------------------------------

        if self.hydrology is not None:

            # Meander noise only where water might be.
            meander = np.zeros_like(directions)

            near = self.hydrology.near_water(directions)

            if np.any(near):

                meander[near] = meander_offsets(
                    directions[near],
                    self.hydrology.grid.cell_angle,
                    self.settings.seed
                )

            elevation, water = self.hydrology.apply(directions, elevation, spacing, meander)

        if dunes is not None:
            elevation = elevation + dunes * (1.0 - water)

        return elevation, water

    def _shape(
        self,
        directions: np.ndarray,
        spacing: float = 0.0
    ) -> np.ndarray:
        """
        The surface before impacts and erosion: continents,
        mountains, hills and volcanoes.
        """

        s = self.settings

        if s.bands:

            # Cloud tops of a giant: no relief on its shape
            # (the flattened ellipsoid).
            return np.zeros(len(directions))

        radius = s.radius

        def octaves(frequency):

            return octaves_for_spacing(
                radius / frequency,
                spacing,
                _MAX_OCTAVES
            )

        if self.elevation_map is not None:
            return self._carve(self._measured(directions, octaves, spacing), directions, spacing, self._measured_relief)

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

        # -------------------------------------------------
        # Gullies and valleys cut by running water
        # -------------------------------------------------

        elevation = self._carve(elevation, directions, spacing, self._relief_only, land)

        # -------------------------------------------------
        # Volcanoes where magma reaches the surface
        # -------------------------------------------------

        if self.volcanoes is not None:

            # A volcano grows until its summit reaches gravity's
            # limit, whatever it stands on (a hot spot on a
            # mountain range adds less).
            volcano = self.volcanoes.height(directions, spacing)

            elevation += np.minimum(
                volcano,
                np.maximum(self.settings.volcano_max_height - elevation, 0.0)
            )

        return elevation

    def _carve(
        self,
        elevation: np.ndarray,
        directions: np.ndarray,
        spacing: float,
        relief,
        land=1.0
    ) -> np.ndarray:
        """
        Gullies on the slopes (planet/erosion.py), where the
        samples are fine enough to show them and the body has
        had running water; not under seas.
        """

        s = self.settings

        if s.gullies <= 0.0 or (spacing > 0.0 and spacing * 4.0 > GULLY_WAVELENGTHS[0]):
            return elevation

        # Over a map, only valleys finer than it shows.
        largest = float("inf")

        if self.elevation_map is not None:

            detail = self.elevation_map.detail

            largest = (detail or self.elevation_map).resolution(s.radius)

            if largest < GULLY_WAVELENGTHS[-1] or spacing * 4.0 > largest:
                return elevation

        gradient = relief_gradient(relief, directions, s.radius, 2.0 * min(GULLY_WAVELENGTHS[0], largest))

        cut = gullies(directions, s.radius, gradient, spacing, s.gullies, s.seed, largest)

        if s.has_liquid:
            cut = cut * _smoothstep(-20.0, 60.0, elevation)

        return elevation + cut * land

    def _relief_only(
        self,
        directions: np.ndarray,
        spacing: float
    ) -> np.ndarray:
        """The relief before carving, at a coarse `spacing` (its gradient)."""

        settings = self.settings

        # (No carving inside, no volcanoes: the larger forms.)
        saved, self.settings = settings, _without_gullies(settings)

        volcanoes, self.volcanoes = self.volcanoes, None

        try:
            return self._shape(directions, spacing)
        finally:
            self.settings = saved
            self.volcanoes = volcanoes

    def _measured_relief(
        self,
        directions: np.ndarray,
        spacing: float
    ) -> np.ndarray:

        radius = self.settings.radius

        def octaves(frequency):
            return octaves_for_spacing(radius / frequency, spacing, _MAX_OCTAVES)

        return self._measured(directions, octaves, spacing)

    def _measured(
        self,
        directions: np.ndarray,
        octaves,
        spacing: float = 0.0
    ) -> np.ndarray:
        """
        A real map's heights, with hills finer than its
        samples (less under seas).
        """

        s = self.settings

        # (Sharper data where the samples are finer than the
        # map: planet/maps.py.)
        elevation = self.elevation_map.sample(directions, spacing, s.radius)

        detail_frequency = s.mountain_frequency * _DETAIL_FREQUENCY_SCALE

        if s.detail_height > 0.0:

            land = _smoothstep(-100.0, 200.0, elevation) if s.has_liquid else 1.0

            elevation = elevation + (
                fbm(self._detail, directions * detail_frequency, octaves(detail_frequency))
                * s.detail_height
                * (0.3 + 0.7 * land)
            )

        return elevation

    def surface_age(
        self,
        directions: np.ndarray
    ):
        """
        Surface age (Myr) per point: the tectonic field's
        crust age, else the planet's.
        """

        if self.field is None:
            return self.settings.surface_age

        return self.field.sample("age", directions)

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

        # Peaks where crust was recently uplifted (young
        # belts); rugged foothills wherever land stands high.
        orogeny = field.sample("orogeny", directions)

        if getattr(field, "regime", "plate_tectonics") != "plate_tectonics":

            # Other regimes have no continents and no ranges
            # beyond what they uplift themselves (Venus's
            # tesserae, Io's blocks); craters roughen the
            # rest.
            land = np.ones_like(elevation)

            return elevation, land, _smoothstep(200.0, 2_000.0, orogeny)

        land = _smoothstep(-100.0, 200.0, elevation)

        # Ranges along the belts, and on the highest plateaus
        # (not on all land: most continental crust is low;
        # checked against ETOPO1 with tools/compare_terrain.py).
        mask = np.maximum(
            _smoothstep(400.0, 2_500.0, orogeny),
            _smoothstep(1_500.0, 4_000.0, elevation)
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

        # Mottled ground on the other regimes' bare worlds:
        # regolith and rock vary in brightness at every scale
        # (from ~500 km patches down to the vertex spacing).
        if getattr(field, "regime", "plate_tectonics") != "plate_tectonics":

            data[:, 2] = np.clip(
                data[:, 2] + 0.12 * fbm(self._mottle, directions * (self.settings.radius / 500_000.0), 6),
                0.0,
                None
            )

        # Dune fields: often darker sand (Mars's basalt,
        # Titan's organics), visible even from orbit.
        if self.dunes is not None and self.settings.dune_darkening > 0.0:

            _, cover = self.dunes.apply(directions, 1e12)

            data[:, 2] = np.clip(data[:, 2] - self.settings.dune_darkening * cover, 0.0, None)

        # Young craters' bright ejecta and rays lighten the
        # crust (the mineral palette colors by it), past the
        # brightest highlands (fresh, unweathered rock).
        if self.craters is not None and self.settings.crater_rays:

            data[:, 2] = np.clip(
                data[:, 2] + 0.6 * self.craters.brightness(directions, self.surface_age(directions)),
                0.0,
                RAY_BRIGHTNESS
            )

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


def _without_gullies(
    settings: TerrainSettings
) -> TerrainSettings:

    return replace(settings, gullies=0.0)


def _smoothstep(
    edge0: float,
    edge1: float,
    x
) -> np.ndarray:

    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)

    return t * t * (3.0 - 2.0 * t)


def body_shape(
    settings: TerrainSettings
) -> BodyShape | None:
    """The settings' shape with the flattening folded in, or None for a sphere."""

    f = min(max(float(settings.oblateness), 0.0), 0.5)

    shape = settings.shape

    if shape is None and f == 0.0:
        return None

    shape = shape or ShapeSettings()

    if f > 0.0:
        shape = replace(shape, axes=(shape.axes[0], shape.axes[1] * (1.0 - f), shape.axes[2]))

    if shape.spherical:
        return None

    return BodyShape(shape, settings.radius)


def oblate_offset(
    directions: np.ndarray,
    radius: float,
    oblateness: float
) -> np.ndarray:
    """
    Height (m, <= 0) of an oblate ellipsoid's surface below
    the equatorial radius along unit directions (y = the
    rotation axis).
    """

    f = min(max(float(oblateness), 0.0), 0.5)

    if f == 0.0:
        return np.zeros(len(directions))

    sin2 = np.clip(directions[:, 1], -1.0, 1.0) ** 2

    polar = 1.0 - f

    return radius * (polar / np.sqrt(polar * polar * (1.0 - sin2) + sin2) - 1.0)
