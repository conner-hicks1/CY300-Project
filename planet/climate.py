import math
import time

from dataclasses import dataclass

import numpy as np

from planet.sphere_grid import SphereGrid, sphere_grid


# =========================================================
# Climate
# =========================================================
#
# An annual-mean climate on the global grid
# (planet/sphere_grid.py), computed from the planet's
# elevation in well under a second:
#
#   Temperature
#       Annual-mean sunlight by latitude for the axial tilt
#       (North 1975: S(x) = 1 + s2 P2(x)), turned into a
#       sea-level temperature, spread by heat transport
#       (diffusion; stronger over oceans, which carry heat
#       poleward and moderate coasts), then cooled with
#       height (6.5 C per km).
#
#   Wind
#       Earth's three-cell circulation: trade winds toward
#       the equator from the east, westerlies toward the
#       poles, polar easterlies.
#
#   Rain
#       Moisture evaporates from oceans (more where warm),
#       is carried downwind step by step, and rains out
#       where air rises: the equatorial convergence zone,
#       the polar fronts, and wherever wind blows uphill
#       (so mountains get a wet windward side and a dry rain
#       shadow). Air sinking around 30 degrees suppresses
#       rain: the desert belts. Cold air holds less water.
#
# Units: C, mm / year, meters. The planet's rotation axis
# is +Y in planet space (as everywhere in planet/).

LAPSE_RATE = 6.5e-3                     # C per m of altitude

# Sea-level temperature (C) from annual sunlight S
# (1 = global mean): ~32 C at Earth's equator before heat transport (S = 1.24),
# ~-20 C at its poles (S = 0.52); ~15 C global mean.
_TEMPERATURE_PER_SUNLIGHT = 72.8
_TEMPERATURE_AT_ZERO_SUNLIGHT = -58.0

# Zonal (east, +) and meridional (poleward, +) surface wind
# by |latitude| in degrees.
_WIND_LATITUDES = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
_WIND_EAST = np.array([-1.0, -0.8, 0.0, 1.0, 0.0, -0.5, 0.0])
_WIND_POLEWARD = np.array([0.0, -0.4, 0.0, 0.3, 0.0, -0.3, 0.0])

# Rain-out multipliers by |latitude|: wet where air rises
# (equator, ~60 deg), dry where it sinks (~30 deg, poles).
_RAIN_LATITUDES = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 60.0, 75.0, 90.0])
_RAIN_FACTOR = np.array([2.2, 1.6, 0.55, 0.3, 0.8, 1.4, 0.7, 0.4])


@dataclass(frozen=True, slots=True)
class ClimateSettings:

    # Degrees (Earth: 23.44). Higher tilt evens out the
    # annual sunlight between equator and poles.
    axial_tilt: float = 23.44

    # Added to every temperature (C): a warmer or colder
    # world.
    temperature_offset: float = 0.0

    # Scales evaporation, so overall rainfall (1 = Earth,
    # ~1,000 mm / year average).
    humidity: float = 1.0

    # Grid cells per cube-face edge (climate is smooth; a
    # coarse grid is plenty).
    resolution: int = 64


@dataclass(slots=True)
class ClimateState:

    # Per cell.
    sea_temperature: np.ndarray     # C at sea level
    temperature: np.ndarray         # C at the surface
    precipitation: np.ndarray       # mm / year
    wind: np.ndarray                # (cells, 3) unit-ish surface wind

    compute_seconds: float = 0.0


# =========================================================
# Building Blocks
# =========================================================

def annual_sunlight(
    sin_latitude,
    axial_tilt_degrees: float
) -> np.ndarray:
    """Annual-mean top-of-atmosphere sunlight, 1 = planet average."""

    x = np.asarray(sin_latitude, dtype=np.float64)

    cos_tilt = math.cos(math.radians(axial_tilt_degrees))

    s2 = -(5.0 / 16.0) * (3.0 * cos_tilt * cos_tilt - 1.0)

    p2 = 0.5 * (3.0 * x * x - 1.0)

    return 1.0 + s2 * p2


def sea_level_temperature(
    sin_latitude,
    settings: ClimateSettings
) -> np.ndarray:
    """Temperature (C) from sunlight alone, before heat transport."""

    return (
        _TEMPERATURE_AT_ZERO_SUNLIGHT
        + _TEMPERATURE_PER_SUNLIGHT * annual_sunlight(sin_latitude, settings.axial_tilt)
        + settings.temperature_offset
    )


def prevailing_wind(
    directions: np.ndarray
) -> np.ndarray:
    """(n, 3) surface wind (tangent, magnitude ~0-1) by latitude belt."""

    directions = np.asarray(directions, dtype=np.float64)

    sin_latitude = np.clip(directions[:, 1], -1.0, 1.0)

    latitude = np.degrees(np.arcsin(np.abs(sin_latitude)))

    east_speed = np.interp(latitude, _WIND_LATITUDES, _WIND_EAST)
    poleward_speed = np.interp(latitude, _WIND_LATITUDES, _WIND_POLEWARD)

    east, north = local_axes(directions)

    hemisphere = np.where(sin_latitude >= 0.0, 1.0, -1.0)

    return (
        east * east_speed[:, None]
        + north * (poleward_speed * hemisphere)[:, None]
    )


def local_axes(
    directions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Unit east and north tangents (poles: an arbitrary frame)."""

    axis = np.array([0.0, 1.0, 0.0])

    east = np.cross(axis, directions)

    length = np.linalg.norm(east, axis=1, keepdims=True)

    fallback = np.array([1.0, 0.0, 0.0])

    east = np.where(length > 1e-9, east / np.maximum(length, 1e-12), fallback)

    north = np.cross(directions, east)

    return east, north


def water_capacity(
    temperature
) -> np.ndarray:
    """
    Relative amount of water air delivers as rain, by
    temperature. Vapor capacity grows ~7% per C
    (Clausius-Clapeyron), but rainfall does not scale that
    steeply (polar regions still get 100-300 mm a year from
    moisture blown in): ~4.5% per C gives Earth's ~12x
    tropics-to-poles contrast.
    """

    return np.exp(0.045 * (np.asarray(temperature) - 15.0))


# =========================================================
# Model
# =========================================================

class ClimateModel:

    # Advection iterations; each carries moisture about one
    # cell (~160 km) downwind, so the furthest inland air
    # comes from ~ITERATIONS cells upwind.
    ITERATIONS = 60

    # Heat transport smoothing passes over land and ocean.
    LAND_DIFFUSION_PASSES = 6
    OCEAN_DIFFUSION_PASSES = 30

    def __init__(
        self,
        settings: ClimateSettings
    ):

        self.settings = settings

        self.grid: SphereGrid = sphere_grid(settings.resolution)

    def compute(
        self,
        elevation: np.ndarray
    ) -> ClimateState:
        """
        elevation: per grid cell, meters above sea level
            (negative = ocean floor).
        """

        started = time.perf_counter()

        grid = self.grid
        s = self.settings

        directions = grid.directions
        neighbors = grid.neighbors

        elevation = np.asarray(elevation, dtype=np.float64)

        ocean = elevation < 0.0

        height = np.maximum(elevation, 0.0)

        sin_latitude = directions[:, 1]

        # -------------------------------------------------
        # Temperature
        # -------------------------------------------------

        sea = sea_level_temperature(sin_latitude, s)

        # Heat transport: everything mixes a little, oceans
        # a lot (currents carry heat poleward and keep
        # coasts mild).
        for _ in range(self.LAND_DIFFUSION_PASSES):
            sea = sea + 0.5 * (sea[neighbors].mean(axis=1) - sea)

        for _ in range(self.OCEAN_DIFFUSION_PASSES):

            mixed = sea + 0.5 * (sea[neighbors].mean(axis=1) - sea)

            sea = np.where(ocean, mixed, sea)

        # Continental interiors far from the sea run colder
        # (in the annual mean, mostly through cold winters
        # at high latitudes).
        continentality = self._distance_inland(ocean)

        sea = sea - 4.0 * np.clip(continentality / 15.0, 0.0, 1.0) * np.abs(sin_latitude)

        temperature = sea - LAPSE_RATE * height

        # -------------------------------------------------
        # Wind
        # -------------------------------------------------

        wind = prevailing_wind(directions)

        speed = np.linalg.norm(wind, axis=1)

        # Where the moisture in each cell came from: one
        # cell upwind.
        step = grid.cell_angle

        upwind = directions - wind * step

        upwind /= np.linalg.norm(upwind, axis=1, keepdims=True)

        upwind_padded = grid.pad

        # Wind blowing uphill lifts air (orographic rain);
        # downhill it sinks and dries (rain shadow).
        upwind_height = grid.sample(grid.pad(height), upwind)

        climb = (height - upwind_height) / 1_000.0                # km per cell

        lift = np.clip(climb * speed, -2.0, 4.0)

        latitude = np.degrees(np.arcsin(np.abs(np.clip(sin_latitude, -1.0, 1.0))))

        rain_out = 0.06 * np.interp(latitude, _RAIN_LATITUDES, _RAIN_FACTOR)

        rain_out = rain_out * np.clip(1.0 + 1.5 * lift, 0.15, 6.0)

        # -------------------------------------------------
        # Moisture transport
        # -------------------------------------------------

        capacity = water_capacity(temperature)

        evaporation = np.where(
            ocean,
            capacity,
            0.6 * capacity                          # plants, lakes, soil (recycling)
        ) * s.humidity

        moisture = np.zeros(grid.cell_count)
        precipitation = np.zeros(grid.cell_count)

        for _ in range(self.ITERATIONS):

            # Carry moisture one cell downwind.
            moisture = grid.sample(upwind_padded(moisture), upwind)

            moisture = moisture + evaporation * 0.1

            rain = moisture * rain_out

            # Air holds only so much: the excess rains out.
            excess = np.maximum(moisture - rain - capacity * 2.0, 0.0)

            rain = rain + excess

            moisture = moisture - rain

            precipitation += rain

        precipitation /= self.ITERATIONS

        # Scale to Earth-like totals: average ~1,000 mm a
        # year times the humidity setting.
        mean = float(precipitation.mean())

        if mean > 0.0:
            precipitation *= 1_000.0 * s.humidity / mean

        return ClimateState(
            sea_temperature=(sea).astype(np.float32),
            temperature=temperature.astype(np.float32),
            precipitation=precipitation.astype(np.float32),
            wind=wind.astype(np.float32),
            compute_seconds=time.perf_counter() - started
        )

    def _distance_inland(
        self,
        ocean: np.ndarray
    ) -> np.ndarray:
        """Cells to the nearest ocean cell (0 at sea), up to 20."""

        distance = np.where(ocean, 0.0, 20.0)

        neighbors = self.grid.neighbors

        for _ in range(20):

            distance = np.minimum(distance, distance[neighbors].min(axis=1) + 1.0)

        return distance


# =========================================================
# Biomes
# =========================================================
#
# Same classes and thresholds as terrainBiome() in
# assets/shaders/include/terrain.glsl (the Biomes view).

BIOMES = (
    "Ice",
    "Tundra",
    "Taiga",
    "Temperate forest",
    "Grassland",
    "Desert",
    "Savanna",
    "Tropical rainforest",
)


def wetness(
    precipitation,
    temperature
) -> np.ndarray:
    """Rain relative to evaporation (< 0.3 desert, > 1.2 forest)."""

    return np.asarray(precipitation) / (300.0 + 30.0 * np.maximum(temperature, 0.0))


def classify_biomes(
    temperature,
    precipitation
) -> np.ndarray:
    """Index into BIOMES per point."""

    t = np.asarray(temperature, dtype=np.float64)
    w = wetness(precipitation, t)

    biome = np.where(t < 20.0, np.where(w > 1.2, 3, 4), np.where(w > 1.5, 7, 6))
    biome = np.where(t < 6.0, np.where(w > 0.9, 2, 4), biome)
    biome = np.where((t >= 0.0) & (w < 0.3), 5, biome)
    biome = np.where(t < 0.0, 1, biome)
    biome = np.where(t < -5.0, 0, biome)

    return biome.astype(np.int32)


# =========================================================
# Field (what the terrain reads)
# =========================================================

@dataclass(frozen=True, slots=True, eq=False)
class ClimateField:

    grid: SphereGrid

    sea_temperature: np.ndarray     # padded, C
    precipitation: np.ndarray       # padded, mm / year

    version: int

    # Summary for the editor (over land).
    mean_temperature: float = 0.0
    mean_precipitation: float = 0.0
    biome_fractions: tuple[float, ...] = ()

    @classmethod
    def from_state(
        cls,
        grid: SphereGrid,
        state: ClimateState,
        version: int,
        land: np.ndarray | None = None
    ) -> "ClimateField":
        """land: per-cell mask for the land statistics."""

        if land is None or not np.any(land):
            land = np.ones(grid.cell_count, dtype=bool)

        biomes = classify_biomes(state.temperature[land], state.precipitation[land])

        fractions = np.bincount(biomes, minlength=len(BIOMES)) / max(len(biomes), 1)

        return cls(
            grid=grid,
            sea_temperature=grid.pad(state.sea_temperature),
            precipitation=grid.pad(state.precipitation),
            version=version,
            mean_temperature=float(state.temperature[land].mean()),
            mean_precipitation=float(state.precipitation[land].mean()),
            biome_fractions=tuple(float(f) for f in fractions)
        )

    def surface(
        self,
        directions: np.ndarray,
        elevation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        (temperature C, precipitation mm/yr) at the given
        points. Temperature follows each point's own height
        (lapse rate), so peaks are colder than the climate
        cell they sit in.
        """

        sea = self.grid.sample(self.sea_temperature, directions)

        temperature = sea - LAPSE_RATE * np.maximum(elevation, 0.0)

        return temperature, np.maximum(self.grid.sample(self.precipitation, directions), 0.0)


def fallback_surface(
    directions: np.ndarray,
    elevation: np.ndarray,
    moisture: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Temperature and precipitation without a climate model:
    sunlight by latitude (Earth tilt), lapse rate, and
    rainfall from a moisture noise value in [0, 1].
    """

    sea = sea_level_temperature(np.asarray(directions)[:, 1], ClimateSettings())

    temperature = sea - LAPSE_RATE * np.maximum(elevation, 0.0)

    precipitation = 100.0 + 2_400.0 * np.asarray(moisture) ** 1.5

    return temperature, precipitation
