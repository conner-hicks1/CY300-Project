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

LAPSE_RATE = 6.5e-3                     # C per m of altitude (Earth)

STEFAN_BOLTZMANN = 5.670374e-8
KELVIN = 273.15

# Heat transport as a diffusivity on the unit sphere
# (W / m^2 per K, as in North's energy-balance models):
# Earth's air and ocean each carry about half. Air scales
# with sqrt(pressure): Venus's thick air evens its
# temperatures out, Mars's thin air barely moves heat.
_AIR_DIFFUSIVITY = 0.08
_OCEAN_DIFFUSIVITY = 0.10

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

    # ---- Physics (Earth by default; planet/bodies.py
    # derives them for other bodies) ----

    # Starlight arriving at the body, W / m^2 (Earth: 1361).
    stellar_flux: float = 1361.0

    # Fraction of sunlight reflected (clouds, ice, ground).
    bond_albedo: float = 0.306

    # Infrared optical depth of the atmosphere (greenhouse):
    # the surface radiates through it with emissivity
    # 1 / (1 + 0.75 tau). Earth ~0.85, Venus ~140, Moon 0.
    greenhouse_depth: float = 0.85

    # Surface pressure, bar: how well the air carries heat
    # around the planet (0 = airless: no transport).
    surface_pressure: float = 1.014

    # How much colder per meter of height (Earth 6.5e-3).
    lapse_rate: float = 6.5e-3

    eccentricity: float = 0.0167

    # A liquid ocean carries heat too (and evaporates).
    ocean: bool = True

    # Where the seas' liquid freezes and boils (C) at this
    # pressure (planet/phases.py liquid_range); None without
    # a liquid (or for lava, molten from volcanic heat).
    # Seas boil away when they average hotter than boiling
    # (a runaway greenhouse: the basins dry out).
    liquid_freezing: float | None = -1.9
    liquid_boiling: float | None = 100.0


@dataclass(slots=True)
class ClimateState:

    # Per cell.
    sea_temperature: np.ndarray     # C at sea level
    temperature: np.ndarray         # C at the surface
    precipitation: np.ndarray       # mm / year
    wind: np.ndarray                # (cells, 3) unit-ish surface wind

    # The seas: "none", "liquid", "partly frozen", "frozen"
    # or "boiled away"; and the share of sea cells colder
    # than the freezing point.
    liquid_state: str = "none"
    frozen_fraction: float = 0.0

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

        sea = self._sea_temperatures(sin_latitude, ocean, s.ocean)

        # -------------------------------------------------
        # The seas' phase
        # -------------------------------------------------

        liquid_state = "none"
        frozen_fraction = 0.0

        if s.liquid_boiling is not None and np.any(ocean):

            if float(sea[ocean].mean()) > s.liquid_boiling:

                # Boiled away: no seas, no ocean heat transport
                # or evaporation; the basins are dry ground.
                liquid_state = "boiled away"

                ocean = np.zeros_like(ocean)

                sea = self._sea_temperatures(sin_latitude, ocean, False)

            else:

                freezing = s.liquid_freezing if s.liquid_freezing is not None else -np.inf

                frozen_fraction = float(np.mean(sea[ocean] < freezing))

                liquid_state = (
                    "frozen" if frozen_fraction > 0.98
                    else "partly frozen" if frozen_fraction > 0.02
                    else "liquid"
                )

        temperature = sea - s.lapse_rate * height

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
            ocean & s.ocean,
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
            liquid_state=liquid_state,
            frozen_fraction=frozen_fraction,
            compute_seconds=time.perf_counter() - started
        )

    def _sea_temperatures(
        self,
        sin_latitude: np.ndarray,
        ocean: np.ndarray,
        has_ocean: bool
    ) -> np.ndarray:
        """Sea-level temperature (C) per cell."""

        sea = self._energy_balance(sin_latitude, ocean)

        # Continental interiors far from the sea run colder
        # (in the annual mean, mostly through cold winters
        # at high latitudes).
        if has_ocean and np.any(ocean):

            continentality = self._distance_inland(ocean)

            sea = sea - 4.0 * np.clip(continentality / 15.0, 0.0, 1.0) * np.abs(sin_latitude)

        return sea + self.settings.temperature_offset

    def sea_level_temperatures(
        self,
        elevation: np.ndarray
    ) -> np.ndarray:
        """Energy-balance temperatures only (C per cell; for calibration)."""

        elevation = np.asarray(elevation, dtype=np.float64)

        return self._energy_balance(self.grid.directions[:, 1], elevation < 0.0)

    def _energy_balance(
        self,
        sin_latitude: np.ndarray,
        ocean: np.ndarray
    ) -> np.ndarray:
        """
        Annual-mean sea-level temperature (C) per cell from
        radiative balance:

            absorbed sunlight  Q (1 - A)
            = emitted heat     e sigma T^4,  e = 1 / (1 + 0.75 tau)
              - heat brought in by transport  D (mean(T_neighbors) - T)

        solved to a steady state by damped Newton steps.
        Q is the annual mean by latitude for the tilt, with
        a 1 / sqrt(1 - e^2) boost for an eccentric orbit.
        """

        s = self.settings
        grid = self.grid
        neighbors = grid.neighbors

        eccentricity = min(max(s.eccentricity, 0.0), 0.99)

        sunlight = (
            s.stellar_flux / 4.0
            * annual_sunlight(sin_latitude, s.axial_tilt)
            / math.sqrt(1.0 - eccentricity ** 2)
        )

        absorbed = sunlight * (1.0 - min(max(s.bond_albedo, 0.0), 0.99))

        emissivity = 1.0 / (1.0 + 0.75 * max(s.greenhouse_depth, 0.0))

        # Transport: air (none in vacuum) plus ocean currents
        # over water, as a diffusivity; per cell it is the
        # discrete Laplacian's 4 / h^2 (h: cell spacing).
        grid_scale = 4.0 / grid.cell_angle ** 2

        transport = (
            _AIR_DIFFUSIVITY * math.sqrt(max(s.surface_pressure, 0.0))
            + (_OCEAN_DIFFUSIVITY * np.asarray(ocean, dtype=np.float64) if s.ocean else 0.0)
        ) * grid_scale

        transport = np.broadcast_to(np.asarray(transport, dtype=np.float64), absorbed.shape)

        # Without transport every cell is in local radiative
        # equilibrium.
        if not np.any(transport > 0.0):
            return np.maximum((absorbed / (emissivity * STEFAN_BOLTZMANN)) ** 0.25, 3.0) - KELVIN

        # Heat flows between neighbors through each shared
        # edge (symmetric, so the linear solves below can use
        # conjugate gradients).
        conductance = (transport[:, None] + transport[neighbors]) * 0.125

        total_conductance = conductance.sum(axis=1)

        def laplacian(values):
            return total_conductance * values - (conductance * values[neighbors]).sum(axis=1)

        # Newton's method from the planet-wide equilibrium;
        # each step solves the linearized balance exactly
        # (a relaxation would take thousands of sweeps to
        # spread heat across a thick atmosphere).
        kelvin = np.full(
            absorbed.shape,
            max((absorbed.mean() / (emissivity * STEFAN_BOLTZMANN)) ** 0.25, 3.0)
        )

        for _ in range(40):

            emitted = emissivity * STEFAN_BOLTZMANN * kelvin ** 4

            residual = absorbed - emitted - laplacian(kelvin)

            radiative_slope = 4.0 * emitted / kelvin

            step = _conjugate_gradient(
                lambda v: radiative_slope * v + laplacian(v),
                residual,
                radiative_slope + total_conductance
            )

            # Keep far-off first guesses from overshooting.
            step = np.clip(step, -0.5 * kelvin, kelvin)

            kelvin = np.maximum(kelvin + step, 3.0)

            if np.abs(step).max() < 0.01:
                break

        return kelvin - KELVIN

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


def _conjugate_gradient(
    apply,
    rhs: np.ndarray,
    diagonal: np.ndarray,
    tolerance: float = 1e-4,
    absolute_tolerance: float = 0.01,
    max_iterations: int = 2000
) -> np.ndarray:
    """
    Solve A x = rhs for a symmetric positive-definite A
    (given as a function), Jacobi-preconditioned.
    """

    x = np.zeros_like(rhs)
    r = rhs.copy()
    z = r / diagonal
    d = z.copy()
    rz = float(r @ z)

    limit = max(tolerance * float(np.abs(rhs).max()), absolute_tolerance)

    for _ in range(max_iterations):

        if np.abs(r).max() < limit:
            break

        ad = apply(d)

        alpha = rz / float(d @ ad)

        x += alpha * d
        r -= alpha * ad

        z = r / diagonal

        rz_next = float(r @ z)

        d = z + (rz_next / rz) * d

        rz = rz_next

    return x


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

    # C per m of height (the planet's, from ClimateSettings).
    lapse_rate: float = LAPSE_RATE

    # Summary for the editor (over land).
    mean_temperature: float = 0.0
    mean_precipitation: float = 0.0
    biome_fractions: tuple[float, ...] = ()

    # Whole planet (land and sea), C.
    min_temperature: float = 0.0
    max_temperature: float = 0.0
    global_mean_temperature: float = 0.0

    # The seas (see ClimateState).
    liquid_state: str = "none"
    frozen_fraction: float = 0.0

    # Rivers, lakes, deltas and glaciers draining this
    # climate's rain (planet/hydrology.py), or None.
    hydrology: object = None

    @property
    def liquid_boiled(
        self
    ) -> bool:

        return self.liquid_state == "boiled away"

    @classmethod
    def from_state(
        cls,
        grid: SphereGrid,
        state: ClimateState,
        version: int,
        land: np.ndarray | None = None,
        lapse_rate: float = LAPSE_RATE
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
            biome_fractions=tuple(float(f) for f in fractions),
            lapse_rate=lapse_rate,
            min_temperature=float(state.temperature.min()),
            max_temperature=float(state.temperature.max()),
            global_mean_temperature=float(state.temperature.mean()),
            liquid_state=state.liquid_state,
            frozen_fraction=state.frozen_fraction
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

        temperature = sea - self.lapse_rate * np.maximum(elevation, 0.0)

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
