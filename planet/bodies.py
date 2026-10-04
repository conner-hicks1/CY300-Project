import json
import math

from dataclasses import dataclass, field
from pathlib import Path

from ecs.components import (
    AtmosphereComponent,
    BodyComponent,
    ClimateComponent,
    OrbitComponent,
    PlanetComponent,
    RingsComponent,
    TectonicsComponent
)


# =========================================================
# Body Profiles
# =========================================================
#
# One JSON file per body in assets/bodies/ (format 1) with
# its real physical data: size and mass, rotation, tilt,
# orbit and star, surface temperature and liquid, surface
# colors, terrain relief, atmosphere (pressure, gases,
# aerosols), geology and climate. Values for the solar
# system come from NASA's planetary fact sheets.
#
# A profile turns into the planet entity's components
# (components_for): Planet, Body, and when they apply
# Atmosphere, Climate and Tectonics. Physical quantities
# the engine needs but the files do not list are derived
# here (surface gravity, sunlight, equilibrium temperature,
# atmospheric scale height, light scattering).

PROFILE_FORMAT = 1

BODIES_DIRECTORY = Path("assets/bodies")

from graphics.rings import flat_bands
from planet.phases import ICES, SUBSTANCES, frost_point
from planet.regimes import DEFAULT_TIME_STEPS
from planet.shape import MAX_BASINS, flat_basins
from planet.volcanoes import max_volcano_height

GRAVITATIONAL_CONSTANT = 6.674e-11
GAS_CONSTANT = 8.314                            # J / (mol K)
STEFAN_BOLTZMANN = 5.670374e-8
SOLAR_CONSTANT = 1361.0                         # W / m^2 at 1 AU
SOLAR_RADIUS_M = 6.957e8
AU_M = 1.495978707e11

KINDS = ("terrestrial", "moon", "dwarf", "gas_giant", "ice_giant", "asteroid", "comet")

KELVIN = 273.15

# Frost point of a body without ice caps (C).
NO_FROST_C = -1000.0
LIQUIDS = ("none", "water", "methane", "lava")
PALETTES = ("biomes", "mineral", "bands")
REGIMES = (
    "plate_tectonics",
    "stagnant_lid",
    "episodic_resurfacing",
    "heat_pipe",
    "ice_shell",
    "none",
)

# Molar mass (g / mol), refractivity (n - 1 at STP, visible)
# and specific heat (J / kg K, near room temperature) of
# common atmospheric gases. Rayleigh scattering per
# molecule goes with refractivity squared.
GASES: dict[str, tuple[float, float, float]] = {
    "N2": (28.014, 2.98e-4, 1040.0),
    "O2": (31.998, 2.72e-4, 918.0),
    "Ar": (39.948, 2.81e-4, 520.0),
    "CO2": (44.009, 4.50e-4, 844.0),
    "CH4": (16.043, 4.44e-4, 2220.0),
    "H2": (2.016, 1.39e-4, 14300.0),
    "He": (4.003, 3.50e-5, 5193.0),
    "CO": (28.010, 3.38e-4, 1040.0),
    "SO2": (64.066, 6.86e-4, 640.0),
    "H2O": (18.015, 2.56e-4, 1860.0),
    "Ne": (20.180, 6.70e-5, 1030.0),
}

# Latent heat of condensing water makes rising moist air
# cool slower than dry air (Earth: 6.5 vs 9.8 C per km).
MOIST_LAPSE_FACTOR = 0.66

# Below ~this pressure (bar) the air barely sets the ground
# temperature (it is set by sunlight and the ground's own
# radiation), so the lapse rate fades out: P / (P + this).
LAPSE_PRESSURE_SCALE = 0.02

_AIR_REFRACTIVITY = 2.93e-4

EARTH_SURFACE_TEMPERATURE_K = 288.0
EARTH_PRESSURE_BAR = 1.01325

# Earth's Rayleigh coefficients at sea level (1 / Mm), the
# reference every atmosphere is scaled from.
EARTH_RAYLEIGH = (5.802, 13.558, 33.1)
EARTH_OZONE = (0.650, 1.881, 0.085)

# Global mean temperature the climate model produces with
# no offset (planet/climate.py); presets shift it to their
# real mean until the model is driven by sunlight itself.
CLIMATE_MODEL_MEAN_C = 13.0


class ProfileError(ValueError):
    """A body profile file that is missing or malformed."""


@dataclass(frozen=True, slots=True)
class Aerosols:

    scattering_per_Mm: float
    absorption_per_Mm: float
    scale_height_km: float
    anisotropy: float

    # Per-channel (red, green, blue) multipliers: colored
    # dust and haze absorb and scatter blue differently.
    scattering_tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
    absorption_tint: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # A deck centered this high (km); 0 = thickest at the
    # ground.
    altitude_km: float = 0.0


@dataclass(frozen=True, slots=True)
class AtmosphereProfile:

    surface_pressure_bar: float

    # (gas, mole fraction), largest first.
    composition: tuple[tuple[str, float], ...]

    aerosols: Aerosols | None
    ozone: bool = False

    # Weather clouds: (coverage, altitude km, optical depth,
    # feature size km, color), or None.
    clouds: tuple | None = None

    @property
    def molar_mass(
        self
    ) -> float:
        """Mean molar mass, g / mol."""

        total = sum(fraction for _, fraction in self.composition)

        return sum(GASES[gas][0] * fraction for gas, fraction in self.composition) / total

    @property
    def scattering_factor(
        self
    ) -> float:
        """Rayleigh scattering per molecule relative to Earth's air."""

        total = sum(fraction for _, fraction in self.composition)

        refractivity_squared = sum(
            GASES[gas][1] ** 2 * fraction
            for gas, fraction in self.composition
        ) / total

        return refractivity_squared / _AIR_REFRACTIVITY ** 2

    def specific_heat(
        self,
        temperature_k: float = 300.0
    ) -> float:
        """
        Mass-weighted heat capacity, J / kg K. Heavy molecules
        (CO2) gain heat capacity when hot: ~(T / 300)^0.3.
        """

        mass = [(GASES[gas][0] * fraction, gas) for gas, fraction in self.composition]

        total = sum(m for m, _ in mass)

        heat = sum(m * GASES[gas][2] for m, gas in mass) / total

        return heat * (max(temperature_k, 50.0) / 300.0) ** (0.3 if self.molar_mass > 35.0 else 0.0)


@dataclass(frozen=True, slots=True)
class BodyProfile:

    id: str
    name: str
    kind: str
    orbits: str
    description: str

    radius_km: float
    mass_kg: float
    rotation_hours: float
    solar_day_hours: float
    axial_tilt_deg: float
    oblateness: float
    bond_albedo: float

    distance_au: float
    eccentricity: float
    period_days: float
    parent_distance_km: float | None

    star_luminosity: float
    star_temperature_k: float
    star_radius: float

    mean_temperature_c: float
    liquid: str

    # What the ice caps are made of (planet/phases.py; "none"
    # = no frost), and the temperature below which they form
    # (C): the liquid's freezing point when the ice is the
    # body's own liquid (Earth's snow), otherwise where the
    # gas's vapor pressure drops below its partial pressure
    # in the air (Mars's water ice at ~-76 C, Pluto's
    # nitrogen at ~-236 C).
    ice: str
    frost_point_c: float

    # Earth-like life (vegetation on the biomes palette).
    life: bool

    palette: str
    colors: dict[str, tuple[float, float, float]] = field(hash=False)

    terrain: dict[str, float] = field(hash=False)

    atmosphere: AtmosphereProfile | None

    regime: str
    plate_count: int
    humidity: float

    band_count: int = 0
    band_colors: tuple[tuple[float, float, float], ...] = ()

    # Rings: (inner km, outer km, ((inner km, outer km,
    # optical depth), ...), color) or None.
    rings: tuple | None = None

    # Giants: the great storm (latitude, longitude deg,
    # size km, color, strength) or None; white ovals;
    # Saturn's polar hexagon.
    storm: tuple | None = None
    ovals: float = 0.0
    polar_hexagon: bool = False

    # Seed of the body's terrain and tectonics (each body
    # its own layout of basins and craters).
    seed: int = 1

    # Irregular shape (planet/shape.py): (semi-axes, center,
    # lobe center, lobe semi-axes, all in units of the
    # radius; lumpiness; basins as (lat, lon deg, diameter,
    # depth, peak m) groups), or None for a sphere flattened
    # by its oblateness.
    shape: tuple | None = None

    # Orbit orientation (deg; J2000) and its reference plane
    # ("ecliptic", or "equator" = the parent's), and spin:
    # the IAU north pole and prime meridian angle (deg; None
    # = from the axial tilt), or locked by tides.
    inclination_deg: float = 0.0
    ascending_node_deg: float = 0.0
    periapsis_arg_deg: float = 0.0
    mean_anomaly_deg: float = 0.0
    orbit_plane: str = "ecliptic"
    node_rate_deg_per_year: float = 0.0
    orbit_theory: str = ""
    periapsis_rate_deg_per_year: float = 0.0
    pole: tuple[float, float, float] | None = None
    tidally_locked: bool = False

    # -----------------------------------------------------
    # Derived physics
    # -----------------------------------------------------

    @property
    def radius_m(
        self
    ) -> float:

        return self.radius_km * 1000.0

    @property
    def surface_gravity(
        self
    ) -> float:
        """m / s^2 (g = G M / r^2)."""

        return GRAVITATIONAL_CONSTANT * self.mass_kg / self.radius_m ** 2

    @property
    def escape_velocity(
        self
    ) -> float:
        """m / s."""

        return math.sqrt(2.0 * GRAVITATIONAL_CONSTANT * self.mass_kg / self.radius_m)

    @property
    def sunlight(
        self
    ) -> float:
        """Stellar flux relative to Earth's (L / d^2)."""

        return self.star_luminosity / self.distance_au ** 2

    @property
    def stellar_flux(
        self
    ) -> float:
        """W / m^2 at the body's distance."""

        return SOLAR_CONSTANT * self.sunlight

    @property
    def equilibrium_temperature_k(
        self
    ) -> float:
        """
        Temperature with no greenhouse effect, heat spread
        over the whole sphere: (F (1 - A) / 4 sigma)^(1/4).
        Earth: ~255 K; its real 288 K is the greenhouse.
        """

        return (self.stellar_flux * (1.0 - self.bond_albedo) / (4.0 * STEFAN_BOLTZMANN)) ** 0.25

    @property
    def greenhouse_warming_c(
        self
    ) -> float:
        """Real mean minus equilibrium temperature (Venus: ~500 C)."""

        return self.mean_temperature_c + 273.15 - self.equilibrium_temperature_k

    @property
    def scale_height_km(
        self
    ) -> float | None:
        """Atmospheric scale height H = R T / (M g)."""

        if self.atmosphere is None:
            return None

        temperature = self.mean_temperature_c + 273.15

        return (
            GAS_CONSTANT * temperature
            / (self.atmosphere.molar_mass / 1000.0 * self.surface_gravity)
        ) / 1000.0

    @property
    def lapse_rate_k_per_km(
        self
    ) -> float:
        """
        How much colder per km of height near the ground:
        the dry adiabatic rate g / cp, slowed by condensing
        water on wet worlds, fading out in thin air (0 in
        vacuum). Earth ~6.4, Venus ~8, Mars ~1, Titan ~1.3.
        """

        air = self.atmosphere

        if air is None or air.surface_pressure_bar <= 0.0:
            return 0.0

        dry = self.surface_gravity / air.specific_heat(self.mean_temperature_c + 273.15) * 1000.0

        moist = MOIST_LAPSE_FACTOR if self.liquid == "water" and self.humidity > 0.0 else 1.0

        pressure = air.surface_pressure_bar

        return dry * moist * pressure / (pressure + LAPSE_PRESSURE_SCALE)

    @property
    def greenhouse_depth_estimate(
        self
    ) -> float:
        """
        Infrared optical depth that warms the equilibrium
        temperature to the observed mean, for a uniform
        planet: T^4 = Teq^4 (1 + 0.75 tau). 0 without air.
        """

        if self.atmosphere is None or self.atmosphere.surface_pressure_bar < VISIBLE_ATMOSPHERE_BAR:
            return 0.0

        ratio = (self.mean_temperature_c + 273.15) / self.equilibrium_temperature_k

        return max(0.0, (ratio ** 4 - 1.0) / 0.75)

    @property
    def sun_angular_radius_deg(
        self
    ) -> float:
        """Apparent radius of the star in this body's sky."""

        return math.degrees(math.atan(self.star_radius * SOLAR_RADIUS_M / (self.distance_au * AU_M)))

    @property
    def has_solid_surface(
        self
    ) -> bool:

        return self.kind not in ("gas_giant", "ice_giant")


# =========================================================
# Loading
# =========================================================

def load_profile(
    path
) -> BodyProfile:

    path = Path(path)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileError(f"{path}: {error}") from error

    return parse_profile(data, path.stem, str(path))


def parse_profile(
    data: dict,
    profile_id: str,
    source: str = "<profile>"
) -> BodyProfile:

    def fail(message):
        raise ProfileError(f"{source}: {message}")

    def section(name):

        value = data.get(name)

        if not isinstance(value, dict):
            fail(f"'{name}' must be an object")

        return value

    def number(parent, name, minimum=None, optional=False, default=None):

        value = parent.get(name, default)

        if value is None and optional:
            return None

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            fail(f"'{name}' must be a number")

        if minimum is not None and value < minimum:
            fail(f"'{name}' must be at least {minimum}")

        return float(value)

    def choice(parent, name, options):

        value = parent.get(name)

        if value not in options:
            fail(f"'{name}' must be one of {', '.join(options)}")

        return value

    def tint(parent, name):

        value = parent.get(name, [1.0, 1.0, 1.0])

        if (
            not isinstance(value, list)
            or len(value) != 3
            or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 10.0
                for v in value
            )
        ):
            fail(f"'{name}' must be three numbers in 0..10")

        return tuple(float(v) for v in value)

    def color(value, name):

        if (
            not isinstance(value, list)
            or len(value) != 3
            or not all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in value)
        ):
            fail(f"'{name}' must be three numbers in 0..1")

        return tuple(float(v) for v in value)

    if data.get("format") != PROFILE_FORMAT:
        fail(f"unsupported format {data.get('format')!r} (expected {PROFILE_FORMAT})")

    if not isinstance(data.get("name"), str) or not data["name"]:
        fail("'name' must be a non-empty string")

    physical = section("physical")
    orbit = section("orbit")
    star = section("star")
    surface = section("surface")
    terrain = section("terrain")
    geology = section("geology")
    climate = section("climate")

    colors = surface.get("colors")

    if not isinstance(colors, dict):
        fail("'surface.colors' must be an object")

    parsed_colors = {
        key: color(colors.get(key), f"colors.{key}")
        for key in ("low", "high", "steep", "ice")
    }

    terrain_values = {
        key: number(terrain, key, minimum=0.0 if key != "land_bias" else None)
        for key in (
            "continent_frequency",
            "continent_height",
            "land_bias",
            "mountain_frequency",
            "mountain_height",
            "detail_height",
        )
    }

    atmosphere = None

    raw_atmosphere = data.get("atmosphere")

    if raw_atmosphere is not None:

        if not isinstance(raw_atmosphere, dict):
            fail("'atmosphere' must be an object or null")

        composition = raw_atmosphere.get("composition")

        if not isinstance(composition, dict) or not composition:
            fail("'atmosphere.composition' must list gases")

        for gas, fraction in composition.items():

            if gas not in GASES:
                fail(f"unknown gas '{gas}' (known: {', '.join(GASES)})")

            if not isinstance(fraction, (int, float)) or fraction < 0:
                fail(f"fraction of '{gas}' must be a non-negative number")

        raw_aerosols = raw_atmosphere.get("aerosols")

        aerosols = None

        if raw_aerosols is not None:

            if not isinstance(raw_aerosols, dict):
                fail("'atmosphere.aerosols' must be an object or null")

            aerosols = Aerosols(
                scattering_per_Mm=number(raw_aerosols, "scattering_per_Mm", 0.0),
                absorption_per_Mm=number(raw_aerosols, "absorption_per_Mm", 0.0),
                scale_height_km=number(raw_aerosols, "scale_height_km", 0.01),
                anisotropy=number(raw_aerosols, "anisotropy", -0.99),
                scattering_tint=tint(raw_aerosols, "scattering_tint"),
                absorption_tint=tint(raw_aerosols, "absorption_tint"),
                altitude_km=number(raw_aerosols, "altitude_km", 0.0, default=0.0)
            )

        raw_clouds = raw_atmosphere.get("clouds")

        clouds = None

        if raw_clouds is not None:

            if not isinstance(raw_clouds, dict):
                fail("'atmosphere.clouds' must be an object or null")

            coverage = number(raw_clouds, "coverage", 0.0)

            if coverage > 1.0:
                fail("'coverage' must be at most 1")

            clouds = (
                coverage,
                number(raw_clouds, "altitude_km", 0.1),
                number(raw_clouds, "optical_depth", 0.0),
                number(raw_clouds, "scale_km", 1.0),
                color(raw_clouds.get("color", [1.0, 1.0, 1.0]), "color"),
            )

        atmosphere = AtmosphereProfile(
            surface_pressure_bar=number(raw_atmosphere, "surface_pressure_bar", 0.0),
            composition=tuple(
                sorted(
                    ((gas, float(fraction)) for gas, fraction in composition.items()),
                    key=lambda item: -item[1]
                )
            ),
            aerosols=aerosols,
            clouds=clouds,
            ozone=bool(raw_atmosphere.get("ozone", False))
        )

    bands = data.get("bands")

    band_count = 0

    storm = None
    ovals = 0.0
    polar_hexagon = False
    band_colors: tuple = ()

    if bands is not None:

        if not isinstance(bands, dict):
            fail("'bands' must be an object")

        band_count = int(number(bands, "count", 1.0))

        raw_colors = bands.get("colors")

        if not isinstance(raw_colors, list) or len(raw_colors) != 2:
            fail("'bands.colors' must be two colors")

        band_colors = tuple(color(c, "bands.colors") for c in raw_colors)

        raw_storm = bands.get("storm")

        if raw_storm is not None:

            if not isinstance(raw_storm, dict):
                fail("'bands.storm' must be an object or null")

            storm = (
                number(raw_storm, "latitude_deg"),
                number(raw_storm, "longitude_deg", default=0.0),
                number(raw_storm, "size_km", 0.0),
                color(raw_storm.get("color", [0.6, 0.3, 0.2]), "storm.color"),
                number(raw_storm, "strength", 0.0, default=1.0),
            )

        ovals = number(bands, "ovals", 0.0, default=0.0)

        polar_hexagon = bands.get("polar_hexagon", False)

        if not isinstance(polar_hexagon, bool):
            fail("'polar_hexagon' must be true or false")

    rings = data.get("rings")

    ring_extent = None

    if rings is not None:

        if not isinstance(rings, dict):
            fail("'rings' must be an object")

        raw_bands = rings.get("bands", [])

        if not isinstance(raw_bands, list) or len(raw_bands) > 8:
            fail("'rings.bands' must be a list of up to 8 [inner_km, outer_km, optical_depth]")

        ring_bands = []

        for band in raw_bands:

            if (
                not isinstance(band, list)
                or len(band) != 3
                or not all(isinstance(v, (int, float)) and v >= 0 for v in band)
                or band[1] <= band[0]
            ):
                fail("each ring band must be [inner_km, outer_km, optical_depth]")

            ring_bands.append(tuple(float(v) for v in band))

        ring_extent = (
            number(rings, "inner_km", 0.0),
            number(rings, "outer_km", 0.0),
            tuple(ring_bands),
            color(rings.get("color", [0.8, 0.72, 0.6]), "rings.color"),
        )

    shape = None

    raw_shape = data.get("shape")

    if raw_shape is not None:

        if not isinstance(raw_shape, dict):
            fail("'shape' must be an object")

        radius_km = number(physical, "radius_km", 0.1)

        def vector(parent, name, default, positive=False):

            value = parent.get(name, default)

            if (
                not isinstance(value, list)
                or len(value) != 3
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
                or (positive and not all(v > 0 for v in value))
            ):
                fail(f"'{name}' must be three {'positive ' if positive else ''}numbers (km)")

            return tuple(float(v) / radius_km for v in value)

        lobe_center = lobe_axes = (0.0, 0.0, 0.0)

        raw_lobe = raw_shape.get("lobe")

        if raw_lobe is not None:

            if not isinstance(raw_lobe, dict):
                fail("'shape.lobe' must be an object")

            lobe_center = vector(raw_lobe, "center_km", None)
            lobe_axes = vector(raw_lobe, "axes_km", None, positive=True)

        raw_basins = raw_shape.get("basins", [])

        if not isinstance(raw_basins, list) or len(raw_basins) > MAX_BASINS:
            fail(f"'shape.basins' must be a list of up to {MAX_BASINS} basins")

        basins = []

        for basin in raw_basins:

            if not isinstance(basin, dict):
                fail("each basin must be an object")

            basins.append((
                number(basin, "latitude_deg", -90.0),
                number(basin, "longitude_deg", -360.0),
                number(basin, "diameter_km", 0.001) * 1000.0,
                number(basin, "depth_km", 0.0) * 1000.0,
                number(basin, "peak_km", 0.0, default=0.0) * 1000.0,
            ))

        shape = (
            vector(raw_shape, "axes_km", [radius_km] * 3, positive=True),
            vector(raw_shape, "center_km", [0.0, 0.0, 0.0]),
            lobe_center,
            lobe_axes,
            number(raw_shape, "lumpiness", 0.0, default=0.0),
            tuple(basins),
        )

    rotation = data.get("rotation", {})

    if not isinstance(rotation, dict):
        fail("'rotation' must be an object")

    locked = rotation.get("tidally_locked", False)

    if not isinstance(locked, bool):
        fail("'tidally_locked' must be true or false")

    pole = None

    if "pole_ra_deg" in rotation:

        pole = (
            number(rotation, "pole_ra_deg", -360.0),
            number(rotation, "pole_dec_deg", -90.0),
            number(rotation, "prime_meridian_deg", default=0.0),
        )

        if pole[1] > 90.0:
            fail("'pole_dec_deg' must be at most 90")

    orbit_plane = orbit.get("plane", "equator" if data.get("orbits", "Sun") != "Sun" else "ecliptic")

    if orbit_plane not in ("ecliptic", "equator"):
        fail("'plane' must be ecliptic or equator")

    # Ice caps and life.
    ice = surface.get("ice", "none")

    if ice not in ICES:
        fail(f"'ice' must be one of {', '.join(ICES)}")

    life = surface.get("life", False)

    if not isinstance(life, bool):
        fail("'life' must be true or false")

    frost = number(surface, "frost_point_c", -KELVIN, optional=True)

    if frost is None:

        frost = derived_frost_point(
            ice,
            choice(surface, "liquid", LIQUIDS),
            atmosphere,
            number(surface, "ice_vapor_bar", 0.0, optional=True)
        )

    profile = BodyProfile(
        id=profile_id,
        name=data["name"],
        kind=choice(data, "kind", KINDS),
        orbits=str(data.get("orbits", "Sun")),
        description=str(data.get("description", "")),
        radius_km=number(physical, "radius_km", 0.1),
        mass_kg=number(physical, "mass_kg", 1.0),
        rotation_hours=number(physical, "rotation_hours"),
        solar_day_hours=number(physical, "solar_day_hours", 0.0),
        axial_tilt_deg=number(physical, "axial_tilt_deg", 0.0),
        oblateness=number(physical, "oblateness", 0.0, default=0.0),
        bond_albedo=number(physical, "bond_albedo", 0.0),
        distance_au=number(orbit, "distance_au", 1e-6),
        eccentricity=number(orbit, "eccentricity", 0.0),
        period_days=number(orbit, "period_days", 0.0),
        parent_distance_km=number(orbit, "parent_distance_km", 0.0, optional=True),
        star_luminosity=number(star, "luminosity_solar", 0.0),
        star_temperature_k=number(star, "temperature_k", 1.0),
        star_radius=number(star, "radius_solar", 0.0, default=1.0),
        mean_temperature_c=number(surface, "mean_temperature_c", -273.15),
        liquid=choice(surface, "liquid", LIQUIDS),
        ice=ice,
        frost_point_c=frost,
        life=life,
        palette=choice(surface, "palette", PALETTES),
        colors=parsed_colors,
        terrain=terrain_values,
        atmosphere=atmosphere,
        regime=choice(geology, "regime", REGIMES),
        plate_count=int(number(geology, "plate_count", 0.0, default=0)),
        humidity=number(climate, "humidity", 0.0),
        band_count=band_count,
        band_colors=band_colors,
        rings=ring_extent,
        storm=storm,
        ovals=ovals,
        polar_hexagon=polar_hexagon,
        seed=int(number(data, "seed", 0.0, default=1)),
        shape=shape,
        inclination_deg=number(orbit, "inclination_deg", default=0.0),
        ascending_node_deg=number(orbit, "ascending_node_deg", default=0.0),
        periapsis_arg_deg=number(orbit, "periapsis_arg_deg", default=0.0),
        mean_anomaly_deg=number(orbit, "mean_anomaly_deg", default=0.0),
        orbit_plane=orbit_plane,
        node_rate_deg_per_year=number(orbit, "node_rate_deg_per_year", default=0.0),
        orbit_theory=str(orbit.get("theory", "")),
        periapsis_rate_deg_per_year=number(orbit, "periapsis_rate_deg_per_year", default=0.0),
        pole=pole,
        tidally_locked=locked
    )

    if profile.bond_albedo >= 1.0:
        fail("'bond_albedo' must be below 1")

    if profile.palette == "bands" and profile.band_count == 0:
        fail("the 'bands' palette needs a 'bands' section")

    return profile


def load_presets(
    directory=BODIES_DIRECTORY
) -> dict[str, BodyProfile]:
    """All profiles in the directory, by id, ordered outward from the Sun."""

    profiles = [load_profile(path) for path in sorted(Path(directory).glob("*.json"))]

    order = {
        profile.id: (
            profile.distance_au,
            profile.parent_distance_km or 0.0
        )
        for profile in profiles
    }

    return {
        profile.id: profile
        for profile in sorted(profiles, key=lambda p: order[p.id])
    }


def system_members(
    presets: dict[str, "BodyProfile"],
    profile_id: str,
    limit: int = 7
) -> list[str]:
    """
    The other bodies of a profile's planetary system among
    the presets: a planet's moons, or a moon's planet and its
    sibling moons (none for bodies orbiting only the star).
    """

    focus = presets[profile_id]

    by_name = {profile.name: profile for profile in presets.values()}

    planet = by_name.get(focus.orbits, focus) if focus.orbits != "Sun" else focus

    members = [planet] + sorted(
        (p for p in presets.values() if p.orbits == planet.name and p.orbits != "Sun"),
        key=lambda p: p.parent_distance_km or 0.0
    )

    return [p.id for p in members if p.id != profile_id][:limit]


def preset_groups(
    presets: dict[str, BodyProfile]
) -> dict[str, list[BodyProfile]]:
    """Planets, moons (by parent) and dwarf planets, for menus."""

    groups: dict[str, list[BodyProfile]] = {
        "Planets": [],
        "Moons": [],
        "Dwarf planets": [],
        "Asteroids & comets": [],
    }

    for profile in presets.values():

        if profile.kind == "moon":
            groups["Moons"].append(profile)
        elif profile.kind == "dwarf":
            groups["Dwarf planets"].append(profile)
        elif profile.kind in ("asteroid", "comet"):
            groups["Asteroids & comets"].append(profile)
        else:
            groups["Planets"].append(profile)

    return {name: members for name, members in groups.items() if members}


# =========================================================
# Components
# =========================================================

@dataclass(slots=True)
class BodyComponents:

    planet: PlanetComponent
    body: BodyComponent
    atmosphere: AtmosphereComponent | None
    climate: ClimateComponent | None
    tectonics: TectonicsComponent | None
    rings: RingsComponent | None = None
    orbit: OrbitComponent | None = None

    def all(
        self
    ) -> list:

        return [
            component
            for component in (
                self.planet, self.body, self.atmosphere, self.climate, self.tectonics, self.rings, self.orbit
            )
            if component is not None
        ]


# Below this surface pressure the air is too thin to scatter
# visible light noticeably: the sky is black.
VISIBLE_ATMOSPHERE_BAR = 1e-4


def components_for(
    profile: BodyProfile,
    seed: int | None = None
) -> BodyComponents:
    """seed: None = the profile's own."""

    if seed is None:
        seed = profile.seed

    terrain = profile.terrain

    bands = profile.palette == "bands"

    colors = profile.colors

    if bands:
        low, high = profile.band_colors
    else:
        low, high = colors["low"], colors["high"]

    planet = PlanetComponent(
        seed=seed,
        radius=profile.radius_m,
        continent_frequency=terrain["continent_frequency"],
        continent_height=terrain["continent_height"],
        land_bias=terrain["land_bias"],
        mountain_frequency=terrain["mountain_frequency"],
        mountain_height=terrain["mountain_height"],
        detail_height=terrain["detail_height"],
        liquid=profile.liquid,
        palette=profile.palette,
        color_low=low,
        color_high=high,
        color_steep=colors["steep"],
        color_ice=colors["ice"],
        ice=profile.ice,
        frost_point=profile.frost_point_c,
        life=profile.life,
        **crater_settings(profile),
        **volcano_settings(profile),
        **erosion_settings(profile),
        bands=profile.band_count if bands else 0,
        **giant_settings(profile),
        **shape_fields(profile),
        max_depth=detail_depth(profile.radius_m)
    )

    body = BodyComponent(
        profile=profile.id,
        name=profile.name,
        kind=profile.kind,
        orbits=profile.orbits,
        mass=profile.mass_kg,
        surface_gravity=profile.surface_gravity,
        rotation_hours=profile.rotation_hours,
        solar_day_hours=profile.solar_day_hours,
        orbit_distance_au=profile.distance_au,
        eccentricity=profile.eccentricity,
        year_days=profile.period_days,
        bond_albedo=profile.bond_albedo,
        star_luminosity=profile.star_luminosity,
        star_radius=profile.star_radius,
        surface_pressure_bar=(
            profile.atmosphere.surface_pressure_bar
            if profile.atmosphere is not None
            else 0.0
        ),
        mean_temperature=profile.mean_temperature_c,
        oblateness=profile.oblateness,
        greenhouse_depth=calibrated_greenhouse(profile, planet),
        lapse_rate=profile.lapse_rate_k_per_km,
        star_temperature=profile.star_temperature_k
    )

    climate = None

    if profile.has_solid_surface:

        climate = ClimateComponent(
            axial_tilt=min(profile.axial_tilt_deg, 180.0 - profile.axial_tilt_deg),
            temperature_offset=0.0,
            humidity=profile.humidity
        )

    tectonics = None

    if profile.regime == "plate_tectonics":

        tectonics = TectonicsComponent(
            seed=seed,
            plate_count=max(2, profile.plate_count or 12)
        )

    elif profile.regime in DEFAULT_TIME_STEPS and profile.has_solid_surface:

        tectonics = TectonicsComponent(
            regime=profile.regime,
            seed=seed,
            time_step=DEFAULT_TIME_STEPS[profile.regime],
            relief_scale=relief_scale(profile.surface_gravity)
        )

    rings = None

    if profile.rings is not None:

        inner_km, outer_km, ring_bands, ring_color = profile.rings

        rings = RingsComponent(
            inner_radius=inner_km * 1000.0,
            outer_radius=outer_km * 1000.0,
            bands=flat_bands([(a * 1000.0, b * 1000.0, tau) for a, b, tau in ring_bands]),
            color=ring_color,
            seed=seed
        )

    return BodyComponents(
        rings=rings,
        orbit=orbit_for(profile),
        planet=planet,
        body=body,
        atmosphere=atmosphere_for(profile),
        climate=climate,
        tectonics=tectonics
    )


def orbit_for(
    profile: BodyProfile
) -> OrbitComponent:
    """
    The body's orbit and spin. Planets orbit the star at
    distance_au; moons their planet at parent_distance_km.
    Without a profile pole, the axis leans by the axial tilt
    from the orbit's pole (toward ecliptic longitude 90 deg).
    """

    moon = profile.orbits != "Sun"

    if moon and profile.parent_distance_km:
        semi_major_axis = profile.parent_distance_km * 1000.0
    else:
        semi_major_axis = profile.distance_au * AU_M

    if profile.pole is not None:

        pole_ra, pole_dec, prime_meridian = profile.pole

    else:

        # Tilt about the vernal equinox direction, as a pole
        # in Earth-equatorial coordinates.
        tilt = math.radians(profile.axial_tilt_deg)

        ecliptic = (0.0, -math.sin(tilt), math.cos(tilt))

        c, s = math.cos(math.radians(23.4392911)), math.sin(math.radians(23.4392911))

        x, y, z = ecliptic[0], ecliptic[1] * c - ecliptic[2] * s, ecliptic[1] * s + ecliptic[2] * c

        pole_ra = math.degrees(math.atan2(y, x)) % 360.0
        pole_dec = math.degrees(math.asin(max(-1.0, min(1.0, z))))
        prime_meridian = 0.0

    return OrbitComponent(
        parent=profile.orbits if moon else "",
        semi_major_axis=semi_major_axis,
        eccentricity=profile.eccentricity,
        inclination=profile.inclination_deg,
        ascending_node=profile.ascending_node_deg,
        periapsis=profile.periapsis_arg_deg,
        mean_anomaly=profile.mean_anomaly_deg,
        period=profile.period_days * 86_400.0,
        plane=profile.orbit_plane,
        node_rate=profile.node_rate_deg_per_year,
        theory=profile.orbit_theory,
        periapsis_rate=profile.periapsis_rate_deg_per_year,
        pole_ra=pole_ra,
        pole_dec=pole_dec,
        prime_meridian=prime_meridian,
        rotation_period=profile.rotation_hours * 3_600.0,
        tidally_locked=profile.tidally_locked
    )


def crater_settings(
    profile: BodyProfile
) -> dict:
    """
    PlanetComponent crater fields from the body's physics:
    - complex craters start at ~15 km on the Moon, scaling
      with 1 / gravity (Mars ~7 km, Earth ~3 km)
    - air burns up small impactors: ~100 m at 1 bar, ~3 km
      under Venus's 92 bar (~P^0.75)
    - liquid weather (Earth's rain, Titan's methane) erases
      craters within a few hundred Myr
    - bright rays survive only without air
    """

    if profile.palette == "bands" or not profile.has_solid_surface:
        return {"crater_density": 0.0}

    air = profile.atmosphere

    pressure = air.surface_pressure_bar if air is not None else 0.0

    return {
        # Comets shed their surface each pass by the Sun:
        # few craters survive.
        "crater_density": 0.15 if profile.kind == "comet" else 1.0,
        "surface_age": 4_000.0,
        "crater_transition": 15_000.0 * 1.62 / max(profile.surface_gravity, 0.05),
        "crater_min_diameter": 100.0 * pressure ** 0.75,
        "crater_erosion": 300.0 if profile.liquid in ("water", "methane") else 0.0,
        "crater_rays": pressure < VISIBLE_ATMOSPHERE_BAR,
    }


def volcano_settings(
    profile: BodyProfile
) -> dict:
    """
    PlanetComponent volcano fields from the body:
    - the tallest volcano scales with 1 / gravity
    - large shields need sustained mantle plumes: plate and
      resurfacing worlds, heat-pipe Io (low shields only by
      its activity), and rocky stagnant lids big enough to
      stay hot (Mars, not the Moon or Mercury, which get
      small domes)
    - icy bodies: none (cryovolcanism is not modeled)
    """

    if profile.palette == "bands" or not profile.has_solid_surface:
        return {"volcanism": 0.0}

    density = profile.mass_kg / (4.0 / 3.0 * math.pi * profile.radius_m ** 3)

    rocky = density > 3_000.0

    if not rocky or profile.regime in ("ice_shell", "none"):
        volcanism = 0.0
    elif profile.regime == "stagnant_lid":
        volcanism = 1.0 if profile.radius_km > 3_000.0 else 0.15
    else:
        volcanism = 1.0

    return {
        "volcanism": volcanism,
        "volcano_max_height": max_volcano_height(profile.surface_gravity),
    }


def erosion_settings(
    profile: BodyProfile
) -> dict:
    """
    PlanetComponent erosion fields from the body:
    - rivers wherever a liquid falls as rain and flows to
      seas (Earth's water, Titan's methane)
    - dunes need air to move sand (~1 mbar and up) and dry
      ground: everywhere dry on rainless worlds (Mars),
      in deserts where it rains (Earth); Venus's sluggish
      surface winds build few
    - Titan: long linear dunes along the wind, ~3 km apart,
      ~100 m high, in its equatorial belt, of dark organic
      sand; Mars: dark basaltic transverse dunes
    """

    air = profile.atmosphere

    pressure = air.surface_pressure_bar if air is not None else 0.0

    settings = {
        "rivers": profile.liquid in ("water", "methane") and profile.humidity > 0.0,
        "dune_density": 0.0,
    }

    if profile.palette == "bands" or not profile.has_solid_surface or pressure < 1e-3:
        return settings

    if profile.liquid == "methane":

        settings.update(
            dune_density=1.2, dune_amplitude=100.0, dune_wavelength=3_000.0,
            dune_linear=True, dune_max_latitude=30.0, dune_darkening=0.45
        )

    elif profile.liquid == "water":

        settings.update(
            dune_density=0.6, dune_amplitude=60.0, dune_wavelength=1_500.0,
            dune_linear=False, dune_max_latitude=50.0, dune_darkening=0.0
        )

    elif pressure > 10.0:

        settings.update(
            dune_density=0.1, dune_amplitude=30.0, dune_wavelength=500.0,
            dune_linear=False, dune_max_latitude=90.0, dune_darkening=0.0
        )

    else:

        settings.update(
            dune_density=1.0, dune_amplitude=40.0, dune_wavelength=700.0,
            dune_linear=False, dune_max_latitude=85.0, dune_darkening=0.35
        )

    return settings


def giant_settings(
    profile: BodyProfile
) -> dict:
    """PlanetComponent giant-planet fields (shape and weather)."""

    if profile.palette != "bands":
        return {}

    settings = {
        "ovals": profile.ovals,
        "polar_hexagon": profile.polar_hexagon,
    }

    if profile.storm is not None:

        latitude, longitude, size_km, color, strength = profile.storm

        settings.update(
            storm_latitude=latitude,
            storm_longitude=longitude,
            storm_size=size_km * 1000.0,
            storm_color=color,
            storm_strength=strength,
        )

    return settings


def shape_fields(
    profile: BodyProfile
) -> dict:
    """
    PlanetComponent shape fields: every body is flattened by
    its spin (Jupiter 6.5%, Earth 0.3%); small ones have
    irregular shapes of their own.
    """

    settings = {"oblateness": profile.oblateness}

    if profile.shape is None:
        return settings

    axes, center, lobe_center, lobe_axes, lumpiness, basins = profile.shape

    settings.update(
        shape_axes=axes,
        shape_center=center,
        lobe_center=lobe_center,
        lobe_axes=lobe_axes,
        lumpiness=lumpiness,
        basins=flat_basins(basins),
    )

    return settings


# Finest vertex spacing worth building: ~10 m on a planet,
# down to half a meter on a comet (the deepest level the
# quadtree goes, at most 15).
def detail_depth(
    radius: float
) -> int:

    target = max(0.5, 1.5e-6 * radius)

    edge = math.pi * 0.5 * radius / 32.0

    return int(min(15, max(6, math.ceil(math.log2(max(edge / target, 1.0))))))


def relief_scale(
    surface_gravity: float
) -> float:
    """
    Relief relative to Earth's: weaker gravity holds up
    taller mountains and deeper basins (roughly ~1/sqrt(g)
    for the planets; capped for small moons).
    """

    return float(min(max((9.81 / max(surface_gravity, 0.1)) ** 0.5, 0.5), 2.2))


def climate_physics(
    body: BodyComponent,
    liquid: str,
    humidity: float
) -> dict:
    """ClimateSettings physics fields for a body."""

    return dict(
        stellar_flux=SOLAR_CONSTANT * body.star_luminosity / max(body.orbit_distance_au, 1e-6) ** 2,
        bond_albedo=body.bond_albedo,
        greenhouse_depth=body.greenhouse_depth,
        surface_pressure=body.surface_pressure_bar,
        lapse_rate=body.lapse_rate / 1000.0,
        eccentricity=body.eccentricity,
        ocean=liquid == "water" and humidity > 0.0
    )


_GREENHOUSE_CACHE: dict[str, float] = {}


def calibrated_greenhouse(
    profile: BodyProfile,
    planet: PlanetComponent | None = None
) -> float:
    """
    Greenhouse optical depth at which the energy-balance
    climate (planet/climate.py) reproduces the body's
    observed global mean temperature. Starts from the
    uniform-planet estimate and corrects it with a few
    secant steps on a coarse grid (~0.1 s, cached).
    Airless bodies: 0 (their temperatures follow from
    sunlight alone).

    planet: the body's planet component; its terrain (land,
    heights) is used so mountains and continents count.
    """

    if profile.id in _GREENHOUSE_CACHE:
        return _GREENHOUSE_CACHE[profile.id]

    estimate = profile.greenhouse_depth_estimate

    if estimate <= 0.0 or not profile.has_solid_surface:

        _GREENHOUSE_CACHE[profile.id] = estimate

        return estimate

    # Deferred: planet.climate is heavier and imports nothing
    # from here, but keep body loading light.
    from planet.climate import ClimateModel, ClimateSettings

    body = BodyComponent(
        orbit_distance_au=profile.distance_au,
        star_luminosity=profile.star_luminosity,
        bond_albedo=profile.bond_albedo,
        surface_pressure_bar=profile.atmosphere.surface_pressure_bar,
        eccentricity=profile.eccentricity,
        lapse_rate=profile.lapse_rate_k_per_km
    )

    target = profile.mean_temperature_c

    elevation = None

    def mean_for(tau):

        settings = ClimateSettings(
            axial_tilt=min(profile.axial_tilt_deg, 180.0 - profile.axial_tilt_deg),
            resolution=16,
            **{
                **climate_physics(body, profile.liquid, profile.humidity),
                "greenhouse_depth": tau,
            }
        )

        model = ClimateModel(settings)

        nonlocal elevation

        if elevation is None:
            elevation = _calibration_terrain(model.grid, planet, profile)

        return float(model.compute(elevation).temperature.mean())

    low, high = estimate, estimate * 1.3 + 0.05

    f_low, f_high = mean_for(low) - target, mean_for(high) - target

    tau = estimate

    for _ in range(6):

        if abs(f_high - f_low) < 1e-9:
            break

        tau = max(0.0, high - f_high * (high - low) / (f_high - f_low))

        f = mean_for(tau) - target

        if abs(f) < 0.2:
            break

        low, f_low, high, f_high = high, f_high, tau, f

    _GREENHOUSE_CACHE[profile.id] = tau

    return tau


def _calibration_terrain(grid, planet, profile):
    """Ground heights on the climate grid for calibration."""

    import numpy as np

    if planet is None:
        return np.full(grid.cell_count, -1000.0 if profile.liquid == "water" else 100.0)

    from planet.terrain import Terrain
    from systems.planet_system import terrain_settings_for

    terrain = Terrain(terrain_settings_for(planet))

    return terrain.elevation(grid.directions, grid.cell_angle * planet.radius)


def derived_frost_point(
    ice: str,
    liquid: str,
    atmosphere: "AtmosphereProfile | None",
    vapor_bar: float | None = None
) -> float:
    """
    Frost point (C) of a body's ice (see BodyProfile.ice).
    vapor_bar: the ice's gas partial pressure, for trace
    gases the composition leaves out (water on Mars).
    """

    s = SUBSTANCES.get(ice)

    if s is None:
        return NO_FROST_C

    pressure = atmosphere.surface_pressure_bar if atmosphere is not None else 0.0

    # Snow of the body's own liquid: freezes where it falls.
    if ice == liquid and pressure >= s.triple_bar:
        return s.freezing_c

    partial = vapor_bar

    if partial is None and atmosphere is not None:

        partial = pressure * sum(
            fraction
            for gas, fraction in atmosphere.composition
            if gas == s.gas
        ) / max(sum(fraction for _, fraction in atmosphere.composition), 1e-12)

    if not partial:
        return NO_FROST_C

    return frost_point(s, partial)


def star_color(
    temperature_k: float
) -> tuple[float, float, float]:
    """
    Approximate linear RGB of a blackbody (normalized so the
    brightest channel is 1): red dwarfs orange, the Sun
    white, hot stars blue-white.
    """

    t = max(1000.0, min(temperature_k, 40000.0)) / 100.0

    if t <= 66.0:
        red = 1.0
        green = 0.390081579 * math.log(t) - 0.631841444
    else:
        red = 1.292936186 * (t - 60.0) ** -0.1332047592
        green = 1.129890861 * (t - 60.0) ** -0.0755148492

    if t >= 66.0:
        blue = 1.0
    elif t <= 19.0:
        blue = 0.0
    else:
        blue = 0.543206789 * math.log(t - 10.0) - 1.196254089

    srgb = [min(max(c, 0.0), 1.0) for c in (red, green, blue)]

    # sRGB -> linear, then normalize.
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]

    peak = max(linear)

    return tuple(c / peak for c in linear)


def sun_intensity(
    sunlight: float
) -> float:
    """
    Directional light intensity for a body's sunlight
    (relative to Earth's). Real sunlight ranges 1000x across
    the solar system; this compresses it (like the eye
    adapting) so Neptune is dim but visible and Mercury
    bright, with Earth at the engine's usual 5.
    """

    return 5.0 * min(max(max(sunlight, 1e-9) ** 0.3, 0.2), 2.5)


def cloud_fields(
    clouds: tuple | None
) -> dict:
    """AtmosphereComponent cloud fields from a profile's clouds."""

    if clouds is None:
        return {"cloud_coverage": 0.0}

    coverage, altitude_km, optical_depth, scale_km, color = clouds

    return {
        "cloud_coverage": coverage,
        "cloud_altitude": altitude_km * 1000.0,
        "cloud_optical_depth": optical_depth,
        "cloud_scale": scale_km * 1000.0,
        "cloud_color": color,
    }


def atmosphere_for(
    profile: BodyProfile
) -> AtmosphereComponent | None:
    """
    Scattering atmosphere from pressure, temperature, gravity
    and gases. None when it is too thin to see (the renderer
    then shows a black sky).

    Rayleigh scattering scales with molecules per volume
    (pressure / temperature, relative to Earth) and with the
    gases' refractivity squared; the density falls off with
    the scale height H = R T / (M g). Aerosols (dust, haze,
    cloud) come from the profile.
    """

    air = profile.atmosphere

    if air is None or air.surface_pressure_bar < VISIBLE_ATMOSPHERE_BAR:
        return None

    temperature = profile.mean_temperature_c + 273.15

    density = (
        air.surface_pressure_bar / EARTH_PRESSURE_BAR
        * EARTH_SURFACE_TEMPERATURE_K / max(temperature, 1.0)
    )

    rayleigh = tuple(
        value * density * air.scattering_factor
        for value in EARTH_RAYLEIGH
    )

    scale_height = profile.scale_height_km

    aerosols = air.aerosols

    # Thick enough to hold ~12 scale heights of either gas
    # or haze, and any cloud deck.
    tallest = max(scale_height, aerosols.scale_height_km if aerosols else 0.0)

    deck = (aerosols.altitude_km + 6.0 * aerosols.scale_height_km) if aerosols and aerosols.altitude_km > 0.0 else 0.0

    height_km = min(max(12.0 * tallest, deck, 20.0), 800.0)

    component = AtmosphereComponent(
        height=height_km * 1000.0,
        rayleigh_scattering=rayleigh,
        rayleigh_scale_height=scale_height * 1000.0,
        mie_scattering=aerosols.scattering_per_Mm if aerosols else 0.0,
        mie_absorption=aerosols.absorption_per_Mm if aerosols else 0.0,
        mie_scale_height=(aerosols.scale_height_km if aerosols else 1.0) * 1000.0,
        mie_anisotropy=aerosols.anisotropy if aerosols else 0.8,
        mie_scattering_tint=aerosols.scattering_tint if aerosols else (1.0, 1.0, 1.0),
        mie_absorption_tint=aerosols.absorption_tint if aerosols else (1.0, 1.0, 1.0),
        mie_layer_altitude=(aerosols.altitude_km if aerosols else 0.0) * 1000.0,
        **cloud_fields(air.clouds),
        ozone_absorption=EARTH_OZONE if air.ozone else (0.0, 0.0, 0.0),
        ground_albedo=profile.bond_albedo
    )

    return component
