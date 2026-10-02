import json
import math

from dataclasses import dataclass, field
from pathlib import Path

from ecs.components import (
    AtmosphereComponent,
    BodyComponent,
    ClimateComponent,
    PlanetComponent,
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
#
# Not modeled yet (stored for later steps): oblateness,
# rings, eccentric-orbit seasons, star brightness in the
# renderer's light.

PROFILE_FORMAT = 1

BODIES_DIRECTORY = Path("assets/bodies")

GRAVITATIONAL_CONSTANT = 6.674e-11
GAS_CONSTANT = 8.314                            # J / (mol K)
STEFAN_BOLTZMANN = 5.670374e-8
SOLAR_CONSTANT = 1361.0                         # W / m^2 at 1 AU
SOLAR_RADIUS_M = 6.957e8
AU_M = 1.495978707e11

KINDS = ("terrestrial", "moon", "dwarf", "gas_giant", "ice_giant")
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

# Molar mass (g / mol) and refractivity (n - 1 at STP, visible)
# of common atmospheric gases. Rayleigh scattering per
# molecule goes with refractivity squared.
GASES: dict[str, tuple[float, float]] = {
    "N2": (28.014, 2.98e-4),
    "O2": (31.998, 2.72e-4),
    "Ar": (39.948, 2.81e-4),
    "CO2": (44.009, 4.50e-4),
    "CH4": (16.043, 4.44e-4),
    "H2": (2.016, 1.39e-4),
    "He": (4.003, 3.50e-5),
    "CO": (28.010, 3.38e-4),
    "SO2": (64.066, 6.86e-4),
    "H2O": (18.015, 2.56e-4),
    "Ne": (20.180, 6.70e-5),
}

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


@dataclass(frozen=True, slots=True)
class AtmosphereProfile:

    surface_pressure_bar: float

    # (gas, mole fraction), largest first.
    composition: tuple[tuple[str, float], ...]

    aerosols: Aerosols | None
    ozone: bool = False

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
    frost_point_c: float
    palette: str
    colors: dict[str, tuple[float, float, float]] = field(hash=False)

    terrain: dict[str, float] = field(hash=False)

    atmosphere: AtmosphereProfile | None

    regime: str
    plate_count: int
    humidity: float

    band_count: int = 0
    band_colors: tuple[tuple[float, float, float], ...] = ()

    rings: tuple[float, float] | None = None

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
                anisotropy=number(raw_aerosols, "anisotropy", -0.99)
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
            ozone=bool(raw_atmosphere.get("ozone", False))
        )

    bands = data.get("bands")

    band_count = 0
    band_colors: tuple = ()

    if bands is not None:

        if not isinstance(bands, dict):
            fail("'bands' must be an object")

        band_count = int(number(bands, "count", 1.0))

        raw_colors = bands.get("colors")

        if not isinstance(raw_colors, list) or len(raw_colors) != 2:
            fail("'bands.colors' must be two colors")

        band_colors = tuple(color(c, "bands.colors") for c in raw_colors)

    rings = data.get("rings")

    ring_extent = None

    if rings is not None:

        if not isinstance(rings, dict):
            fail("'rings' must be an object")

        ring_extent = (number(rings, "inner_km", 0.0), number(rings, "outer_km", 0.0))

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
        frost_point_c=number(surface, "frost_point_c"),
        palette=choice(surface, "palette", PALETTES),
        colors=parsed_colors,
        terrain=terrain_values,
        atmosphere=atmosphere,
        regime=choice(geology, "regime", REGIMES),
        plate_count=int(number(geology, "plate_count", 0.0, default=0)),
        humidity=number(climate, "humidity", 0.0),
        band_count=band_count,
        band_colors=band_colors,
        rings=ring_extent
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


def preset_groups(
    presets: dict[str, BodyProfile]
) -> dict[str, list[BodyProfile]]:
    """Planets, moons (by parent) and dwarf planets, for menus."""

    groups: dict[str, list[BodyProfile]] = {"Planets": [], "Moons": [], "Dwarf planets": []}

    for profile in presets.values():

        if profile.kind == "moon":
            groups["Moons"].append(profile)
        elif profile.kind == "dwarf":
            groups["Dwarf planets"].append(profile)
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

    def all(
        self
    ) -> list:

        return [
            component
            for component in (self.planet, self.body, self.atmosphere, self.climate, self.tectonics)
            if component is not None
        ]


# Below this surface pressure the air is too thin to scatter
# visible light noticeably: the sky is black.
VISIBLE_ATMOSPHERE_BAR = 1e-4


def components_for(
    profile: BodyProfile,
    seed: int = 1
) -> BodyComponents:

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
        frost_point=profile.frost_point_c,
        bands=profile.band_count if bands else 0
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
        oblateness=profile.oblateness
    )

    climate = None

    if profile.has_solid_surface:

        climate = ClimateComponent(
            axial_tilt=min(profile.axial_tilt_deg, 180.0 - profile.axial_tilt_deg),
            temperature_offset=profile.mean_temperature_c - CLIMATE_MODEL_MEAN_C,
            humidity=profile.humidity
        )

    tectonics = None

    if profile.regime == "plate_tectonics":

        tectonics = TectonicsComponent(
            seed=seed,
            plate_count=max(2, profile.plate_count or 12)
        )

    return BodyComponents(
        planet=planet,
        body=body,
        atmosphere=atmosphere_for(profile),
        climate=climate,
        tectonics=tectonics
    )


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
    # or haze.
    tallest = max(scale_height, aerosols.scale_height_km if aerosols else 0.0)

    height_km = min(max(12.0 * tallest, 20.0), 800.0)

    component = AtmosphereComponent(
        height=height_km * 1000.0,
        rayleigh_scattering=rayleigh,
        rayleigh_scale_height=scale_height * 1000.0,
        mie_scattering=aerosols.scattering_per_Mm if aerosols else 0.0,
        mie_absorption=aerosols.absorption_per_Mm if aerosols else 0.0,
        mie_scale_height=(aerosols.scale_height_km if aerosols else 1.0) * 1000.0,
        mie_anisotropy=aerosols.anisotropy if aerosols else 0.8,
        ozone_absorption=EARTH_OZONE if air.ozone else (0.0, 0.0, 0.0),
        ground_albedo=profile.bond_albedo
    )

    return component
