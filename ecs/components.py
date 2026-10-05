from dataclasses import dataclass, field

import numpy as np

from core.handle import Handle
from ecs.entity import Entity
from math3d.transform import Transform


# =========================================================
# Name
# =========================================================
#
# Human-readable label used by the debug UI and logs.

@dataclass(slots=True)
class NameComponent:
    name: str


# =========================================================
# Transform
# =========================================================
#
# `transform` is the local transform (relative to the
# parent, or to the world when there is no parent).
#
# `world_matrix` is computed each frame by TransformSystem.
# Anything that needs a world-space position/direction
# (rendering, lights, cameras) should use the world_*
# helpers rather than `transform` directly.

def _identity() -> np.ndarray:

    return np.identity(
        4,
        dtype=np.float64
    )


@dataclass(slots=True)
class TransformComponent:

    transform: Transform = field(
        default_factory=Transform
    )

    world_matrix: np.ndarray = field(
        default_factory=_identity
    )

    # -----------------------------------------------------
    # World-Space Helpers
    # -----------------------------------------------------

    @property
    def world_position(
        self
    ) -> np.ndarray:

        return np.array(
            self.world_matrix[:3, 3],
            dtype=np.float64
        )

    @property
    def world_forward(
        self
    ) -> np.ndarray:

        return self._world_direction(
            (0.0, 0.0, -1.0)
        )

    @property
    def world_up(
        self
    ) -> np.ndarray:

        return self._world_direction(
            (0.0, 1.0, 0.0)
        )

    def _world_direction(
        self,
        local
    ) -> np.ndarray:

        world = (
            self.world_matrix[:3, :3]
            @ np.asarray(local, dtype=np.float64)
        )

        length = float(
            np.linalg.norm(world)
        )

        # A zero-scaled parent can collapse directions;
        # fall back to the unrotated local axis.

        if length <= 1e-8:

            return np.asarray(
                local,
                dtype=np.float64
            )

        return (
            world
            / length
        ).astype(
            np.float64
        )


# =========================================================
# Hierarchy
# =========================================================
#
# Makes the owning entity's transform relative to
# `parent`. The parent must have a TransformComponent.

@dataclass(slots=True)
class HierarchyComponent:
    parent: Entity


# =========================================================
# Mesh Renderer
# =========================================================

@dataclass(slots=True)
class MeshRendererComponent:
    mesh: Handle
    material: Handle
    casts_shadows: bool = True


# =========================================================
# Camera
# =========================================================

@dataclass(slots=True)
class CameraComponent:
    fov: float = 45.0
    near: float = 0.1
    far: float = 100.0
    primary: bool = False


# =========================================================
# Directional Light
# =========================================================
#
# Direction comes from the owning entity's world forward.
# It also places the sun in the sky. (Ambient light comes
# from the sky through image-based lighting; see
# RenderSettings.ibl_intensity.)

@dataclass(slots=True)
class DirectionalLightComponent:
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 5.0
    casts_shadows: bool = True


# =========================================================
# Point Light
# =========================================================
#
# Position comes from the owning entity's world position.
#
# range: distance at which the light fades to zero.

@dataclass(slots=True)
class PointLightComponent:
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    range: float = 10.0


# =========================================================
# Spot Light
# =========================================================
#
# Position and direction come from the owning entity's
# world position / forward.
#
# Angles are half-angles in degrees measured from the
# spot direction. Full intensity inside inner_angle,
# fading to zero at outer_angle.

@dataclass(slots=True)
class SpotLightComponent:
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    range: float = 10.0
    inner_angle: float = 15.0
    outer_angle: float = 25.0
    casts_shadows: bool = True


# =========================================================
# Rotator
# =========================================================
#
# Spins the entity's local transform at a constant rate
# (degrees per second around X, Y, Z). Driven by
# RotatorSystem on the fixed timestep.

@dataclass(slots=True)
class RotatorComponent:
    degrees_per_second: tuple[float, float, float] = (
        0.0,
        30.0,
        0.0
    )


# =========================================================
# Camera Controller
# =========================================================

@dataclass(slots=True)
class CameraControllerComponent:

    movement_speed: float = 3.0
    mouse_sensitivity: float = 0.1

    # Prevent looking exactly straight up/down.
    min_pitch: float = -89.0
    max_pitch: float = 89.0

    # Planet mode: "up" points away from planet_center
    # instead of world +Y, so the horizon stays level
    # anywhere on a sphere, W/S/A/D move along the ground
    # (around the planet from orbit) and Space/Shift move
    # along the zenith.
    planet_mode: bool = False
    planet_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    planet_radius: float = 0.0

    # Speed grows with altitude above planet_radius:
    # speed = max(movement_speed, altitude * altitude_speed).
    # 0 disables scaling.
    altitude_speed: float = 0.0

    # Closest the camera may get to the surface.
    min_altitude: float = 1.0

    # How far below planet_radius the camera may go (m):
    # giant planets have no ground, only ever-thicker air.
    descent: float = 0.0


# =========================================================
# Planet
# =========================================================
#
# A procedurally generated planet centered on the entity
# (rotation applies; scale is ignored). PlanetSystem builds
# and streams its terrain chunks. Changing any terrain
# field rebuilds the planet.

@dataclass(slots=True)
class PlanetComponent:

    # Terrain (see planet/terrain.py TerrainSettings).
    seed: int = 1
    radius: float = 6_371_000.0
    continent_frequency: float = 1.2
    continent_height: float = 2_500.0
    land_bias: float = 0.05
    mountain_frequency: float = 150.0
    mountain_height: float = 5_000.0
    detail_height: float = 300.0

    # Level of detail: vertices per chunk edge, deepest
    # quadtree level, and how close (in chunk edge lengths)
    # the camera must be before a chunk splits.
    resolution: int = 33
    max_depth: int = 15
    split_factor: float = 1.5

    # Surface appearance.
    #   liquid:  "water", "methane", "lava" or "none" (dry
    #            worlds keep their basins instead of seas)
    #   palette: "biomes" (Earth-like life, from climate),
    #            "mineral" (bare ground: color_low / high /
    #            steep, color_ice where colder than
    #            frost_point), or "bands" (gas giant cloud
    #            belts in color_low / color_high; no relief)
    liquid: str = "water"
    palette: str = "biomes"
    color_low: tuple[float, float, float] = (0.30, 0.30, 0.30)
    color_high: tuple[float, float, float] = (0.40, 0.40, 0.40)
    color_steep: tuple[float, float, float] = (0.16, 0.14, 0.12)
    color_ice: tuple[float, float, float] = (0.80, 0.82, 0.86)

    # Ice caps: what they are made of (planet/phases.py:
    # water, carbon_dioxide, nitrogen, methane or none), and
    # the annual mean temperature (C) below which the ground
    # frosts over (derived from the body's air; editable).
    ice: str = "water"
    frost_point: float = -1.9

    # Earth-like life: vegetation on the biomes palette
    # (without it, the same climate zones are bare ground).
    life: bool = True

    # Number of cloud bands ("bands" palette).
    bands: int = 0

    # Shape (planet/shape.py): flattening by the spin
    # ((equatorial - polar) / equatorial radius; Jupiter
    # 0.065, Earth 0.0034), and for irregular bodies the
    # semi-axes in units of the radius (x, y = spin axis, z
    # = longitude 0) and their center, a second lobe
    # (contact binary; axes 0 = none), large-scale lumps (a
    # fraction of the radius), and up to 4 giant basins
    # (flattened lat, lon deg, diameter, depth, central peak
    # m; diameter 0 = unused). Relief and seas stand on it.
    oblateness: float = 0.0
    shape_axes: tuple[float, float, float] = (1.0, 1.0, 1.0)
    shape_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lobe_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lobe_axes: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lumpiness: float = 0.0
    basins: tuple[float, ...] = (0.0,) * 20

    # Real maps of the body (planet/maps.py dataset ids; ""
    # = generated): measured heights as the terrain's base,
    # and surface colors. Needs tools/fetch_maps.py.
    elevation_map: str = ""
    color_map: str = ""

    # Giant planets ("bands" palette): the great storm
    # (latitude, longitude in degrees, east-west size in m,
    # 0 = none; color; strength 0..1: Jupiter's Great Red
    # Spot, Neptune's Great Dark Spot), how many small white
    # ovals, and Saturn's hexagonal north polar jet.
    storm_latitude: float = 0.0
    storm_longitude: float = 0.0
    storm_size: float = 0.0
    storm_color: tuple[float, float, float] = (0.62, 0.28, 0.16)
    storm_strength: float = 1.0
    ovals: float = 0.0
    polar_hexagon: bool = False

    # Impact craters (planet/craters.py): impact rate
    # relative to the Moon's (0 = none); how many show
    # depends on the surface's age (the tectonic field's, or
    # surface_age in Myr). Erosion (Myr, 0 = never) wears
    # old ones away; air stops impactors smaller than
    # crater_min_diameter (m); craters above
    # crater_transition (m) get flat floors and central
    # peaks; airless bodies keep bright rays.
    crater_density: float = 1.0
    surface_age: float = 4_000.0
    crater_erosion: float = 300.0
    crater_min_diameter: float = 100.0
    crater_transition: float = 3_000.0
    crater_rays: bool = False

    # Volcanoes (planet/volcanoes.py; need a tectonic
    # simulation, which says where magma comes up): 0..1 how
    # much the body builds them (large shields need >= 0.5),
    # and the tallest its gravity allows (m).
    volcanism: float = 1.0
    volcano_max_height: float = 10_000.0

    # Erosion matched to the body: rivers, lakes, deltas and
    # glaciers draining the climate's rain into the seas
    # (planet/hydrology.py), and wind-blown dune fields on
    # dry ground (planet/dunes.py): how common, crest
    # height and spacing (m), linear ridges along the wind
    # (Titan) or transverse crests across it, the latitude
    # band, and how much darker the sand is.
    rivers: bool = True
    dune_density: float = 0.4
    dune_amplitude: float = 60.0
    dune_wavelength: float = 1_500.0
    dune_linear: bool = False
    dune_max_latitude: float = 50.0
    dune_darkening: float = 0.0


# =========================================================
# Rings
# =========================================================
#
# A ring system in the equatorial plane of the planet on the
# same entity (graphics/rings.py): from inner_radius to
# outer_radius (m), made of up to 8 bands (inner m, outer m,
# normal optical depth; flattened, 0 = unused), particles of
# `color` albedo; `opacity` scales the depths.

@dataclass(slots=True)
class RingsComponent:

    inner_radius: float = 0.0
    outer_radius: float = 0.0
    bands: tuple[float, ...] = (0.0,) * 24
    color: tuple[float, float, float] = (0.8, 0.72, 0.6)
    opacity: float = 1.0
    seed: int = 1


# =========================================================
# Orbit
# =========================================================
#
# How the body on this entity moves and turns
# (planet/orbits.py; OrbitSystem places it): a Keplerian
# orbit around another body (by its entity's name) or the
# star (""), and its spin. Elements are relative to the
# ecliptic, or for "equator" to the parent's equator; the
# mean anomaly is at J2000. Spin: the IAU north pole (right
# ascension, declination) and prime meridian angle at J2000
# (deg), the sidereal rotation period (s; negative =
# retrograde), or locked by tides (its prime meridian
# always toward its parent). A planet's elements are those
# of the barycenter it shares with its moons.

@dataclass(slots=True)
class OrbitComponent:

    parent: str = ""
    semi_major_axis: float = 1.495978707e11     # m
    eccentricity: float = 0.0
    inclination: float = 0.0                    # deg
    ascending_node: float = 0.0                 # deg
    periapsis: float = 0.0                      # deg (argument)
    mean_anomaly: float = 0.0                   # deg at J2000
    period: float = 31_558_149.8                # s (anomalistic)
    plane: str = "ecliptic"

    # Precession (deg per year): the Moon's nodes regress
    # once in 18.6 years and its perigee advances, which
    # sets when eclipses happen.
    node_rate: float = 0.0
    periapsis_rate: float = 0.0

    # "moon": our Moon's own theory instead of an ellipse
    # (the Sun's pull; planet/orbits.py moon_offset).
    theory: str = ""

    pole_ra: float = 0.0                        # deg
    pole_dec: float = 90.0                      # deg
    prime_meridian: float = 0.0                 # deg at J2000
    rotation_period: float = 86_164.1           # s
    tidally_locked: bool = False


# =========================================================
# Star and Clock
# =========================================================
#
# On the sun entity: the star the bodies orbit (its light's
# brightness and color, its apparent size), and the
# simulation clock that moves them: seconds since J2000
# (2000-01-01 12:00 UTC) and how many simulated seconds
# pass per real second (0 = stopped). With these, the sun's
# direction, day and night, seasons and the moons in the
# sky all follow the time.

@dataclass(slots=True)
class StarComponent:

    luminosity: float = 1.0                     # Suns
    temperature: float = 5772.0                 # K
    radius: float = 6.957e8                     # m


@dataclass(slots=True)
class ClockComponent:

    time: float = 0.0                           # s since J2000
    rate: float = 0.0                           # simulated s per real s


# =========================================================
# Atmosphere
# =========================================================
#
# Physically based atmosphere around the planet on the same
# entity (needs a PlanetComponent; its radius is sea
# level). Defaults are Earth's (Hillaire 2020). Scattering
# and absorption coefficients are per megameter (1e-6/m).

@dataclass(slots=True)
class AtmosphereComponent:

    # Top of the atmosphere above sea level.
    height: float = 100_000.0

    # Rayleigh (air molecules): scatters blue most.
    rayleigh_scattering: tuple[float, float, float] = (5.802, 13.558, 33.1)
    rayleigh_scale_height: float = 8_000.0

    # Mie (aerosols, haze): strongly forward. Grey by
    # default; the tints color it per channel (Mars's
    # butterscotch dust, Titan's orange haze absorb blue).
    mie_scattering: float = 3.996
    mie_absorption: float = 4.40
    mie_scale_height: float = 1_200.0
    mie_anisotropy: float = 0.8
    mie_scattering_tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mie_absorption_tint: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Weather clouds (graphics/clouds.py): the mean share of
    # the sky they cover (0 = none; where comes from the
    # climate's rain), the layer's height (m), optical depth
    # at the thickest, the largest cloud features (m), their
    # color, and how fast they drift (m / s). Defaults:
    # Earth.
    cloud_coverage: float = 0.62
    cloud_altitude: float = 4_000.0
    cloud_optical_depth: float = 14.0
    cloud_scale: float = 700_000.0
    cloud_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    cloud_speed: float = 10.0

    # Aerosols in a layer centered this high (m), thinning
    # above and below by the scale height (a cloud deck:
    # Venus's sulfuric acid clouds at ~57 km). 0 = densest at
    # the ground (dust, haze).
    mie_layer_altitude: float = 0.0

    # Ozone: absorbs orange-red, deepening twilight blue.
    ozone_absorption: tuple[float, float, float] = (0.650, 1.881, 0.085)
    ozone_altitude: float = 25_000.0
    ozone_thickness: float = 30_000.0

    # Average ground reflectance (light bounced into the
    # sky, and the ground seen in reflections).
    ground_albedo: float = 0.3

    # Cloud decks (up to 4; flattened groups of base m, top
    # m, optical depth, single-scattering albedo rgb, texture
    # 0..1, texture size m, lightning flashes per s per
    # million km^2; optical depth 0 = unused): Venus's
    # sulfuric acid layers, the giants' ammonia, ammonium
    # hydrosulfide and water clouds.
    decks: tuple[float, ...] = (0.0,) * 36

    # Temperature at sea level / a giant's cloud tops (K),
    # how fast it falls with height (K/km; on a giant, how
    # fast it rises below the tops: the adiabat), cp / R
    # (density ~ T^(cp/R - 1) down the adiabat), and the deep
    # air's absorption (1/km at the tops' density; it rises
    # as density^2, so the hot depths glow).
    temperature: float = 288.0
    lapse_rate: float = 6.5
    adiabatic_exponent: float = 3.5
    deep_absorption: float = 0.0


# =========================================================
# Tectonics
# =========================================================
#
# Plate tectonics for the planet on the same entity
# (planet/tectonics.py; driven by TectonicsSystem). The
# simulation's continents, ocean basins and mountain belts
# replace the planet's noise continents.
#
# The simulation state itself is not saved: simulated_time
# is, and loading re-simulates from the seed to that time
# (deterministic).

@dataclass(slots=True)
class TectonicsComponent:

    # How the outer shell moves (planet/regimes.py):
    #   plate_tectonics       Earth: plates, ridges, collisions
    #   stagnant_lid          Mars, Mercury, the Moon: one
    #                         rigid shell, ancient relief
    #   episodic_resurfacing  Venus: global lava floods
    #   heat_pipe             Io: constant volcanism
    #   ice_shell             Europa: tidal cracks in ice
    regime: str = "plate_tectonics"

    seed: int = 1

    # Plates only.
    plate_count: int = 12

    # Fraction of the surface that starts as continent.
    land_fraction: float = 0.3

    # Typical plate speed, cm / year (Earth: 2-10).
    plate_speed: float = 5.0

    # Simulated time per step (million years).
    time_step: float = 5.0

    # Other regimes: relief multiplier (weaker gravity
    # holds up taller mountains).
    relief_scale: float = 1.0

    # Stagnant lids: strength of the crustal dichotomy (one
    # hemisphere low, 1 = default), and the share of the
    # surface in the low one.
    dichotomy: float = 1.0
    lowlands: float = 0.5

    # Grid cells per cube-face edge (6 * n^2 cells).
    resolution: int = 128

    # Million years simulated so far.
    simulated_time: float = 0.0


# =========================================================
# Climate
# =========================================================
#
# Annual-mean climate for the planet on the same entity
# (planet/climate.py; ClimateSystem): temperature from
# sunlight and height, winds, and rainfall carried from the
# oceans. Drives the biomes (forests, deserts, tundra...)
# and where snow lies. Recomputed whenever the land changes.

@dataclass(slots=True)
class ClimateComponent:

    # Degrees (Earth: 23.44): how unevenly the year's
    # sunlight is spread between equator and poles.
    axial_tilt: float = 23.44

    # Added to every temperature (C): a warmer or colder
    # world (ice ages, hothouse climates).
    temperature_offset: float = 0.0

    # Scales evaporation, so rainfall (1 = Earth-like).
    humidity: float = 1.0

    # Grid cells per cube-face edge.
    resolution: int = 64


# =========================================================
# Body
# =========================================================
#
# What the planet entity is, physically: the facts a body
# profile (assets/bodies/*.json, planet/bodies.py) brings,
# kept on the entity for display and for the physics that
# will use them (energy-balance climate, gravity, orbits).

@dataclass(slots=True)
class BodyComponent:

    # Profile id ("mars") and display name.
    profile: str = ""
    name: str = ""

    # "terrestrial", "moon", "dwarf", "gas_giant", "ice_giant".
    kind: str = "terrestrial"

    # What it orbits ("Sun", "Saturn", ...).
    orbits: str = "Sun"

    mass: float = 5.972e24                      # kg
    surface_gravity: float = 9.81               # m/s^2
    rotation_hours: float = 23.934              # sidereal; < 0 = retrograde
    solar_day_hours: float = 24.0
    orbit_distance_au: float = 1.0              # from the star
    eccentricity: float = 0.0167
    year_days: float = 365.256
    bond_albedo: float = 0.306
    star_luminosity: float = 1.0                # Suns
    star_radius: float = 1.0                    # Suns
    surface_pressure_bar: float = 1.014
    mean_temperature: float = 15.0              # C (observed)
    oblateness: float = 0.0

    # Climate physics (planet/climate.py energy balance):
    # infrared optical depth (greenhouse), and how much
    # colder the air gets per km of height.
    greenhouse_depth: float = 0.85
    lapse_rate: float = 6.5                     # C per km

    # Star surface temperature (K): the color of its light.
    star_temperature: float = 5772.0

    # Seen from afar (planet/bodies.py): brightness at full
    # phase, and the disc's color (relative).
    geometric_albedo: float = 0.204
    disc_color: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Comets: dust activity A f rho at 1 AU (m; 0 = not
    # active), gas relative to dust.
    comet_afrho: float = 0.0
    comet_gas: float = 0.0
