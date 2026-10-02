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

    # Ozone: absorbs orange-red, deepening twilight blue.
    ozone_absorption: tuple[float, float, float] = (0.650, 1.881, 0.085)
    ozone_altitude: float = 25_000.0
    ozone_thickness: float = 30_000.0

    # Average ground reflectance (light bounced into the
    # sky, and the ground seen in reflections).
    ground_albedo: float = 0.3


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

    seed: int = 1
    plate_count: int = 12

    # Fraction of the surface that starts as continent.
    land_fraction: float = 0.3

    # Typical plate speed, cm / year (Earth: 2-10).
    plate_speed: float = 5.0

    # Simulated time per step (million years).
    time_step: float = 5.0

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
