import math

from dataclasses import dataclass

import numpy as np

from ecs.components import (
    AtmosphereComponent,
    BodyComponent,
    CameraComponent,
    ClockComponent,
    DirectionalLightComponent,
    NameComponent,
    PlanetComponent,
    TransformComponent
)
from ecs.entity import Entity

from math3d import quaternion

from planet import solar
from planet.air import AirColumn, air_column_of


# =========================================================
# Flight Context
# =========================================================
#
# Where the player is: the primary camera (the vessel, once
# there are vessels), the body it is nearest to (its
# surface, not its center: a moon close by beats a giant
# far away), the sun and the clock, with the geometry the
# HUD and the camera moves need.

class FlightContext:

    def __init__(
        self,
        scene,
        body: Entity | None,
        camera: Entity | None,
        sun: Entity | None
    ):

        self.scene = scene
        self.body = body
        self.camera = camera
        self.sun = sun

    @classmethod
    def find(
        cls,
        scene,
        focus=None,
        orbiting=None
    ) -> "FlightContext":
        """
        focus: where the player is (world m; the vessel), else
        the camera. orbiting: the body the flown vessel orbits
        (its sphere of influence's): that body, not the
        nearest surface (a moon passing closer than its
        planet's ground); None, or the Sun: the nearest.
        """

        registry = scene.registry

        camera = next(
            (entity for entity, component in registry.view_with(CameraComponent) if component.primary),
            None
        )

        sun = next((entity for entity, _ in registry.view_with(DirectionalLightComponent)), None)

        body = orbiting if orbiting is not None and registry.has(orbiting, PlanetComponent) else None

        if body is None and (focus is not None or camera is not None):

            position = (
                np.asarray(focus, dtype=np.float64)
                if focus is not None
                else np.asarray(scene.get_component(camera, TransformComponent).transform.position, dtype=np.float64)
            )

            nearest = math.inf

            for entity, transform, planet in registry.view_with(TransformComponent, PlanetComponent):

                center = np.asarray(transform.world_matrix, dtype=np.float64)[:3, 3]

                height = float(np.linalg.norm(position - center)) - planet.radius

                if height < nearest:
                    body, nearest = entity, height

        return cls(scene, body, camera, sun)

    # -----------------------------------------------------
    # The body
    # -----------------------------------------------------

    @property
    def planet(
        self
    ) -> PlanetComponent:

        return self.scene.get_component(self.body, PlanetComponent)

    def name(
        self
    ) -> str:

        component = self.scene.try_get_component(self.body, NameComponent)

        return component.name if component is not None else "Planet"

    def _body_world(
        self
    ) -> np.ndarray:

        matrix = np.asarray(
            self.scene.get_component(self.body, TransformComponent).world_matrix,
            dtype=np.float64
        )

        basis = matrix[:3, :3]

        rigid = np.eye(4)
        rigid[:3, :3] = basis / np.linalg.norm(basis, axis=0, keepdims=True)
        rigid[:3, 3] = matrix[:3, 3]

        return rigid

    def body_center(
        self
    ) -> np.ndarray:

        return self._body_world()[:3, 3]

    def body_rotation(
        self
    ) -> np.ndarray:

        return self._body_world()[:3, :3]

    def clock(
        self
    ) -> ClockComponent | None:

        return next((c for _, c in self.scene.registry.view_with(ClockComponent)), None)

    def camera_transform(self):

        return self.scene.get_component(self.camera, TransformComponent).transform

    def camera_direction(
        self
    ) -> np.ndarray | None:
        """Body-space unit direction from the center to the camera."""

        offset = np.asarray(self.camera_transform().position, dtype=np.float64) - self.body_center()

        length = float(np.linalg.norm(offset))

        if length <= 0.0:
            return None

        return self.body_rotation().T @ (offset / length)

    # -----------------------------------------------------
    # Solar time
    # -----------------------------------------------------
    #
    # In body space (y = the spin axis), the sun's direction
    # has a latitude (declination = season) and a longitude.
    # Local solar time at the camera is the sun's longitude
    # relative to the camera's: 12:00 when they match.

    def solar_time(
        self,
        point: np.ndarray | None = None
    ) -> tuple[float, float, float] | None:
        """
        (hour 0-24, declination rad, sun elevation rad) at a
        body-space direction (default: under the camera).
        """

        if self.body is None or self.sun is None or (point is None and self.camera is None):
            return None

        if point is None:
            point = self.camera_direction()

        if point is None:
            return None

        sun_transform = self.scene.get_component(self.sun, TransformComponent).transform

        sun_world = -np.asarray(sun_transform.forward, dtype=np.float64)

        body = self.scene.try_get_component(self.body, BodyComponent)

        retrograde = body is not None and body.rotation_hours < 0.0

        return solar.solar_time(point, self.body_rotation().T @ sun_world, retrograde)

    # -----------------------------------------------------
    # The air
    # -----------------------------------------------------

    def air(
        self
    ) -> AirColumn | None:
        """The body's air column (planet/air.py), None without air."""

        return air_column_of(
            self.scene.try_get_component(self.body, AtmosphereComponent),
            self.scene.try_get_component(self.body, BodyComponent),
            giant=self.planet.palette == "bands"
        )


# =========================================================
# Flight Readout
# =========================================================

@dataclass(slots=True)
class FlightReadout:

    body: str

    # m above the body's datum (sea level; a giant's cloud
    # tops) and above the ground under the camera (None:
    # no ground, a giant).
    altitude: float
    above_ground: float | None

    latitude: float                 # deg
    longitude: float                # deg east

    # The air around the camera (None: vacuum).
    pressure: float | None          # bar
    temperature: float | None       # C

    # Local solar time and the sun's height (deg); None
    # without a sun.
    local_hour: float | None
    sun_elevation: float | None

    # s since J2000 (None without a clock).
    time: float | None


def read_flight(
    context: FlightContext,
    ground=None,
    position=None
) -> FlightReadout | None:
    """
    The readout at `position` (world m; the vessel), else at
    the camera. ground(body entity, body-space direction) ->
    (shape height m, relief m) or None (PlanetSystem.ground).
    """

    if context.body is None or (context.camera is None and position is None):
        return None

    if position is None:
        position = context.camera_transform().position

    offset = np.asarray(position, dtype=np.float64) - context.body_center()

    distance = float(np.linalg.norm(offset))

    local = context.body_rotation().T @ (offset / max(distance, 1e-9))

    planet = context.planet

    surface = ground(context.body, local) if ground is not None else None

    # Sea level (the datum on dry worlds): the body's shape.
    base, relief = surface if surface is not None else (0.0, 0.0)

    altitude = distance - planet.radius - base

    giant = planet.palette == "bands"

    above_ground = None if giant or surface is None else altitude - relief

    pressure = temperature = None

    air = context.air()

    if air is not None:

        bar = air.pressure(altitude / 1000.0)

        if bar > 1e-9:
            pressure = bar
            temperature = air.temperature(altitude / 1000.0) - 273.15

    hour = elevation = None

    # (Where the player is, not the camera: the map's camera
    # is far away.)
    current = context.solar_time(local)

    if current is not None:
        hour, _, sun = current
        elevation = math.degrees(sun)

    clock = context.clock()

    return FlightReadout(
        body=context.name(),
        altitude=altitude,
        above_ground=above_ground,
        latitude=math.degrees(math.asin(float(np.clip(local[1], -1.0, 1.0)))),
        longitude=math.degrees(math.atan2(local[0], local[2])),
        pressure=pressure,
        temperature=temperature,
        local_hour=hour,
        sun_elevation=elevation,
        time=clock.time if clock is not None else None
    )


# =========================================================
# Camera Moves
# =========================================================

def frame_body(
    context: FlightContext,
    extent: float
):
    """Back the camera off until the whole body (`extent` m) is in view."""

    scene = context.scene

    transform = context.camera_transform()

    camera = scene.get_component(context.camera, CameraComponent)

    center = context.body_center()

    offset = np.asarray(transform.position, dtype=np.float64) - center

    length = float(np.linalg.norm(offset))

    direction = offset / length if length > 0.0 else np.array([0.0, 0.0, 1.0])

    # Fits the sphere in the vertical field of view, with a
    # margin.
    distance = extent / math.sin(math.radians(camera.fov) * 0.5) * 1.25

    transform.position = center + direction * distance

    transform.orientation = quaternion.look_rotation(
        -direction,
        np.array([0.0, 1.0, 0.0]) if abs(direction[1]) < 0.99 else np.array([0.0, 0.0, -1.0])
    )


def go_to_orbit(
    context: FlightContext,
    altitude: float
):
    """Straight up from where the camera is, to `altitude` m, the horizon in view."""

    center = context.body_center()

    transform = context.camera_transform()

    offset = np.asarray(transform.position, dtype=np.float64) - center

    up = offset / max(float(np.linalg.norm(offset)), 1e-9)

    heading = transform.forward - np.dot(transform.forward, up) * up

    if np.linalg.norm(heading) < 1e-6:
        heading = transform.up - np.dot(transform.up, up) * up

    heading /= np.linalg.norm(heading)

    radius = context.planet.radius

    transform.position = center + up * (radius + altitude)

    # Pitched down so the curved horizon sits in view.
    dip = math.degrees(math.acos(radius / (radius + altitude)))

    transform.orientation = quaternion.multiply(
        quaternion.look_rotation(heading, up),
        quaternion.from_euler((-(dip + 8.0), 0.0, 0.0))
    )


def go_to_surface(
    context: FlightContext,
    spawn,
    ground
):
    """
    To a scenic spot (planet/spawn.py Spawn), 150 m up,
    looking along its view. ground: as in read_flight.
    """

    rotation = context.body_rotation()
    center = context.body_center()

    up = rotation @ spawn.direction
    view = rotation @ spawn.view_direction

    base, surface = ground(context.body, spawn.direction) or (0.0, max(spawn.elevation, 0.0))

    height = context.planet.radius + base + surface

    transform = context.camera_transform()

    transform.position = center + up * (height + 150.0)

    transform.orientation = quaternion.multiply(
        quaternion.look_rotation(view, up),
        quaternion.from_euler((-4.0, 0.0, 0.0))
    )
