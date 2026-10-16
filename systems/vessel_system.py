import math

from dataclasses import dataclass

import numpy as np

from core.handle import Handle
from core.logger import Logger

from ecs.components import (
    AtmosphereComponent,
    BodyComponent,
    HierarchyComponent,
    MeshRendererComponent,
    NameComponent,
    PlanetComponent,
    TransformComponent,
    VesselComponent
)
from ecs.entity import Entity

from graphics.material import Material
from graphics.mesh import Mesh
from graphics.mesh_factory import MeshFactory

from math3d import quaternion
from math3d.transform import Transform

from planet.air import air_column_of

from resources.resources import Resources

from scene.scene import Scene

from vessel import kepler
from vessel.parts import VesselDesign
from vessel.physics import (
    Controls,
    Environment,
    VesselState,
    can_coast,
    coast,
    land,
    step
)
from vessel.vessel import Vessel


# Gravitational constant (m^3 / kg s^2).
G = 6.6743e-11

# Pressure (bar) below which the air is not worth counting.
AIR_LIMIT_BAR = 1e-7


@dataclass(frozen=True, slots=True)
class LaunchSite:
    """On the ground: a body-frame direction, the ground's radius there (m)."""

    direction: np.ndarray
    radius: float


@dataclass(frozen=True, slots=True)
class OrbitStart:
    """In a circular orbit over a body-frame direction, at `radius` m."""

    direction: np.ndarray
    radius: float


@dataclass(slots=True)
class VesselReadout:

    name: str
    situation: str
    throttle: float
    sas: bool
    next_stage: int
    stages_left: int
    propellant: float               # kg left in the burning section
    propellant_capacity: float
    mass: float                     # kg
    thrust: float                   # N, now
    twr: float                      # full-throttle thrust / weight here
    delta_v: float                  # m/s left (vacuum)
    surface_speed: float            # m/s through the air / over the ground
    orbital_speed: float            # m/s in the body's non-rotating frame
    vertical_speed: float           # m/s, up positive
    apoapsis: float | None          # m above sea level; None escaping
    periapsis: float
    destroyed: bool
    impact_speed: float


class VesselSystem:

    # =====================================================
    # Vessels
    # =====================================================
    #
    # Flies the vessel (vessel/physics.py) in simulated time
    # (the clock's step, time warp included) and places it
    # in the world:
    #
    #   * physics in fixed steps of PHYSICS_STEP while it is
    #     thrusting, in the air or near the ground: time
    #     warp is held to PHYSICS_WARP (5x) there;
    #   * coasting above the air on an orbit that clears the
    #     ground: exact orbits, any warp ("on rails"); a
    #     falling arc goes on rails in pieces short enough
    #     not to pass through the ground;
    #   * landed: carried round with the body, any warp.
    #
    # The vessel's frame is its body's: centered on it, not
    # turning (planet/orbits.py states). Its parts are child
    # entities stacked along its +Y; burning engines get an
    # exhaust plume.

    PHYSICS_STEP = 0.02
    PHYSICS_WARP = 5.0

    # Most steps in one frame (beyond: bigger steps).
    MAX_STEPS = 200

    # Plume: length (m) at full throttle, and its widening
    # in vacuum.
    PLUME_LENGTH = 9.0

    def __init__(
        self,
        resources: Resources,
        lit_shader: Handle
    ):

        self._resources = resources
        self._lit = lit_shader

        # body entity -> (radius of its farthest ground m).
        self._extents: dict[Entity, float] = {}

        # Ground under the vessel (shape height, surface
        # above it), refreshed once a frame.
        self._ground: tuple[float, float] | None = None

        # Before the body's terrain is built: the launch
        # site's ground radius.
        self._fallback_ground: float | None = None

    # =====================================================
    # Spawning
    # =====================================================

    def spawn(
        self,
        scene: Scene,
        design: VesselDesign,
        body: Entity,
        start: LaunchSite | OrbitStart,
        orbit_system
    ) -> Entity:

        vessel = Vessel.from_design(design)

        position, frame = orbit_system.states[body]

        spin = orbit_system.spin(scene, body)

        mu = self._mu(scene, body)

        up = frame @ np.asarray(start.direction, dtype=np.float64)
        up /= np.linalg.norm(up)

        # East: along the spin at the site (the way a rocket
        # launches to orbit); any horizontal without spin.
        axis = spin / np.linalg.norm(spin) if np.linalg.norm(spin) > 0.0 else frame[:, 1]

        east = np.cross(axis, up)

        if np.linalg.norm(east) < 1e-6:
            east = np.cross(frame[:, 0], up)

        east /= np.linalg.norm(east)

        if isinstance(start, LaunchSite):

            # The nose up, its "forward" (-Z) east: pitching
            # forward turns it east.
            orientation = quaternion.look_rotation(east, up)

            state = VesselState(
                position=up * start.radius,
                velocity=np.zeros(3),
                orientation=orientation,
                angular_velocity=np.zeros(3)
            )

            self._fallback_ground = start.radius

            environment = Environment(mu, spin, lambda r, d: (0.0, 0.0), lambda d: start.radius)

            land(state, environment)

        else:

            speed = kepler.circular_speed(mu, start.radius)

            state = VesselState(
                position=up * start.radius,
                velocity=east * speed,
                # Nose prograde.
                orientation=quaternion.look_rotation(-up, east),
                angular_velocity=np.zeros(3)
            )

            self._fallback_ground = None

        entity = scene.create_entity()

        scene.add_component(entity, NameComponent(vessel.name))
        scene.add_component(entity, TransformComponent(transform=Transform()))

        component = VesselComponent(
            vessel=vessel,
            state=state,
            body=body,
            controls=Controls()
        )

        scene.add_component(entity, component)

        self._build_parts(scene, entity, component)

        self._ground = None

        self.place(scene, orbit_system)

        Logger.info(
            "[Vessels] %s: %.1f t, %.0f m/s of delta-v.",
            vessel.name,
            vessel.mass / 1000.0,
            vessel.delta_v()
        )

        return entity

    # =====================================================
    # Queries
    # =====================================================

    @staticmethod
    def find(
        scene: Scene
    ) -> tuple[Entity, VesselComponent] | None:

        return next(iter(scene.registry.view_with(VesselComponent)), None)

    @staticmethod
    def _mu(
        scene: Scene,
        body: Entity
    ) -> float:

        component = scene.try_get_component(body, BodyComponent)

        if component is not None and component.mass > 0.0:
            return G * component.mass

        planet = scene.get_component(body, PlanetComponent)

        # Earth's surface gravity on this radius.
        return 9.81 * planet.radius ** 2

    def environment(
        self,
        scene: Scene,
        body: Entity,
        orbit_system,
        planet_system,
        direction: np.ndarray | None = None
    ) -> Environment:
        """
        The body for physics. direction: inertial unit
        direction of the vessel, where the ground is looked
        up (once; physics steps within the frame reuse it).
        """

        planet = scene.get_component(body, PlanetComponent)

        _, frame = orbit_system.states[body]

        if direction is not None:

            ground = planet_system.ground(body, frame.T @ direction)

            if ground is not None:
                self._ground = ground

        if self._ground is not None:

            base, surface = self._ground

            ground_radius = planet.radius + base + surface
            sea_radius = planet.radius + base

        else:

            ground_radius = self._fallback_ground or planet.radius
            sea_radius = planet.radius

        # A giant has no ground: its air thickens without end
        # (and the vessel is crushed long before; not yet).
        if planet.palette == "bands":
            ground_radius = 0.5 * planet.radius

        column = air_column_of(
            scene.try_get_component(body, AtmosphereComponent),
            scene.try_get_component(body, BodyComponent),
            giant=planet.palette == "bands"
        )

        if column is not None:

            gas_constant = column.gas_constant

            def air(radius, direction):

                altitude = (radius - sea_radius) / 1000.0

                pressure = column.pressure(altitude)

                if pressure < AIR_LIMIT_BAR:
                    return 0.0, 0.0

                return pressure, pressure * 1e5 / (gas_constant * column.temperature(altitude))

            air_top = sea_radius + 1000.0 * column.altitude_of_pressure(AIR_LIMIT_BAR)

        else:

            def air(radius, direction):
                return 0.0, 0.0

            air_top = 0.0

        return Environment(
            mu=self._mu(scene, body),
            spin=orbit_system.spin(scene, body),
            air=air,
            ground=lambda d: ground_radius,
            air_top=air_top
        )

    def _extent(
        self,
        body: Entity,
        planet_system,
        scene: Scene
    ) -> float:
        """Farthest the ground reaches from the center (m)."""

        extent = planet_system.extent(body)

        if extent is None:
            extent = scene.get_component(body, PlanetComponent).radius * 1.02

        return extent

    # =====================================================
    # Time Warp
    # =====================================================

    def max_warp(
        self,
        scene: Scene,
        orbit_system,
        planet_system,
        frame_time: float = 1.0 / 30.0
    ) -> float:
        """
        The fastest the clock may run for the vessel (inf: no
        limit). frame_time: a slow frame's length (s).
        """

        found = self.find(scene)

        if found is None or found[1].body not in orbit_system.states:
            return math.inf

        _, component = found

        state = component.state

        thrusting = component.vessel.throttle > 0.0 and bool(component.vessel.burning_engines())

        if state.destroyed or (state.landed and not thrusting):
            return math.inf

        environment = self.environment(scene, component.body, orbit_system, planet_system)

        if not can_coast(component.vessel, state, environment):
            return self.PHYSICS_WARP

        extent = self._extent(component.body, planet_system, scene)

        shape = kepler.orbit_shape(state.position, state.velocity, environment.mu)

        if shape.periapsis > max(extent, environment.air_top):
            return math.inf

        # Falling: no faster than reaching the ground (or the
        # air) in about four frames.
        radius = float(np.linalg.norm(state.position))
        speed = max(float(np.linalg.norm(state.velocity)), 1.0)

        clearance = radius - max(extent, environment.air_top)

        return max(self.PHYSICS_WARP, clearance / speed / (4.0 * frame_time))

    # =====================================================
    # Update
    # =====================================================

    def update(
        self,
        scene: Scene,
        dt: float,
        orbit_system,
        planet_system
    ) -> list[str]:
        """
        Fly the vessel `dt` simulated seconds. Returns events
        for the HUD ("destroyed", "landed", "staged").
        """

        found = self.find(scene)

        if found is None:
            return []

        entity, component = found

        if component.body not in orbit_system.states:
            return []

        vessel = component.vessel
        state = component.state

        events = []

        was_landed = state.landed

        if dt > 0.0 and not state.destroyed:

            direction = state.position / np.linalg.norm(state.position)

            environment = self.environment(scene, component.body, orbit_system, planet_system, direction)

            self._fly(vessel, state, environment, component.controls, dt, self._extent(component.body, planet_system, scene))

            if state.destroyed:

                events.append("destroyed")

                Logger.info("[Vessels] %s hit the ground at %.0f m/s.", vessel.name, state.impact_speed)

                self._clear_parts(scene, component)

            elif state.landed and not was_landed:

                events.append("landed")

        if vessel.dropped:

            self._drop_parts(scene, component)

            events.append("staged")

        self._update_plumes(scene, entity, component, orbit_system, planet_system)

        self.place(scene, orbit_system)

        return events

    def _fly(
        self,
        vessel: Vessel,
        state: VesselState,
        environment: Environment,
        controls: Controls,
        dt: float,
        extent: float
    ):

        remaining = dt

        steps = 0

        while remaining > 1e-9 and not state.destroyed:

            steps += 1

            if state.landed and (vessel.throttle <= 0.0 or not vessel.burning_engines()):

                # Riding the ground: one exact step.
                step(vessel, state, environment, controls, remaining)

                return

            if can_coast(vessel, state, environment):

                shape = kepler.orbit_shape(state.position, state.velocity, environment.mu)

                floor = max(extent, environment.air_top)

                if shape.periapsis > floor:

                    coast(state, environment, remaining)

                    return

                # A falling arc: on rails while well clear.
                radius = float(np.linalg.norm(state.position))
                speed = max(float(np.linalg.norm(state.velocity)), 1.0)

                piece = 0.25 * (radius - floor) / speed

                if piece > 10.0 * self.PHYSICS_STEP:

                    piece = min(piece, remaining)

                    coast(state, environment, piece)

                    remaining -= piece

                    continue

            # Physics; bigger steps past MAX_STEPS.
            delta = self.PHYSICS_STEP if steps < self.MAX_STEPS else remaining

            delta = min(delta, remaining)

            step(vessel, state, environment, controls, delta)

            remaining -= delta

    # =====================================================
    # Placing and Drawing
    # =====================================================

    def place(
        self,
        scene: Scene,
        orbit_system
    ):
        """The vessel's entity where its state says, in the world."""

        found = self.find(scene)

        if found is None:
            return

        entity, component = found

        body_state = orbit_system.states.get(component.body)

        if body_state is None or orbit_system.inertial_to_world is None:
            return

        body_position, _ = body_state

        world = orbit_system.world_point(body_position + component.state.position)

        if world is None:
            return

        transform = scene.get_component(entity, TransformComponent).transform

        transform.position = world
        transform.orientation = quaternion.normalize(
            quaternion.multiply(
                quaternion.from_matrix3(orbit_system.inertial_to_world),
                component.state.orientation
            )
        )

    def world_up(
        self,
        scene: Scene,
        orbit_system
    ) -> np.ndarray | None:
        """The radial "up" at the vessel, in the world."""

        found = self.find(scene)

        if found is None or orbit_system.inertial_to_world is None:
            return None

        position = found[1].state.position

        return orbit_system.inertial_to_world @ (position / np.linalg.norm(position))

    def _material(
        self,
        key: str,
        color,
        metallic: float,
        roughness: float,
        emissive=(0.0, 0.0, 0.0)
    ) -> Handle:

        lit = self._lit

        def create():

            material = Material(lit)

            material.set_vec3("uBaseColor", tuple(color))
            material.set_float("uMetallic", metallic)
            material.set_float("uRoughness", roughness)
            material.set_vec3("uEmissive", tuple(emissive))

            return material

        return self._resources.materials.load(key, create)

    def _frustum(
        self,
        radius_top: float,
        radius_bottom: float,
        height: float
    ) -> Handle:

        key = f"vessel/frustum_{radius_top:.3f}_{radius_bottom:.3f}_{height:.3f}"

        return self._resources.meshes.load(
            key,
            lambda: Mesh.from_data(MeshFactory.frustum_data(radius_top, radius_bottom, height))
        )

    def _build_parts(
        self,
        scene: Scene,
        entity: Entity,
        component: VesselComponent
    ):

        for state in component.vessel.parts:

            part = state.part

            child = scene.create_entity()

            scene.add_component(child, NameComponent(part.name))
            scene.add_component(child, HierarchyComponent(entity))
            scene.add_component(child, TransformComponent(transform=Transform(position=(0.0, state.bottom, 0.0))))
            scene.add_component(
                child,
                MeshRendererComponent(
                    mesh=self._frustum(part.radius_top, part.radius_bottom, part.height),
                    material=self._material(f"vessel/{part.id}", part.color, part.metallic, part.roughness)
                )
            )

            component.part_entities.append(child)

    def _restack_parts(
        self,
        scene: Scene,
        component: VesselComponent
    ):

        for state, child in zip(component.vessel.parts, component.part_entities):

            transform = scene.get_component(child, TransformComponent).transform

            transform.position = (0.0, state.bottom, 0.0)

    def _drop_parts(
        self,
        scene: Scene,
        component: VesselComponent
    ):
        """Separated parts leave (not flown on: they fall away out of sight)."""

        count = len(component.vessel.dropped)

        for child in component.part_entities[-count:]:
            scene.destroy_entity(child)

        del component.part_entities[-count:]

        component.vessel.dropped.clear()

        self._restack_parts(scene, component)

    def _clear_parts(
        self,
        scene: Scene,
        component: VesselComponent
    ):

        for child in component.part_entities + component.plume_entities:
            scene.destroy_entity(child)

        component.part_entities.clear()
        component.plume_entities.clear()

    def _update_plumes(
        self,
        scene: Scene,
        entity: Entity,
        component: VesselComponent,
        orbit_system,
        planet_system
    ):
        """A glowing cone under each burning engine, longer with the throttle, wider in thin air."""

        vessel = component.vessel

        engines = [] if component.state.destroyed or vessel.throttle <= 0.0 else vessel.burning_engines()

        while len(component.plume_entities) > len(engines):
            scene.destroy_entity(component.plume_entities.pop())

        if not engines:
            return

        state = component.state

        environment = self.environment(scene, component.body, orbit_system, planet_system)

        radius = float(np.linalg.norm(state.position))

        pressure, _ = environment.air(radius, state.position / radius)

        # Exhaust: a narrowing flame at sea level, spreading
        # wider than the nozzle as the air thins.
        thin = 1.0 - min(pressure / 1.01325, 1.0)

        mesh = self._frustum(1.0, 0.35, 1.0) if thin < 0.5 else self._frustum(1.0, 1.8, 1.0)

        widen = 1.0 + 0.6 * thin

        material = self._material("vessel/plume", (0.0, 0.0, 0.0), 0.0, 1.0, emissive=(2.6, 1.3, 0.45))

        while len(component.plume_entities) < len(engines):

            child = scene.create_entity()

            scene.add_component(child, NameComponent("Exhaust"))
            scene.add_component(child, HierarchyComponent(entity))
            scene.add_component(child, TransformComponent(transform=Transform()))
            scene.add_component(child, MeshRendererComponent(mesh=mesh, material=material, casts_shadows=False))

            component.plume_entities.append(child)

        for (_, engine), child in zip(engines, component.plume_entities):

            length = self.PLUME_LENGTH * (0.3 + 0.7 * vessel.throttle) * (1.0 + 0.5 * thin)

            nozzle = engine.part.radius_bottom * 0.85

            transform = scene.get_component(child, TransformComponent).transform

            transform.position = (0.0, engine.bottom - length, 0.0)
            transform.scale = (nozzle * widen, length, nozzle * widen)

            # (Narrow or spreading, as the air is now.)
            scene.get_component(child, MeshRendererComponent).mesh = mesh

    # =====================================================
    # Readout
    # =====================================================

    def readout(
        self,
        scene: Scene,
        orbit_system,
        planet_system
    ) -> VesselReadout | None:

        found = self.find(scene)

        if found is None or found[1].body not in orbit_system.states:
            return None

        _, component = found

        vessel = component.vessel
        state = component.state

        environment = self.environment(scene, component.body, orbit_system, planet_system)

        planet = scene.get_component(component.body, PlanetComponent)

        sea_radius = planet.radius + (self._ground[0] if self._ground is not None else 0.0)

        radius = float(np.linalg.norm(state.position))

        up = state.position / radius

        pressure, _ = environment.air(radius, up)

        relative = state.velocity - np.cross(environment.spin, state.position)

        shape = kepler.orbit_shape(state.position, state.velocity, environment.mu)

        weight = vessel.mass * environment.mu / radius ** 2

        left, capacity = vessel.section_propellant()

        if state.destroyed:
            situation = "Destroyed"
        elif state.landed:
            situation = "Landed"
        elif pressure > 0.0:
            situation = "Flying"
        elif shape.apoapsis is None:
            situation = "Escaping"
        elif shape.periapsis > max(self._extent(component.body, planet_system, scene), environment.air_top):
            situation = "Orbiting"
        else:
            situation = "Sub-orbital"

        return VesselReadout(
            name=vessel.name,
            situation=situation,
            throttle=vessel.throttle,
            sas=component.controls.sas,
            next_stage=vessel.next_stage,
            stages_left=vessel.stages_left,
            propellant=left,
            propellant_capacity=capacity,
            mass=vessel.mass,
            thrust=vessel.thrust(pressure)[0],
            twr=vessel.max_thrust(pressure) / weight if weight > 0.0 else 0.0,
            delta_v=vessel.delta_v(),
            surface_speed=float(np.linalg.norm(relative)),
            orbital_speed=float(np.linalg.norm(state.velocity)),
            vertical_speed=float(np.dot(state.velocity, up)),
            apoapsis=shape.apoapsis - sea_radius if shape.apoapsis is not None else None,
            periapsis=shape.periapsis - sea_radius,
            destroyed=state.destroyed,
            impact_speed=state.impact_speed
        )
