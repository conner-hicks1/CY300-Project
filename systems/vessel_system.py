import math

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np

from core.handle import Handle
from core.logger import Logger

from ecs.components import (
    AtmosphereComponent,
    BodyComponent,
    DebrisComponent,
    HierarchyComponent,
    MeshRendererComponent,
    NameComponent,
    OrbitComponent,
    PlanetComponent,
    StarComponent,
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

from vessel import kepler, soi
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

# The Sun's key in patched conics (it is not a body entity:
# the solar system's positions are measured from it).
SUN = None

SUN_RADIUS = 6.957e8


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


@dataclass(frozen=True, slots=True)
class _BodyPhysics:

    radius: float
    giant: bool
    solid: bool                     # has ground (not a giant, not the Sun)
    mu: float
    spin: np.ndarray
    column: object | None           # planet/air.py AirColumn
    air_top: float | None           # m above sea level


@dataclass(slots=True)
class VesselReadout:

    name: str
    body: str                       # what it flies around
    situation: str
    throttle: float
    sas: bool
    sas_mode: str
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
    apoapsis: float | None          # m above sea level; None escaping (or landed)
    periapsis: float | None         # None landed
    destroyed: bool
    impact_speed: float
    parachute: str                  # "", "armed", "open"


def copy_positions(
    state
) -> list[tuple[float, float, float]]:
    """
    Where a part's copies stand in the vessel's frame: the
    core's on the axis; a radial group's evenly round it,
    the first on +X (to the sides as the camera first sees
    a vessel on the pad).
    """

    if not state.radial:
        return [(0.0, state.bottom, 0.0)]

    return [
        (
            state.distance * math.cos(2.0 * math.pi * k / state.count),
            state.bottom,
            state.distance * math.sin(2.0 * math.pi * k / state.count)
        )
        for k in range(state.count)
    ]


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
    # Patched conics (vessel/soi.py): the vessel feels one
    # body at a time, the one whose sphere of influence it is
    # in; leaving it (or entering a moon's) it is handed over,
    # its state re-expressed about the new body. Time warp
    # slows ahead of a boundary so the handover is not
    # skipped. The Sun is a body too (SUN), for a vessel that
    # escapes its planet.
    #
    # The vessel's frame is its body's: centered on it, not
    # turning (planet/orbits.py states). Its parts are child
    # entities stacked along its +Y; burning engines get an
    # exhaust plume, open parachutes a canopy. Spent stages
    # fall as debris until they hit the ground or are left
    # DEBRIS_RANGE behind.

    PHYSICS_STEP = 0.02
    PHYSICS_WARP = 5.0

    # Most steps in one frame (beyond: bigger steps).
    MAX_STEPS = 200

    # Plume: length (m) at full throttle.
    PLUME_LENGTH = 9.0

    # Debris: dropped this far behind (m) or this old (s).
    DEBRIS_RANGE = 25_000.0
    DEBRIS_LIFETIME = 600.0

    # Separated parts are pushed apart at (m/s).
    SEPARATION_SPEED = 2.0

    def __init__(
        self,
        resources: Resources,
        lit_shader: Handle
    ):

        self._resources = resources
        self._lit = lit_shader

        # Ground under the vessel (shape height, surface
        # above it), refreshed once a frame.
        self._ground: tuple[float, float] | None = None

        # Before the body's terrain is built: the launch
        # site's ground radius.
        self._fallback_ground: float | None = None

        # Height above the ground last frame (m): how
        # accurately the ground needs looking up.
        self._height = 0.0

        # body key -> _BodyPhysics.
        self._bodies: dict[Hashable, _BodyPhysics] = {}

        # The map's prediction: (clock time, body key,
        # Prediction).
        self._prediction = None

        # Where the vessel was launched from: (body, start),
        # for the part editor to rebuild it there.
        self._launch: tuple | None = None

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

        self._launch = (body, start)

        # A new scene: its entities may reuse old ones' ids.
        self._bodies.clear()
        self._height = 0.0
        self._prediction = None

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

            self._height = start.radius

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

    def rebuild(
        self,
        scene: Scene,
        design: VesselDesign | None,
        orbit_system
    ) -> Entity | None:
        """
        A different vessel where the last one was launched
        from (the part editor): the old one and its debris
        go. design None or empty: none at all.
        """

        found = self.find(scene)

        if found is not None:
            self._destroy_tree(scene, found[0])

        for entity, _ in list(scene.registry.view_with(DebrisComponent)):
            self._destroy_tree(scene, entity)

        if self._launch is None or design is None or not design.stack:
            return None

        body, start = self._launch

        if body not in orbit_system.states:
            return None

        return self.spawn(scene, design, body, start, orbit_system)

    def highlight(
        self,
        scene: Scene,
        selection: tuple | None
    ):
        """
        Light up the editor's selection: a part by its place
        in the design (PartState.source), or a whole radial
        group by (core index, group); None: nothing.
        """

        found = self.find(scene)

        if found is None:
            return

        _, component = found

        for state in component.vessel.parts:

            part = state.part

            lit = selection is not None and (
                state.source == tuple(selection)
                or (len(selection) == 2 and state.source[:2] == tuple(selection))
            )

            for child in component.part_entities.get(id(state), []):

                renderer = scene.try_get_component(child, MeshRendererComponent)

                if renderer is None:
                    continue

                if lit:
                    renderer.material = self._material(
                        f"vessel/{part.id}/selected",
                        tuple(min(1.0, c * 0.6 + 0.4) for c in part.color),
                        part.metallic,
                        part.roughness,
                        emissive=(0.35, 0.22, 0.05)
                    )
                else:
                    renderer.material = self._material(f"vessel/{part.id}", part.color, part.metallic, part.roughness)

    # =====================================================
    # Queries
    # =====================================================

    @property
    def launch_body(
        self
    ) -> Entity | None:
        """The body the vessel was launched from (None: none yet)."""

        return self._launch[0] if self._launch is not None else None

    @property
    def ground_accuracy(
        self
    ) -> float:
        """How closely (m) the ground under the vessel is worth knowing now."""

        return 0.02 * max(self._height, 0.0)

    @staticmethod
    def find(
        scene: Scene
    ) -> tuple[Entity, VesselComponent] | None:

        return next(iter(scene.registry.view_with(VesselComponent)), None)

    @staticmethod
    def _mu(
        scene: Scene,
        body: Hashable
    ) -> float:

        if body is SUN:
            return soi.SUN_MU

        component = scene.try_get_component(body, BodyComponent)

        if component is not None and component.mass > 0.0:
            return G * component.mass

        planet = scene.get_component(body, PlanetComponent)

        # Earth's surface gravity on this radius.
        return 9.81 * planet.radius ** 2

    @staticmethod
    def body_name(
        scene: Scene,
        body: Hashable
    ) -> str:

        if body is SUN:
            return "the Sun"

        name = scene.try_get_component(body, NameComponent)

        return name.name if name is not None else "body"

    @staticmethod
    def reference_state(
        orbit_system,
        body: Hashable
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """(position about the star m, frame) of a body; the Sun's: the origin."""

        if body is SUN:
            return np.zeros(3), np.eye(3)

        return orbit_system.states.get(body)

    @staticmethod
    def _star_radius(
        scene: Scene
    ) -> float:

        star = next((component for _, component in scene.registry.view_with(StarComponent)), None)

        return star.radius if star is not None else SUN_RADIUS

    def _body_physics(
        self,
        scene: Scene,
        body: Hashable,
        orbit_system
    ) -> _BodyPhysics:
        """What does not change about a body (cached per body)."""

        physics = self._bodies.get(body)

        if physics is not None:
            return physics

        if body is SUN:

            physics = _BodyPhysics(
                radius=self._star_radius(scene),
                giant=False,
                solid=False,
                mu=soi.SUN_MU,
                spin=np.zeros(3),
                column=None,
                air_top=None
            )

        else:

            planet = scene.get_component(body, PlanetComponent)

            giant = planet.palette == "bands"

            column = air_column_of(
                scene.try_get_component(body, AtmosphereComponent),
                scene.try_get_component(body, BodyComponent),
                giant=giant
            )

            physics = _BodyPhysics(
                radius=planet.radius,
                giant=giant,
                solid=not giant,
                mu=self._mu(scene, body),
                spin=orbit_system.spin(scene, body),
                column=column,
                # Where the air stops counting, m above sea level.
                air_top=1000.0 * column.altitude_of_pressure(AIR_LIMIT_BAR) if column is not None else None
            )

        self._bodies[body] = physics

        return physics

    def environment(
        self,
        scene: Scene,
        body: Hashable,
        orbit_system,
        planet_system,
        direction: np.ndarray | None = None
    ) -> Environment:
        """
        The body for physics. direction: inertial unit
        direction of the vessel, where the ground is looked
        up (once; physics steps within the frame reuse it),
        as accurately as the height above it needs.
        """

        physics = self._body_physics(scene, body, orbit_system)

        if direction is not None and physics.solid:

            _, frame = orbit_system.states[body]

            # Within ~2% of the height above the ground: exact
            # near it, coarse (and cheap) high above.
            ground = planet_system.ground(body, frame.T @ direction, self.ground_accuracy)

            if ground is not None:
                self._ground = ground

        if physics.solid and self._ground is not None:

            base, surface = self._ground

            ground_radius = physics.radius + base + surface
            sea_radius = physics.radius + base

        else:

            ground_radius = (self._fallback_ground or physics.radius) if physics.solid else physics.radius
            sea_radius = physics.radius

        # A giant has no ground: its air thickens without end
        # (and the vessel is crushed long before; not yet).
        if physics.giant:
            ground_radius = 0.5 * physics.radius

        column = physics.column

        if column is not None:

            gas_constant = column.gas_constant

            def air(radius, direction):

                altitude = (radius - sea_radius) / 1000.0

                pressure = column.pressure(altitude)

                if pressure < AIR_LIMIT_BAR:
                    return 0.0, 0.0

                return pressure, pressure * 1e5 / (gas_constant * column.temperature(altitude))

            air_top = sea_radius + physics.air_top

        else:

            def air(radius, direction):
                return 0.0, 0.0

            air_top = 0.0

        return Environment(
            mu=physics.mu,
            spin=physics.spin,
            air=air,
            ground=lambda d: ground_radius,
            air_top=air_top
        )

    def _extent(
        self,
        body: Hashable,
        planet_system,
        scene: Scene,
        orbit_system
    ) -> float:
        """Farthest the ground reaches from the center (m)."""

        if body is SUN:
            return self._star_radius(scene)

        extent = planet_system.extent(body)

        if extent is None:
            extent = scene.get_component(body, PlanetComponent).radius * 1.02

        return extent

    # =====================================================
    # Spheres of Influence
    # =====================================================

    def soi_bodies(
        self,
        scene: Scene
    ) -> dict[Hashable, soi.Body]:
        """Every body for patched conics (the Sun included)."""

        registry = scene.registry

        by_name = {
            name.name: entity
            for entity, name in registry.view_with(NameComponent)
            if registry.has(entity, OrbitComponent)
        }

        masses = {}

        for entity, body in registry.view_with(BodyComponent):
            masses[entity] = body.mass

        sun_mass = soi.SUN_MU / G

        bodies = {
            SUN: soi.Body(SUN, "the Sun", soi.SUN_MU, self._star_radius(scene), math.inf, SUN)
        }

        for entity, orbit, planet in registry.view_with(OrbitComponent, PlanetComponent):

            parent = by_name.get(orbit.parent) if orbit.parent else SUN

            mass = masses.get(entity, 0.0)

            parent_mass = masses.get(parent, 0.0) if parent is not SUN else sun_mass

            radius = (
                soi.soi_radius(orbit.semi_major_axis, mass, parent_mass)
                if mass > 0.0 and parent_mass > 0.0
                else math.inf
            )

            bodies[entity] = soi.Body(
                entity,
                self.body_name(scene, entity),
                self._mu(scene, entity),
                planet.radius,
                radius,
                parent
            )

        return bodies

    def _children(
        self,
        bodies: dict[Hashable, soi.Body],
        reference: Hashable,
        orbit_system
    ) -> list[Hashable]:

        return [
            key
            for key, body in bodies.items()
            if key is not SUN and body.parent == reference and key != reference and key in orbit_system.states
        ]

    def relative_states(
        self,
        scene: Scene,
        orbit_system,
        reference: Hashable,
        keys
    ) -> dict[Hashable, tuple[np.ndarray, np.ndarray]]:
        """(position, velocity) of bodies about `reference`, now."""

        velocities = orbit_system.velocities(scene)

        origin, _ = self.reference_state(orbit_system, reference)
        origin_velocity = velocities.get(reference, np.zeros(3)) if reference is not SUN else np.zeros(3)

        return {
            key: (
                orbit_system.states[key][0] - origin,
                velocities.get(key, np.zeros(3)) - origin_velocity
            )
            for key in keys
        }

    def _hand_over(
        self,
        scene: Scene,
        component: VesselComponent,
        orbit_system
    ) -> str | None:
        """Into another body's sphere of influence, if the vessel has crossed one. Returns its name."""

        state = component.state

        if state.landed or state.destroyed:
            return None

        bodies = self.soi_bodies(scene)

        reference = component.body

        if reference not in bodies:
            return None

        children = self._children(bodies, reference, orbit_system)

        origin, _ = self.reference_state(orbit_system, reference)

        child_positions = {key: orbit_system.states[key][0] - origin for key in children}

        changed, target = soi.transition(state.position, bodies[reference], bodies, child_positions, state.velocity)

        if not changed or (target is not SUN and target not in orbit_system.states):
            return None

        velocities = orbit_system.velocities(scene)

        def frame_of(key):

            position, _ = self.reference_state(orbit_system, key)

            velocity = np.zeros(3) if key is SUN else velocities.get(key, np.zeros(3))

            return position, velocity

        old_position, old_velocity = frame_of(reference)
        new_position, new_velocity = frame_of(target)

        state.position, state.velocity = soi.rebase(
            state.position,
            state.velocity,
            old_position,
            old_velocity,
            new_position,
            new_velocity
        )

        component.body = target

        # A new body: its ground, unknown yet; far from it.
        self._ground = None
        self._fallback_ground = None
        self._height = float(np.linalg.norm(state.position))
        self._prediction = None

        name = self.body_name(scene, target)

        Logger.info("[Vessels] %s is now in %s's sphere of influence.", component.vessel.name, name)

        return name

    def prediction(
        self,
        scene: Scene,
        orbit_system
    ) -> soi.Prediction | None:
        """The vessel's path ahead (the map's orbit lines); recomputed as the clock moves."""

        found = self.find(scene)

        if found is None:
            return None

        _, component = found

        if component.state.destroyed:
            return None

        key = (orbit_system.time, component.body)

        if self._prediction is not None and self._prediction[0] == key:
            return self._prediction[1]

        bodies = self.soi_bodies(scene)

        if component.body not in bodies:
            return None

        children = self._children(bodies, component.body, orbit_system)

        # Round the Sun: years, finely (a planet's sphere is
        # crossed in days).
        around_sun = component.body is SUN

        prediction = soi.predict(
            component.state.position,
            component.state.velocity,
            bodies[component.body],
            bodies,
            self.relative_states(scene, orbit_system, component.body, children),
            samples=2048 if around_sun else 256,
            horizon=3.0 * 365.25 * 86_400.0 if around_sun else 30.0 * 86_400.0
        )

        self._prediction = (key, prediction)

        return prediction

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

        if found is None or self.reference_state(orbit_system, found[1].body) is None:
            return math.inf

        _, component = found

        state = component.state

        thrusting = component.vessel.throttle > 0.0 and bool(component.vessel.burning_engines())

        if state.destroyed or (state.landed and not thrusting):
            return math.inf

        environment = self.environment(scene, component.body, orbit_system, planet_system)

        if not can_coast(component.vessel, state, environment):
            return self.PHYSICS_WARP

        limit = math.inf

        radius = float(np.linalg.norm(state.position))
        speed = max(float(np.linalg.norm(state.velocity)), 1.0)

        # No faster than reaching a boundary in about four
        # frames: the ground (or the air)...
        floor = max(self._extent(component.body, planet_system, scene, orbit_system), environment.air_top)

        shape = kepler.orbit_shape(state.position, state.velocity, environment.mu)

        if shape.periapsis <= floor:
            limit = min(limit, max(self.PHYSICS_WARP, (radius - floor) / speed / (4.0 * frame_time)))

        # ...or another body's sphere of influence.
        bodies = self.soi_bodies(scene)

        reference = bodies.get(component.body)

        outbound = float(np.dot(state.position, state.velocity)) > 0.0

        if reference is not None:

            if outbound and (shape.apoapsis is None or shape.apoapsis > reference.soi):
                limit = min(limit, max(self.PHYSICS_WARP, (reference.soi - radius) / speed / (4.0 * frame_time)))

            children = self._children(bodies, component.body, orbit_system)

            relative = self.relative_states(scene, orbit_system, component.body, children)

            for key in children:

                where, moving = relative[key]

                gap = float(np.linalg.norm(state.position - where)) - bodies[key].soi

                # Closing at about the speed between them now
                # (with room for the paths' curving and the
                # child's pull near it); a planet met at a few
                # km/s, not the ~50 km/s of both round the Sun.
                closing = 1.5 * float(np.linalg.norm(state.velocity - moving)) + math.sqrt(
                    2.0 * bodies[key].mu / max(bodies[key].soi, 1.0)
                ) + 1.0

                limit = min(limit, max(self.PHYSICS_WARP, gap / closing / (4.0 * frame_time)))

        return limit

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
        for the HUD ("destroyed", "landed", "staged",
        "soi").
        """

        found = self.find(scene)

        if found is None:
            return []

        entity, component = found

        if self.reference_state(orbit_system, component.body) is None:
            return []

        vessel = component.vessel
        state = component.state

        events = []

        # Separated parts first: the vessel goes on from
        # where its new bottom is.
        if vessel.dropped:

            self._drop_parts(scene, entity, component)

            events.append("staged")

        was_landed = state.landed

        environment = None

        if dt > 0.0 and not state.destroyed:

            direction = state.position / np.linalg.norm(state.position)

            environment = self.environment(scene, component.body, orbit_system, planet_system, direction)

            extent = self._extent(component.body, planet_system, scene, orbit_system)

            self._fly(vessel, state, environment, component.controls, dt, extent)

            self._height = float(np.linalg.norm(state.position)) - environment.ground(direction)

            if state.destroyed:

                events.append("destroyed")

                Logger.info("[Vessels] %s hit the ground at %.0f m/s.", vessel.name, state.impact_speed)

                self._clear_parts(scene, component)

            elif state.landed and not was_landed:

                events.append("landed")

            elif self._hand_over(scene, component, orbit_system) is not None:

                events.append("soi")

        self._update_debris(scene, component, environment, dt, planet_system, orbit_system)

        self._update_plumes(scene, entity, component, orbit_system, planet_system)
        self._update_canopies(scene, entity, component)

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
    # Debris
    # =====================================================

    def _update_debris(
        self,
        scene: Scene,
        component: VesselComponent,
        environment: Environment | None,
        dt: float,
        planet_system,
        orbit_system
    ):
        """Spent stages fall (gravity, air) until they land, are left behind, or the vessel moves on."""

        for entity, debris in list(scene.registry.view_with(DebrisComponent)):

            gone = (
                debris.body is not component.body
                and debris.body != component.body
            )

            if not gone and dt > 0.0 and environment is not None:

                debris.age += dt

                self._fly(
                    debris.vessel,
                    debris.state,
                    environment,
                    Controls(sas=False),
                    dt,
                    self._extent(debris.body, planet_system, scene, orbit_system)
                )

                distance = float(np.linalg.norm(debris.state.position - component.state.position))

                gone = (
                    debris.state.landed
                    or debris.state.destroyed
                    or distance > self.DEBRIS_RANGE
                    or debris.age > self.DEBRIS_LIFETIME
                )

            if gone:

                self._destroy_tree(scene, entity)

    def _destroy_tree(
        self,
        scene: Scene,
        entity: Entity
    ):

        for child, hierarchy in list(scene.registry.view_with(HierarchyComponent)):

            if hierarchy.parent == entity:
                scene.destroy_entity(child)

        scene.destroy_entity(entity)

    # =====================================================
    # Placing and Drawing
    # =====================================================

    def _world_transform(
        self,
        orbit_system,
        body: Hashable,
        state: VesselState
    ) -> tuple[np.ndarray, np.ndarray] | None:

        reference = self.reference_state(orbit_system, body)

        if reference is None or orbit_system.inertial_to_world is None:
            return None

        world = orbit_system.world_point(reference[0] + state.position)

        if world is None:
            return None

        orientation = quaternion.normalize(
            quaternion.multiply(
                quaternion.from_matrix3(orbit_system.inertial_to_world),
                state.orientation
            )
        )

        return world, orientation

    def place(
        self,
        scene: Scene,
        orbit_system
    ):
        """The vessel's (and its debris's) entities where their states say, in the world."""

        placed = [(entity, component.body, component.state) for entity, component in scene.registry.view_with(VesselComponent)]
        placed += [(entity, debris.body, debris.state) for entity, debris in scene.registry.view_with(DebrisComponent)]

        for entity, body, state in placed:

            where = self._world_transform(orbit_system, body, state)

            if where is None:
                continue

            transform = scene.get_component(entity, TransformComponent).transform

            transform.position, transform.orientation = where

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

        component.part_entities = {}

        for state in component.vessel.parts:

            part = state.part

            copies = []

            for position in copy_positions(state):

                child = scene.create_entity()

                scene.add_component(child, NameComponent(part.name))
                scene.add_component(child, HierarchyComponent(entity))
                scene.add_component(child, TransformComponent(transform=Transform(position=position)))
                scene.add_component(
                    child,
                    MeshRendererComponent(
                        mesh=self._frustum(part.radius_top, part.radius_bottom, part.height),
                        material=self._material(f"vessel/{part.id}", part.color, part.metallic, part.roughness)
                    )
                )

                copies.append(child)

            component.part_entities[id(state)] = copies

    def _restack_parts(
        self,
        scene: Scene,
        component: VesselComponent
    ):

        for state in component.vessel.parts:

            for child, position in zip(component.part_entities.get(id(state), []), copy_positions(state)):

                scene.get_component(child, TransformComponent).transform.position = position

    def _drop_parts(
        self,
        scene: Scene,
        entity: Entity,
        component: VesselComponent
    ):
        """
        Separated parts fall away as debris; the vessel goes
        on from where its new bottom part is (its origin moves
        up by what was dropped, so nothing jumps).
        """

        vessel = component.vessel
        state = component.state

        dropped = list(vessel.dropped)

        vessel.dropped.clear()

        dropped_entities = [
            child
            for part in dropped
            for child in component.part_entities.pop(id(part), [])
        ]

        # (Only the core's height moves the origin: dropped
        # side boosters leave the stack as it was.)
        lift = sum(part.part.height for part in dropped if not part.radial)

        up = state.up

        # The debris: where the stack was, pushed back.
        debris_state = state.copy()
        debris_state.velocity = state.velocity - up * self.SEPARATION_SPEED
        debris_state.landed = False
        debris_state.angular_velocity = np.zeros(3)

        for part in dropped:
            part.ignited = False

        state.position = state.position + up * lift

        if not state.destroyed and not state.landed:

            debris = scene.create_entity()

            scene.add_component(debris, NameComponent("Debris"))
            scene.add_component(debris, TransformComponent(transform=Transform()))
            scene.add_component(
                debris,
                DebrisComponent(
                    vessel=Vessel("Debris", dropped),
                    state=debris_state,
                    body=component.body
                )
            )

            for child in dropped_entities:
                scene.get_component(child, HierarchyComponent).parent = debris

        else:

            for child in dropped_entities:
                scene.destroy_entity(child)

        self._restack_parts(scene, component)

    def _clear_parts(
        self,
        scene: Scene,
        component: VesselComponent
    ):

        parts = [child for copies in component.part_entities.values() for child in copies]

        for child in parts + component.plume_entities + component.canopy_entities:
            scene.destroy_entity(child)

        component.part_entities.clear()
        component.plume_entities.clear()
        component.canopy_entities.clear()

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

        engines = (
            []
            if component.state.destroyed or vessel.throttle <= 0.0
            else [
                (state, position)
                for _, state in vessel.burning_engines()
                for position in copy_positions(state)
            ]
        )

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

        for (engine, (x, _, z)), child in zip(engines, component.plume_entities):

            length = self.PLUME_LENGTH * (0.3 + 0.7 * vessel.throttle) * (1.0 + 0.5 * thin) * min(1.0, engine.part.radius_bottom / 0.9)

            nozzle = engine.part.radius_bottom * 0.85

            transform = scene.get_component(child, TransformComponent).transform

            transform.position = (x, engine.bottom - length, z)
            transform.scale = (nozzle * widen, length, nozzle * widen)

            # (Narrow or spreading, as the air is now.)
            scene.get_component(child, MeshRendererComponent).mesh = mesh

    def _update_canopies(
        self,
        scene: Scene,
        entity: Entity,
        component: VesselComponent
    ):
        """An orange dome over each opening parachute, its lines' length above the vessel."""

        chutes = [] if component.state.destroyed else [p for p in component.vessel.parachutes if p.deployed > 0.0]

        while len(component.canopy_entities) > len(chutes):
            scene.destroy_entity(component.canopy_entities.pop())

        if not chutes:
            return

        mesh = self._frustum(0.3, 1.0, 0.45)

        material = self._material("vessel/canopy", (0.95, 0.42, 0.1), 0.0, 0.85)

        while len(component.canopy_entities) < len(chutes):

            child = scene.create_entity()

            scene.add_component(child, NameComponent("Canopy"))
            scene.add_component(child, HierarchyComponent(entity))
            scene.add_component(child, TransformComponent(transform=Transform()))
            scene.add_component(child, MeshRendererComponent(mesh=mesh, material=material))

            component.canopy_entities.append(child)

        for chute, child in zip(chutes, component.canopy_entities):

            if chute.radial:
                continue

            full = math.sqrt(chute.part.parachute.area / math.pi)

            # Streams out, then blooms.
            radius = full * max(chute.deployed, 0.05) ** 1.5

            lines = 1.2 * full * min(1.0, chute.deployed * 3.0)

            transform = scene.get_component(child, TransformComponent).transform

            transform.position = (0.0, chute.bottom + chute.part.height + lines, 0.0)
            transform.scale = (radius, radius, radius)

    def navball_frame(
        self,
        scene: Scene,
        orbit_system,
        planet_system
    ) -> tuple | None:
        """
        (orientation, up, north, motion) for the attitude ball,
        all in the body's non-rotating frame: motion is through
        the air while in it, along the orbit above it.
        """

        found = self.find(scene)

        if found is None or self.reference_state(orbit_system, found[1].body) is None:
            return None

        _, component = found

        state = component.state

        radius = float(np.linalg.norm(state.position))

        up = state.position / radius

        physics = self._body_physics(scene, component.body, orbit_system)

        axis = physics.spin if np.linalg.norm(physics.spin) > 0.0 else np.cross(state.position, state.velocity)

        if np.linalg.norm(axis) < 1e-12:
            axis = np.array([0.0, 1.0, 0.0])

        axis = axis / np.linalg.norm(axis)

        north = axis - np.dot(axis, up) * up

        if np.linalg.norm(north) < 1e-6:
            north = np.cross(up, [1.0, 0.0, 0.0])

        north /= np.linalg.norm(north)

        environment = self.environment(scene, component.body, orbit_system, planet_system)

        pressure, _ = environment.air(radius, up)

        motion = state.velocity - np.cross(environment.spin, state.position) if pressure > 0.0 else state.velocity

        return state.orientation, up, north, motion

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

        if found is None or self.reference_state(orbit_system, found[1].body) is None:
            return None

        _, component = found

        vessel = component.vessel
        state = component.state

        environment = self.environment(scene, component.body, orbit_system, planet_system)

        physics = self._body_physics(scene, component.body, orbit_system)

        sea_radius = physics.radius + (self._ground[0] if self._ground is not None and physics.solid else 0.0)

        radius = float(np.linalg.norm(state.position))

        up = state.position / radius

        pressure, _ = environment.air(radius, up)

        relative = state.velocity - np.cross(environment.spin, state.position)

        shape = kepler.orbit_shape(state.position, state.velocity, environment.mu)

        weight = vessel.mass * environment.mu / radius ** 2

        left, capacity = vessel.section_propellant()

        bodies = self.soi_bodies(scene)

        reference = bodies.get(component.body)

        escaping = shape.apoapsis is None or (reference is not None and shape.apoapsis > reference.soi)

        floor = max(self._extent(component.body, planet_system, scene, orbit_system), environment.air_top)

        if state.destroyed:
            situation = "Destroyed"
        elif state.landed:
            situation = "Landed"
        elif pressure > 0.0:
            situation = "Flying"
        elif escaping:
            # Leaving, or still coming in: past the body
            # (a flyby) or into it.
            if float(np.dot(state.position, state.velocity)) >= 0.0:
                situation = "Escaping"
            elif shape.periapsis > floor:
                situation = "Flyby"
            else:
                situation = "Sub-orbital"
        elif shape.periapsis > floor:
            situation = "Orbiting"
        else:
            situation = "Sub-orbital"

        chutes = vessel.parachutes

        if any(chute.deployed > 0.0 for chute in chutes):
            parachute = "open"
        elif any(chute.armed for chute in chutes):
            parachute = "armed"
        else:
            parachute = ""

        orbit_known = not state.landed and not state.destroyed

        return VesselReadout(
            name=vessel.name,
            body=self.body_name(scene, component.body),
            situation=situation,
            throttle=vessel.throttle,
            sas=component.controls.sas,
            sas_mode=component.controls.sas_mode,
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
            apoapsis=shape.apoapsis - sea_radius if orbit_known and not escaping else None,
            periapsis=shape.periapsis - sea_radius if orbit_known else None,
            destroyed=state.destroyed,
            impact_speed=state.impact_speed,
            parachute=parachute
        )
