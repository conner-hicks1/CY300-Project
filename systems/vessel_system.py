import math

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np

from core.handle import Handle
from core.logger import Logger

from ecs.components import (
    AtmosphereComponent,
    GlowComponent,
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
from graphics.mesh_data import MeshData
from graphics.mesh_factory import MeshFactory
from graphics.texture import Texture2D

from math3d import quaternion
from math3d.transform import Transform

from planet.air import air_column_of

from resources.resources import Resources

from scene.scene import Scene

from planet import weather
from science.survey import latitude_longitude

from vessel import docking, heating, kepler, landing, looks, maneuver, power, propellants, soi
from vessel.parts import VesselDesign
from vessel.physics import (
    Controls,
    Environment,
    VesselState,
    can_coast,
    coast,
    cool,
    land,
    rcs_push,
    steer,
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

    # Mars: its weather (planet/weather.py) moves its air.
    mars: bool = False


@dataclass(frozen=True, slots=True)
class Suggestion:
    """A burn worth planning: what, why, when (clock s), the inertial velocity change (m/s)."""

    label: str
    why: str | None
    time: float
    vector: np.ndarray


def format_distance(
    meters: float
) -> str:
    """A height for a suggestion's line (as game/format.py)."""

    return f"{meters:,.0f} m" if abs(meters) < 1_000.0 else f"{meters / 1000.0:,.0f} km"


# Coming down into the air, the periapsis at about this
# pressure (bar): deep enough to be caught (Earth ~45 km,
# Mars ~10 km), not so deep the heat is fierce.
ENTRY_PRESSURE_BAR = 0.002


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

    # Burned up: the part's name ("" otherwise).
    burned_up: str = ""

    # The part nearest its heat limit: (skin K, limit K,
    # name); None without parts.
    heat: tuple[float, float, str] | None = None

    # Landing (vessel/landing.py): the ground's slope under
    # it (deg; None: not near the ground), the slope it still
    # stands on, its legs ("", "up", "lowering", "down",
    # "raising"); fallen over, and why.
    slope: float | None = None
    standing: float = 0.0
    legs: str = ""
    toppled: bool = False
    fell: str = ""

    # What the burning (or next) engines burn: its short name
    # (vessel/propellants.py), "" without engines.
    propellant_kind: str = ""

    # Mars's weather over it (planet/weather.py), low or
    # landed there; else None.
    weather: object = None


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

        # (Turned over in a docked stack: its own bottom at
        # the top of its place.)
        return [(0.0, state.bottom + (state.part.height if state.flipped else 0.0), 0.0)]

    base = math.radians(state.angle)

    return [
        (
            state.distance * math.cos(base + 2.0 * math.pi * k / state.count),
            state.bottom,
            state.distance * math.sin(base + 2.0 * math.pi * k / state.count)
        )
        for k in range(state.count)
    ]


def copy_orientations(
    state
) -> list:
    """
    How a part's copies are turned in the vessel's frame:
    upright, or (sideways: side ports, modules docked to them)
    their own axis straight out from the core (or in).
    """

    if not state.outward:

        if state.flipped:
            return [quaternion.from_axis_angle(np.array([1.0, 0.0, 0.0]), 180.0)] * max(state.count, 1)

        return [np.array([0.0, 0.0, 0.0, 1.0])] * max(state.count, 1)

    base = math.radians(state.angle)

    turned = []

    for k in range(state.count):

        angle = base + 2.0 * math.pi * k / state.count

        out = np.array([math.cos(angle), 0.0, math.sin(angle)]) * state.outward

        turned.append(quaternion.look_rotation(np.array([0.0, 1.0, 0.0]), out))

    return turned


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

    # Translating with RCS on rails: pieces this long (s).
    RCS_STEP = 0.1

    # Most steps in one frame (beyond: bigger steps).
    MAX_STEPS = 200

    # Plume: length (m) at full throttle (a 2.5 m engine's;
    # smaller engines', shorter).
    PLUME_LENGTH = 24.0

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
        # Prediction); the path after the maneuver node: (key,
        # Prediction or None); the node's burn time: (key, s).
        self._prediction = None
        self._planned = None
        self._burn_time = None

        # The landing burn's estimate: (clock time, Braking or
        # None).
        self._braking = None

        # The ground's slope under the vessel (deg), near the
        # ground; else None. Where it was found: (body, position).
        self._slope: float | None = None
        self._slope_at: tuple | None = None

        # Where the vessel was launched from: (body, start),
        # for the part editor to rebuild it there.
        self._launch: tuple | None = None

        # The vessel flown now (others fly on rails), or None:
        # the first there is.
        self.active: Entity | None = None

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

        entity = self.add_vessel(scene, vessel, state, body, orbit_system)

        self.active = entity

        self._ground = None

        Logger.info(
            "[Vessels] %s: %.1f t, %.0f m/s of delta-v.",
            vessel.name,
            vessel.mass / 1000.0,
            vessel.delta_v()
        )

        return entity

    def add_vessel(
        self,
        scene: Scene,
        vessel: Vessel,
        state: VesselState,
        body: Hashable,
        orbit_system,
        **fields
    ) -> Entity:
        """A vessel in flight, as it is (not flown: switch to fly it). fields: more VesselComponent fields."""

        entity = scene.create_entity()

        scene.add_component(entity, NameComponent(vessel.name))
        scene.add_component(entity, TransformComponent(transform=Transform()))

        component = VesselComponent(
            vessel=vessel,
            state=state,
            body=body,
            controls=Controls(),
            **fields
        )

        scene.add_component(entity, component)

        self._build_parts(scene, entity, component)

        self.place(scene, orbit_system)

        return entity

    # =====================================================
    # Docking
    # =====================================================

    # Pushed apart undocking (m/s, each).
    UNDOCK_SPEED = 0.2

    def port_points(
        self,
        component: VesselComponent
    ) -> list[tuple[docking.Port, tuple[np.ndarray, np.ndarray]]]:
        """A vessel's free ports: (port, (face point, facing)), about its body, inertial."""

        return [
            (port, docking.port_face(component.vessel, component.state.position, component.state.orientation, port))
            for port in docking.free_ports(component.vessel)
        ]

    def docking_pair(
        self,
        scene: Scene,
        component: VesselComponent,
        target_entity: Entity
    ) -> tuple | None:
        """
        The closest pair of free ports between this vessel and
        the target (same body): (own port, own point, target
        port, target point, target component), or None.
        """

        target = scene.try_get_component(target_entity, VesselComponent)

        if target is None or target is component or target.body != component.body or target.state.destroyed:
            return None

        best = None

        for own, own_point in self.port_points(component):

            for theirs, their_point in self.port_points(target):

                distance = float(np.linalg.norm(own_point[0] - their_point[0]))

                if best is None or distance < best[0]:
                    best = (distance, own, own_point, theirs, their_point, target)

        return best[1:] if best is not None else None

    @staticmethod
    def _port_at(
        component: VesselComponent,
        end: str,
        copy: int
    ):

        return next(
            (port for port in docking.free_ports(component.vessel) if port.end == end and (end != "side" or port.copy == copy)),
            None
        )

    def dock(
        self,
        scene: Scene,
        entity: Entity,
        own_end: str,
        target_entity: Entity,
        target_end: str,
        orbit_system,
        own_copy: int = 0,
        target_copy: int = 0
    ) -> bool:
        """Join the target to this vessel at their ports: one vessel (this one), the target's parts in its stack."""

        component = scene.try_get_component(entity, VesselComponent)
        target = scene.try_get_component(target_entity, VesselComponent)

        if component is None or target is None:
            return False

        own_mass = component.vessel.mass
        their_mass = target.vessel.mass

        up = component.state.up

        state = component.state

        velocity = (state.velocity * own_mass + target.state.velocity * their_mass) / (own_mass + their_mass)

        if target_end == "side" or own_end == "side":

            # Onto a side port: the one with it keeps its core;
            # the other hangs off it, straight out.
            host_is_target = target_end == "side"

            host, guest = (target, component) if host_is_target else (component, target)

            port_end, guest_end = (target_end, own_end) if host_is_target else (own_end, target_end)

            port = self._port_at(host, port_end, target_copy if host_is_target else own_copy)

            if port is None or docking.can_hang(guest.vessel):
                return False

            joined = docking.attach_side(host.vessel, port, guest.vessel, guest_end)

            if host_is_target:
                state.position = target.state.position.copy()
                state.orientation = target.state.orientation.copy()

        else:

            joined, below, rise = docking.merge(component.vessel, own_end, target.vessel, target_end)

            # The joined stack's origin: below, at the target's
            # bottom.
            state.position = state.position - up * rise

        # Both now move as one (momentum kept).
        state.velocity = velocity
        state.angular_velocity = np.zeros(3)

        component.vessel = joined
        component.science.extend(target.science)
        component.landed_on |= target.landed_on
        component.paid = component.paid or target.paid

        self._clear_parts(scene, component)
        self._build_parts(scene, entity, component)

        self.remove(scene, target_entity)

        self._prediction = None
        self._planned = None

        self.place(scene, orbit_system)

        Logger.info("[Vessels] Docked: %s is %d vessels in one.", joined.name, docking.modules(joined))

        return True

    def undock(
        self,
        scene: Scene,
        entity: Entity,
        orbit_system
    ) -> Entity | None:
        """Split the vessel at its last joint: the lower half goes its own way (a vessel of its own). Returns it."""

        component = scene.try_get_component(entity, VesselComponent)

        if component is None:
            return None

        if docking.side_module(component.vessel):
            return self._undock_side(scene, entity, component, orbit_system)

        at = docking.joint(component.vessel)

        if at is None:
            return None

        state = component.state

        up = state.up

        origin = state.position.copy()

        upper, lower, lift, upper_turned, lower_turned = docking.split(component.vessel, at)

        flip = quaternion.from_axis_angle(np.array([1.0, 0.0, 0.0]), 180.0)

        lower_state = state.copy()
        lower_state.velocity = state.velocity - up * self.UNDOCK_SPEED
        lower_state.angular_velocity = np.zeros(3)

        if lower_turned:

            # Its old top (the joint) is its bottom now.
            lower_state.position = origin + up * lift
            lower_state.orientation = quaternion.normalize(quaternion.multiply(state.orientation, flip))

        state.velocity = state.velocity + up * self.UNDOCK_SPEED

        if upper_turned:
            state.position = origin + up * (lift + upper.height)
            state.orientation = quaternion.normalize(quaternion.multiply(state.orientation, flip))
        else:
            state.position = origin + up * lift

        component.vessel = upper

        self._clear_parts(scene, component)
        self._build_parts(scene, entity, component)

        other = self.add_vessel(scene, lower, lower_state, component.body, orbit_system, paid=component.paid)

        self._prediction = None
        self._planned = None

        self.place(scene, orbit_system)

        Logger.info("[Vessels] Undocked: %s and %s.", upper.name, lower.name)

        return other

    def _undock_side(
        self,
        scene: Scene,
        entity: Entity,
        component: VesselComponent,
        orbit_system
    ) -> Entity | None:
        """The last module on a side port goes its own way: straight out from it."""

        detached = docking.detach_side(component.vessel)

        if detached is None:
            return None

        state = component.state

        rotate = lambda vector: quaternion.rotate_vector(state.orientation, vector)

        out = rotate(detached.out)

        axis_point = state.position + rotate(np.array([0.0, detached.height, 0.0]))

        guest = detached.guest

        length = guest.height

        # Its stack straight out: the docked end at the port.
        if detached.sign < 0:
            up = -out
            origin = axis_point + out * (detached.inner + length)
        else:
            up = out
            origin = axis_point + out * detached.inner

        guest_state = state.copy()
        guest_state.position = origin
        guest_state.velocity = state.velocity + out * self.UNDOCK_SPEED
        guest_state.angular_velocity = np.zeros(3)
        guest_state.orientation = quaternion.look_rotation(rotate(np.array([0.0, 1.0, 0.0])), up)

        state.velocity = state.velocity - out * self.UNDOCK_SPEED * guest.mass / max(detached.host.mass, 1.0)

        component.vessel = detached.host

        self._clear_parts(scene, component)
        self._build_parts(scene, entity, component)

        other = self.add_vessel(scene, guest, guest_state, component.body, orbit_system, paid=component.paid)

        self._prediction = None
        self._planned = None

        self.place(scene, orbit_system)

        Logger.info("[Vessels] Undocked %s from %s's side port.", guest.name, detached.host.name)

        return other

    def remove(
        self,
        scene: Scene,
        entity: Entity
    ):
        """Take a vessel out of the game (terminated, or recovered)."""

        if scene.try_get_component(entity, VesselComponent) is None:
            return

        self._destroy_tree(scene, entity)

        if self.active == entity:
            self.active = None

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

        # The one on the pad goes; one in flight (launched)
        # stays where it is, on rails.
        if found is not None and not found[1].paid:
            self._destroy_tree(scene, found[0])

        self.active = None

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

            lit = selection is not None and (
                state.source == tuple(selection)
                or (len(selection) == 2 and state.source[:2] == tuple(selection))
            )

            for child, finish in component.pieces.get(id(state), []):

                renderer = scene.try_get_component(child, MeshRendererComponent)

                if renderer is not None:
                    renderer.material = self._finish_material(finish, selected=lit)

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

    def find(
        self,
        scene: Scene
    ) -> tuple[Entity, VesselComponent] | None:
        """The active vessel (the one flown and followed)."""

        if self.active is not None:

            component = scene.try_get_component(self.active, VesselComponent)

            if component is not None:
                return self.active, component

        first = next(iter(scene.registry.view_with(VesselComponent)), None)

        self.active = first[0] if first is not None else None

        return first

    @staticmethod
    def vessels(
        scene: Scene
    ) -> list[tuple[Entity, VesselComponent]]:
        """Every vessel in flight (the active one too)."""

        return list(scene.registry.view_with(VesselComponent))

    def position_of(
        self,
        component: VesselComponent,
        orbit_system
    ) -> np.ndarray | None:
        """A vessel's position about the star (m), or None."""

        reference = self.reference_state(orbit_system, component.body)

        if reference is None:
            return None

        return np.asarray(reference[0], dtype=np.float64) + component.state.position

    def brief_situation(
        self,
        scene: Scene,
        component: VesselComponent,
        orbit_system
    ) -> str:
        """Landed / orbiting / sub-orbital / escaping, for a list (vessels not flown have no readout)."""

        state = component.state

        if state.destroyed:
            return "Lost"

        if state.landed:
            return "Landed"

        bodies = self.soi_bodies(scene)

        body = bodies.get(component.body)

        if body is None:
            return "In flight"

        shape = kepler.orbit_shape(state.position, state.velocity, body.mu)

        if shape.apoapsis is None or shape.apoapsis > body.soi:
            return "Escaping"

        if shape.periapsis < self._body_physics(scene, component.body, orbit_system).radius:
            return "Sub-orbital"

        return "Orbiting"

    # Close enough (m) and still enough (m/s) to each other to
    # move propellant across.
    ALONGSIDE_DISTANCE = 200.0
    ALONGSIDE_SPEED = 2.0

    def alongside(
        self,
        scene: Scene,
        component: VesselComponent,
        orbit_system
    ) -> tuple | None:
        """The nearest vessel alongside this one: (entity, name, distance m, relative speed m/s), or None."""

        best = None

        for entity, other in self.vessels(scene):

            if other is component or other.state.destroyed or other.body != component.body:
                continue

            distance = float(np.linalg.norm(other.state.position - component.state.position))

            speed = float(np.linalg.norm(other.state.velocity - component.state.velocity))

            if distance <= self.ALONGSIDE_DISTANCE and speed <= self.ALONGSIDE_SPEED and (best is None or distance < best[2]):
                best = (entity, other.vessel.name, distance, speed)

        return best

    def switch(
        self,
        scene: Scene,
        entity: Entity,
        orbit_system,
        planet_system
    ) -> str:
        """
        Fly another vessel. Returns "" or why not: the one
        flown now must be coasting above the air or landed
        (no thrust), as the others fly on rails.
        """

        target = scene.try_get_component(entity, VesselComponent)

        if target is None or target.state.destroyed:
            return "No such vessel"

        found = self.find(scene)

        if found is not None and found[0] != entity:

            _, component = found

            state = component.state

            if not state.destroyed:

                environment = self.environment(scene, component.body, orbit_system, planet_system)

                thrusting = component.vessel.throttle > 0.0 and bool(component.vessel.burning_engines())

                radius = float(np.linalg.norm(state.position))

                if thrusting:
                    return "Not under thrust: cut the throttle first"

                if not state.landed and radius < environment.air_top:
                    return "Not while flying through the air"

            component.vessel.throttle = 0.0
            component.hover = None

        self.active = entity

        # Another vessel: its own ground, predictions.
        self._ground = None
        self._fallback_ground = None
        self._height = float(np.linalg.norm(target.state.position))
        self._prediction = None
        self._planned = None
        self._braking = None
        self._burn_time = None

        return ""

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
                air_top=1000.0 * column.altitude_of_pressure(AIR_LIMIT_BAR) if column is not None else None,
                mars=self.body_name(scene, body) == "Mars"
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

        # The ground's slope (vessel/landing.py), from the
        # heights round a point (as fine as the vessel's height
        # needs).
        normal = None

        if physics.solid and planet_system is not None and body in orbit_system.states:

            _, body_frame = orbit_system.states[body]

            accuracy = self.ground_accuracy

            def ground_at(where):

                found = planet_system.ground(body, body_frame.T @ where, accuracy)

                return physics.radius + found[0] + found[1] if found is not None else ground_radius

            def normal_at(where):
                return landing.ground_normal(ground_at, where, max(3.0, 4.0 * accuracy))

            normal = normal_at

        # Mars: the season thickens or thins the whole air, and
        # storms blow (planet/weather.py).
        thicken = 1.0
        wind = None

        if physics.mars and body in orbit_system.states:

            now = orbit_system.time

            thicken = weather.pressure_factor(weather.solar_longitude(now))

            _, mars_frame = orbit_system.states[body]

            def wind_of(position):
                return self.wind_at(mars_frame, now, position)

            wind = wind_of

        if column is not None:

            gas_constant = column.gas_constant

            def air(radius, direction):

                altitude = (radius - sea_radius) / 1000.0

                pressure = column.pressure(altitude) * thicken

                if pressure < AIR_LIMIT_BAR:
                    return 0.0, 0.0

                return pressure, pressure * 1e5 / (gas_constant * column.temperature(altitude))

            def temperature(radius):
                return column.temperature((radius - sea_radius) / 1000.0)

            air_top = sea_radius + physics.air_top

        else:

            def air(radius, direction):
                return 0.0, 0.0

            temperature = None

            air_top = 0.0

        return Environment(
            mu=physics.mu,
            spin=physics.spin,
            air=air,
            ground=lambda d: ground_radius,
            air_top=air_top,
            temperature=temperature,
            normal=normal,
            wind=wind
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
        orbit_system,
        active: bool = True
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
        if active:
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

        key = (orbit_system.time, component.body, id(component))

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
    # Maneuver Nodes
    # =====================================================
    #
    # A planned burn on the vessel's path (vessel/maneuver.py),
    # on its current conic (before the next sphere of
    # influence): the path after it drawn on the map, time
    # warp slowing to arrive at its burn's start, done once
    # the burn has made its velocity change.

    # Time warp comes back to real time this long (s) before
    # a node's burn starts.
    NODE_LEAD = 15.0

    # Done within (m/s).
    NODE_DONE = 0.5

    def node_state(
        self,
        scene: Scene,
        orbit_system,
        time: float
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        The vessel's (position, velocity) about its body at a
        clock time on its current conic (now, if past); None
        landed, destroyed, or when the path leaves the conic
        first (another sphere).
        """

        found = self.find(scene)

        if found is None:
            return None

        _, component = found

        state = component.state

        if state.landed or state.destroyed:
            return None

        ahead = time - orbit_system.time

        if ahead <= 0.0:
            return state.position.copy(), state.velocity.copy()

        prediction = self.prediction(scene, orbit_system)

        if prediction is None or not prediction.patches:
            return None

        first = prediction.patches[0]

        if first.end != "orbit" and first.times is not None and ahead > float(first.times[-1]):
            return None

        return kepler.propagate(state.position, state.velocity, self._body_physics(scene, component.body, orbit_system).mu, ahead)

    def set_node(
        self,
        scene: Scene,
        orbit_system,
        node: maneuver.ManeuverNode | None
    ) -> bool:
        """
        Put (or move, or change: call again) the vessel's
        node; None removes it. False when there is no conic
        there to burn on.
        """

        found = self.find(scene)

        if found is None:
            return False

        _, component = found

        if node is None:
            component.node = None
            return True

        where = self.node_state(scene, orbit_system, node.time)

        if where is None:
            return False

        node.vector = maneuver.to_inertial(*where, *node.components)
        node.baseline = component.state.thrust_dv.copy()

        component.node = node

        return True

    @staticmethod
    def node_remaining(
        component: VesselComponent
    ) -> np.ndarray | None:

        if component.node is None:
            return None

        return component.node.remaining(component.state.thrust_dv)

    def node_burn_time(
        self,
        component: VesselComponent
    ) -> float:
        """s at full throttle for what is left of the node's burn."""

        remaining = self.node_remaining(component)

        if remaining is None:
            return 0.0

        size = float(np.linalg.norm(remaining))

        key = (round(size, 1), round(component.vessel.mass), component.vessel.next_stage)

        if self._burn_time is None or self._burn_time[0] != key:
            self._burn_time = (key, maneuver.burn_time(component.vessel, size))

        return self._burn_time[1]

    def planned(
        self,
        scene: Scene,
        orbit_system
    ) -> soi.Prediction | None:
        """The path after the node's burn (times from now), or None."""

        found = self.find(scene)

        if found is None or found[1].node is None:
            return None

        _, component = found

        node = component.node

        remaining = self.node_remaining(component)

        key = (orbit_system.time, component.body, id(component), node.time, tuple(np.round(remaining, 3)))

        if self._planned is not None and self._planned[0] == key:
            return self._planned[1]

        planned = None

        where = self.node_state(scene, orbit_system, node.time)

        bodies = self.soi_bodies(scene)

        if where is not None and component.body in bodies:

            ahead = max(node.time - orbit_system.time, 0.0)

            mu = bodies[component.body].mu

            children = self._children(bodies, component.body, orbit_system)

            # The moons (or planets) where they will be then.
            moved = {
                key: kepler.propagate(position, velocity, mu, ahead)
                for key, (position, velocity) in self.relative_states(scene, orbit_system, component.body, children).items()
            }

            around_sun = component.body is SUN

            planned = soi.predict(
                where[0],
                where[1] + remaining,
                bodies[component.body],
                bodies,
                moved,
                samples=2048 if around_sun else 256,
                horizon=3.0 * 365.25 * 86_400.0 if around_sun else 30.0 * 86_400.0
            )

            for patch in planned.patches:

                patch.start += ahead

                if patch.times is not None:
                    patch.times = patch.times + ahead

        self._planned = (key, planned)

        return planned

    def suggested_burns(
        self,
        scene: Scene,
        orbit_system,
        planet_system
    ) -> list[Suggestion]:
        """
        Burns worth a node from where the vessel is:
        circularizing at either apsis, capturing at the
        periapsis coming in, coming down into the air (or to
        the ground).
        """

        found = self.find(scene)

        if found is None:
            return []

        _, component = found

        state = component.state

        if state.landed or state.destroyed:
            return []

        bodies = self.soi_bodies(scene)

        body = bodies.get(component.body)

        if body is None:
            return []

        physics = self._body_physics(scene, component.body, orbit_system)

        mu = physics.mu

        now = orbit_system.time

        shape = kepler.orbit_shape(state.position, state.velocity, mu)

        floor = max(self._extent(component.body, planet_system, scene, orbit_system), self.environment(scene, component.body, orbit_system, planet_system).air_top)

        closed = shape.apoapsis is not None and shape.apoapsis < body.soi

        suggestions = []

        def at(seconds):

            return kepler.propagate(state.position, state.velocity, mu, seconds)

        sea = physics.radius + (self._ground[0] if self._ground is not None and physics.solid else 0.0)

        if closed:

            for apsis, short in (("apoapsis", "Ap"), ("periapsis", "Pe")):

                wait = maneuver.time_to_apsis(state.position, state.velocity, mu, apsis)

                if wait is None:
                    continue

                # Not one already behind (a near-circular
                # orbit's apsis is anywhere): a little ahead.
                wait = max(wait, 60.0)

                where = at(wait)

                height = float(np.linalg.norm(where[0])) - sea

                if apsis == "periapsis" and float(np.linalg.norm(where[0])) < floor:
                    continue

                change = maneuver.circularize(*where, mu)

                suggestions.append(Suggestion(
                    f"Circularize at {short}",
                    f"{float(np.linalg.norm(change)):,.0f} m/s, at {format_distance(height)}",
                    now + wait,
                    change
                ))

        else:

            # Coming in to a body (not the Sun): capture at the
            # periapsis, if the path passes it above the air.
            wait = maneuver.time_to_apsis(state.position, state.velocity, mu, "periapsis")

            if component.body is not SUN and wait is not None and wait > 0.0 and shape.periapsis > floor:

                where = at(wait)

                circular = maneuver.circularize(*where, mu)

                suggestions.append(Suggestion(
                    "Capture: circular at Pe",
                    f"{float(np.linalg.norm(circular)):,.0f} m/s",
                    now + wait,
                    circular
                ))

                # A long ellipse costs much less (and can be
                # lowered later, or aerobraked).
                loose = maneuver.capture(*where, mu, 0.4 * body.soi)

                suggestions.append(Suggestion(
                    "Capture: long orbit",
                    f"{float(np.linalg.norm(loose)):,.0f} m/s, Ap {format_distance(0.4 * body.soi - sea)}",
                    now + wait,
                    loose
                ))

        # Coming down from orbit: at the apoapsis (or soon,
        # near circular), the periapsis into the air or to the
        # ground.
        if closed and physics.solid and shape.periapsis > floor:

            column = physics.column

            if column is not None and physics.air_top:
                goal = sea + min(1000.0 * column.altitude_of_pressure(ENTRY_PRESSURE_BAR), 0.5 * physics.air_top)
                why = f"Pe {format_distance(goal - sea)} in the air"
            else:
                goal = sea
                why = "Pe at the ground: brake to land (HUD)"

            wait = maneuver.time_to_apsis(state.position, state.velocity, mu, "apoapsis")

            wait = 300.0 if wait is None or shape.eccentricity < 0.01 else max(wait, 60.0)

            where = at(wait)

            change = maneuver.lower_periapsis(*where, mu, max(goal, 1.0))

            if change is not None:
                suggestions.append(Suggestion(
                    "Come down",
                    f"{float(np.linalg.norm(change)):,.0f} m/s: {why}",
                    now + wait,
                    change
                ))

        return suggestions

    # Braking help shows below this height (m) and in air no
    # thicker than this (bar; thicker, parachutes do it).
    BRAKING_HEIGHT = 60_000.0
    BRAKING_PRESSURE = 0.05

    def braking(
        self,
        scene: Scene,
        orbit_system,
        planet_system
    ) -> maneuver.Braking | None:
        """Falling toward the ground on engines: when to start the burn that stops at it (HUD)."""

        cached = self._braking

        key = (orbit_system.time, self.active)

        if cached is not None and cached[0] == key:
            return cached[1]

        found = self._find_braking(scene, orbit_system, planet_system)

        self._braking = (key, found)

        return found

    def _find_braking(
        self,
        scene: Scene,
        orbit_system,
        planet_system
    ) -> maneuver.Braking | None:

        found = self.find(scene)

        if found is None:
            return None

        _, component = found

        state = component.state

        if state.landed or state.destroyed or self.reference_state(orbit_system, component.body) is None:
            return None

        physics = self._body_physics(scene, component.body, orbit_system)

        if not physics.solid:
            return None

        environment = self.environment(scene, component.body, orbit_system, planet_system)

        radius = float(np.linalg.norm(state.position))

        up = state.position / radius

        height = radius - environment.ground(up)

        pressure, _ = environment.air(radius, up)

        if height > self.BRAKING_HEIGHT or pressure > self.BRAKING_PRESSURE:
            return None

        # The ground's frame: what the landing has to stop.
        relative = state.velocity - np.cross(environment.spin, state.position)

        vessel = component.vessel

        thrust = vessel.max_thrust(pressure)

        impact = self._impact_time(scene, component, environment, orbit_system, planet_system)

        return maneuver.braking(
            impact,
            height,
            float(np.dot(relative, up)),
            float(np.linalg.norm(relative)),
            environment.mu / radius ** 2,
            thrust / vessel.mass if vessel.mass > 0.0 else 0.0
        )

    # Samples of the path ahead against the terrain, and how
    # finely the ground is looked up for it (m).
    IMPACT_SAMPLES = 96
    IMPACT_SPACING = 2_000.0

    def _impact_time(
        self,
        scene: Scene,
        component: VesselComponent,
        environment: Environment,
        orbit_system,
        planet_system
    ) -> float | None:
        """
        s until the path hits the ground: the real terrain
        along it (planet_system.ground_heights, the body
        turning meanwhile), between where it is low enough to
        hit the highest ground and where it passes the lowest
        (or its periapsis). None: it passes over.
        """

        state = component.state

        mu = environment.mu

        radius = float(np.linalg.norm(state.position))

        up = state.position / radius

        highest = self._extent(component.body, planet_system, scene, orbit_system)

        start = maneuver.time_to_ground(state.position, state.velocity, mu, highest)

        if start is None:
            return None

        physics = self._body_physics(scene, component.body, orbit_system)

        terrain = planet_system.terrain(component.body) if planet_system is not None else None

        if terrain is None or component.body not in orbit_system.states:

            # (No terrain to ask: the ground below now.)
            return maneuver.time_to_ground(state.position, state.velocity, mu, environment.ground(up))

        lowest = physics.radius + terrain.base_bounds[0] + min(terrain.min_elevation, 0.0)

        end = maneuver.time_to_ground(state.position, state.velocity, mu, lowest)

        if end is None:
            end = maneuver.time_to_apsis(state.position, state.velocity, mu, "periapsis")

        if end is None or end <= start:
            end = start + 600.0

        times = np.linspace(start, end, self.IMPACT_SAMPLES)

        points, _ = kepler.propagate_many(state.position, state.velocity, mu, times)

        distances = np.linalg.norm(points, axis=1)

        directions = points / np.maximum(distances, 1.0)[:, None]

        # Into the body's frame then: it turns meanwhile.
        _, frame = orbit_system.states[component.body]

        spin = physics.spin

        rate = float(np.linalg.norm(spin))

        if rate > 0.0:

            axis = spin / rate

            angles = -rate * times

            cos = np.cos(angles)[:, None]
            sin = np.sin(angles)[:, None]

            directions = directions * cos + np.cross(axis, directions) * sin + axis * (directions @ axis)[:, None] * (1.0 - cos)

        heights = planet_system.ground_heights(component.body, directions @ frame, self.IMPACT_SPACING)

        if heights is None:
            return maneuver.time_to_ground(state.position, state.velocity, mu, environment.ground(up))

        clearance = distances - (physics.radius + heights)

        below = np.nonzero(clearance < 0.0)[0]

        if not len(below):
            return None

        k = int(below[0])

        if k == 0:
            return float(times[0])

        # Between the samples, straight.
        share = clearance[k - 1] / (clearance[k - 1] - clearance[k])

        return float(times[k - 1] + (times[k] - times[k - 1]) * share)

    def _node_warp(
        self,
        component: VesselComponent,
        orbit_system,
        frame_time: float
    ) -> float:
        """Fastest the clock may run not to pass the start of the node's burn."""

        node = component.node

        if node is None:
            return math.inf

        burn = self.node_burn_time(component)

        if not math.isfinite(burn):
            burn = 0.0

        gap = node.time - 0.5 * burn - self.NODE_LEAD - orbit_system.time

        if gap <= 0.0:
            # Burning (or late): real time until it is done
            # or long past.
            return 1.0 if orbit_system.time < node.time + burn + self.NODE_LEAD else math.inf

        return max(1.0, gap / (4.0 * frame_time))

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

        node_limit = self._node_warp(component, orbit_system, frame_time)

        if not can_coast(component.vessel, state, environment) or component.controls.translating:
            return min(self.PHYSICS_WARP, node_limit)

        limit = node_limit

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

            was_toppled = state.toppled

            landing.move_legs(vessel, dt)

            self._fly(vessel, state, environment, component.controls, dt, extent)

            self._height = float(np.linalg.norm(state.position)) - environment.ground(direction)

            # The slope below, near the ground (looked up again
            # once it has moved a little: five lookups of the
            # fine ground are not cheap).
            if environment.normal is not None and self._height < self.SLOPE_HEIGHT and not state.destroyed:

                last = self._slope_at

                # (In the body's own turning frame: landed, it
                # stays put.)
                local = orbit_system.states[component.body][1].T @ state.position

                moved = math.inf if last is None or last[0] is not component.body else float(np.linalg.norm(local - last[1]))

                if moved > max(5.0, 0.05 * self._height):

                    up = state.position / float(np.linalg.norm(state.position))

                    self._slope = landing.slope_of(environment.normal(up), up)
                    self._slope_at = (component.body, local)

            else:

                self._slope = None
                self._slope_at = None

            if state.toppled and not was_toppled:

                events.append("toppled")

                Logger.info("[Vessels] %s fell over: %s.", vessel.name, state.fell)

            if state.destroyed:

                events.append("destroyed")

                if state.burned_up:
                    Logger.info("[Vessels] %s burned up (its %s overheated).", vessel.name, state.burned_up)
                elif state.fell:
                    Logger.info("[Vessels] %s %s.", vessel.name, state.fell)
                else:
                    Logger.info("[Vessels] %s hit the ground at %.0f m/s.", vessel.name, state.impact_speed)

                self._clear_parts(scene, component)

            elif state.landed and not was_landed:

                events.append("landed")

            elif self._hand_over(scene, component, orbit_system) is not None:

                events.append("soi")

            node = component.node

            if node is not None and not state.destroyed:

                remaining = node.remaining(state.thrust_dv)

                done = state.thrust_dv - node.baseline

                # Made (or overshot) once burning.
                if float(np.linalg.norm(done)) > 0.0 and (
                    float(np.linalg.norm(remaining)) < self.NODE_DONE
                    or float(np.dot(remaining, node.vector)) < 0.0
                ):

                    component.node = None

                    events.append("node done")

        self._update_debris(scene, component, environment, dt, planet_system, orbit_system)

        for other, background in self.vessels(scene):

            if other != entity:

                events.extend(self._background(scene, other, background, dt, orbit_system))

                # Lost on rails: gone.
                if background.state.destroyed:
                    self.remove(scene, other)

        self._update_plumes(scene, entity, component, orbit_system, planet_system)
        self._update_canopies(scene, entity, component)
        self._update_reentry_glow(scene, entity, component)

        for _, any_component in self.vessels(scene):
            self._update_wings(scene, any_component)
            self._update_legs(scene, any_component)

        self.place(scene, orbit_system)

        return events

    def _background(
        self,
        scene: Scene,
        entity: Entity,
        component: VesselComponent,
        dt: float,
        orbit_system
    ) -> list[str]:
        """
        A vessel not flown, on rails: landed, it turns with
        its body; else it follows its conic (into other
        spheres), until it comes down: with a parachute in
        air, it lands; else it is lost. Events "lost:<name>",
        "landed:<name>".
        """

        state = component.state

        if state.destroyed or dt <= 0.0 or self.reference_state(orbit_system, component.body) is None:
            return []

        physics = self._body_physics(scene, component.body, orbit_system)

        radius = float(np.linalg.norm(state.position))

        if state.landed:

            environment = Environment(physics.mu, physics.spin, lambda r, d: (0.0, 0.0), lambda d: radius)

            step(component.vessel, state, environment, component.controls, dt)

            return []

        ground = physics.radius if physics.solid else 0.5 * physics.radius

        impact = maneuver.time_to_ground(state.position, state.velocity, physics.mu, ground) if physics.solid or component.body is SUN else None

        if impact is not None and impact <= dt:

            there, _ = kepler.propagate(state.position, state.velocity, physics.mu, impact)

            up = there / float(np.linalg.norm(there))

            chute = any(part.armed or part.deployed > 0.0 for part in component.vessel.parachutes)

            if chute and physics.column is not None and component.body is not SUN:

                environment = Environment(physics.mu, physics.spin, lambda r, d: (0.0, 0.0), lambda d: ground)

                state.position = up * ground

                land(state, environment)

                Logger.info("[Vessels] %s came down under its parachute.", component.vessel.name)

                return [f"landed:{component.vessel.name}"]

            state.destroyed = True

            self._clear_parts(scene, component)

            Logger.info("[Vessels] %s (not flown) hit %s.", component.vessel.name, self.body_name(scene, component.body))

            return [f"lost:{component.vessel.name}"]

        coast(state, Environment(physics.mu, physics.spin, lambda r, d: (0.0, 0.0), lambda d: ground), dt)

        self._hand_over(scene, component, orbit_system, active=False)

        return []

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

                if shape.periapsis > floor and controls.translating:

                    # Translating (docking): the orbit exactly,
                    # the thrusters' push in small pieces.
                    piece = min(remaining, self.RCS_STEP)

                    coast(state, environment, piece)

                    push = rcs_push(vessel, state, controls, piece)

                    state.velocity = state.velocity + push * piece
                    state.position = state.position + push * (0.5 * piece * piece)
                    state.thrust_dv = state.thrust_dv + push * piece

                    self._steer(vessel, state, controls, piece, environment)

                    remaining -= piece

                    continue

                if shape.periapsis > floor:

                    coast(state, environment, remaining)
                    cool(vessel, state, remaining)

                    self._steer(vessel, state, controls, remaining, environment)

                    return

                # A falling arc: on rails while well clear.
                radius = float(np.linalg.norm(state.position))
                speed = max(float(np.linalg.norm(state.velocity)), 1.0)

                piece = 0.25 * (radius - floor) / speed

                if piece > 10.0 * self.PHYSICS_STEP:

                    piece = min(piece, remaining)

                    coast(state, environment, piece)
                    cool(vessel, state, piece)

                    self._steer(vessel, state, controls, piece, environment)

                    remaining -= piece

                    continue

            # Physics; bigger steps past MAX_STEPS.
            delta = self.PHYSICS_STEP if steps < self.MAX_STEPS else remaining

            delta = min(delta, remaining)

            step(vessel, state, environment, controls, delta)

            remaining -= delta

    # Coasting, attitude still turns at warps up to this much
    # simulated time a frame (s); beyond, it holds (on rails).
    STEER_SPAN = 2.0

    def _steer(
        self,
        vessel: Vessel,
        state: VesselState,
        controls: Controls,
        span: float,
        environment: Environment | None = None
    ):
        """Turn the coasting vessel over `span` s in physics-sized steps."""

        if span > self.STEER_SPAN:

            # Too fast to steer: stop any spin, as KSP does on
            # rails (the wheels idle).
            state.angular_velocity = np.zeros(3)
            vessel.wheel_use = 0.0

            return

        steps = max(1, math.ceil(span / self.PHYSICS_STEP))

        for _ in range(steps):
            steer(vessel, state, controls, span / steps, environment)

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

        children: dict = {}

        for child, hierarchy in scene.registry.view_with(HierarchyComponent):
            children.setdefault(hierarchy.parent, []).append(child)

        doomed = [entity]

        for item in doomed:
            doomed.extend(children.get(item, ()))

        for item in reversed(doomed):
            scene.destroy_entity(item)

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
        component.pieces = {}
        component.panel_entities = {}

        for state in component.vessel.parts:

            part = state.part

            copies = []
            pieces = []

            shapes = self._part_meshes(part, self._skirt(component.vessel, state))

            for position, turned in zip(copy_positions(state), copy_orientations(state)):

                child = scene.create_entity()

                scene.add_component(child, NameComponent(part.name))
                scene.add_component(child, HierarchyComponent(entity))
                scene.add_component(child, TransformComponent(transform=Transform(position=position, orientation=turned)))

                # The first finish on the copy itself, the others
                # as its children (one draw each).
                for index, (finish, mesh) in enumerate(shapes):

                    holder = child

                    if index:

                        holder = scene.create_entity()

                        scene.add_component(holder, NameComponent(f"{part.name} ({finish})"))
                        scene.add_component(holder, HierarchyComponent(child))
                        scene.add_component(holder, TransformComponent(transform=Transform()))

                    scene.add_component(holder, MeshRendererComponent(mesh=mesh, material=self._finish_material(finish)))

                    pieces.append((holder, finish))

                copies.append(child)

            if part.solar is not None:
                self._build_wings(scene, component, state, copies)

            component.part_entities[id(state)] = copies
            component.pieces[id(state)] = pieces

    @staticmethod
    def _skirt(
        vessel: Vessel,
        state
    ) -> float:
        """A core separator right under an engine: the interstage skirt's height (the engine's); else 0."""

        if state.part.kind != "decoupler" or state.radial or state.flipped:
            return 0.0

        core = [item for item in vessel.parts if not item.radial]

        index = next((k for k, item in enumerate(core) if item is state), None)

        if not index:
            return 0.0

        above = core[index - 1]

        return above.part.height if above.part.kind == "engine" and not above.flipped else 0.0

    def _part_meshes(
        self,
        part,
        skirt: float = 0.0
    ) -> list[tuple[str, Handle]]:
        """The part's look (vessel/looks.py) as one mesh per finish (cached by part, and skirt)."""

        meshes = []

        for finish, (pieces, colors) in looks.merged(part, skirt).items():

            def build(pieces=pieces, colors=colors):

                datas = [
                    MeshFactory.lathe_data(
                        piece.profile,
                        piece.segments,
                        piece.tile,
                        colors=[color] * (len(piece.profile) - 1),
                        # (An octagon: a face to the front.)
                        angle_offset=math.pi / piece.segments if piece.segments < 12 else 0.0
                    )
                    for piece, color in zip(pieces, colors)
                ]

                return Mesh.from_data(MeshData.merge(datas))

            meshes.append((finish, self._resources.meshes.load(f"vessel/look/{part.id}/{finish}/{skirt:.2f}", build)))

        return meshes

    def _finish_material(
        self,
        finish: str,
        selected: bool = False
    ) -> Handle:
        """A finish's material (shared by every part: vertex colors carry the part's color), lit up when selected."""

        metallic, roughness, surface = looks.FINISHES[finish]

        key = f"vessel/finish/{finish}" + ("/selected" if selected else "")

        def create():

            material = Material(self._lit)

            material.set_vec3("uBaseColor", (1.35, 1.25, 1.0) if selected else (1.0, 1.0, 1.0))
            material.set_float("uMetallic", metallic)
            material.set_float("uRoughness", roughness)
            material.set_vec3("uEmissive", (0.35, 0.22, 0.05) if selected else (0.0, 0.0, 0.0))

            if surface is not None:

                textures = self._resources.textures

                material.set_texture("uBaseColorMap", textures.load(f"vessel/{surface}_albedo", lambda: Texture2D(f"assets/textures/{surface}_albedo.png", srgb=True)))
                material.set_texture("uNormalMap", textures.load(f"vessel/{surface}_normal", lambda: Texture2D(f"assets/textures/{surface}_normal.png", srgb=False)))

                orm = textures.load(f"vessel/{surface}_orm", lambda: Texture2D(f"assets/textures/{surface}_orm.png", srgb=False))

                material.set_texture("uMetallicRoughnessMap", orm)
                material.set_texture("uOcclusionMap", orm)
                material.set_float("uNormalStrength", 1.0)
                material.set_float("uOcclusionStrength", 1.0)

            return material

        return self._resources.materials.load(key, create)

    def _restack_parts(
        self,
        scene: Scene,
        component: VesselComponent
    ):

        for state in component.vessel.parts:

            for child, position, turned in zip(component.part_entities.get(id(state), []), copy_positions(state), copy_orientations(state)):

                transform = scene.get_component(child, TransformComponent).transform

                transform.position = position
                transform.orientation = turned

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

        dropped_map = {id(part): component.part_entities.pop(id(part), []) for part in dropped}

        dropped_pieces = {id(part): component.pieces.pop(id(part), []) for part in dropped}

        dropped_panels = {id(part): component.panel_entities.pop(id(part)) for part in dropped if id(part) in component.panel_entities}

        dropped_entities = [child for copies in dropped_map.values() for child in copies]

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

        separated = any(part.part.kind == "command" for part in dropped)

        if separated and not state.destroyed and not state.landed:

            # It can fly itself: a new vessel (a relay, a probe).
            others = sum(1 for _ in scene.registry.view_with(VesselComponent))

            section = Vessel(f"{vessel.name} {others + 1}", dropped)

            section._restack()

            child_entity = scene.create_entity()

            scene.add_component(child_entity, NameComponent(section.name))
            scene.add_component(child_entity, TransformComponent(transform=Transform()))

            child = VesselComponent(
                vessel=section,
                state=debris_state,
                body=component.body,
                controls=Controls(),
                paid=True
            )

            scene.add_component(child_entity, child)

            child.part_entities = dropped_map
            child.pieces = dropped_pieces
            child.panel_entities = dropped_panels

            for child_part in dropped_entities:
                scene.get_component(child_part, HierarchyComponent).parent = child_entity

            # (Its parts' places in the new stack.)
            self._restack_parts(scene, child)

            Logger.info("[Vessels] %s separated: a vessel of its own.", section.name)

        elif not state.destroyed and not state.landed:

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
                self._destroy_tree(scene, child)

        self._restack_parts(scene, component)

    def _clear_parts(
        self,
        scene: Scene,
        component: VesselComponent
    ):

        parts = [child for copies in component.part_entities.values() for child in copies]

        for child in parts + component.plume_entities + component.canopy_entities + component.glow_entities:
            self._destroy_tree(scene, child)

        component.part_entities.clear()
        component.panel_entities.clear()
        component.pieces.clear()
        component.plume_entities.clear()
        component.canopy_entities.clear()
        component.glow_entities.clear()

    # Plumes (unit meshes: the nozzle's radius 1, 1 long,
    # from the nozzle at 0 down; v along them): a narrowing
    # flame in thick air, a wide spreading one in vacuum;
    # inside each a short bright core.
    PLUME_SHAPES = {
        "air": ((1.0, 0.0), (0.97, -0.12), (0.8, -0.45), (0.45, -0.8), (0.0, -1.0)),
        "air_core": ((0.62, 0.0), (0.5, -0.25), (0.22, -0.6), (0.0, -0.75)),
        "vacuum": ((1.0, 0.0), (1.55, -0.18), (2.3, -0.55), (2.9, -1.0)),
        "vacuum_core": ((0.7, 0.0), (0.9, -0.25), (1.05, -0.55)),
    }

    # HDR colors at full throttle (they bloom).
    PLUME_COLORS = {
        "air": (9.0, 3.6, 0.9),
        "air_core": (28.0, 20.0, 11.0),
        "vacuum": (1.6, 0.9, 0.55),
        "vacuum_core": (9.0, 7.0, 4.6),
    }

    def _glow_mesh(
        self,
        name: str,
        profile
    ) -> Handle:

        return self._resources.meshes.load(
            f"vessel/glow/{name}",
            lambda: Mesh.from_data(MeshFactory.lathe_data(profile, 32, None, crease_degrees=80.0))
        )

    def _update_plumes(
        self,
        scene: Scene,
        entity: Entity,
        component: VesselComponent,
        orbit_system,
        planet_system
    ):
        """Glowing exhaust under each burning engine: longer with the throttle, spreading as the air thins."""

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

        # Two glows an engine: the flame, its core.
        wanted = 2 * len(engines)

        while len(component.plume_entities) > wanted:
            self._destroy_tree(scene, component.plume_entities.pop())

        if not engines:
            return

        state = component.state

        environment = self.environment(scene, component.body, orbit_system, planet_system)

        radius = float(np.linalg.norm(state.position))

        pressure, _ = environment.air(radius, state.position / radius)

        # 0 at sea level, 1 in vacuum.
        thin = 1.0 - min(pressure / 1.01325, 1.0)

        kind = "vacuum" if thin > 0.6 else "air"

        while len(component.plume_entities) < wanted:

            child = scene.create_entity()

            scene.add_component(child, NameComponent("Exhaust"))
            scene.add_component(child, HierarchyComponent(entity))
            scene.add_component(child, TransformComponent(transform=Transform()))
            scene.add_component(child, GlowComponent(mesh=self._glow_mesh("air", self.PLUME_SHAPES["air"])))

            component.plume_entities.append(child)

        throttle = vessel.throttle

        for index, (engine, (x, _, z)) in enumerate(engines):

            nozzle = engine.part.radius_bottom * 0.9

            length = self.PLUME_LENGTH * (0.35 + 0.65 * throttle) * (1.0 + 0.6 * thin) * min(1.0, engine.part.radius_bottom / 0.9)

            for layer, name in enumerate((kind, f"{kind}_core")):

                child = component.plume_entities[2 * index + layer]

                glow = scene.get_component(child, GlowComponent)

                glow.mesh = self._glow_mesh(name, self.PLUME_SHAPES[name])
                glow.color = tuple(c * (0.25 + 0.75 * throttle) for c in self.PLUME_COLORS[name])
                glow.falloff = 1.2 if layer == 0 else 2.0
                glow.tail = 1.2 if layer == 0 else 1.6

                transform = scene.get_component(child, TransformComponent).transform

                transform.position = (x, engine.bottom, z)
                transform.scale = (nozzle, length * (1.0 if layer == 0 else 0.55), nozzle)

    # The glow of coming in: a bow shock round the leading
    # end and its wake behind (unit: the end's radius 1; the
    # shock's front at +y, the wake trailing to -y).
    SHOCK_SHAPE = ((0.0, 0.32), (0.55, 0.24), (1.02, 0.02), (1.3, -0.5), (1.5, -1.4))
    WAKE_SHAPE = ((1.15, 0.0), (1.0, -1.5), (0.55, -5.0), (0.0, -8.0))

    # Flux (W/m^2) the glow starts at, and where it is at its
    # brightest.
    GLOW_START = 1.5e5
    GLOW_FULL = 3.0e6

    def _update_reentry_glow(
        self,
        scene: Scene,
        entity: Entity,
        component: VesselComponent
    ):
        """Plasma round the end meeting the air, brighter and whiter with the heat flux."""

        vessel = component.vessel

        flux = vessel.heat_flux if not component.state.destroyed else 0.0

        core = vessel.core

        if flux < self.GLOW_START or not core:

            while component.glow_entities:
                self._destroy_tree(scene, component.glow_entities.pop())

            return

        while len(component.glow_entities) < 2:

            child = scene.create_entity()

            scene.add_component(child, NameComponent("Plasma"))
            scene.add_component(child, HierarchyComponent(entity))
            scene.add_component(child, TransformComponent(transform=Transform()))
            scene.add_component(child, GlowComponent(mesh=self._glow_mesh("shock", self.SHOCK_SHAPE)))

            component.glow_entities.append(child)

        strength = min((flux - self.GLOW_START) / (self.GLOW_FULL - self.GLOW_START), 1.0)

        # Orange, then pink-white as it grows fierce.
        hot = np.array([1.0, 0.36, 0.1]) * (1.0 - strength) + np.array([1.0, 0.75, 0.95]) * strength

        top = vessel.heat_end == "top"

        end = core[0] if top else core[-1]

        size = max(end.part.radius_top if top else end.part.radius_bottom, end.part.radius * 0.6)

        # Facing the air: up the stack from the top, down it
        # from the bottom.
        turned = np.array([0.0, 0.0, 0.0, 1.0]) if top else quaternion.from_axis_angle(np.array([1.0, 0.0, 0.0]), 180.0)

        where = (0.0, vessel.height if top else 0.0, 0.0)

        for child, (name, shape, scale, brightness, falloff) in zip(component.glow_entities, (
            ("shock", self.SHOCK_SHAPE, 1.0, 6.0 + 30.0 * strength, 0.8),
            ("wake", self.WAKE_SHAPE, 1.0 + strength, 1.5 + 6.0 * strength, 1.4),
        )):

            glow = scene.get_component(child, GlowComponent)

            glow.mesh = self._glow_mesh(name, shape)
            glow.color = tuple(float(c) * brightness for c in hot)
            glow.falloff = falloff
            glow.tail = 1.0

            transform = scene.get_component(child, TransformComponent).transform

            transform.position = where
            transform.orientation = turned
            transform.scale = (size * scale, size * scale, size * scale)

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

    # Below this height (m) the slope under the vessel is
    # looked up (for the HUD, and the landing).
    SLOPE_HEIGHT = 3_000.0

    # Below this height (m) over Mars the HUD shows its
    # weather.
    WEATHER_HEIGHT = 150_000.0

    @staticmethod
    def weather_here(
        frame: np.ndarray,
        now: float,
        position: np.ndarray
    ) -> weather.Weather:
        """Mars's weather over a point (inertial, about Mars; frame: Mars's)."""

        local = frame.T @ (position / float(np.linalg.norm(position)))

        latitude, longitude = latitude_longitude(local)

        return weather.weather_at(now, float(latitude[0]), float(longitude[0]))

    @classmethod
    def wind_at(
        cls,
        frame: np.ndarray,
        now: float,
        position: np.ndarray
    ) -> np.ndarray:
        """Mars's wind at a point (inertial m/s)."""

        here = cls.weather_here(frame, now, position)

        up = position / float(np.linalg.norm(position))

        pole = frame @ np.array([0.0, 1.0, 0.0])

        east = np.cross(pole, up)

        length = float(np.linalg.norm(east))

        if length < 1e-9:
            return np.zeros(3)

        east = east / length

        north = np.cross(up, east)

        return east * here.wind_east + north * here.wind_north

    # =====================================================
    # Landing Legs
    # =====================================================
    #
    # Each leg hangs from the top of its place (its pivot);
    # swinging down, its foot reaches out its span and down
    # level with the vessel's bottom (where it stands);
    # folded, it lies along the side.

    def _update_legs(
        self,
        scene: Scene,
        component: VesselComponent
    ):

        up = np.array([0.0, 1.0, 0.0])

        for state in component.vessel.parts:

            spec = state.part.legs

            if spec is None or not state.radial:
                continue

            copies = component.part_entities.get(id(state), [])

            out = state.extended

            height = state.part.height

            pivot = state.bottom + height

            reach = spec.span * out
            foot = state.bottom * (1.0 - out)

            base = math.radians(state.angle)

            for k, child in enumerate(copies):

                angle = base + 2.0 * math.pi * k / max(state.count, 1)

                radial = np.array([math.cos(angle), 0.0, math.sin(angle)])

                # From the foot up to the pivot.
                along = up * (pivot - foot) - radial * reach

                length = float(np.linalg.norm(along))

                if length <= 1e-6:
                    continue

                along = along / length

                turn = np.cross(up, along)

                sine = float(np.linalg.norm(turn))

                orientation = (
                    quaternion.from_axis_angle(turn / sine, math.degrees(math.atan2(sine, float(up @ along))))
                    if sine > 1e-9
                    else np.array([0.0, 0.0, 0.0, 1.0])
                )

                where = radial * (state.distance + reach) + up * foot

                transform = scene.get_component(child, TransformComponent).transform

                transform.position = (float(where[0]), float(where[1]), float(where[2]))
                transform.orientation = orientation
                transform.scale = (1.0, length / height, 1.0)

    # =====================================================
    # Solar Wings
    # =====================================================
    #
    # Each copy of a solar part grows its wings (vessel/
    # power.py): a stack part's out both sides, a radial
    # copy's straight out from the core, thin panels of
    # cells from its middle. Unfolding they grow out to
    # their length; out, each turns about its long axis to
    # face the Sun. Torn off: gone.

    WING_THICKNESS = 0.035

    def _build_wings(
        self,
        scene: Scene,
        component: VesselComponent,
        state,
        copies: list
    ):

        solar = state.part.solar

        def build():

            # A unit box, its cells a metre to a texture tile
            # on the wing at full size.
            data = MeshFactory.cube_data()

            data.vertices[:, 9] *= solar.length
            data.vertices[:, 10] *= solar.width

            return Mesh.from_data(data)

        mesh = self._resources.meshes.load(f"vessel/wing/{state.part.id}", build)

        material = self._finish_material("cells")

        wings = []

        for child, axis in zip(copies, power.wing_axes(state)):

            sides = (1.0, -1.0) if not state.radial else (1.0,)

            for side in sides:

                wing = scene.create_entity()

                scene.add_component(wing, NameComponent(f"{state.part.name} wing"))
                scene.add_component(wing, HierarchyComponent(child))
                scene.add_component(wing, TransformComponent(transform=Transform(scale=(0.0, 0.0, 0.0))))
                scene.add_component(wing, MeshRendererComponent(mesh=mesh, material=material))

                wings.append((wing, axis * side))

        component.panel_entities[id(state)] = wings

    def _update_wings(
        self,
        scene: Scene,
        component: VesselComponent
    ):
        """Each wing out as far as it is unfolded, turned to the Sun."""

        if not component.panel_entities:
            return

        status = component.power

        sun = getattr(status, "sun_local", None) if status is not None else None

        up = np.array([0.0, 1.0, 0.0])

        for state in component.vessel.parts:

            wings = component.panel_entities.get(id(state))

            if not wings:
                continue

            solar = state.part.solar

            out = 0.0 if state.broken else state.extended

            for wing, axis in wings:

                transform = scene.get_component(wing, TransformComponent).transform

                if out <= 0.0:

                    transform.scale = (0.0, 0.0, 0.0)

                    continue

                # Its face: square to the Sun about its axis;
                # without one, flat along the stack.
                if sun is not None:
                    normal = sun - axis * float(sun @ axis)
                else:
                    normal = np.cross(axis, up)

                length = float(np.linalg.norm(normal))

                normal = normal / length if length > 1e-6 else np.cross(axis, up)

                across = np.cross(normal, axis)

                # (Its copy stands upright: these are the
                # copy's own axes.)
                reach = solar.length * out

                center = axis * (state.part.radius + 0.5 * reach)

                transform.position = (float(center[0]), state.part.height * 0.5, float(center[2]))
                transform.orientation = quaternion.from_matrix3(np.column_stack((axis, across, normal)))
                transform.scale = (reach, solar.width, self.WING_THICKNESS)

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
            parachute=parachute,
            burned_up=state.burned_up,
            heat=(hot.temperature, hot.part.max_temperature, hot.part.name) if (hot := heating.hottest(vessel)) is not None else None,
            slope=self._slope if not state.destroyed else None,
            standing=landing.standing_angle(vessel),
            legs=landing.leg_state(vessel),
            toppled=state.toppled,
            fell=state.fell,
            propellant_kind=propellants.of(kind).short if (kind := vessel.section_kind()) is not None else "",
            weather=(
                self.weather_here(orbit_system.states[component.body][1], orbit_system.time, state.position)
                if physics.mars and component.body in orbit_system.states and radius - physics.radius < self.WEATHER_HEIGHT
                else None
            )
        )
