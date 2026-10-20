import math

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from math3d import quaternion

from vessel import heating, kepler, landing, rover
from vessel.vessel import BODY_DRAG_COEFFICIENT, G0, Vessel


# =========================================================
# Flight Physics
# =========================================================
#
# One vessel, one body: positions and velocities relative to
# the body's center in its non-rotating frame (m, m/s),
# orientation in the same frame. The vessel's own axes:
# +Y along the stack (thrust pushes this way), X and Z
# across it.
#
# Each step:
#   gravity   -mu r / |r|^3 (the body as a point mass)
#   thrust    along +Y, at the throttle, the push set by the
#             outside pressure (Isp); propellant burns
#   drag      -1/2 rho |u| u Cd A, u the velocity through the
#             air, which turns with the body (u = v - w x r)
#   attitude  pitch / yaw / roll commands spin the vessel
#             (reaction wheels and engine gimbal); SAS holds
#             it still
#   ground    below the surface: landed (at most what it
#             lands on stands: its legs, else its bottom
#             part) or destroyed; standing square to the
#             slope, or fallen over (vessel/landing.py); a
#             landed vessel turns with the body until the
#             engines lift it (not on its side)
#   heat      fast through the air, the leading part's skin
#             heats (vessel/heating.py); past its limit the
#             vessel burns up
#
# Coasting outside the air without thrust, the orbit is
# followed exactly instead (vessel/kepler.py): any time
# warp.

# (The stack's own drag coefficient: vessel/vessel.py.)
DRAG_COEFFICIENT = BODY_DRAG_COEFFICIENT

# Air thinner than this (kg/m^3) does not matter: vacuum.
VACUUM_DENSITY = 1e-9


@dataclass(slots=True)
class VesselState:

    position: np.ndarray            # m from the body's center
    velocity: np.ndarray            # m/s
    orientation: np.ndarray         # quaternion xyzw: vessel -> inertial
    angular_velocity: np.ndarray    # rad/s about the vessel's own axes

    landed: bool = False
    destroyed: bool = False

    # Speed it hit the ground at, when destroyed (m/s).
    impact_speed: float = 0.0

    # Destroyed by heating: the part that burned up's name.
    burned_up: str = ""

    # The thrust's velocity change so far (m/s, inertial;
    # gravity and drag not counted): what a maneuver node's
    # burn has done.
    thrust_dv: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Fallen over landing (vessel/landing.py): on its side,
    # its origin this far (m) above the ground; why (also
    # why, when the fall broke it).
    toppled: bool = False
    rest_height: float = 0.0
    fell: str = ""

    # Driving (vessel/rover.py): m/s along its front, its
    # wheels' turn (rad), m driven since the slope was looked
    # at.
    drive: float = 0.0
    wheel_turn: float = 0.0
    driven: float = 0.0

    def copy(self) -> "VesselState":

        return VesselState(
            self.position.copy(),
            self.velocity.copy(),
            np.asarray(self.orientation, dtype=np.float64).copy(),
            self.angular_velocity.copy(),
            self.landed,
            self.destroyed,
            self.impact_speed,
            self.burned_up,
            self.thrust_dv.copy(),
            self.toppled,
            self.rest_height,
            self.fell,
            self.drive,
            self.wheel_turn,
            self.driven
        )

    @property
    def up(
        self
    ) -> np.ndarray:
        """The stack's axis (thrust direction), inertial."""

        return quaternion.rotate_vector(self.orientation, np.array([0.0, 1.0, 0.0]))


SAS_MODES = ("hold", "prograde", "retrograde", "maneuver")


@dataclass(slots=True)
class Controls:

    pitch: float = 0.0              # -1..1
    yaw: float = 0.0
    roll: float = 0.0

    # Stability assist when not steering: "hold" the
    # attitude, or point the nose "prograde" / "retrograde"
    # (along / against the motion: through the air while in
    # it, along the orbit above it), or along a "maneuver"
    # (a planned velocity change, below; holds without one).
    sas: bool = True
    sas_mode: str = "hold"

    # The planned velocity change (m/s, inertial), or None.
    maneuver: np.ndarray | None = None

    # Translation (RCS), -1..1 along the vessel's own X, Y
    # (the stack), Z: for docking.
    translate: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def translating(
        self
    ) -> bool:

        return bool(np.any(np.abs(self.translate) > 1e-6))


@dataclass(slots=True)
class Environment:
    """
    The body, as physics needs it:

    mu: G M (m^3/s^2)
    spin: angular velocity (rad/s, inertial)
    air(radius m, direction) -> (pressure bar, density kg/m^3)
        at a distance from the center
    ground(direction) -> radius (m) of the surface below a
        unit inertial direction (the sea's over water)
    """

    mu: float
    spin: np.ndarray
    air: Callable[[float, np.ndarray], tuple[float, float]]
    ground: Callable[[np.ndarray], float]

    # Radius above which there is no air worth counting.
    air_top: float = 0.0

    # The air's temperature (K) at a radius (m); None: no air.
    temperature: Callable[[float], float] | None = None

    # The ground's unit normal under a direction (inertial);
    # None: level (square to the direction).
    normal: Callable[[np.ndarray], np.ndarray] | None = None

    # The wind at a position (m/s, inertial, on top of the air
    # turning with the body; planet/weather.py); None: still.
    wind: Callable[[np.ndarray], np.ndarray] | None = None


# =========================================================
# Stepping
# =========================================================

def air_velocity(
    environment: Environment,
    state: VesselState
) -> np.ndarray:
    """The vessel's velocity through the air: less the air's turning with the body, and its wind."""

    relative = state.velocity - surface_velocity(environment, state.position)

    if environment.wind is not None:
        relative = relative - environment.wind(state.position)

    return relative


def surface_velocity(
    environment: Environment,
    position: np.ndarray
) -> np.ndarray:
    """Velocity of the ground (and the air) at a point (m/s)."""

    return np.cross(environment.spin, position)


def gravity(
    environment: Environment,
    position: np.ndarray
) -> np.ndarray:

    r = float(np.linalg.norm(position))

    return -environment.mu * position / r ** 3


def can_coast(
    vessel: Vessel,
    state: VesselState,
    environment: Environment
) -> bool:
    """On rails: flying, no thrust, above the air."""

    return (
        not state.landed
        and not state.destroyed
        and (vessel.throttle <= 0.0 or not vessel.burning_engines())
        and float(np.linalg.norm(state.position)) > environment.air_top
    )


def step(
    vessel: Vessel,
    state: VesselState,
    environment: Environment,
    controls: Controls,
    dt: float
):
    """Advance the vessel `dt` seconds (small: ~0.02 s)."""

    if state.destroyed or dt <= 0.0:
        return

    radius = float(np.linalg.norm(state.position))
    direction = state.position / radius

    pressure, density = environment.air(radius, direction)

    through_air = air_velocity(environment, state) if density > VACUUM_DENSITY else state.velocity - surface_velocity(environment, state.position)

    # -------------------------------------------------
    # Attitude
    # -------------------------------------------------

    # Hanging under an open parachute, the vessel turns to
    # trail it (nose against the motion through the air),
    # whatever the pilot asks.
    if density > VACUUM_DENSITY and any(chute.deployed > 0.2 for chute in vessel.parachutes):
        controls = Controls(sas=True, sas_mode="retrograde")

    _turn(vessel, state, controls, pressure, dt, motion(state, environment, density))

    # -------------------------------------------------
    # Forces
    # -------------------------------------------------

    thrust, flow = vessel.thrust(pressure)

    # (Driving sets it again, landed on wheels.)
    vessel.drive_power = 0.0

    # Rotors: lift by the air's density, on battery power.
    if vessel.powered and vessel.throttle > 0.0:
        thrust += rover.rotor_lift(vessel, density) * vessel.throttle

    mass = vessel.mass

    push = state.up * (thrust / mass) + rcs_push(vessel, state, controls, dt)

    acceleration = gravity(environment, state.position) + push

    state.thrust_dv = state.thrust_dv + push * dt

    if density > VACUUM_DENSITY:

        speed = float(np.linalg.norm(through_air))

        vessel.open_parachutes(pressure, speed, dt)

        acceleration -= 0.5 * density * speed * through_air * vessel.drag_area() / mass

    ambient = environment.temperature(radius) if environment.temperature is not None and density > VACUUM_DENSITY else heating.SPACE_TEMPERATURE

    burned = heating.heat(vessel, state.up, through_air, density if density > VACUUM_DENSITY else 0.0, ambient, dt)

    if burned is not None:

        state.destroyed = True
        state.burned_up = burned.part.name

        return

    vessel.burn(dt, pressure)

    # -------------------------------------------------
    # On the ground
    # -------------------------------------------------

    if state.landed:

        # Lifts off once the push up beats the weight (never
        # lying on its side).
        if state.toppled or float(np.dot(acceleration, direction)) <= 0.0:

            # Driving on its wheels (W / S, A / D).
            if not state.toppled and rover.wheels(vessel):
                _drive(vessel, state, environment, controls, dt)

            _ride_the_ground(state, environment, dt)

            return

        state.landed = False
        state.rest_height = 0.0

    # -------------------------------------------------
    # Moving (semi-implicit Euler: keeps orbits closed)
    # -------------------------------------------------

    state.velocity = state.velocity + acceleration * dt
    state.position = state.position + state.velocity * dt

    _touch_ground(vessel, state, environment)


def coast(
    state: VesselState,
    environment: Environment,
    dt: float
):
    """Follow the orbit exactly (no thrust, no air)."""

    state.position, state.velocity = kepler.propagate(state.position, state.velocity, environment.mu, dt)

    # Still spinning, if it was.
    _spin(state, dt)


def steer(
    vessel: Vessel,
    state: VesselState,
    controls: Controls,
    dt: float,
    environment: Environment | None = None
):
    """Coasting: the pilot's (and SAS's) turning, in space (no air), without moving it."""

    _turn(vessel, state, controls, 0.0, dt, motion(state, environment) if environment is not None else state.velocity)


# Below this height above the ground (m), prograde and
# retrograde are over the ground, not round the body: to
# land, the nose must point against the motion the ground
# sees.
SURFACE_MODE_HEIGHT = 30_000.0


def motion(
    state: VesselState,
    environment: Environment,
    density: float | None = None
) -> np.ndarray:
    """What prograde means now: through the air in it, over the ground near it, along the orbit above."""

    radius = float(np.linalg.norm(state.position))

    if density is None:
        _, density = environment.air(radius, state.position / radius)

    if density > VACUUM_DENSITY:
        return air_velocity(environment, state)

    if radius - environment.ground(state.position / radius) < SURFACE_MODE_HEIGHT:
        return state.velocity - surface_velocity(environment, state.position)

    return state.velocity


# The translation thrusters' specific impulse (s): they burn
# the tanks' propellant (cold hydrazine-like, ~220 s).
RCS_ISP = 220.0


def rcs_push(
    vessel: Vessel,
    state: VesselState,
    controls: Controls,
    dt: float = 0.0
) -> np.ndarray:
    """
    The translation thrusters' acceleration (inertial, m/s^2);
    dt given, burning their propellant for that long (none
    left: no push).
    """

    if not controls.translating:
        return np.zeros(3)

    thrust = vessel.rcs_thrust

    if thrust <= 0.0 or vessel.propellant <= 0.0:
        return np.zeros(3)

    command = np.clip(np.asarray(controls.translate, dtype=np.float64), -1.0, 1.0)

    if dt > 0.0:

        used = thrust * float(np.sum(np.abs(command)))

        vessel.draw_propellant(used / (RCS_ISP * G0) * dt)

    return quaternion.rotate_vector(state.orientation, command) * (thrust / vessel.mass)


def cool(
    vessel: Vessel,
    state: VesselState,
    dt: float
):
    """Coasting in space: the skins radiate their heat away."""

    heating.heat(vessel, state.up, np.zeros(3), 0.0, heating.SPACE_TEMPERATURE, dt)


def _turn(
    vessel: Vessel,
    state: VesselState,
    controls: Controls,
    pressure: float,
    dt: float,
    motion: np.ndarray | None = None
):
    """
    motion: what prograde means now (through the air while
    in it, along the orbit above it).
    """

    if state.landed:

        state.angular_velocity = np.zeros(3)
        vessel.wheel_use = 0.0

        return

    torque = vessel.control_torque(pressure)

    pitch_inertia, roll_inertia = vessel.moments_of_inertia

    # Rad/s^2 available about each axis (x pitch, y roll,
    # z yaw).
    limit = np.array([torque / pitch_inertia, torque / roll_inertia, torque / pitch_inertia])

    command = np.array([controls.pitch, controls.roll, controls.yaw], dtype=np.float64)

    accel = command * limit

    if controls.sas:

        # Brake whatever axis is not being steered; in a
        # pointing mode, turn the nose to the target.
        wanted = np.zeros(3)

        speed = float(np.linalg.norm(motion)) if motion is not None else 0.0

        target = None

        if controls.sas_mode == "maneuver":

            if controls.maneuver is not None and float(np.linalg.norm(controls.maneuver)) > 1e-6:
                target = controls.maneuver / float(np.linalg.norm(controls.maneuver))

        elif controls.sas_mode != "hold" and speed > 0.5:

            target = motion / speed * (1.0 if controls.sas_mode == "prograde" else -1.0)

        if target is not None:

            # The target in the vessel's own axes.
            local = quaternion.rotate_vector(quaternion.conjugate(state.orientation), target)

            # About the axis from the nose (+Y) to it; no
            # faster than it can stop in time (v = sqrt(2 a
            # angle)); no roll.
            axis = np.cross([0.0, 1.0, 0.0], local)

            sine = float(np.linalg.norm(axis))

            angle = math.atan2(sine, float(local[1]))

            if sine > 1e-9:

                rate = math.sqrt(2.0 * 0.8 * min(limit[0], limit[2]) * angle)

                wanted = axis / sine * min(rate, 0.6)

            elif local[1] < 0.0:

                # Exactly backwards: any way round.
                wanted = np.array([0.3, 0.0, 0.0])

        idle = np.abs(command) < 1e-3

        brake = np.clip((wanted - state.angular_velocity) / max(dt, 1e-6), -limit, limit)

        accel = np.where(idle, brake, accel)

    state.angular_velocity = state.angular_velocity + accel * dt

    # How hard the wheels worked (their power draw).
    vessel.wheel_use = min(float(np.max(np.abs(accel) / np.maximum(limit, 1e-12))), 1.0) if torque > 0.0 else 0.0

    _spin(state, dt)


def _spin(
    state: VesselState,
    dt: float
):

    rate = float(np.linalg.norm(state.angular_velocity))

    if rate * dt < 1e-12:
        return

    axis = state.angular_velocity / rate

    # About the vessel's own axes: multiply on the right.
    state.orientation = quaternion.normalize(
        quaternion.multiply(state.orientation, quaternion.from_axis_angle(axis, math.degrees(rate * dt)))
    )


def _ride_the_ground(
    state: VesselState,
    environment: Environment,
    dt: float
):
    """Landed: carried round with the body's spin (exact for any dt)."""

    spin = float(np.linalg.norm(environment.spin))

    if spin > 0.0:

        turn = quaternion.from_axis_angle(environment.spin / spin, math.degrees(spin * dt))

        state.position = quaternion.rotate_vector(turn, state.position)
        state.orientation = quaternion.normalize(quaternion.multiply(turn, state.orientation))

    radius = float(np.linalg.norm(state.position))

    direction = state.position / radius

    state.position = direction * (environment.ground(direction) + state.rest_height)
    state.velocity = surface_velocity(environment, state.position)
    state.angular_velocity = np.zeros(3)


def _drive(
    vessel: Vessel,
    state: VesselState,
    environment: Environment,
    controls: Controls,
    dt: float
):
    """A landed vessel's wheels driving it; every SLOPE_CHECK m the slope it is on (steeper than it stands: over it goes)."""

    up = state.position / float(np.linalg.norm(state.position))

    # (In time warp: not across the map in a frame.)
    state.driven += rover.drive(vessel, state, -controls.pitch, controls.yaw, up, min(dt, 0.5))

    vessel.drive_power = rover.wheel_power(vessel, state.drive)

    if state.driven < rover.SLOPE_CHECK or environment.normal is None:
        return

    state.driven = 0.0

    direction = state.position / float(np.linalg.norm(state.position))

    normal = environment.normal(direction)

    slope = landing.slope_of(normal, direction)

    if slope > landing.standing_angle(vessel):

        lying = landing.fallen_axis(normal, direction, state.up, state.velocity - surface_velocity(environment, state.position))

        state.orientation = quaternion.look_rotation(-normal, lying)
        state.toppled = True
        state.drive = 0.0
        state.fell = f"drove onto a {slope:,.0f} deg slope and rolled over"
        state.rest_height = max((part.part.radius for part in vessel.core), default=0.5)

        return

    # Square to the ground it is on.
    axis = state.up

    turn = np.cross(axis, normal)

    sine = float(np.linalg.norm(turn))

    if sine > 1e-9:

        angle = math.degrees(math.atan2(sine, float(axis @ normal)))

        state.orientation = quaternion.normalize(quaternion.multiply(quaternion.from_axis_angle(turn / sine, angle), state.orientation))


def _touch_ground(
    vessel: Vessel,
    state: VesselState,
    environment: Environment
):

    radius = float(np.linalg.norm(state.position))

    direction = state.position / radius

    ground = environment.ground(direction)

    if radius >= ground:
        return

    relative = state.velocity - surface_velocity(environment, state.position)

    state.position = direction * ground

    normal = environment.normal(direction) if environment.normal is not None else direction

    weight = environment.mu / (ground * ground)

    axis = state.up

    result = landing.touch_down(vessel, axis, normal, direction, relative, weight)

    state.velocity = surface_velocity(environment, state.position)
    state.angular_velocity = np.zeros(3)

    if result.outcome == "crash":

        state.destroyed = True
        state.impact_speed = result.impact

        return

    if result.outcome == "topple":

        fall = landing.fall_speed(vessel, weight)

        if fall > landing.TOPPLE_CRASH:

            state.destroyed = True
            state.impact_speed = fall
            state.fell = f"fell over ({result.reason}) and broke up"

            return

        # On its side, along the ground.
        lying = landing.fallen_axis(normal, direction, axis, relative)

        state.orientation = quaternion.look_rotation(-normal, lying)
        state.toppled = True
        state.fell = result.reason
        state.rest_height = max((part.part.radius for part in vessel.core), default=0.5)
        state.position = direction * (ground + state.rest_height)
        state.landed = True

        return

    # Standing: square to the ground.
    turn = np.cross(axis, normal)

    sine = float(np.linalg.norm(turn))

    if sine > 1e-9:

        angle = math.degrees(math.atan2(sine, float(axis @ normal)))

        state.orientation = quaternion.normalize(
            quaternion.multiply(quaternion.from_axis_angle(turn / sine, angle), state.orientation)
        )

    state.landed = True

    # On its wheels: held up by them.
    state.rest_height = rover.clearance(vessel)


def land(
    state: VesselState,
    environment: Environment,
    vessel: Vessel | None = None
):
    """Settle a vessel on the ground under it (a launch; on its wheels, if it has them)."""

    radius = float(np.linalg.norm(state.position))

    direction = state.position / radius

    state.position = direction * environment.ground(direction)
    state.velocity = surface_velocity(environment, state.position)
    state.angular_velocity = np.zeros(3)
    state.landed = True
    state.toppled = False
    state.rest_height = rover.clearance(vessel) if vessel is not None else 0.0

    state.position = direction * (environment.ground(direction) + state.rest_height)


def upright(
    direction: np.ndarray
) -> np.ndarray:
    """Orientation standing the stack along `direction`."""

    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)

    hint = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([1.0, 0.0, 0.0])

    # look_rotation(forward, up) puts up along +Y.
    return quaternion.look_rotation(np.cross(hint, direction), direction)
