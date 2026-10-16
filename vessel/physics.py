import math

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from math3d import quaternion

from vessel import kepler
from vessel.vessel import Vessel


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
#   ground    below the surface: landed (at most the bottom
#             part's crash speed) or destroyed; a landed
#             vessel turns with the body until the engines
#             lift it
#
# Coasting outside the air without thrust, the orbit is
# followed exactly instead (vessel/kepler.py): any time
# warp.

# Drag coefficient of a rocket nose-first (a cone on a
# cylinder, ~0.3-0.5).
DRAG_COEFFICIENT = 0.45

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

    def copy(self) -> "VesselState":

        return VesselState(
            self.position.copy(),
            self.velocity.copy(),
            np.asarray(self.orientation, dtype=np.float64).copy(),
            self.angular_velocity.copy(),
            self.landed,
            self.destroyed,
            self.impact_speed
        )

    @property
    def up(
        self
    ) -> np.ndarray:
        """The stack's axis (thrust direction), inertial."""

        return quaternion.rotate_vector(self.orientation, np.array([0.0, 1.0, 0.0]))


@dataclass(slots=True)
class Controls:

    pitch: float = 0.0              # -1..1
    yaw: float = 0.0
    roll: float = 0.0

    # Stability assist: hold the attitude when not steering.
    sas: bool = True


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


# =========================================================
# Stepping
# =========================================================

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

    # -------------------------------------------------
    # Attitude
    # -------------------------------------------------

    _turn(vessel, state, controls, pressure, dt)

    # -------------------------------------------------
    # Forces
    # -------------------------------------------------

    thrust, flow = vessel.thrust(pressure)

    mass = vessel.mass

    acceleration = gravity(environment, state.position) + state.up * (thrust / mass)

    if density > VACUUM_DENSITY:

        air_velocity = state.velocity - surface_velocity(environment, state.position)

        speed = float(np.linalg.norm(air_velocity))

        area = math.pi * vessel.radius ** 2

        acceleration -= 0.5 * density * speed * air_velocity * DRAG_COEFFICIENT * area / mass

    vessel.burn(dt, pressure)

    # -------------------------------------------------
    # On the ground
    # -------------------------------------------------

    if state.landed:

        # Lifts off once the push up beats the weight.
        if float(np.dot(acceleration, direction)) <= 0.0:

            _ride_the_ground(state, environment, dt)

            return

        state.landed = False

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


def _turn(
    vessel: Vessel,
    state: VesselState,
    controls: Controls,
    pressure: float,
    dt: float
):

    if state.landed:

        state.angular_velocity = np.zeros(3)

        return

    torque = vessel.control_torque(pressure)

    pitch_inertia, roll_inertia = vessel.moments_of_inertia

    # Rad/s^2 available about each axis (x pitch, y roll,
    # z yaw).
    limit = np.array([torque / pitch_inertia, torque / roll_inertia, torque / pitch_inertia])

    command = np.array([controls.pitch, controls.roll, controls.yaw], dtype=np.float64)

    accel = command * limit

    if controls.sas:

        # Brake whatever axis is not being steered.
        idle = np.abs(command) < 1e-3

        brake = np.clip(-state.angular_velocity / max(dt, 1e-6), -limit, limit)

        accel = np.where(idle, brake, accel)

    state.angular_velocity = state.angular_velocity + accel * dt

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

    state.position = direction * environment.ground(direction)
    state.velocity = surface_velocity(environment, state.position)
    state.angular_velocity = np.zeros(3)


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

    impact = float(np.linalg.norm(relative))

    state.position = direction * ground

    if impact > vessel.crash_speed:

        state.destroyed = True
        state.impact_speed = impact
        state.velocity = surface_velocity(environment, state.position)

        return

    state.landed = True
    state.velocity = surface_velocity(environment, state.position)
    state.angular_velocity = np.zeros(3)


def land(
    state: VesselState,
    environment: Environment
):
    """Settle a vessel on the ground under it (a launch)."""

    radius = float(np.linalg.norm(state.position))

    direction = state.position / radius

    state.position = direction * environment.ground(direction)
    state.velocity = surface_velocity(environment, state.position)
    state.angular_velocity = np.zeros(3)
    state.landed = True


def upright(
    direction: np.ndarray
) -> np.ndarray:
    """Orientation standing the stack along `direction`."""

    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)

    hint = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([1.0, 0.0, 0.0])

    # look_rotation(forward, up) puts up along +Y.
    return quaternion.look_rotation(np.cross(hint, direction), direction)
