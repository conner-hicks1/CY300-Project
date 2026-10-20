import math

import numpy as np

from math3d import quaternion

from vessel.vessel import PartState, Vessel


# =========================================================
# Wheels and Rotors
# =========================================================
#
# Wheels: a landed vessel with wheels (and power) drives over
# the ground: W / S forward and back (its front its -Z, as a
# pitch forward), A / D steer (turning on the spot when
# stopped). It speeds up at ACCELERATION to its wheels' top
# speed, coasts down when let go, draws its wheels' power as
# hard as it drives. It goes where the ground goes (the real
# heights); now and then the slope under it is looked at,
# and steeper than it stands it rolls over (vessel/
# landing.py).
#
# Rotors: lift along the stack in proportion to the air's
# density (N per kg/m^3 at full throttle): plenty on Earth,
# just enough on Mars (Ingenuity flew in air 1.5% as thick),
# none in vacuum; they draw their power as hard as they
# turn. No propellant: as long as the batteries last.

ACCELERATION = 0.6                  # m/s^2
COAST = 2.0                         # s, the speed's e-folding let go
TURN_RADIUS = 4.0                   # m at speed
SPIN_TURN = 0.5                     # rad/s turning on the spot

# Every this many m driven, the slope under it is checked.
SLOPE_CHECK = 5.0


def wheels(
    vessel: Vessel
) -> list[PartState]:

    return [state for state in vessel.parts if state.part.wheels is not None]


def rotors(
    vessel: Vessel
) -> list[PartState]:

    return [state for state in vessel.parts if state.part.rotor is not None]


def top_speed(
    vessel: Vessel
) -> float:

    found = wheels(vessel)

    return min((state.part.wheels.speed for state in found), default=0.0)


def clearance(
    vessel: Vessel
) -> float:
    """m its wheels hold its origin above the ground (0 without wheels, or wheels above its bottom)."""

    return max((state.part.radius - state.bottom for state in wheels(vessel)), default=0.0)


def wheel_power(
    vessel: Vessel,
    speed: float
) -> float:
    """W its wheels draw driving at `speed`."""

    found = wheels(vessel)

    top = top_speed(vessel)

    if not found or top <= 0.0:
        return 0.0

    share = min(abs(speed) / top, 1.0)

    return sum(state.part.wheels.power * state.count for state in found) * share


def rotor_lift(
    vessel: Vessel,
    density: float
) -> float:
    """N of lift its rotors give at full throttle in air of `density` kg/m^3."""

    return sum(state.part.rotor.lift * state.count for state in rotors(vessel)) * max(density, 0.0)


def rotor_power(
    vessel: Vessel
) -> float:
    """W its rotors draw at its throttle."""

    return sum(state.part.rotor.power * state.count for state in rotors(vessel)) * vessel.throttle


def drive(
    vessel: Vessel,
    state,
    forward_input: float,
    steer_input: float,
    up: np.ndarray,
    dt: float
) -> float:
    """
    `dt` s of driving (landed): the speed (state.drive) and
    heading change, the vessel moved along the ground (its
    height left to the caller). Returns the m driven.
    """

    top = top_speed(vessel)

    speed = state.drive

    if not wheels(vessel) or top <= 0.0 or not vessel.powered:
        forward_input = 0.0

    if abs(forward_input) > 1e-3:
        speed += forward_input * ACCELERATION * dt
    else:
        speed *= math.exp(-dt / COAST)

        if abs(speed) < 0.01:
            speed = 0.0

    speed = min(max(speed, -0.5 * top), top)

    state.drive = speed

    # Turning: as a car at speed, on the spot when stopped.
    if abs(speed) > 0.2:
        rate = steer_input * speed / TURN_RADIUS
    else:
        rate = steer_input * SPIN_TURN if wheels(vessel) and vessel.powered else 0.0

    if abs(rate) > 1e-9:

        turn = quaternion.from_axis_angle(up, math.degrees(rate * dt))

        state.orientation = quaternion.normalize(quaternion.multiply(turn, state.orientation))

    if speed == 0.0:
        return 0.0

    # Its front (-Z) along the ground.
    front = quaternion.rotate_vector(state.orientation, np.array([0.0, 0.0, -1.0]))

    front = front - up * float(front @ up)

    length = float(np.linalg.norm(front))

    if length < 1e-6:
        return 0.0

    step = front / length * speed * dt

    state.position = state.position + step

    state.wheel_turn += speed * dt / max(min(state_radius(vessel), 10.0), 0.05)

    return abs(speed * dt)


def state_radius(
    vessel: Vessel
) -> float:
    """Its wheels' radius (for their turning)."""

    return max((state.part.radius for state in wheels(vessel)), default=0.3)
