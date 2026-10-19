import math

from dataclasses import dataclass

import numpy as np

from vessel.vessel import PartState, Vessel


# =========================================================
# Landing
# =========================================================
#
# How a vessel meets the ground. It stands on its feet: its
# landing legs' pads, out round it, when they are down (G),
# else the bottom of its lowest part (an engine's bell, a
# heat shield). Touching down it
#
#   * breaks up, coming in faster than what it lands on
#     stands (legs take more than a bare engine);
#   * settles onto the slope under it (from the real heights
#     and the fine detail on them), standing square to the
#     ground, if it came down near enough upright;
#   * or tips over: the slope steeper than its feet hold
#     (its center of mass past the edge of its feet: the
#     wider the feet and the lower the mass, the steeper it
#     stands), coming down tilted past that, or sliding
#     sideways hard enough to roll over its downhill feet.
#     Fallen over it lies on its side: it cannot lift off
#     again, but a probe can still do its science and
#     report (as IM-1 Odysseus did in 2024). A tall vessel
#     falling in strong gravity breaks up.

# Seconds the legs take to swing down (or up).
LEG_TIME = 3.0

# m/s the top of a falling vessel may hit the ground at and
# survive lying there.
TOPPLE_CRASH = 9.0


def legs(
    vessel: Vessel
) -> list[PartState]:

    return [state for state in vessel.parts if state.part.legs is not None]


def legs_down(
    vessel: Vessel
) -> bool:
    """Landing legs, all fully down."""

    found = legs(vessel)

    return bool(found) and all(state.extended >= 0.999 for state in found)


def feet(
    vessel: Vessel
) -> float:
    """
    m from the axis to the edge of what it stands on: its legs'
    pads (down), else its lowest part's bottom.
    """

    reach = 0.0

    # (Legs not all the way down stand on nothing: they are
    # still up the side.)
    if legs_down(vessel):
        for state in legs(vessel):
            reach = max(reach, state.distance + state.part.legs.span)

    core = vessel.core

    bottom = core[-1].part.radius_bottom if core else 0.5

    return max(reach, bottom)


def crash_speed(
    vessel: Vessel
) -> float:
    """m/s it survives touching down at: on its legs, theirs; else its lowest part's."""

    speed = vessel.crash_speed

    if legs_down(vessel):
        speed = max(speed, max(state.part.crash_speed for state in legs(vessel)))

    return speed


def standing_angle(
    vessel: Vessel
) -> float:
    """Degrees of slope it still stands on: its mass over the edge of its feet beyond."""

    height = max(vessel.center_of_mass, 0.05)

    return math.degrees(math.atan2(feet(vessel), height))


def move_legs(
    vessel: Vessel,
    dt: float
):
    """The legs swinging down (or up) over LEG_TIME."""

    step = dt / LEG_TIME

    for state in legs(vessel):

        if state.extending:
            state.extended = min(state.extended + step, 1.0)
        else:
            state.extended = max(state.extended - step, 0.0)


def set_legs(
    vessel: Vessel,
    down: bool
):

    for state in legs(vessel):
        state.extending = down


def toggle_legs(
    vessel: Vessel
) -> bool | None:
    """Legs down (all up or going up) or up; the new direction, None without any."""

    found = legs(vessel)

    if not found:
        return None

    down = not any(state.extending for state in found)

    set_legs(vessel, down)

    return down


def leg_state(
    vessel: Vessel
) -> str:

    found = legs(vessel)

    if not found:
        return ""

    if all(state.extending and state.extended >= 0.999 for state in found):
        return "down"

    if any(state.extending for state in found):
        return "lowering"

    if any(state.extended > 0.0 for state in found):
        return "raising"

    return "up"


@dataclass(frozen=True, slots=True)
class Touchdown:
    """What touching down did."""

    outcome: str                    # "stand", "topple", "crash"
    reason: str = ""
    axis: np.ndarray | None = None  # the stack's axis after (inertial)
    impact: float = 0.0             # m/s


def slope_of(
    normal: np.ndarray,
    up: np.ndarray
) -> float:
    """Degrees between the ground's normal and the local vertical."""

    return math.degrees(math.acos(min(max(float(normal @ up), -1.0), 1.0)))


def touch_down(
    vessel: Vessel,
    axis: np.ndarray,
    normal: np.ndarray,
    up: np.ndarray,
    velocity: np.ndarray,
    gravity: float
) -> Touchdown:
    """
    The vessel meeting the ground: axis its stack's direction,
    normal the ground's (both unit, inertial), up the local
    vertical, velocity relative to the ground (m/s), gravity
    m/s^2.
    """

    impact = float(np.linalg.norm(velocity))

    if impact > crash_speed(vessel):
        return Touchdown("crash", f"hit the ground at {impact:,.0f} m/s", None, impact)

    stand = standing_angle(vessel)

    tilt = math.degrees(math.acos(min(max(float(axis @ normal), -1.0), 1.0)))

    slope = slope_of(normal, up)

    if slope > stand:
        return Touchdown("topple", f"a {slope:,.0f} deg slope, steeper than its feet hold ({stand:,.0f} deg)", None, impact)

    # Tilted against the ground past its stance: it lands on
    # its edge and goes over.
    if tilt > stand:
        return Touchdown("topple", f"came down {tilt:,.0f} deg off square to the ground", None, impact)

    # Sliding: rolls over its feet if the sideways motion's
    # energy (horizontal: coming straight down a slope is the
    # slope's business) lifts its mass over their edge.
    across = velocity - up * float(velocity @ up)

    sideways = float(np.linalg.norm(across))

    height = max(vessel.center_of_mass, 0.05)

    reach = feet(vessel)

    lever = math.hypot(height, reach)

    critical = math.atan2(reach, height)

    barrier = lever * (1.0 - math.cos(max(critical - math.radians(slope), 0.0)))

    if 0.5 * sideways * sideways > gravity * barrier:
        return Touchdown("topple", f"sliding {sideways:,.1f} m/s sideways, it rolled over", None, impact)

    return Touchdown("stand", "", normal.copy(), impact)


def fallen_axis(
    normal: np.ndarray,
    up: np.ndarray,
    axis: np.ndarray,
    velocity: np.ndarray
) -> np.ndarray:
    """The stack's axis lying on the ground: downhill, else the way it leaned, else the way it slid."""

    downhill = -(up - normal * float(up @ normal))

    for guess in (downhill, axis - normal * float(axis @ normal), velocity - normal * float(velocity @ normal)):

        length = float(np.linalg.norm(guess))

        if length > 1e-3:
            return guess / length

    return _any_across(normal)


def fall_speed(
    vessel: Vessel,
    gravity: float
) -> float:
    """m/s its top hits the ground falling over (a rod falling from upright: about sqrt(3 g h), h its mass's height)."""

    return math.sqrt(3.0 * gravity * max(vessel.center_of_mass, 0.0))


def _any_across(
    normal: np.ndarray
) -> np.ndarray:

    hint = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 0.0, 1.0])

    across = np.cross(normal, hint)

    return across / float(np.linalg.norm(across))


def ground_normal(
    ground,
    direction: np.ndarray,
    spacing: float
) -> np.ndarray:
    """
    The ground's normal under a unit direction (inertial):
    ground(direction) -> the surface's radius (m) there,
    sampled `spacing` m round it.
    """

    across = _any_across(direction)
    along = np.cross(direction, across)

    radius = ground(direction)

    angle = spacing / max(radius, 1.0)

    points = []

    for offset in (across, along, -across, -along):

        sample = direction + offset * angle
        sample = sample / float(np.linalg.norm(sample))

        points.append(sample * ground(sample))

    # Across the samples: two diagonals' cross product.
    normal = np.cross(points[0] - points[2], points[1] - points[3])

    length = float(np.linalg.norm(normal))

    if length <= 0.0:
        return direction.copy()

    normal = normal / length

    return normal if float(normal @ direction) > 0.0 else -normal
