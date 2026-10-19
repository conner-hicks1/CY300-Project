import math
import random

import numpy as np

from vessel import kepler
from vessel.parts import PartDefinition
from vessel.vessel import PartState, Vessel


# =========================================================
# Orbital Debris
# =========================================================
#
# What reaches orbit stays there: a spent upper stage, a
# fairing, the fragments of a collision. They go round on
# their own (on rails), saved with the fleet, until the thin
# upper air brings them down (Earth: days from 200 km, a year
# or two from 400 km, decades from 600, centuries higher).
# Two things meeting at orbital speed break into a cloud of
# fragments flung apart at a share of the speed they met
# at; each can hit something else (the Kessler syndrome).
#
#   thermosphere  the upper air's density (kg/m^3) from 200 to
#                 1,000 km: an exponential between measured
#                 points (a middling Sun)
#   decay         drag slows the orbit's speed, lowering its
#                 semi-major axis by sqrt(mu a) rho B per s
#                 (B: drag coefficient x area / mass); kept the
#                 same shape and turned the same way

# (km, ln of kg/m^3) at a middling solar activity.
_EARTH_THERMOSPHERE = (
    (120.0, math.log(2.2e-8)),
    (200.0, math.log(2.5e-10)),
    (300.0, math.log(1.9e-11)),
    (400.0, math.log(2.8e-12)),
    (500.0, math.log(5.2e-13)),
    (600.0, math.log(1.1e-13)),
    (800.0, math.log(1.1e-14)),
    (1000.0, math.log(3.0e-15)),
)

DRAG_COEFFICIENT = 2.2

# Below this height (m) an object comes down (reenters).
REENTRY_HEIGHT = 120_000.0

# A fragment.
FRAGMENT = PartDefinition(
    id="fragment",
    name="Debris fragment",
    kind="structure",
    mass=40.0,
    radius_top=0.25,
    radius_bottom=0.35,
    height=0.6,
    color=(0.55, 0.55, 0.56),
    metallic=0.8,
    roughness=0.5,
    crash_speed=1.0
)


def thermosphere(
    height: float
) -> float:
    """The Earth's upper air's density (kg/m^3) at a height (m)."""

    km = height / 1000.0

    points = _EARTH_THERMOSPHERE

    if km <= points[0][0]:
        return math.exp(points[0][1])

    for (k0, l0), (k1, l1) in zip(points, points[1:]):

        if km <= k1:
            return math.exp(l0 + (l1 - l0) * (km - k0) / (k1 - k0))

    # Beyond: on with the last scale height.
    (k0, l0), (k1, l1) = points[-2], points[-1]

    return math.exp(l1 + (l1 - l0) * (km - k1) / (k1 - k0))


def ballistic(
    vessel: Vessel
) -> float:
    """Drag coefficient x area / mass (m^2/kg): tumbling, its side's and its end's areas averaged."""

    height = max(vessel.height, 0.3)

    radius = max(vessel.radius, 0.2)

    area = 0.5 * (2.0 * radius * height) + 0.5 * math.pi * radius * radius

    return DRAG_COEFFICIENT * area / max(vessel.mass, 1.0)


def decay(
    position: np.ndarray,
    velocity: np.ndarray,
    mu: float,
    body_radius: float,
    b: float,
    dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    `dt` s of the upper air dragging an orbit down (the
    density where it is lowest, weighted toward the
    periapsis): the orbit scaled, its shape and phase kept.
    """

    shape = kepler.orbit_shape(position, velocity, mu)

    if shape.period is None or shape.semi_major_axis <= 0.0:
        return position, velocity

    a = shape.semi_major_axis

    # (Most of the drag is near the periapsis.)
    low = shape.periapsis + 0.15 * ((shape.apoapsis or shape.periapsis) - shape.periapsis)

    rho = thermosphere(low - body_radius)

    shrink = math.sqrt(mu * a) * rho * b * dt

    after = max(a - shrink, body_radius)

    k = after / a

    # A Kepler orbit scaled: positions by k, speeds by 1/sqrt(k).
    return position * k, velocity / math.sqrt(k)


def lifetime(
    height: float,
    b: float,
    mu: float = 3.986004418e14,
    body_radius: float = 6_371_000.0
) -> float:
    """About how long (s) a circular orbit at `height` lasts (drag only), for the notices."""

    total = 0.0

    h = height

    while h > REENTRY_HEIGHT:

        a = body_radius + h

        rate = math.sqrt(mu * a) * thermosphere(h) * b

        step = max(5_000.0, 0.02 * h)

        total += step / max(rate, 1e-30)

        h -= step

        if total > 1e12:
            break

    return total


def fragments(
    position: np.ndarray,
    velocity: np.ndarray,
    speed: float,
    mass: float,
    seed: int
) -> list[tuple[Vessel, np.ndarray, np.ndarray]]:
    """
    A collision's cloud: (vessel, position, velocity) for each
    fragment, flung every way at up to a fifth of the speed
    things met at (the larger the mass broken, the more).
    """

    rng = random.Random(seed)

    count = int(min(max(mass / 2_000.0, 4), 16))

    found = []

    for number in range(count):

        direction = np.array([rng.gauss(0.0, 1.0) for _ in range(3)])
        direction /= max(float(np.linalg.norm(direction)), 1e-9)

        kick = direction * speed * rng.uniform(0.02, 0.2)

        piece = Vessel(f"Fragment {number + 1}", [PartState(FRAGMENT, None)])

        piece._restack()

        found.append((piece, position + direction * rng.uniform(1.0, 6.0), velocity + kick))

    return found


def closest_approach(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    within: float
) -> tuple[float, float] | None:
    """(s from now, m) of the closest approach in a straight line, if it comes within `within` s; else None."""

    speed2 = float(relative_velocity @ relative_velocity)

    if speed2 <= 1e-12:
        return None

    t = -float(relative_position @ relative_velocity) / speed2

    if t < 0.0 or t > within:
        return None

    return t, float(np.linalg.norm(relative_position + relative_velocity * t))


def swept_hit(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    dt: float,
    reach: float
) -> bool:
    """Whether two things passed within `reach` m over the last `dt` s (now `relative_position` apart): fast hits."""

    speed2 = float(relative_velocity @ relative_velocity)

    if speed2 <= 1e-12:
        return float(np.linalg.norm(relative_position)) < reach

    # Going back over the step: the nearest point of
    # p - v t for t in [0, dt].
    t = min(max(float(relative_position @ relative_velocity) / speed2, 0.0), dt)

    return float(np.linalg.norm(relative_position - relative_velocity * t)) < reach
