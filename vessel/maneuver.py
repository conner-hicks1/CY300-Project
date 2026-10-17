import math

from dataclasses import dataclass, field

import numpy as np

from vessel import kepler


# =========================================================
# Maneuver Nodes
# =========================================================
#
# A planned burn: when (on the clock), and how much along the
# orbit's own directions there, as in most spaceflight games:
#
#   prograde   along the motion
#   normal     along the orbit's pole (r x v)
#   radial     out from the body, square to the motion
#
# Set or edited, the node fixes its velocity change as an
# inertial vector (from the orbit at the node); burning, the
# thrust's own velocity change (VesselState.thrust_dv) is
# taken off it, so what is left is what the SAS maneuver mode
# points along and the HUD counts down, whatever gravity does
# meanwhile.
#
# And the burns worth planning: circularizing, capturing at a
# planet, coming down (deorbiting into the air or to the
# ground), and when to start braking to land.


@dataclass(slots=True)
class ManeuverNode:

    time: float                     # on the orbit clock (s)
    prograde: float = 0.0           # m/s
    normal: float = 0.0
    radial: float = 0.0

    # Inertial velocity change (m/s), fixed when last set,
    # and the thrust's velocity change counter then.
    vector: np.ndarray = field(default_factory=lambda: np.zeros(3))
    baseline: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def components(
        self
    ) -> tuple[float, float, float]:

        return self.prograde, self.normal, self.radial

    @property
    def size(
        self
    ) -> float:
        """m/s planned."""

        return math.sqrt(self.prograde ** 2 + self.normal ** 2 + self.radial ** 2)

    def remaining(
        self,
        thrust_dv: np.ndarray
    ) -> np.ndarray:
        """The velocity change still to make (inertial, m/s)."""

        return self.vector - (np.asarray(thrust_dv, dtype=np.float64) - self.baseline)


def node_frame(
    position,
    velocity
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(prograde, normal, radial) unit vectors of an orbit at a point."""

    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)

    prograde = velocity / max(float(np.linalg.norm(velocity)), 1e-9)

    normal = np.cross(position, velocity)

    length = float(np.linalg.norm(normal))

    if length < 1e-9:

        # Straight up or down: any square direction.
        hint = np.array([0.0, 0.0, 1.0]) if abs(prograde[2]) < 0.9 else np.array([1.0, 0.0, 0.0])

        normal = np.cross(prograde, hint)
        length = float(np.linalg.norm(normal))

    normal = normal / length

    radial = np.cross(prograde, normal)

    return prograde, normal, radial


def to_inertial(
    position,
    velocity,
    prograde: float,
    normal: float,
    radial: float
) -> np.ndarray:

    p, n, r = node_frame(position, velocity)

    return prograde * p + normal * n + radial * r


def from_inertial(
    position,
    velocity,
    vector
) -> tuple[float, float, float]:

    p, n, r = node_frame(position, velocity)

    vector = np.asarray(vector, dtype=np.float64)

    return float(vector @ p), float(vector @ n), float(vector @ r)


def burn_time(
    vessel,
    delta_v: float
) -> float:
    """
    Seconds at full throttle to change speed by `delta_v`
    (vacuum): what burns now, then stage by stage (staging
    as the vessel would). inf when it cannot.
    """

    if delta_v <= 0.0:
        return 0.0

    copy = vessel._copy()

    left = delta_v
    total = 0.0

    for _ in range(len(copy.parts) + 16):

        if not copy.burning_engines():

            if copy.stage() is None:
                return math.inf

            continue

        thrust, flow = copy.thrust(0.0)

        if flow <= 0.0 or thrust <= 0.0:
            return math.inf

        start = copy.mass

        burned = copy._burn_to_burnout(0.0)

        if burned is None:
            continue

        segment, _, seconds = burned

        if segment >= left:

            exhaust = thrust / flow

            return total + start / flow * (1.0 - math.exp(-left / exhaust))

        total += seconds
        left -= segment

    return math.inf


# ---------------------------------------------------------
# Burns worth planning
# ---------------------------------------------------------

def time_to_apsis(
    position,
    velocity,
    mu: float,
    which: str
) -> float | None:
    """
    s until the orbit next reaches its "periapsis" or
    "apoapsis" (None: an open orbit has no apoapsis, or it
    is already past its periapsis going out).
    """

    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)

    r = float(np.linalg.norm(position))

    energy = 0.5 * float(velocity @ velocity) - mu / r

    h = np.cross(position, velocity)

    e_vector = np.cross(velocity, h) / mu - position / r

    e = float(np.linalg.norm(e_vector))

    if e < 1e-6:
        # Circular: everywhere is both; now.
        return 0.0

    # True anomaly now.
    cos_nu = float(np.clip(e_vector @ position / (e * r), -1.0, 1.0))

    nu = math.acos(cos_nu)

    if float(position @ velocity) < 0.0:
        nu = 2.0 * math.pi - nu

    if energy < 0.0 and e < 1.0:

        a = -mu / (2.0 * energy)

        n = math.sqrt(mu / a ** 3)

        def mean(true_anomaly):

            E = 2.0 * math.atan2(
                math.sqrt(1.0 - e) * math.sin(true_anomaly / 2.0),
                math.sqrt(1.0 + e) * math.cos(true_anomaly / 2.0)
            )

            return (E - e * math.sin(E)) % (2.0 * math.pi)

        goal = 0.0 if which == "periapsis" else math.pi

        return ((mean(goal) - mean(nu)) % (2.0 * math.pi)) / n

    if which == "apoapsis":
        return None

    # Open: only coming in (nu in (pi, 2 pi)).
    if nu <= math.pi:
        return None

    a = mu / (2.0 * energy)

    n = math.sqrt(mu / a ** 3)

    nu_signed = nu - 2.0 * math.pi

    F = 2.0 * math.atanh(math.sqrt((e - 1.0) / (e + 1.0)) * math.tan(nu_signed / 2.0))

    M = e * math.sinh(F) - F

    return -M / n


def circularize(
    position,
    velocity,
    mu: float
) -> np.ndarray:
    """The inertial velocity change making the orbit circular through this point."""

    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)

    r = float(np.linalg.norm(position))

    up = position / r

    horizontal = velocity - (velocity @ up) * up

    length = float(np.linalg.norm(horizontal))

    if length < 1e-9:
        _, normal, _ = node_frame(position, velocity)
        horizontal = np.cross(normal, up)
        length = float(np.linalg.norm(horizontal))

    return horizontal / length * math.sqrt(mu / r) - velocity


def capture(
    position,
    velocity,
    mu: float,
    apoapsis: float
) -> np.ndarray:
    """
    At a periapsis (or any point taken as one): the
    retrograde change leaving an orbit with its farthest
    point `apoapsis` m from the center (no farther than the
    sphere of influence, for a capture).
    """

    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)

    r = float(np.linalg.norm(position))

    apoapsis = max(apoapsis, r)

    speed = math.sqrt(2.0 * mu * apoapsis / (r * (r + apoapsis)))

    now = float(np.linalg.norm(velocity))

    return velocity / max(now, 1e-9) * (speed - now)


def lower_periapsis(
    position,
    velocity,
    mu: float,
    periapsis: float
) -> np.ndarray | None:
    """
    The retrograde change bringing the orbit's lowest point
    to `periapsis` m from the center (to come down into the
    air, or to the ground). None when it is low enough.
    """

    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)

    speed = float(np.linalg.norm(velocity))

    direction = velocity / max(speed, 1e-9)

    def lowest(change: float) -> float:

        shape = kepler.orbit_shape(position, velocity - direction * change, mu)

        return shape.periapsis

    if lowest(0.0) <= periapsis:
        return None

    low, high = 0.0, speed

    if lowest(high) > periapsis:
        return -direction * high

    for _ in range(60):

        middle = 0.5 * (low + high)

        if lowest(middle) > periapsis:
            low = middle
        else:
            high = middle

    return -direction * high


@dataclass(frozen=True, slots=True)
class Braking:
    """Coming down on engines: when to start (s), how high then (m), speed at the ground without braking (m/s), when it gets there (s)."""

    start_in: float
    height: float
    impact_speed: float
    impact_in: float


def time_to_ground(
    position,
    velocity,
    mu: float,
    ground_radius: float,
    samples: int = 64
) -> float | None:
    """
    s until the conic comes down to `ground_radius` from the
    center (0 if below it already); None if it never does
    (it clears the ground, or is climbing away for good).
    """

    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)

    r = float(np.linalg.norm(position))

    if r <= ground_radius:
        return 0.0

    horizon = time_to_apsis(position, velocity, mu, "periapsis")

    if horizon is None or horizon <= 0.0 or not math.isfinite(horizon):

        # (Straight down, or nearly: half a circular orbit
        # at this height is surely long enough.)
        horizon = math.pi * math.sqrt(r ** 3 / mu)

    times = np.linspace(0.0, horizon, samples + 1)

    points, _ = kepler.propagate_many(position, velocity, mu, times)

    distances = np.linalg.norm(points, axis=1)

    below = np.nonzero(np.isfinite(distances) & (distances < ground_radius))[0]

    if not len(below):
        return None

    high = float(times[below[0]])
    low = float(times[below[0] - 1]) if below[0] > 0 else 0.0

    for _ in range(40):

        middle = 0.5 * (low + high)

        there, _ = kepler.propagate(position, velocity, mu, middle)

        if float(np.linalg.norm(there)) < ground_radius:
            high = middle
        else:
            low = middle

    return high


def braking(
    time_to_impact: float | None,
    height: float,
    vertical_speed: float,
    surface_speed: float,
    gravity: float,
    acceleration: float
) -> Braking | None:
    """
    Coming down toward the ground (time_to_impact s on the
    path, height m above the ground now, speeds m/s over the
    ground, up positive): a full-throttle burn (acceleration
    m/s^2 of thrust) against the motion that stops it just
    at the ground must start when the path left is what the
    burn takes: it stops in speed / a, covering speed^2 / 2a,
    the time the free path would cover that in being
    speed / 2a. Gravity along the path takes off from a.
    None when not coming down.
    """

    if time_to_impact is None or height <= 0.0:
        return None

    speed = max(surface_speed, 1e-6)

    impact_speed = math.sqrt(max(surface_speed * surface_speed + 2.0 * gravity * height, 0.0))

    # Gravity's share along the path (falling: all of it).
    net = acceleration - gravity * max(-vertical_speed, 0.0) / speed

    if net <= 0.0:
        return Braking(math.inf, 0.0, impact_speed, time_to_impact)

    start_in = max(time_to_impact - speed / (2.0 * net), 0.0)

    start_height = max(height + vertical_speed * start_in, 0.0)

    return Braking(start_in, start_height, impact_speed, time_to_impact)
