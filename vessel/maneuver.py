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
    """Coming down on engines: when to start (s), how high then (m), speed at the ground without braking (m/s)."""

    start_in: float
    height: float
    impact_speed: float
    impact_in: float


def braking(
    height: float,
    vertical_speed: float,
    surface_speed: float,
    gravity: float,
    acceleration: float
) -> Braking | None:
    """
    Falling toward the ground (height m above it, speeds
    m/s, up positive): a full-throttle burn (acceleration
    m/s^2 of thrust) against the motion must start by the
    height where it just stops at the ground (the "suicide
    burn"). None when not falling or the engines cannot hold
    the weight.
    """

    if vertical_speed >= 0.0 or height <= 0.0:
        return None

    net = acceleration - gravity

    # Falling freely to the ground.
    fall = -vertical_speed

    impact_in = (-fall + math.sqrt(fall * fall + 2.0 * gravity * height)) / gravity if gravity > 0.0 else height / fall

    impact_speed = math.sqrt(max(surface_speed * surface_speed + 2.0 * gravity * height, 0.0))

    if net <= 0.0:
        return Braking(math.inf, 0.0, impact_speed, impact_in)

    # Stopping all the motion over the height: v^2 = 2 a h,
    # the speed then as it falls from here.
    #   v(h)^2 = s^2 + 2 g (H - h);  h_b = v(h_b)^2 / (2 net)
    #   => h_b = (s^2 + 2 g H) / (2 net + 2 g) = (s^2 + 2 g H) / (2 a)
    start_height = (surface_speed * surface_speed + 2.0 * gravity * height) / (2.0 * acceleration)

    start_height = min(start_height, height)

    # Time to fall from here to there.
    drop = height - start_height

    start_in = (-fall + math.sqrt(fall * fall + 2.0 * gravity * drop)) / gravity if gravity > 0.0 else drop / fall

    return Braking(start_in, start_height, impact_speed, impact_in)
