import math

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np


# =========================================================
# Orbits and Spin
# =========================================================
#
# Where bodies are and which way they face at a moment in
# time (seconds since the J2000 epoch, 2000-01-01 12:00).
#
# Frames (all right-handed, engine axes: y up):
#
#   inertial   the ecliptic of J2000, the star at the
#              origin: x toward the vernal equinox, y the
#              ecliptic's north pole.
#   body       a body's own frame, turning with it: y its
#              north pole, z its prime meridian (longitude
#              0; planet/solar.py measures longitude as
#              atan2(x, z)), x 90 degrees east.
#
# Orbits are Keplerian ellipses: fixed elements (from JPL's
# mean elements for the planets) give the position at any
# time through Kepler's equation; no perturbations, so the
# Moon's nodes do not regress and Mercury's perihelion stays
# put. Planets orbit the star; a moon's elements are
# relative to its planet's equator (or the ecliptic, for our
# Moon), its position relative to the planet.
#
# Spin follows the IAU convention: a north pole given in
# right ascension and declination, and the prime meridian's
# angle W = W0 + 360 deg * t / rotation period, measured
# east along the equator from where it crosses Earth's
# equator. Moons locked by tides keep their prime meridian
# toward their planet instead.

J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

DAY = 86_400.0

GRAVITATIONAL_CONSTANT = 6.674e-11

# Tilt of Earth's equator to the ecliptic at J2000.
OBLIQUITY = math.radians(23.4392911)


# ---------------------------------------------------------
# Time
# ---------------------------------------------------------

def seconds_since_j2000(
    moment: datetime
) -> float:

    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)

    return (moment - J2000).total_seconds()


def date_of(
    seconds: float
) -> datetime:
    """The UTC date and time of a simulation time."""

    return J2000 + timedelta(seconds=float(seconds))


# ---------------------------------------------------------
# Frames
# ---------------------------------------------------------

def ecliptic_to_engine(
    v
) -> np.ndarray:
    """Ecliptic coordinates (z north) -> engine axes (y north)."""

    v = np.asarray(v, dtype=np.float64)

    return np.array([v[0], v[2], -v[1]])


def equatorial_to_engine(
    v
) -> np.ndarray:
    """Earth-equatorial (ICRF) coordinates -> engine axes."""

    x, y, z = np.asarray(v, dtype=np.float64)

    c, s = math.cos(OBLIQUITY), math.sin(OBLIQUITY)

    return ecliptic_to_engine((x, y * c + z * s, -y * s + z * c))


def pole_direction(
    right_ascension_deg: float,
    declination_deg: float
) -> np.ndarray:
    """A north pole given in right ascension and declination, in engine axes."""

    ra = math.radians(right_ascension_deg)
    dec = math.radians(declination_deg)

    return equatorial_to_engine((math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)))


def equator_node(
    right_ascension_deg: float
) -> np.ndarray:
    """
    Where a body's equator crosses Earth's, ascending (the
    IAU origin of W): 90 degrees of right ascension past
    the pole's.
    """

    ra = math.radians(right_ascension_deg)

    return equatorial_to_engine((-math.sin(ra), math.cos(ra), 0.0))


def body_frame(
    pole: np.ndarray,
    prime_meridian: np.ndarray
) -> np.ndarray:
    """
    Rotation (3x3, columns = body axes in inertial
    coordinates) of a body with this pole and prime
    meridian direction (made perpendicular to the pole).
    """

    y = pole / np.linalg.norm(pole)

    z = prime_meridian - np.dot(prime_meridian, y) * y

    length = float(np.linalg.norm(z))

    if length < 1e-12:

        # Prime meridian along the pole: any perpendicular.
        z = np.cross(y, [1.0, 0.0, 0.0] if abs(y[0]) < 0.9 else [0.0, 0.0, 1.0])
        length = float(np.linalg.norm(z))

    z = z / length

    x = np.cross(y, z)

    return np.stack((x, y, z), axis=1)


# ---------------------------------------------------------
# Kepler
# ---------------------------------------------------------

def solve_kepler(
    mean_anomaly: float,
    eccentricity: float
) -> float:
    """Eccentric anomaly E with E - e sin E = M (radians)."""

    m = math.remainder(mean_anomaly, 2.0 * math.pi)

    e = min(max(eccentricity, 0.0), 0.99)

    E = m if e < 0.8 else math.pi * math.copysign(1.0, m)

    for _ in range(30):

        step = (E - e * math.sin(E) - m) / (1.0 - e * math.cos(E))

        E -= step

        if abs(step) < 1e-13:
            break

    return E


@dataclass(frozen=True, slots=True)
class OrbitElements:

    semi_major_axis: float          # m
    eccentricity: float
    inclination: float              # rad
    ascending_node: float           # rad
    periapsis: float                # rad (argument of periapsis)
    mean_anomaly: float             # rad at J2000
    period: float                   # s (sidereal)


def orbit_offset(
    elements: OrbitElements,
    seconds: float
) -> np.ndarray:
    """
    Position (m) relative to the focus at a time, in the
    elements' reference frame (x toward the node origin, z
    north: ecliptic-style coordinates).
    """

    e = elements.eccentricity
    a = elements.semi_major_axis

    mean = elements.mean_anomaly + 2.0 * math.pi * seconds / max(elements.period, 1.0)

    E = solve_kepler(mean, e)

    # In the orbit's plane, x toward periapsis.
    px = a * (math.cos(E) - e)
    py = a * math.sqrt(max(1.0 - e * e, 0.0)) * math.sin(E)

    cos_node, sin_node = math.cos(elements.ascending_node), math.sin(elements.ascending_node)
    cos_peri, sin_peri = math.cos(elements.periapsis), math.sin(elements.periapsis)
    cos_i, sin_i = math.cos(elements.inclination), math.sin(elements.inclination)

    return np.array([
        px * (cos_node * cos_peri - sin_node * sin_peri * cos_i)
        - py * (cos_node * sin_peri + sin_node * cos_peri * cos_i),
        px * (sin_node * cos_peri + cos_node * sin_peri * cos_i)
        - py * (sin_node * sin_peri - cos_node * cos_peri * cos_i),
        px * sin_peri * sin_i + py * cos_peri * sin_i,
    ])


# ---------------------------------------------------------
# The Moon
# ---------------------------------------------------------
#
# The Sun tugs the Moon off a Kepler ellipse by over a
# degree (evection, variation, the annual equation): enough
# to miss every eclipse. Its main periodic terms (Meeus,
# Astronomical Algorithms ch. 47; good to ~0.02 deg) put
# the Moon where it really is, so eclipses fall on their
# real dates.

# (D, M, M', F multipliers, amplitude): longitude and
# latitude (deg), distance (km; cosines).
_MOON_LONGITUDE = (
    (0, 0, 1, 0, 6.288774), (2, 0, -1, 0, 1.274027), (2, 0, 0, 0, 0.658314),
    (0, 0, 2, 0, 0.213618), (0, 1, 0, 0, -0.185116), (0, 0, 0, 2, -0.114332),
    (2, 0, -2, 0, 0.058793), (2, -1, -1, 0, 0.057066), (2, 0, 1, 0, 0.053322),
    (2, -1, 0, 0, 0.045758), (0, 1, -1, 0, -0.040923), (1, 0, 0, 0, -0.034720),
    (0, 1, 1, 0, -0.030383),
)

_MOON_LATITUDE = (
    (0, 0, 0, 1, 5.128122), (0, 0, 1, 1, 0.280602), (0, 0, 1, -1, 0.277693),
    (2, 0, 0, -1, 0.173237), (2, 0, -1, 1, 0.055413), (2, 0, -1, -1, 0.046271),
    (2, 0, 0, 1, 0.032573),
)

_MOON_DISTANCE = (
    (0, 0, 1, 0, -20905.355), (2, 0, -1, 0, -3699.111), (2, 0, 0, 0, -2955.968),
    (0, 0, 2, 0, -569.925), (0, 1, 0, 0, 48.888), (0, 0, 0, 2, -3.149),
    (2, 0, -2, 0, 246.158), (2, -1, -1, 0, -152.138), (2, 0, 1, 0, -170.733),
    (2, -1, 0, 0, -204.586), (0, 1, -1, 0, -129.620), (1, 0, 0, 0, 108.743),
    (0, 1, 1, 0, 104.755),
)


def moon_offset(
    seconds: float
) -> np.ndarray:
    """The Moon's position relative to Earth (m), inertial."""

    T = seconds / (36_525.0 * DAY)

    longitude = 218.3164477 + 481267.88123421 * T

    arguments = np.radians((
        297.8501921 + 445267.1114034 * T,           # D: elongation
        357.5291092 + 35999.0502909 * T,            # M: the Sun's anomaly
        134.9633964 + 477198.8675055 * T,           # M': the Moon's
        93.2720950 + 483202.0175233 * T,            # F: from the node
    ))

    def series(terms, function):

        return sum(
            amplitude * function(d * arguments[0] + m * arguments[1] + mp * arguments[2] + f * arguments[3])
            for d, m, mp, f, amplitude in terms
        )

    # Of the equinox of date -> J2000 (precession).
    lam = math.radians(longitude + series(_MOON_LONGITUDE, math.sin) - 1.3969713 * T)
    beta = math.radians(series(_MOON_LATITUDE, math.sin))

    distance = (385_000.56 + series(_MOON_DISTANCE, math.cos)) * 1000.0

    return ecliptic_to_engine((
        distance * math.cos(beta) * math.cos(lam),
        distance * math.cos(beta) * math.sin(lam),
        distance * math.sin(beta),
    ))


def period_from_masses(
    semi_major_axis: float,
    central_mass: float
) -> float:
    """Kepler's third law: seconds per orbit."""

    return 2.0 * math.pi * math.sqrt(semi_major_axis ** 3 / (GRAVITATIONAL_CONSTANT * max(central_mass, 1.0)))


def prime_meridian_angle(
    w0_deg: float,
    rotation_period: float,
    seconds: float
) -> float:
    """W (radians) at a time; a negative period turns backward (retrograde)."""

    turns = seconds / rotation_period if rotation_period != 0.0 else 0.0

    return math.radians(w0_deg) + 2.0 * math.pi * (turns - math.floor(turns))
