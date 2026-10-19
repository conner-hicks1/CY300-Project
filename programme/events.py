import math

from datetime import datetime, timezone

import numpy as np

from planet.orbits import ecliptic_to_engine, seconds_since_j2000
from vessel import kepler


# =========================================================
# Real Events
# =========================================================
#
# Things on the real calendar a programme can be there for
# (missions in assets/programme/missions.json):
#
#   eclipses   12 August 2026 (total over Greenland, Iceland
#              and Spain) and 2 August 2027 (the century's
#              longest, over Spain, north Africa and Egypt):
#              be where the Moon covers the Sun. 31 December
#              2028: the Moon in the Earth's shadow, a lander
#              on it in the dark for hours. The bodies' real
#              positions decide where the shadows fall
#              (sun_coverage below)
#   Apophis    the 340 m asteroid passing 38,000 km from the
#              Earth's center (inside the geostationary ring)
#              at 21:46 UTC on 13 April 2029, at 7.4 km/s:
#              modeled as its hyperbola about the Earth (the
#              closest point over the mid-Atlantic, its path
#              near the ecliptic), close enough to the real
#              pass to plan for

MU_EARTH = 3.986004418e14

APOPHIS_TIME = seconds_since_j2000(datetime(2029, 4, 13, 21, 46, 0, tzinfo=timezone.utc))
APOPHIS_PERIGEE = 38_017_000.0      # m from the Earth's center
APOPHIS_EXCESS = 5_840.0            # m/s far from the Earth

# The closest point's direction (ecliptic longitude and
# latitude, deg) and the way it is going there.
APOPHIS_DIRECTION = (128.0, 8.0)
APOPHIS_HEADING = 218.0

# Shown on the map this long either side of the pass (s).
APOPHIS_SHOWN = 40.0 * 86_400.0


def _ecliptic(
    longitude: float,
    latitude: float
) -> np.ndarray:

    lon, lat = math.radians(longitude), math.radians(latitude)

    return np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])


def apophis_perigee_state() -> tuple[np.ndarray, np.ndarray]:
    """Apophis at its closest: (position, velocity) about the Earth, engine inertial axes."""

    out = _ecliptic(*APOPHIS_DIRECTION)

    heading = _ecliptic(APOPHIS_HEADING, 0.0)

    # Square to the closest point's direction.
    heading = heading - out * float(heading @ out)
    heading /= float(np.linalg.norm(heading))

    speed = math.sqrt(APOPHIS_EXCESS ** 2 + 2.0 * MU_EARTH / APOPHIS_PERIGEE)

    return ecliptic_to_engine(out * APOPHIS_PERIGEE), ecliptic_to_engine(heading * speed)


def apophis_position(
    seconds: float
) -> np.ndarray:
    """Apophis about the Earth (m, engine inertial axes) at a time."""

    position, velocity = apophis_perigee_state()

    there, _ = kepler.propagate(position, velocity, MU_EARTH, seconds - APOPHIS_TIME)

    return there


def apophis_near(
    seconds: float
) -> bool:
    """Close enough to the pass to show it."""

    return abs(seconds - APOPHIS_TIME) < APOPHIS_SHOWN


def apophis_path(
    seconds: float,
    span: float = 3.0 * 86_400.0,
    points: int = 97
) -> np.ndarray:
    """Its path about the Earth round a time ((n, 3) m)."""

    position, velocity = apophis_perigee_state()

    times = np.linspace(seconds - span, seconds + span, points) - APOPHIS_TIME

    path, _ = kepler.propagate_many(position, velocity, MU_EARTH, times)

    return path[np.isfinite(path).all(axis=1)]


# ---------------------------------------------------------
# Eclipses
# ---------------------------------------------------------

def sun_coverage(
    where: np.ndarray,
    sun: np.ndarray,
    sun_radius: float,
    center: np.ndarray,
    radius: float
) -> float:
    """0..1 of the Sun's disk a body (center, radius) hides, seen from `where`."""

    to_sun = sun - where
    to_body = center - where

    sun_distance = float(np.linalg.norm(to_sun))
    body_distance = float(np.linalg.norm(to_body))

    if body_distance <= radius:
        return 1.0

    # (Beyond the Sun: hides nothing.)
    if body_distance > sun_distance:
        return 0.0

    a = math.asin(min(sun_radius / sun_distance, 1.0))
    b = math.asin(min(radius / body_distance, 1.0))

    cosine = float(to_sun @ to_body) / (sun_distance * body_distance)

    d = math.acos(min(max(cosine, -1.0), 1.0))

    if d >= a + b:
        return 0.0

    if d <= abs(b - a):
        return 1.0 if b >= a else (b * b) / (a * a)

    # The two disks' lens of overlap (small angles: flat).
    part_a = a * a * math.acos((d * d + a * a - b * b) / (2.0 * d * a))
    part_b = b * b * math.acos((d * d + b * b - a * a) / (2.0 * d * b))
    middle = 0.5 * math.sqrt(max((-d + a + b) * (d + a - b) * (d - a + b) * (d + a + b), 0.0))

    return min((part_a + part_b - middle) / (math.pi * a * a), 1.0)
