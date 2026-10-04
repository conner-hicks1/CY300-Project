import math

import numpy as np


# =========================================================
# Solar Time
# =========================================================
#
# In planet space (y = the planet's rotation axis), the
# sun's direction has a latitude, its declination (the
# season: +23.4 deg = northern summer solstice on Earth),
# and a longitude. Local solar time at a point is the
# point's longitude relative to the sun's (east positive):
#
#     hour = 12 + (point longitude - sun longitude) / 15 deg
#
# so 12:00 is when the sun crosses the point's meridian
# (highest in the sky), 0:00 is midnight, and in the
# morning the sun is to the east. As a planet turns (x is
# east of z), local time runs forward.
#
# Longitude is measured as atan2(x, z) in planet space.

MAX_DECLINATION = 23.44     # Earth's axial tilt, degrees


def _longitude(
    direction
) -> float:

    return math.atan2(float(direction[0]), float(direction[2]))


def solar_time(
    point_direction,
    sun_direction
) -> tuple[float, float, float]:
    """
    point_direction: unit planet-space direction of the
        observer (from the planet center).
    sun_direction: unit planet-space direction toward the sun.

    Returns (hour in [0, 24), declination (rad), sun
    elevation above the observer's horizon (rad)).
    """

    point_direction = np.asarray(point_direction, dtype=np.float64)
    sun_direction = np.asarray(sun_direction, dtype=np.float64)

    declination = math.asin(float(np.clip(sun_direction[1], -1.0, 1.0)))

    hour = (
        12.0
        + (_longitude(point_direction) - _longitude(sun_direction)) * 12.0 / math.pi
    ) % 24.0

    elevation = math.asin(float(np.clip(np.dot(point_direction, sun_direction), -1.0, 1.0)))

    return hour, declination, elevation


def sun_direction(
    point_direction,
    hour: float,
    declination: float
) -> np.ndarray:
    """Planet-space unit direction toward the sun (inverse of solar_time)."""

    longitude = _longitude(point_direction) - (hour - 12.0) * math.pi / 12.0

    return np.array([
        math.cos(declination) * math.sin(longitude),
        math.sin(declination),
        math.cos(declination) * math.cos(longitude),
    ])
