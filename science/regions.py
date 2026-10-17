from dataclasses import dataclass

import numpy as np


# =========================================================
# Regions
# =========================================================
#
# Where on a body an experiment was done, from the real
# maps: away from the named sites (science/sites.py), the
# ground is classed by its measured height against the
# body's own range (Mars's northern lowlands and southern
# highlands fall out of MOLA by themselves; Earth's ocean
# floor from ETOPO1) and by latitude (the polar regions).
# Bodies without an elevation map have only their poles
# and the rest.

# Poleward of this (deg): a polar region.
POLAR_LATITUDE = 75.0

# Shares of the ground below the lowlands' top and above the
# highlands' bottom.
LOW_SHARE = 0.3
HIGH_SHARE = 0.8


@dataclass(frozen=True, slots=True)
class Region:

    id: str
    name: str
    value: float = 1.0


NORTH_POLE = Region("north_polar", "Northern polar region", 1.5)
SOUTH_POLE = Region("south_polar", "Southern polar region", 1.5)
LOWLANDS = Region("lowlands", "Lowlands", 1.0)
PLAINS = Region("plains", "Plains", 1.0)
HIGHLANDS = Region("highlands", "Highlands", 1.2)
SEA = Region("sea", "Sea", 0.8)
SURFACE = Region("surface", "Surface", 1.0)

# Experiments that do not care where (high above the body).
GLOBAL = Region("global", "", 1.0)


def elevation_levels(
    heights: np.ndarray,
    has_sea: bool
) -> tuple[float, float]:
    """
    (lowlands' top, highlands' bottom) heights (m) from a
    body's measured heights: shares of the ground (of the
    land only, with a sea).
    """

    heights = np.asarray(heights, dtype=np.float64).ravel()

    if has_sea:

        land = heights[heights > 0.0]

        if len(land):
            heights = land

    low, high = np.quantile(heights, (LOW_SHARE, HIGH_SHARE))

    return float(low), float(high)


def region_of(
    latitude: float,
    elevation: float | None,
    levels: tuple[float, float] | None,
    has_sea: bool = False
) -> Region:
    """
    latitude: deg; elevation: m above the map's datum (or
    None: no map); levels: from elevation_levels.
    """

    if latitude >= POLAR_LATITUDE:
        return NORTH_POLE

    if latitude <= -POLAR_LATITUDE:
        return SOUTH_POLE

    if elevation is None or levels is None:
        return SURFACE

    if has_sea and elevation < 0.0:
        return SEA

    low, high = levels

    if elevation <= low:
        return LOWLANDS

    if elevation >= high:
        return HIGHLANDS

    return PLAINS
