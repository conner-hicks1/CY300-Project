import numpy as np


# =========================================================
# Surface Climate (analytic)
# =========================================================
#
# Annual mean temperature and rainfall over a body's surface,
# for the ground's shading (snow lines, deserts and forests:
# assets/shaders/include/terrain.glsl) and what grows where
# (planet/scatter.py). No simulation: the body's measured
# mean temperature, spread from equator to poles by how much
# its air evens heat out, and the lapse rate with height.
#
#   temperature   T = T_eq - D sin^2(latitude) - lapse * height,
#                 with T_eq set so the surface averages the
#                 body's mean (sin^2 averages 1/3 over a sphere)
#                 and D the equator-to-pole drop: ~45 C on
#                 Earth, ~60 C with thin or no air (Mars), ~10 C
#                 under thick air (Venus) or cold dense air
#                 (Titan)
#   rainfall      Earth's belts: the wet tropics, the dry
#                 subtropics near 25 deg, the wet mid-latitude
#                 storm tracks, the dry poles; patchy with a
#                 noise value (moisture, 0..1)

# Equator-to-pole drop (C) with no air to carry heat.
AIRLESS_DROP = 60.0


def equator_to_pole_drop(
    mean_temperature: float,
    pressure: float
) -> float:
    """C: how much colder the poles are than the equator, on average."""

    # Heat carried by air rises with its density (pressure
    # over temperature).
    density = max(pressure, 0.0) * (288.15 / max(mean_temperature + 273.15, 30.0)) ** 2

    return AIRLESS_DROP / (1.0 + 0.33 * density)


def surface_temperature(
    directions: np.ndarray,
    elevation: np.ndarray,
    mean_temperature: float = 15.0,
    pressure: float = 1.0,
    lapse_rate: float = 6.5
) -> np.ndarray:
    """Annual mean temperature (C) at unit directions and heights (m)."""

    directions = np.asarray(directions, dtype=np.float64)

    sin2 = np.clip(directions[:, 1], -1.0, 1.0) ** 2

    drop = equator_to_pole_drop(mean_temperature, pressure)

    equator = mean_temperature + drop / 3.0

    return equator - drop * sin2 - lapse_rate * 0.001 * np.maximum(np.asarray(elevation, dtype=np.float64), 0.0)


def surface_precipitation(
    directions: np.ndarray,
    moisture: np.ndarray
) -> np.ndarray:
    """Annual rainfall (mm) by latitude belt, patchy with `moisture` (0..1)."""

    latitude = np.degrees(np.abs(np.arcsin(np.clip(np.asarray(directions)[:, 1], -1.0, 1.0))))

    belts = (
        250.0
        + 2_200.0 * np.exp(-(latitude / 10.0) ** 2)                 # the tropics
        + 800.0 * np.exp(-((latitude - 50.0) / 12.0) ** 2)          # storm tracks
        - 200.0 * np.exp(-((latitude - 25.0) / 7.0) ** 2)           # subtropical deserts
        - 150.0 * np.clip((latitude - 65.0) / 25.0, 0.0, 1.0)       # polar deserts
    )

    return np.maximum(belts, 50.0) * (0.25 + 1.5 * np.asarray(moisture, dtype=np.float64) ** 1.5)


def wetness(
    precipitation,
    temperature
) -> np.ndarray:
    """Rain relative to evaporation (< 0.3 desert, > 1.2 forest)."""

    return np.asarray(precipitation) / (300.0 + 30.0 * np.maximum(temperature, 0.0))


# Surface winds by latitude (deg): east speed, poleward speed
# (trade winds, westerlies, polar easterlies).
_WIND_LATITUDES = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
_WIND_EAST = np.array([-1.0, -0.8, 0.0, 1.0, 0.0, -0.5, 0.0])
_WIND_POLEWARD = np.array([0.0, -0.4, 0.0, 0.3, 0.0, -0.3, 0.0])


def prevailing_wind(
    directions: np.ndarray
) -> np.ndarray:
    """(n, 3) surface wind (tangent, magnitude ~0-1) by latitude belt."""

    directions = np.asarray(directions, dtype=np.float64)

    sin_latitude = np.clip(directions[:, 1], -1.0, 1.0)

    latitude = np.degrees(np.arcsin(np.abs(sin_latitude)))

    east_speed = np.interp(latitude, _WIND_LATITUDES, _WIND_EAST)
    poleward_speed = np.interp(latitude, _WIND_LATITUDES, _WIND_POLEWARD)

    east, north = local_axes(directions)

    hemisphere = np.where(sin_latitude >= 0.0, 1.0, -1.0)

    return (
        east * east_speed[:, None]
        + north * (poleward_speed * hemisphere)[:, None]
    )


def local_axes(
    directions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Unit east and north tangents (poles: an arbitrary frame)."""

    axis = np.array([0.0, 1.0, 0.0])

    east = np.cross(axis, directions)

    length = np.linalg.norm(east, axis=1, keepdims=True)

    fallback = np.array([1.0, 0.0, 0.0])

    east = np.where(length > 1e-9, east / np.maximum(length, 1e-12), fallback)

    north = np.cross(directions, east)

    return east, north
