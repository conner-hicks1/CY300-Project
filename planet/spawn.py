import math

from dataclasses import dataclass

import numpy as np

from planet.terrain import Terrain


# =========================================================
# Spawn Point
# =========================================================
#
# Picks a scenic place to start: lowland at moderate
# latitude with high mountains in view. Used by the demo
# scene, which turns the planet so this spot is on top.

@dataclass(slots=True)
class SpawnPoint:

    # Planet-space unit direction of the spawn point.
    direction: np.ndarray

    # Unit tangent pointing toward the mountains.
    view_direction: np.ndarray

    elevation: float


def fibonacci_sphere(
    count: int
) -> np.ndarray:
    """`count` nearly evenly spaced unit vectors."""

    i = np.arange(count) + 0.5

    y = 1.0 - 2.0 * i / count

    ring = np.sqrt(1.0 - y * y)

    angle = i * math.pi * (3.0 - math.sqrt(5.0))

    return np.stack((ring * np.cos(angle), y, ring * np.sin(angle)), axis=1)


def find_spawn(
    terrain: Terrain,
    samples: int = 20_000
) -> SpawnPoint:
    """
    1. Coarse pass over the whole planet: the highest peak
       (moderate latitude) that has lowland nearby.
    2. Fine pass around it to find the actual summit.
    3. Walk from the summit toward that lowland and stand
       where the summit looks steepest; face it.
    """

    settings = terrain.settings

    radius = settings.radius

    directions = fibonacci_sphere(samples)

    coarse_spacing = math.sqrt(4.0 * math.pi / samples) * radius

    elevation = terrain.elevation(directions, coarse_spacing)

    latitude = np.abs(directions[:, 1])

    lowland = np.flatnonzero((elevation > 50.0) & (elevation < 1200.0) & (latitude < 0.6))
    peaks = np.flatnonzero((elevation > 0.4 * settings.mountain_height) & (latitude < 0.6))

    if len(lowland) == 0 or len(peaks) == 0:

        best = int(np.argmax(elevation))

        return SpawnPoint(
            direction=directions[best],
            view_direction=_tangent(directions[best], np.array([0.0, 1.0, 0.0])),
            elevation=float(elevation[best])
        )

    # Peaks with lowland within ~2.5 coarse samples.
    cosines = directions[peaks] @ directions[lowland].T

    reach = math.cos(2.5 * coarse_spacing / radius)

    has_lowland = cosines.max(axis=1) > reach

    if has_lowland.any():
        peaks_near = peaks[has_lowland]
        rows = np.flatnonzero(has_lowland)
    else:
        peaks_near = peaks
        rows = np.arange(len(peaks))

    choice = int(np.argmax(elevation[peaks_near]))

    peak = directions[peaks_near[choice]]

    toward = directions[lowland[int(np.argmax(cosines[rows[choice]]))]]

    # -----------------------------------------------------
    # Summit: fine grid around the coarse peak
    # -----------------------------------------------------

    east = _tangent(peak, toward)
    north = np.cross(peak, east)

    span = 1.5 * coarse_spacing / radius

    offsets = np.linspace(-span, span, 61)

    grid_e, grid_n = np.meshgrid(offsets, offsets)

    local = (
        peak
        + grid_e.reshape(-1, 1) * east
        + grid_n.reshape(-1, 1) * north
    )

    local /= np.linalg.norm(local, axis=1, keepdims=True)

    local_elevation = terrain.elevation(local, 2.0 * span * radius / 60.0)

    summit_index = int(np.argmax(local_elevation))

    summit = local[summit_index]

    summit_elevation = float(local_elevation[summit_index])

    # -----------------------------------------------------
    # Walk down to the lowland
    # -----------------------------------------------------

    away = _tangent(summit, toward)

    distances = np.arange(1, 801) * 500.0          # 0.5 .. 400 km

    angles = distances / radius

    path = (
        np.cos(angles)[:, None] * summit
        + np.sin(angles)[:, None] * away
    )

    path_elevation = terrain.elevation(path, 500.0)

    # Stand where the summit rises most steeply above the
    # horizon (at least 5 km out, on dry land).
    rise = np.arctan2(summit_elevation - path_elevation, distances)

    rise[(distances < 5_000.0) | (path_elevation < 20.0)] = -np.inf

    # Prefer green valleys below the snow.
    below_snow = path_elevation < 2_000.0

    if np.isfinite(rise[below_snow]).any():
        rise[~below_snow] = -np.inf

    index = int(np.argmax(rise))

    spawn = path[index]

    return SpawnPoint(
        direction=spawn,
        view_direction=_tangent(spawn, summit),
        elevation=float(path_elevation[index])
    )


def _tangent(
    up: np.ndarray,
    toward: np.ndarray
) -> np.ndarray:
    """Horizontal direction at `up` facing `toward`."""

    tangent = toward - np.dot(toward, up) * up

    length = float(np.linalg.norm(tangent))

    if length < 1e-9:

        helper = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 0.0, 1.0])

        tangent = helper - np.dot(helper, up) * up
        length = float(np.linalg.norm(tangent))

    return tangent / length
