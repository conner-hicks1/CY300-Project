from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


# =========================================================
# Mouse Picking
# =========================================================
#
# Click selection casts a ray from the camera through the
# mouse position and tests it against each candidate's
# local-space bounding box. Transforming the ray into the
# object's local space (rather than the box into world
# space) keeps the test exact for rotated and
# non-uniformly scaled objects.
#
# Pure numpy; no GL.


@dataclass(slots=True)
class Ray:

    origin: np.ndarray
    direction: np.ndarray


@dataclass(slots=True)
class PickCandidate:

    key: object

    # Object -> world.
    world_matrix: np.ndarray

    # Local-space box.
    bounds_min: np.ndarray
    bounds_max: np.ndarray


def screen_ray(
    mouse_x: float,
    mouse_y: float,
    width: float,
    height: float,
    view: np.ndarray,
    projection: np.ndarray
) -> Ray:
    """
    World-space ray through a window position (pixels,
    origin top-left, as reported by ImGui/GLFW).
    """

    ndc_x = 2.0 * mouse_x / width - 1.0
    ndc_y = 1.0 - 2.0 * mouse_y / height

    inverse = np.linalg.inv(
        np.asarray(projection, dtype=np.float64)
        @ np.asarray(view, dtype=np.float64)
    )

    near = inverse @ np.array([ndc_x, ndc_y, -1.0, 1.0])
    far = inverse @ np.array([ndc_x, ndc_y, 1.0, 1.0])

    near = near[:3] / near[3]
    far = far[:3] / far[3]

    direction = far - near

    return Ray(
        origin=near,
        direction=direction / np.linalg.norm(direction)
    )


def ray_box_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray
) -> float | None:
    """
    Slab test. Returns the ray parameter t >= 0 of the
    nearest hit (t = 0 when the origin is inside), or None.
    Works for flat boxes (e.g. a plane with zero height).
    """

    t_near = -np.inf
    t_far = np.inf

    for axis in range(3):

        d = direction[axis]
        o = origin[axis]

        lo = bounds_min[axis]
        hi = bounds_max[axis]

        if abs(d) < 1e-12:

            # Parallel to this slab: must already be inside.

            if o < lo or o > hi:
                return None

            continue

        t1 = (lo - o) / d
        t2 = (hi - o) / d

        if t1 > t2:
            t1, t2 = t2, t1

        t_near = max(t_near, t1)
        t_far = min(t_far, t2)

        if t_near > t_far:
            return None

    if t_far < 0.0:
        return None

    return float(
        max(t_near, 0.0)
    )


def pick(
    ray: Ray,
    candidates: Iterable[PickCandidate]
) -> object | None:
    """
    Key of the nearest candidate the ray hits, or None.
    """

    best_key = None
    best_distance = np.inf

    origin = np.append(ray.origin, 1.0)
    direction = np.append(ray.direction, 0.0)

    for candidate in candidates:

        try:

            to_local = np.linalg.inv(
                np.asarray(candidate.world_matrix, dtype=np.float64)
            )

        except np.linalg.LinAlgError:

            # Zero-scaled objects cannot be clicked.
            continue

        local_origin = (to_local @ origin)[:3]
        local_direction = (to_local @ direction)[:3]

        # The local direction is not normalized, so t is
        # still measured in world-ray units and distances
        # are comparable across candidates.

        distance = ray_box_distance(
            local_origin,
            local_direction,
            np.asarray(candidate.bounds_min, dtype=np.float64),
            np.asarray(candidate.bounds_max, dtype=np.float64)
        )

        if distance is not None and distance < best_distance:

            best_distance = distance
            best_key = candidate.key

    return best_key
