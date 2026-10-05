from dataclasses import dataclass

import numpy as np

from core.assertions import engine_assert

from math3d.camera import Camera
from math3d.matrices import (
    DEPTH_ZERO_TO_ONE,
    look_at,
    orthographic,
    perspective,
    translation
)


# =========================================================
# Shadow Math
# =========================================================
#
# Pure numpy; the GL side is graphics/shadow_map.py.
#
# Cascaded shadow maps (directional light)
# ----------------------------------------
# The view frustum up to `distance` is cut into slices,
# each covered by its own orthographic shadow map, so
# nearby shadows get many texels and distant ones few.
#
# Stabilization (no shimmering when the camera moves or
# turns):
#
#   * each slice is covered by its bounding *sphere*,
#     whose size does not change as the camera rotates;
#   * the shadow projection is shifted so world-space
#     texels stay on a fixed grid as the camera translates.


@dataclass(slots=True)
class Cascade:

    # World -> shadow clip space.
    matrix: np.ndarray

    # View-space distance where this cascade ends.
    split_far: float

    # World-space size of one shadow-map texel.
    texel_world_size: float


def cascade_splits(
    near: float,
    far: float,
    count: int,
    split_lambda: float
) -> list[float]:
    """
    Far distance of each cascade, blending logarithmic
    (lambda = 1) and uniform (lambda = 0) splits (the
    "practical split scheme").
    """

    engine_assert(
        0.0 < near < far,
        "Cascade range must satisfy 0 < near < far."
    )

    engine_assert(
        count >= 1,
        "Need at least one cascade."
    )

    split_lambda = min(max(split_lambda, 0.0), 1.0)

    splits = []

    for index in range(1, count + 1):

        fraction = index / count

        logarithmic = near * (far / near) ** fraction
        uniform = near + (far - near) * fraction

        splits.append(
            split_lambda * logarithmic
            + (1.0 - split_lambda) * uniform
        )

    return splits


def frustum_slice_corners(
    camera: Camera,
    near: float,
    far: float
) -> np.ndarray:
    """(8, 3) world-space corners of a view-frustum slice."""

    position = np.asarray(camera.position, dtype=np.float64)

    view = camera.view_matrix.astype(np.float64)

    right = view[0, :3]
    up = view[1, :3]
    forward = -view[2, :3]

    tan_half = np.tan(np.radians(camera.fov) / 2.0)

    corners = []

    for distance in (near, far):

        half_height = distance * tan_half
        half_width = half_height * camera.aspect_ratio

        center = position + forward * distance

        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                corners.append(
                    center
                    + right * (sx * half_width)
                    + up * (sy * half_height)
                )

    return np.array(corners)


def compute_cascades(
    camera: Camera,
    direction,
    count: int,
    distance: float,
    split_lambda: float,
    map_size: int,
    caster_margin: float = 25.0
) -> list[Cascade]:
    """
    Shadow matrices for a directional light traveling along
    `direction`.

    caster_margin: extra depth toward the light, so objects
        outside the view (e.g. behind the camera) still cast
        shadows into it.
    """

    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)

    far = min(distance, camera.far)

    engine_assert(
        far > camera.near,
        "Shadow distance must exceed the camera near plane."
    )

    splits = cascade_splits(
        camera.near,
        far,
        count,
        split_lambda
    )

    # Up must not be parallel to the light direction.
    up = (
        (0.0, 0.0, 1.0)
        if abs(direction[1]) > 0.99
        else (0.0, 1.0, 0.0)
    )

    cascades = []

    slice_near = camera.near

    for split_far in splits:

        corners = frustum_slice_corners(
            camera,
            slice_near,
            split_far
        )

        center = corners.mean(axis=0)

        radius = float(
            np.max(np.linalg.norm(corners - center, axis=1))
        )

        # Quantize so tiny float changes do not change the
        # projection size frame to frame.
        radius = np.ceil(radius * 16.0) / 16.0

        eye = center - direction * (radius + caster_margin)

        view = look_at(eye, center, up).astype(np.float64)

        projection = orthographic(
            -radius, radius,
            -radius, radius,
            0.0,
            2.0 * radius + caster_margin
        ).astype(np.float64)

        # Snap the world origin to a texel boundary: then
        # every world point lands on the same sub-texel
        # position no matter where the camera is.

        origin = (projection @ view) @ np.array([0.0, 0.0, 0.0, 1.0])

        texels = origin[:2] * (map_size / 2.0)

        offset = (np.round(texels) - texels) * (2.0 / map_size)

        projection[0, 3] += offset[0]
        projection[1, 3] += offset[1]

        cascades.append(
            Cascade(
                matrix=(projection @ view).astype(np.float32),
                split_far=float(split_far),
                texel_world_size=2.0 * radius / map_size
            )
        )

        slice_near = split_far

    return cascades


# =========================================================
# Terrain Shadow
# =========================================================
#
# The cascades cover tens or hundreds of meters, for props
# and the ground close by. Mountains shade valleys kilometers
# away, and at sunrise and sunset for tens of kilometers: one
# more map covers the land in view around the camera, coarse
# (~1/2000 of its width per texel) but far-reaching, and only
# the terrain draws into it. The lit shader takes the darker
# of the two.

def terrain_shadow_radius(
    altitude: float,
    planet_radius: float
) -> float:
    """
    Half-width (m) of the terrain shadow map: most of the
    ground to the horizon (horizon distance sqrt(2 R h)),
    no less than 20 km (the hills around you at ground
    level), no more than 1,500 km (from orbit, the ranges
    along the terminator).
    """

    altitude = max(altitude, 0.0)

    horizon = float(np.sqrt(2.0 * planet_radius * altitude + altitude * altitude))

    return min(max(0.8 * horizon, 20_000.0), 1_500_000.0, max(planet_radius, 20_000.0))


def terrain_shadow(
    center,
    direction,
    radius: float,
    map_size: int
) -> Cascade:
    """
    An orthographic shadow map around `center` (world, m),
    `radius` across, deep enough for casters a radius toward
    the light beyond it (a low sun's mountains).
    """

    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)

    center = np.asarray(center, dtype=np.float64)

    up = (
        (0.0, 0.0, 1.0)
        if abs(direction[1]) > 0.99
        else (0.0, 1.0, 0.0)
    )

    margin = radius

    eye = center - direction * (radius + margin)

    view = look_at(eye, center, up).astype(np.float64)

    projection = orthographic(
        -radius, radius,
        -radius, radius,
        0.0,
        2.0 * radius + margin
    ).astype(np.float64)

    # Texel-snapped like the cascades (no shimmer).
    origin = (projection @ view) @ np.array([0.0, 0.0, 0.0, 1.0])

    texels = origin[:2] * (map_size / 2.0)

    offset = (np.round(texels) - texels) * (2.0 / map_size)

    projection[0, 3] += offset[0]
    projection[1, 3] += offset[1]

    return Cascade(
        matrix=(projection @ view).astype(np.float32),
        split_far=float("inf"),
        texel_world_size=2.0 * radius / map_size
    )


# =========================================================
# Render Space
# =========================================================

def to_render_space(
    matrix,
    origin
) -> np.ndarray:
    """
    World -> shadow clip matrix  ->  camera-relative ->
    shadow clip with [0, 1] depth.

    Geometry reaches the GPU relative to the render origin
    (camera), so the shadow matrix must undo that shift; and
    with clip control GL_ZERO_TO_ONE, OpenGL-style [-1, 1]
    depth must be remapped. Done in float64, then narrowed.
    """

    return (
        DEPTH_ZERO_TO_ONE
        @ np.asarray(matrix, dtype=np.float64)
        @ translation(origin)
    ).astype(np.float32)


# =========================================================
# Spot Light Shadows
# =========================================================

@dataclass(slots=True)
class SpotShadow:

    matrix: np.ndarray

    # World-space texel size at distance d is
    # d * texel_scale (used for the normal offset).
    texel_scale: float


def spot_shadow(
    position,
    direction,
    outer_angle: float,
    light_range: float,
    map_size: int
) -> SpotShadow:

    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)

    position = np.asarray(position, dtype=np.float64)

    up = (
        (0.0, 0.0, 1.0)
        if abs(direction[1]) > 0.99
        else (0.0, 1.0, 0.0)
    )

    # A few degrees wider than the cone so PCF at the edge
    # stays inside the map.
    fov = min(2.0 * outer_angle + 6.0, 170.0)

    view = look_at(position, position + direction, up)

    projection = perspective(
        fov,
        1.0,
        0.05,
        max(light_range, 0.1)
    )

    return SpotShadow(
        matrix=(projection @ view).astype(np.float32),
        texel_scale=float(
            2.0 * np.tan(np.radians(fov) / 2.0) / map_size
        )
    )
