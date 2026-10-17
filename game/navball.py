import math

import numpy as np

from imgui_bundle import imgui

from math3d import quaternion


# =========================================================
# Attitude Ball
# =========================================================
#
# Which way the vessel points, against the sky: a ball seen
# from behind the nose (the center is where the nose
# points), blue sky above the local horizon and brown ground
# below, the compass points on the horizon, and markers for
# prograde (along the motion: through the air in it, along
# the orbit above it) and retrograde.
#
# In the vessel's own axes the nose is +Y; on the ball +X
# is right and +Z up (so pitching forward, the nose toward
# -Z, brings the ground up from the bottom). The front
# hemisphere is drawn as seen straight on.

SKY = (0.22, 0.45, 0.75, 1.0)
GROUND = (0.45, 0.3, 0.16, 1.0)
HORIZON = (0.95, 0.95, 0.95, 1.0)
NOSE = (1.0, 0.65, 0.1, 1.0)
PROGRADE = (0.85, 1.0, 0.2, 1.0)

ROWS = 40


def to_ball(
    orientation: np.ndarray,
    directions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inertial unit directions -> (n, 2) ball coordinates (x
    right, y up, within the unit disc) and (n,) on the
    front half.
    """

    local = np.atleast_2d(directions) @ quaternion.to_matrix3(orientation)

    return np.stack((local[:, 0], local[:, 2]), axis=1), local[:, 1] > 0.0


def from_ball(
    orientation: np.ndarray,
    x: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Ball coordinates (front half) -> inertial unit directions."""

    z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))

    local = np.stack((x, z, y), axis=-1)

    return local @ quaternion.to_matrix3(orientation).T


def draw_navball(
    center: tuple[float, float],
    radius: float,
    orientation: np.ndarray,
    up: np.ndarray,
    north: np.ndarray,
    motion: np.ndarray | None
):
    """
    orientation: vessel -> inertial; up / north: the local
    vertical and north (inertial units); motion: what
    prograde follows (None: no marker).
    """

    draw = imgui.get_foreground_draw_list()

    cx, cy = center

    up = np.asarray(up, dtype=np.float64)

    # -------------------------------------------------
    # Sky and ground, a row at a time
    # -------------------------------------------------

    draw.add_circle_filled((cx, cy), radius + 3.0, imgui.get_color_u32((0.05, 0.05, 0.06, 0.85)))

    sky = imgui.get_color_u32(SKY)
    ground = imgui.get_color_u32(GROUND)

    rows = (np.arange(ROWS) + 0.5) / ROWS * 2.0 - 1.0
    columns = (np.arange(ROWS) + 0.5) / ROWS * 2.0 - 1.0

    xs, ys = np.meshgrid(columns, rows)

    inside = xs * xs + ys * ys <= 1.0

    is_sky = from_ball(orientation, xs, ys) @ up > 0.0

    cell = 2.0 * radius / ROWS

    for row in range(ROWS):

        y = rows[row]

        top = cy - y * radius - cell * 0.5

        start = None

        for column in range(ROWS + 1):

            current = (
                (bool(is_sky[row, column]) if inside[row, column] else None)
                if column < ROWS
                else None
            )

            if start is not None and current != start[1]:

                left = cx + columns[start[0]] * radius - cell * 0.5
                right = cx + columns[column - 1] * radius + cell * 0.5

                draw.add_rect_filled((left, top), (right, top + cell + 0.5), sky if start[1] else ground)

                start = None

            if start is None and current is not None:
                start = (column, current)

    # -------------------------------------------------
    # Horizon and compass
    # -------------------------------------------------

    north = np.asarray(north, dtype=np.float64)
    east = np.cross(north, up)

    angles = np.linspace(0.0, 2.0 * math.pi, 97)

    ring = np.cos(angles)[:, None] * north[None, :] + np.sin(angles)[:, None] * east[None, :]

    points, front = to_ball(orientation, ring)

    color = imgui.get_color_u32(HORIZON)

    for i in range(len(points) - 1):

        if front[i] and front[i + 1]:
            draw.add_line(
                (cx + points[i, 0] * radius, cy - points[i, 1] * radius),
                (cx + points[i + 1, 0] * radius, cy - points[i + 1, 1] * radius),
                color,
                1.5
            )

    for label, direction in (("N", north), ("E", east), ("S", -north), ("W", -east)):

        point, visible = to_ball(orientation, direction[None, :])

        if visible[0]:
            draw.add_text((cx + point[0, 0] * radius - 4.0, cy - point[0, 1] * radius - 14.0), color, label)

    # -------------------------------------------------
    # Prograde / retrograde
    # -------------------------------------------------

    if motion is not None and float(np.linalg.norm(motion)) > 0.5:

        heading = np.asarray(motion, dtype=np.float64) / float(np.linalg.norm(motion))

        marker = imgui.get_color_u32(PROGRADE)

        for sign, retro in ((1.0, False), (-1.0, True)):

            point, visible = to_ball(orientation, (sign * heading)[None, :])

            if not visible[0]:
                continue

            x, y = cx + point[0, 0] * radius, cy - point[0, 1] * radius

            if retro:
                draw.add_circle((x, y), 6.0, marker, 0, 1.5)
                draw.add_line((x - 4.0, y - 4.0), (x + 4.0, y + 4.0), marker, 1.5)
                draw.add_line((x - 4.0, y + 4.0), (x + 4.0, y - 4.0), marker, 1.5)
            else:
                draw.add_circle((x, y), 6.0, marker, 0, 1.5)
                draw.add_line((x, y - 6.0), (x, y - 11.0), marker, 1.5)
                draw.add_line((x - 6.0, y), (x - 11.0, y), marker, 1.5)
                draw.add_line((x + 6.0, y), (x + 11.0, y), marker, 1.5)

    # -------------------------------------------------
    # The nose
    # -------------------------------------------------

    nose = imgui.get_color_u32(NOSE)

    draw.add_line((cx - 16.0, cy), (cx - 6.0, cy), nose, 2.5)
    draw.add_line((cx - 6.0, cy), (cx, cy + 6.0), nose, 2.5)
    draw.add_line((cx, cy + 6.0), (cx + 6.0, cy), nose, 2.5)
    draw.add_line((cx + 6.0, cy), (cx + 16.0, cy), nose, 2.5)

    draw.add_circle((cx, cy), radius, imgui.get_color_u32((0.6, 0.6, 0.65, 1.0)), 0, 2.0)


def attitude(
    orientation: np.ndarray,
    up: np.ndarray,
    north: np.ndarray
) -> tuple[float, float]:
    """(heading deg from north, pitch deg above the horizon) of the nose."""

    nose = quaternion.rotate_vector(orientation, np.array([0.0, 1.0, 0.0]))

    up = np.asarray(up, dtype=np.float64)
    north = np.asarray(north, dtype=np.float64)
    east = np.cross(north, up)

    pitch = math.degrees(math.asin(float(np.clip(np.dot(nose, up), -1.0, 1.0))))

    heading = math.degrees(math.atan2(float(np.dot(nose, east)), float(np.dot(nose, north)))) % 360.0

    return heading, pitch
