import math

from dataclasses import dataclass

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
# the orbit above it) and retrograde, normal and antinormal
# (along the orbit's pole), radial out and in, the maneuver
# node's burn, the target and away from it. The maneuver and
# the target, behind the ball's face, show as an arrow at its
# rim pointing round to them.
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
NORMAL = (0.82, 0.5, 1.0, 1.0)
RADIAL = (0.35, 0.9, 1.0, 1.0)
MANEUVER = (0.35, 0.62, 1.0, 1.0)
TARGET = (1.0, 0.42, 0.8, 1.0)


@dataclass(slots=True)
class NavballFrame:
    """What the ball shows (all inertial, the body's non-rotating frame)."""

    orientation: np.ndarray         # vessel -> inertial
    up: np.ndarray                  # the local vertical
    north: np.ndarray
    motion: np.ndarray | None       # what prograde follows
    position: np.ndarray | None = None   # about the body (for normal / radial)
    velocity: np.ndarray | None = None
    maneuver: np.ndarray | None = None   # the burn left
    target: np.ndarray | None = None     # toward the target

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
    frame: NavballFrame
):
    """The ball at a screen point, radius px."""

    orientation, up, north, motion = frame.orientation, frame.up, frame.north, frame.motion

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
    # Markers
    # -------------------------------------------------

    def place(direction):
        point, visible = to_ball(orientation, np.asarray(direction, dtype=np.float64)[None, :])
        return (cx + point[0, 0] * radius, cy - point[0, 1] * radius), bool(visible[0]), point[0]

    def unit(vector):
        length = float(np.linalg.norm(vector)) if vector is not None else 0.0
        return None if length < 1e-9 else np.asarray(vector, dtype=np.float64) / length

    heading = unit(motion) if motion is not None and float(np.linalg.norm(motion)) > 0.5 else None

    if heading is not None:

        for sign, glyph in ((1.0, "prograde"), (-1.0, "retrograde")):

            spot, visible, _ = place(sign * heading)

            if visible:
                _glyph(draw, spot, glyph, PROGRADE)

    if frame.position is not None and frame.velocity is not None:

        normal = unit(np.cross(frame.position, frame.velocity))

        velocity = unit(frame.velocity)

        if normal is not None and velocity is not None:

            radial = unit(np.cross(velocity, normal))

            for vector, glyph, color in (
                (normal, "normal", NORMAL),
                (-normal, "antinormal", NORMAL),
                (radial, "radial_out", RADIAL),
                (-radial if radial is not None else None, "radial_in", RADIAL),
            ):

                if vector is None:
                    continue

                spot, visible, _ = place(vector)

                if visible:
                    _glyph(draw, spot, glyph, color)

    for vector, glyph, color, away in (
        (unit(frame.target), "target", TARGET, "away"),
        (unit(frame.maneuver), "maneuver", MANEUVER, None),
    ):

        if vector is None:
            continue

        spot, visible, point = place(vector)

        if visible:
            _glyph(draw, spot, glyph, color)
        else:
            _rim_arrow(draw, (cx, cy), radius, point, color)

        if away is not None:

            spot, visible, _ = place(-vector)

            if visible:
                _glyph(draw, spot, away, color)

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


def _glyph(
    draw,
    spot,
    kind: str,
    color
):
    """A navball marker, KSP's shapes."""

    x, y = spot

    c = imgui.get_color_u32(color)

    if kind == "prograde":
        draw.add_circle((x, y), 6.0, c, 0, 1.5)
        draw.add_line((x, y - 6.0), (x, y - 11.0), c, 1.5)
        draw.add_line((x - 6.0, y), (x - 11.0, y), c, 1.5)
        draw.add_line((x + 6.0, y), (x + 11.0, y), c, 1.5)

    elif kind == "retrograde":
        draw.add_circle((x, y), 6.0, c, 0, 1.5)
        draw.add_line((x - 4.0, y - 4.0), (x + 4.0, y + 4.0), c, 1.5)
        draw.add_line((x - 4.0, y + 4.0), (x + 4.0, y - 4.0), c, 1.5)
        draw.add_line((x, y - 6.0), (x, y - 11.0), c, 1.5)

    elif kind == "normal":
        draw.add_triangle((x, y - 7.0), (x - 6.5, y + 5.0), (x + 6.5, y + 5.0), c, 1.5)
        draw.add_circle_filled((x, y), 1.8, c)

    elif kind == "antinormal":
        draw.add_triangle((x, y + 7.0), (x - 6.5, y - 5.0), (x + 6.5, y - 5.0), c, 1.5)
        draw.add_line((x, y - 5.0), (x, y - 10.0), c, 1.5)

    elif kind == "radial_out":
        draw.add_circle((x, y), 5.0, c, 0, 1.5)
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            draw.add_line((x + dx * 5.0, y + dy * 5.0), (x + dx * 10.0, y + dy * 10.0), c, 1.5)

    elif kind == "radial_in":
        draw.add_circle((x, y), 7.0, c, 0, 1.5)
        for dx, dy in ((0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)):
            draw.add_line((x + dx * 7.0, y + dy * 7.0), (x + dx * 2.5, y + dy * 2.5), c, 1.5)

    elif kind == "maneuver":
        draw.add_quad((x, y - 8.0), (x + 8.0, y), (x, y + 8.0), (x - 8.0, y), c, 2.0)
        draw.add_circle_filled((x, y), 2.2, c)

    elif kind == "target":
        draw.add_circle((x, y), 7.0, c, 0, 2.0)
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            draw.add_line((x + dx * 3.0, y + dy * 3.0), (x + dx * 7.0, y + dy * 7.0), c, 2.0)

    elif kind == "away":
        draw.add_circle((x, y), 7.0, c, 0, 1.5)
        draw.add_line((x - 5.0, y - 5.0), (x + 5.0, y + 5.0), c, 1.5)
        draw.add_line((x - 5.0, y + 5.0), (x + 5.0, y - 5.0), c, 1.5)


def _rim_arrow(
    draw,
    center,
    radius: float,
    point: np.ndarray,
    color
):
    """Behind the ball's face: an arrow at its rim, the way round to it."""

    length = float(np.hypot(point[0], point[1]))

    if length < 1e-6:
        dx, dy = 0.0, -1.0
    else:
        dx, dy = point[0] / length, -point[1] / length

    cx, cy = center

    tip = (cx + dx * (radius + 9.0), cy + dy * (radius + 9.0))

    back = (cx + dx * (radius + 1.0), cy + dy * (radius + 1.0))

    side = (-dy * 5.0, dx * 5.0)

    draw.add_triangle_filled(tip, (back[0] + side[0], back[1] + side[1]), (back[0] - side[0], back[1] - side[1]), imgui.get_color_u32(color))
