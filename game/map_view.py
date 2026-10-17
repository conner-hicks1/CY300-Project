import math

from dataclasses import dataclass

import numpy as np

from imgui_bundle import imgui

from game import format
from math3d import quaternion


# =========================================================
# Map View
# =========================================================
#
# M: the camera pulls far back round the body the vessel
# flies about, and the orbits are drawn over the view:
#
#   * the vessel's path ahead (vessel/soi.py prediction):
#     its orbit, or its arc to the ground, or out of the
#     sphere of influence; an encounter's path round the
#     moon it reaches, drawn round that moon
#   * Ap / Pe: the highest and lowest points, with heights
#   * the moons' orbits, faint, with their names
#
# Lines are drawn in screen space (ImGui's background draw
# list): world points projected through the camera, dimmed
# where the body hides them.

PATH_COLOR = (0.35, 0.85, 1.0, 1.0)
ENCOUNTER_COLOR = (1.0, 0.6, 0.2, 1.0)
MOON_COLOR = (0.75, 0.75, 0.8, 0.45)
MARKER_COLOR = (1.0, 0.9, 0.45, 1.0)
VESSEL_COLOR = (1.0, 1.0, 1.0, 1.0)


@dataclass(slots=True)
class ScreenProjection:
    """World points to window pixels, for one camera."""

    position: np.ndarray            # camera, world m
    rotation: np.ndarray            # camera -> world (3x3)
    fov: float                      # vertical, deg
    width: float                    # window, px
    height: float
    origin: tuple[float, float] = (0.0, 0.0)

    @classmethod
    def of(
        cls,
        transform,
        fov: float,
        width: float,
        height: float,
        origin=(0.0, 0.0)
    ) -> "ScreenProjection":

        return cls(
            np.asarray(transform.position, dtype=np.float64),
            quaternion.to_matrix3(transform.orientation),
            fov,
            width,
            height,
            origin
        )

    def project(
        self,
        points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """(n, 2) pixels and (n,) in front of the camera."""

        local = (np.asarray(points, dtype=np.float64) - self.position) @ self.rotation

        # The camera looks along its -Z.
        depth = -local[:, 2]

        scale = 0.5 * self.height / math.tan(math.radians(self.fov) * 0.5)

        safe = np.where(depth > 1e-6, depth, 1e-6)

        x = self.origin[0] + 0.5 * self.width + local[:, 0] / safe * scale
        y = self.origin[1] + 0.5 * self.height - local[:, 1] / safe * scale

        return np.stack((x, y), axis=1), depth > 1e-6


def hidden_by_sphere(
    camera: np.ndarray,
    points: np.ndarray,
    center: np.ndarray,
    radius: float
) -> np.ndarray:
    """(n,) True where the sphere stands between the camera and the point."""

    camera = np.asarray(camera, dtype=np.float64)

    to_point = np.asarray(points, dtype=np.float64) - camera

    length = np.linalg.norm(to_point, axis=1)

    direction = to_point / np.maximum(length, 1e-9)[:, None]

    to_center = np.asarray(center, dtype=np.float64) - camera

    # Closest approach of each sight line to the center.
    along = direction @ to_center

    closest = np.linalg.norm(to_center[None, :] - direction * along[:, None], axis=1)

    return (closest < radius) & (along > 0.0) & (along < length)


def _color(
    rgba,
    alpha_scale: float = 1.0
) -> int:

    r, g, b, a = rgba

    return imgui.get_color_u32((r, g, b, a * alpha_scale))


class MapView:

    def __init__(self):

        self.open = False

        self.yaw = 0.0                  # deg
        self.pitch = 35.0               # deg above the body's equator
        self.distance = 0.0             # m from the body's center (0: fit)

    def toggle(self):

        self.open = not self.open

        # Fit the orbit afresh each time.
        self.distance = 0.0

    def orbit(
        self,
        mouse_delta: tuple[float, float],
        scroll: float
    ):

        dx, dy = mouse_delta

        self.yaw = (self.yaw + dx * 0.25) % 360.0
        self.pitch = min(max(self.pitch + dy * 0.25, -89.0), 89.0)

        if scroll and self.distance > 0.0:
            self.distance *= 0.85 ** scroll

    def place_camera(
        self,
        transform,
        center: np.ndarray,
        axis: np.ndarray,
        fit: float,
        minimum: float
    ):
        """
        Round the body at `center` (world), its spin `axis`
        up. fit: the radius (m) to fit in view the first
        time; minimum: closest allowed (beyond the body).
        """

        if self.distance <= 0.0:
            self.distance = fit * 2.6

        self.distance = max(self.distance, minimum)

        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)

        reference = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 0.0, 1.0])

        east = np.cross(axis, reference)
        east /= np.linalg.norm(east)

        north = np.cross(east, axis)

        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)

        direction = math.cos(pitch) * (math.cos(yaw) * north + math.sin(yaw) * east) + math.sin(pitch) * axis

        position = np.asarray(center, dtype=np.float64) + direction * self.distance

        transform.position = position
        transform.orientation = quaternion.look_rotation(np.asarray(center) - position, axis)

    # -----------------------------------------------------
    # Drawing
    # -----------------------------------------------------

    def draw(
        self,
        projection: ScreenProjection,
        to_world,
        body_world: dict,
        body_radius: dict,
        body_names: dict,
        prediction,
        vessel_world: np.ndarray | None,
        moon_orbits: dict,
        sea_radius: float
    ):
        """
        to_world: inertial directions/offsets -> world (3x3).
        body_world: key -> world center now; body_radius:
        key -> m; moon_orbits: key -> (n, 3) points about
        the reference body (inertial). sea_radius: the
        reference body's sea level radius (heights of Ap/Pe).
        """

        if not self.open:
            return

        draw = imgui.get_background_draw_list()

        # The moons' orbits and names.
        for key, points in moon_orbits.items():

            reference = prediction.patches[0].body if prediction is not None and prediction.patches else None

            center = body_world.get(reference)

            if center is None:
                continue

            self._polyline(draw, projection, center + points @ to_world.T, MOON_COLOR, body_world, body_radius, 1.0)

        for key, center in body_world.items():

            pixel, front = projection.project(center[None, :])

            if front[0]:
                draw.add_text((pixel[0, 0] + 8.0, pixel[0, 1] - 8.0), _color((0.85, 0.85, 0.9, 0.9)), body_names.get(key, ""))

        if prediction is None:
            return

        # The vessel's path, patch by patch.
        for index, patch in enumerate(prediction.patches):

            center = body_world.get(patch.body)

            if center is None:
                continue

            world = center + patch.points @ to_world.T

            color = PATH_COLOR if index == 0 else ENCOUNTER_COLOR

            self._polyline(draw, projection, world, color, body_world, body_radius, 2.0)

            if index == 0:
                self._apsides(draw, projection, patch, world, sea_radius, body_world, body_radius)

            if patch.end == "encounter" and len(world):
                self._marker(draw, projection, world[-1], ENCOUNTER_COLOR, f"{body_names.get(patch.next_body, '')} encounter")

            if index > 0 and math.isfinite(patch.periapsis):

                lowest = int(np.argmin(np.linalg.norm(patch.points, axis=1)))

                self._marker(
                    draw,
                    projection,
                    world[lowest],
                    ENCOUNTER_COLOR,
                    f"Pe {format.distance(patch.periapsis - body_radius.get(patch.body, 0.0))}"
                )

            if patch.end == "impact" and len(world):
                self._marker(draw, projection, world[-1], (1.0, 0.4, 0.3, 1.0), "Impact")

            if patch.end == "escape" and len(world):
                self._marker(draw, projection, world[-1], MARKER_COLOR, "Escape")

        if vessel_world is not None:

            pixel, front = projection.project(vessel_world[None, :])

            if front[0]:

                x, y = pixel[0]

                draw.add_triangle_filled((x, y - 7.0), (x - 6.0, y + 5.0), (x + 6.0, y + 5.0), _color(VESSEL_COLOR))

    def _polyline(
        self,
        draw,
        projection: ScreenProjection,
        world: np.ndarray,
        color,
        body_world: dict,
        body_radius: dict,
        thickness: float
    ):

        if len(world) < 2:
            return

        pixels, front = projection.project(world)

        hidden = np.zeros(len(world), dtype=bool)

        for key, center in body_world.items():
            hidden |= hidden_by_sphere(projection.position, world, center, body_radius.get(key, 0.0))

        bright = _color(color)
        dim = _color(color, 0.25)

        for i in range(len(world) - 1):

            if not (front[i] and front[i + 1]):
                continue

            draw.add_line(
                tuple(pixels[i]),
                tuple(pixels[i + 1]),
                dim if hidden[i] or hidden[i + 1] else bright,
                thickness
            )

    def _apsides(
        self,
        draw,
        projection,
        patch,
        world,
        sea_radius,
        body_world,
        body_radius
    ):

        distances = np.linalg.norm(patch.points, axis=1)

        if patch.end in ("orbit", "time"):

            high = int(np.argmax(distances))

            self._marker(draw, projection, world[high], MARKER_COLOR, f"Ap {format.distance(distances[high] - sea_radius)}")

        if patch.end != "impact":

            low = int(np.argmin(distances))

            # (On an open path, only if the lowest point is
            # still ahead.)
            if low > 0:
                self._marker(draw, projection, world[low], MARKER_COLOR, f"Pe {format.distance(distances[low] - sea_radius)}")

    @staticmethod
    def _marker(
        draw,
        projection: ScreenProjection,
        world_point: np.ndarray,
        color,
        label: str
    ):

        pixel, front = projection.project(np.asarray(world_point)[None, :])

        if not front[0]:
            return

        x, y = pixel[0]

        draw.add_circle_filled((x, y), 4.0, _color(color))
        draw.add_text((x + 7.0, y - 7.0), _color(color), label)
