import math

import numpy as np

from math3d import quaternion


# =========================================================
# Chase Camera
# =========================================================
#
# Circles the vessel: hold the right mouse button and move
# to swing round it, scroll to come closer or back off.
# Angles are measured in the vessel's horizon (up = away
# from the body, yaw 0 = looking east), so the horizon
# stays level whatever the vessel does.

class ChaseCamera:

    MIN_DISTANCE = 8.0
    MAX_DISTANCE = 5.0e7

    def __init__(
        self,
        distance: float = 70.0
    ):

        # Looking east from the west, a little from above.
        self.yaw = 0.0                  # deg
        self.pitch = 12.0               # deg above the vessel's horizon
        self.distance = distance        # m

    def orbit(
        self,
        mouse_delta: tuple[float, float],
        scroll: float,
        sensitivity: float = 0.2
    ):

        dx, dy = mouse_delta

        self.yaw = (self.yaw + dx * sensitivity) % 360.0
        self.pitch = min(max(self.pitch + dy * sensitivity, -85.0), 85.0)

        if scroll:
            self.distance = min(max(self.distance * 0.85 ** scroll, self.MIN_DISTANCE), self.MAX_DISTANCE)

    def place(
        self,
        transform,
        target: np.ndarray,
        up: np.ndarray,
        axis: np.ndarray,
        floor: float | None = None
    ):
        """
        Put the camera `transform` round `target` (world).
        up: radial up there; axis: the body's spin axis
        (world), for east; floor: the lowest the camera may
        be, in m above the target along `up` (keeps it out
        of the ground beside a landed vessel; None: any).
        """

        up = np.asarray(up, dtype=np.float64)
        up = up / np.linalg.norm(up)

        east = np.cross(axis, up)

        if np.linalg.norm(east) < 1e-6:
            east = np.cross(np.array([1.0, 0.0, 0.0]), up)

        east /= np.linalg.norm(east)

        north = np.cross(up, east)

        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)

        # The way the camera looks (horizontally).
        look = math.cos(yaw) * east + math.sin(yaw) * north

        offset = (-math.cos(pitch) * look + math.sin(pitch) * up) * self.distance

        if floor is not None:

            # Not below the ground around the vessel.
            height = float(np.dot(offset, up))

            if height < floor:
                offset += up * (floor - height)

        position = np.asarray(target, dtype=np.float64) + offset

        transform.position = position

        transform.orientation = quaternion.look_rotation(np.asarray(target) - position, up)
