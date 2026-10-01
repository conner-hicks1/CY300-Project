import math

import numpy as np

from core.assertions import engine_assert
from core.input import Input
from core.key_codes import Key

from ecs.components import (
    CameraComponent,
    CameraControllerComponent,
    TransformComponent
)

from math3d import quaternion

from scene.scene import Scene


WORLD_UP = np.array([0.0, 1.0, 0.0])


class CameraControllerSystem:

    # =====================================================
    # Update
    # =====================================================
    #
    # Two modes, chosen per controller:
    #
    # Flat (default): yaw about world +Y, pitch clamped,
    #     stored as Euler angles. W/S/A/D fly along the
    #     view; Space/Shift move along world Y.
    #
    # Planet (planet_mode): "up" is the direction away from
    #     planet_center at the camera's position. Yaw turns
    #     about that up, pitch is measured from the local
    #     horizon, and the orientation is rebuilt with
    #     look_rotation every frame, so flying around a
    #     sphere keeps the horizon level. W/S/A/D move
    #     along the ground (around the planet, view turning
    #     with it; capped at MAX_ORBIT_RATE), Space/Shift
    #     along the zenith. Speed scales with altitude, and
    #     the camera cannot go below planet_radius +
    #     min_altitude.

    def update(
        self,
        scene: Scene,
        delta_time: float,
        look_enabled: bool = True,
        move_enabled: bool = True
    ):
        """
        look_enabled: apply mouse look. The application
            enables this only while the cursor is captured,
            so moving the mouse over the window (or the
            debug UI) does not spin the camera.

        move_enabled: apply WASD / Space / Shift movement. Disabled
            while the debug UI has keyboard focus.
        """

        engine_assert(
            scene is not None,
            "CameraControllerSystem requires a Scene."
        )

        engine_assert(
            not scene.is_shutdown,
            "CameraControllerSystem cannot update a shutdown Scene."
        )

        engine_assert(
            delta_time >= 0.0,
            "Delta time cannot be negative."
        )

        # -------------------------------------------------
        # Controlled Cameras
        # -------------------------------------------------

        for entity in scene.view(
            TransformComponent,
            CameraComponent,
            CameraControllerComponent
        ):

            transform_component = (
                scene.get_component(
                    entity,
                    TransformComponent
                )
            )

            controller = (
                scene.get_component(
                    entity,
                    CameraControllerComponent
                )
            )

            transform = (
                transform_component.transform
            )

            if controller.planet_mode:

                self._update_planet(
                    transform,
                    controller,
                    delta_time,
                    look_enabled,
                    move_enabled
                )

                continue

            # ---------------------------------------------
            # Rotation
            # ---------------------------------------------

            if look_enabled:

                self._update_rotation(
                    transform,
                    controller
                )

            # ---------------------------------------------
            # Movement
            # ---------------------------------------------

            if move_enabled:

                self._update_movement(
                    transform,
                    controller,
                    delta_time
                )

    # =====================================================
    # Rotation
    # =====================================================

    @staticmethod
    def _update_rotation(
        transform,
        controller: CameraControllerComponent
    ):

        mouse_x, mouse_y = (
            Input.mouse_delta()
        )

        # -------------------------------------------------
        # Yaw
        # -------------------------------------------------
        #
        # Our Transform uses the standard right-handed
        # rotation matrix. Positive Y rotation turns -Z
        # toward -X.
        #
        # Moving the mouse right should normally turn the
        # camera toward +X, so mouse X is subtracted.

        pitch, yaw, _ = (
            float(angle)
            for angle in transform.rotation
        )

        yaw -= (
            mouse_x
            * controller.mouse_sensitivity
        )

        # -------------------------------------------------
        # Pitch
        # -------------------------------------------------
        #
        # GLFW cursor Y increases downward.
        #
        # With our rotation convention, positive X rotation
        # turns -Z toward +Y. Therefore subtracting positive
        # mouse Y makes moving the mouse down look downward.

        pitch -= (
            mouse_y
            * controller.mouse_sensitivity
        )

        # -------------------------------------------------
        # Pitch Clamp
        # -------------------------------------------------

        pitch = float(
            np.clip(
                pitch,
                controller.min_pitch,
                controller.max_pitch
            )
        )

        # -------------------------------------------------
        # No Roll
        # -------------------------------------------------
        #
        # A conventional free-fly FPS-style controller
        # keeps the camera upright. Yaw is wrapped to
        # [-180, 180) by the Transform setter.

        transform.rotation = (
            pitch,
            yaw,
            0.0
        )

    # =====================================================
    # Movement
    # =====================================================

    @staticmethod
    def _update_movement(
        transform,
        controller: CameraControllerComponent,
        delta_time: float
    ):

        # Space/Shift use world-space Y rather than
        # transform.up, so they remain vertical even when
        # the camera is pitched.

        movement = _movement_input(
            transform,
            WORLD_UP
        )

        if movement is None:
            return

        transform.translate(
            movement
            * controller.movement_speed
            * delta_time
        )

    # =====================================================
    # Planet Mode
    # =====================================================

    @staticmethod
    def _update_planet(
        transform,
        controller: CameraControllerComponent,
        delta_time: float,
        look_enabled: bool,
        move_enabled: bool
    ):

        yaw = 0.0
        pitch = 0.0

        if look_enabled:

            mouse_x, mouse_y = Input.mouse_delta()

            # Same signs as flat mode: mouse right turns
            # right, mouse down looks down.
            yaw = -mouse_x * controller.mouse_sensitivity
            pitch = -mouse_y * controller.mouse_sensitivity

        if move_enabled:

            forward_input, right_input, up_input = _key_axes()

            if forward_input or right_input or up_input:

                position, rotation = planet_motion(
                    position=transform.position,
                    center=controller.planet_center,
                    forward=transform.forward,
                    camera_up=transform.up,
                    forward_input=forward_input,
                    right_input=right_input,
                    up_input=up_input,
                    speed=planet_speed(controller, transform.position),
                    delta_time=delta_time
                )

                transform.position = clamp_altitude(
                    position,
                    controller
                )

                # The view travels with the camera around
                # the planet (no swinging toward where it
                # started).
                transform.orientation = quaternion.multiply(
                    rotation,
                    transform.orientation
                )

        # Always re-level: moving across the sphere changes
        # "up" even without mouse input.

        transform.orientation = planet_look(
            transform.forward,
            planet_up(
                transform.position,
                controller.planet_center
            ),
            yaw,
            pitch,
            controller.min_pitch,
            controller.max_pitch
        )


# =========================================================
# Helpers
# =========================================================

def _key_axes() -> tuple[float, float, float]:
    """
    (forward, right, up) movement input in {-1, 0, 1}:
    W/S, D/A, Space/Shift.
    """

    def axis(positive, negative):

        return float(positive) - float(negative)

    shift = Input.is_key_down(Key.LEFT_SHIFT) or Input.is_key_down(Key.RIGHT_SHIFT)

    return (
        axis(Input.is_key_down(Key.W), Input.is_key_down(Key.S)),
        axis(Input.is_key_down(Key.D), Input.is_key_down(Key.A)),
        axis(Input.is_key_down(Key.SPACE), shift),
    )


def _movement_input(
    transform,
    vertical: np.ndarray
) -> np.ndarray | None:
    """Unit fly direction (flat mode), or None without input."""

    forward_input, right_input, up_input = _key_axes()

    movement = (
        forward_input * np.asarray(transform.forward, dtype=np.float64)
        + right_input * np.asarray(transform.right, dtype=np.float64)
        + up_input * vertical
    )

    length = float(np.linalg.norm(movement))

    if length <= 0.0:
        return None

    return movement / length


# Fastest a planet-mode camera may circle the planet
# (rad/s): about 18 s per orbit. Altitude-scaled speed
# alone would whip the planet past in orbit.
MAX_ORBIT_RATE = 0.35


def planet_motion(
    position,
    center,
    forward,
    camera_up,
    forward_input: float,
    right_input: float,
    up_input: float,
    speed: float,
    delta_time: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Planet-mode movement. Returns (new position, rotation
    quaternion applied to the camera's orientation).

    W/S/A/D move along the ground: the camera circles the
    planet center at constant altitude (a rotation, so the
    view turns with it). "Forward" is the view direction
    projected onto the horizon; looking straight down it is
    the top of the screen. Space/Shift move along the
    zenith.
    """

    position = np.asarray(position, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    forward = np.asarray(forward, dtype=np.float64)
    camera_up = np.asarray(camera_up, dtype=np.float64)

    offset = position - center

    radius = float(np.linalg.norm(offset))

    up = planet_up(position, center)

    rotation = quaternion.identity()

    # -----------------------------------------------------
    # Horizontal: around the planet
    # -----------------------------------------------------

    def horizontal(vector):

        return vector - np.dot(vector, up) * up

    # Pitched down, the camera's up leans forward (and
    # backward when pitched up), so this blend stays
    # pointing "ahead" at any pitch, even straight down.
    heading = horizontal(forward) - np.dot(forward, up) * horizontal(camera_up)

    heading_length = float(np.linalg.norm(heading))

    if (forward_input or right_input) and heading_length > 1e-9 and radius > 0.0:

        heading /= heading_length

        right = np.cross(heading, up)

        direction = forward_input * heading + right_input * right

        direction /= np.linalg.norm(direction)

        distance = min(speed, MAX_ORBIT_RATE * radius) * delta_time

        axis = np.cross(up, direction)

        rotation = quaternion.from_axis_angle(
            axis / np.linalg.norm(axis),
            np.degrees(distance / radius)
        )

        offset = quaternion.rotate_vector(rotation, offset)

        up = offset / np.linalg.norm(offset)

    # -----------------------------------------------------
    # Vertical: along the zenith
    # -----------------------------------------------------

    offset = offset + up * (up_input * speed * delta_time)

    return center + offset, rotation


def planet_up(
    position,
    center
) -> np.ndarray:
    """Unit vector from `center` to `position` (+Y at the center)."""

    offset = (
        np.asarray(position, dtype=np.float64)
        - np.asarray(center, dtype=np.float64)
    )

    length = float(np.linalg.norm(offset))

    if length < 1e-9:
        return WORLD_UP.copy()

    return offset / length


def planet_look(
    forward,
    up,
    yaw_degrees: float,
    pitch_degrees: float,
    min_pitch: float,
    max_pitch: float
) -> np.ndarray:
    """
    Orientation after turning `forward` by `yaw_degrees`
    about `up` and adding `pitch_degrees` of pitch
    (measured from the horizon, clamped). No roll.
    """

    forward = np.asarray(forward, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    # Split forward into horizon direction + pitch angle.

    sine = float(np.clip(np.dot(forward, up), -1.0, 1.0))

    pitch = math.degrees(math.asin(sine))

    horizontal = forward - sine * up

    length = float(np.linalg.norm(horizontal))

    if length < 1e-6:

        # Looking straight along up: any horizon direction
        # works; take one perpendicular to up.
        helper = WORLD_UP if abs(up[1]) < 0.9 else np.array([0.0, 0.0, -1.0])

        horizontal = helper - np.dot(helper, up) * up
        length = float(np.linalg.norm(horizontal))

    horizontal /= length

    if yaw_degrees:

        horizontal = quaternion.rotate_vector(
            quaternion.from_axis_angle(up, yaw_degrees),
            horizontal
        )

    pitch = math.radians(
        float(np.clip(pitch + pitch_degrees, min_pitch, max_pitch))
    )

    new_forward = (
        math.cos(pitch) * horizontal
        + math.sin(pitch) * up
    )

    return quaternion.look_rotation(
        new_forward,
        up
    )


def planet_speed(
    controller: CameraControllerComponent,
    position
) -> float:

    if controller.altitude_speed <= 0.0:
        return controller.movement_speed

    return max(
        controller.movement_speed,
        altitude(controller, position) * controller.altitude_speed
    )


def altitude(
    controller: CameraControllerComponent,
    position
) -> float:

    distance = float(
        np.linalg.norm(
            np.asarray(position, dtype=np.float64)
            - np.asarray(controller.planet_center, dtype=np.float64)
        )
    )

    return distance - controller.planet_radius


def clamp_altitude(
    position,
    controller: CameraControllerComponent
) -> np.ndarray:
    """Push `position` out to planet_radius + min_altitude if below it."""

    position = np.asarray(position, dtype=np.float64)

    floor = controller.planet_radius + controller.min_altitude

    if floor <= 0.0:
        return position

    center = np.asarray(controller.planet_center, dtype=np.float64)

    if np.linalg.norm(position - center) >= floor:
        return position

    return center + planet_up(position, center) * floor
