import numpy as np

from core.assertions import engine_assert
from core.input import Input
from core.key_codes import Key

from ecs.components import (
    CameraComponent,
    CameraControllerComponent,
    TransformComponent
)

from scene.scene import Scene


class CameraControllerSystem:

    # =====================================================
    # Update
    # =====================================================

    def update(
        self,
        scene: Scene,
        delta_time: float
    ):

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

            # ---------------------------------------------
            # Rotation
            # ---------------------------------------------

            self._update_rotation(
                transform,
                controller
            )

            # ---------------------------------------------
            # Movement
            # ---------------------------------------------

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

        transform.rotation[1] -= (
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

        transform.rotation[0] -= (
            mouse_y
            * controller.mouse_sensitivity
        )

        # -------------------------------------------------
        # Pitch Clamp
        # -------------------------------------------------

        transform.rotation[0] = np.clip(
            transform.rotation[0],
            controller.min_pitch,
            controller.max_pitch
        )

        # -------------------------------------------------
        # No Roll
        # -------------------------------------------------
        #
        # A conventional free-fly FPS-style controller
        # keeps the camera upright.

        transform.rotation[2] = 0.0

    # =====================================================
    # Movement
    # =====================================================

    @staticmethod
    def _update_movement(
        transform,
        controller: CameraControllerComponent,
        delta_time: float
    ):

        movement = np.zeros(
            3,
            dtype=np.float32
        )

        # -------------------------------------------------
        # Forward / Backward
        # -------------------------------------------------

        if Input.is_key_down(
            Key.W
        ):
            movement += transform.forward

        if Input.is_key_down(
            Key.S
        ):
            movement -= transform.forward

        # -------------------------------------------------
        # Left / Right
        # -------------------------------------------------

        if Input.is_key_down(
            Key.D
        ):
            movement += transform.right

        if Input.is_key_down(
            Key.A
        ):
            movement -= transform.right

        # -------------------------------------------------
        # Vertical
        # -------------------------------------------------
        #
        # This uses world-space Y rather than transform.up.
        #
        # That means Q/E remain vertical even when the
        # camera is pitched.

        if Input.is_key_down(
            Key.E
        ):
            movement += np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float32
            )

        if Input.is_key_down(
            Key.Q
        ):
            movement -= np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float32
            )

        # -------------------------------------------------
        # Normalize
        # -------------------------------------------------

        length = np.linalg.norm(
            movement
        )

        if length <= 0.0:
            return

        movement /= length

        # -------------------------------------------------
        # Apply Movement
        # -------------------------------------------------

        transform.position += (
            movement
            * controller.movement_speed
            * delta_time
        )