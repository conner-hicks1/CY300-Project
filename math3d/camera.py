import numpy as np

from core.assertions import engine_assert


class Camera:

    # =====================================================
    # Construction
    # =====================================================

    def __init__(
        self,
        position=(0.0, 0.0, 3.0),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov=45.0,
        aspect_ratio=1.0,
        near=0.1,
        far=100.0
    ):

        self.position = np.array(
            position,
            dtype=np.float32
        )

        self.target = np.array(
            target,
            dtype=np.float32
        )

        self.up = np.array(
            up,
            dtype=np.float32
        )

        self.fov = float(fov)
        self.aspect_ratio = float(aspect_ratio)
        self.near = float(near)
        self.far = float(far)

        self._validate_projection()

    # =====================================================
    # Projection Validation
    # =====================================================

    def _validate_projection(self):

        engine_assert(
            0.0 < self.fov < 180.0,
            "Camera FOV must be between 0 and 180 degrees."
        )

        engine_assert(
            self.aspect_ratio > 0.0,
            "Camera aspect ratio must be positive."
        )

        engine_assert(
            self.near > 0.0,
            "Camera near plane must be positive."
        )

        engine_assert(
            self.far > self.near,
            "Camera far plane must be greater than near plane."
        )

    # =====================================================
    # View Matrix
    # =====================================================

    @property
    def view_matrix(self):

        forward = (
            self.target
            - self.position
        )

        forward_length = np.linalg.norm(
            forward
        )

        engine_assert(
            forward_length > 0.0,
            "Camera position and target cannot be identical."
        )

        forward = (
            forward
            / forward_length
        )

        right = np.cross(
            forward,
            self.up
        )

        right_length = np.linalg.norm(
            right
        )

        engine_assert(
            right_length > 0.0,
            "Camera up vector cannot be parallel to forward direction."
        )

        right = (
            right
            / right_length
        )

        camera_up = np.cross(
            right,
            forward
        )

        view = np.array(
            [
                [
                    right[0],
                    right[1],
                    right[2],
                    -np.dot(
                        right,
                        self.position
                    )
                ],
                [
                    camera_up[0],
                    camera_up[1],
                    camera_up[2],
                    -np.dot(
                        camera_up,
                        self.position
                    )
                ],
                [
                    -forward[0],
                    -forward[1],
                    -forward[2],
                    np.dot(
                        forward,
                        self.position
                    )
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            dtype=np.float32
        )

        return view

    # =====================================================
    # Projection Matrix
    # =====================================================

    @property
    def projection_matrix(self):

        self._validate_projection()

        fov_radians = np.radians(
            self.fov
        )

        f = (
            1.0
            / np.tan(
                fov_radians / 2.0
            )
        )

        near = self.near
        far = self.far

        return np.array(
            [
                [
                    f / self.aspect_ratio,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    f,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    (far + near) / (near - far),
                    (2.0 * far * near) / (near - far)
                ],
                [
                    0.0,
                    0.0,
                    -1.0,
                    0.0
                ]
            ],
            dtype=np.float32
        )