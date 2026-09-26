import numpy as np

from core.assertions import engine_assert

from math3d.matrices import (
    look_at,
    perspective
)


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

        return look_at(
            self.position,
            self.target,
            self.up
        )

    # =====================================================
    # Projection Matrix
    # =====================================================

    @property
    def projection_matrix(self):

        self._validate_projection()

        return perspective(
            self.fov,
            self.aspect_ratio,
            self.near,
            self.far
        )
