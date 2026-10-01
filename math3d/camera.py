import numpy as np

from core.assertions import engine_assert

from math3d.matrices import (
    look_at,
    perspective,
    perspective_reversed_infinite
)


class Camera:

    # =====================================================
    # Camera
    # =====================================================
    #
    # Position and target are float64 world coordinates.
    #
    # The GPU never sees world-space view matrices: the
    # renderer draws camera-relative (world position minus
    # camera position, in float64, then converted to
    # float32), so it uses `view_rotation_matrix` with the
    # camera at the origin. `view_matrix` (full world view)
    # is for CPU work such as picking and shadow fitting.
    #
    # Projection is reversed-Z with an infinite far plane;
    # `far` only limits what is drawn or shadowed by choice
    # (e.g. the shadow distance), not depth precision.

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
            dtype=np.float64
        )

        self.target = np.array(
            target,
            dtype=np.float64
        )

        self.up = np.array(
            up,
            dtype=np.float64
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
    # View Matrices
    # =====================================================

    @property
    def view_matrix(self) -> np.ndarray:
        """World -> view (float64). CPU use only."""

        return look_at(
            self.position,
            self.target,
            self.up
        ).astype(np.float64)

    @property
    def view_rotation_matrix(self) -> np.ndarray:
        """
        Camera-relative -> view: the view matrix with the
        camera at the origin. What the GPU gets.
        """

        view = self.view_matrix

        view[:3, 3] = 0.0

        return view

    # =====================================================
    # Projection Matrices
    # =====================================================

    @property
    def projection_matrix(self) -> np.ndarray:
        """Reversed-Z, infinite far plane (rendering)."""

        self._validate_projection()

        return perspective_reversed_infinite(
            self.fov,
            self.aspect_ratio,
            self.near
        )

    @property
    def conventional_projection_matrix(self) -> np.ndarray:
        """
        Standard OpenGL perspective using near/far, for
        tools that assume it (ImGuizmo).
        """

        self._validate_projection()

        return perspective(
            self.fov,
            self.aspect_ratio,
            self.near,
            self.far
        ).astype(np.float64)
