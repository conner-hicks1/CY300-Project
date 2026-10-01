import numpy as np

from core.assertions import engine_assert

from math3d import quaternion
from math3d.matrices import normal_matrix as compute_normal_matrix


class Transform:

    # =====================================================
    # Construction
    # =====================================================
    #
    # Position, orientation and scale of an object.
    #
    # Precision: everything is float64. At planet scale a
    # float32 position only resolves about half a metre, so
    # world positions stay in double precision and the
    # renderer subtracts the camera position before
    # converting to float32 (camera-relative rendering).
    #
    # Orientation is a unit quaternion (`orientation`,
    # x y z w). `rotation` presents it as Euler angles in
    # degrees (X, Y, Z; applied X first) for the inspector
    # and scene files. The Euler angles last *set* are kept
    # verbatim, so a typed (0, 190, 0) does not come back as
    # the equivalent (180, -10, 180).
    #
    # Components are exposed as read-only arrays. Change
    # them through the setters or helpers so the cached
    # matrices stay valid:
    #
    #     transform.position = (1.0, 2.0, 3.0)
    #     transform.translate(delta)
    #     transform.rotate((0.0, 30.0 * dt, 0.0))
    #     transform.orientation = quaternion.look_rotation(f, up)
    #
    # In-place writes such as `transform.position[0] = 1`
    # raise ValueError instead of silently desyncing the
    # cache.

    def __init__(
        self,
        position=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        orientation=None
    ):
        """
        orientation: quaternion (x, y, z, w); overrides
            `rotation` when given.
        """

        self._position = self._as_vec3(
            position,
            "position"
        )

        if orientation is not None:

            self._orientation = self._as_quaternion(orientation)
            self._euler = None

        else:

            euler = self._wrap_angles(
                self._as_vec3(rotation, "rotation")
            )

            self._orientation = quaternion.from_euler(euler)
            self._euler = euler

        self._scale = self._as_vec3(
            scale,
            "scale"
        )

        # Incremented on every change. Lets other systems
        # detect changes cheaply.

        self._version = 0

        self._invalidate()

    # =====================================================
    # Position
    # =====================================================

    @property
    def position(
        self
    ) -> np.ndarray:

        return self._read_only(
            self._position
        )

    @position.setter
    def position(
        self,
        value
    ):

        self._position = self._as_vec3(
            value,
            "position"
        )

        self._invalidate()

    # =====================================================
    # Orientation
    # =====================================================

    @property
    def orientation(
        self
    ) -> np.ndarray:
        """Unit quaternion (x, y, z, w)."""

        return self._read_only(
            self._orientation
        )

    @orientation.setter
    def orientation(
        self,
        value
    ):

        self._orientation = self._as_quaternion(
            value
        )

        # Euler view is recomputed on demand.
        self._euler = None

        self._invalidate()

    @property
    def rotation(
        self
    ) -> np.ndarray:
        """
        Orientation as Euler angles in degrees (X, Y, Z),
        each in [-180, 180).
        """

        if self._euler is None:

            self._euler = self._wrap_angles(
                quaternion.to_euler(self._orientation)
            )

        return self._read_only(
            self._euler
        )

    @rotation.setter
    def rotation(
        self,
        value
    ):

        euler = self._wrap_angles(
            self._as_vec3(
                value,
                "rotation"
            )
        )

        self._orientation = quaternion.from_euler(euler)
        self._euler = euler

        self._invalidate()

    # =====================================================
    # Scale
    # =====================================================

    @property
    def scale(
        self
    ) -> np.ndarray:

        return self._read_only(
            self._scale
        )

    @scale.setter
    def scale(
        self,
        value
    ):

        self._scale = self._as_vec3(
            value,
            "scale"
        )

        self._invalidate()

    @property
    def version(
        self
    ) -> int:

        return self._version

    # =====================================================
    # Mutation Helpers
    # =====================================================

    def translate(
        self,
        delta
    ):

        self.position = (
            self._position
            + self._as_vec3(delta, "translation")
        )

    def rotate(
        self,
        delta_degrees
    ):
        """Add to the Euler angles (spin about local X/Y/Z)."""

        self.rotation = (
            self.rotation
            + self._as_vec3(delta_degrees, "rotation delta")
        )

    def rotate_about_axis(
        self,
        axis,
        degrees: float,
        world_space: bool = True
    ):
        """
        Rotate by `degrees` about `axis`, given in world
        space (default) or in this transform's local space.
        """

        delta = quaternion.from_axis_angle(
            axis,
            degrees
        )

        self.orientation = (
            quaternion.multiply(delta, self._orientation)
            if world_space
            else quaternion.multiply(self._orientation, delta)
        )

    # =====================================================
    # Matrices
    # =====================================================

    @property
    def matrix(
        self
    ) -> np.ndarray:
        """Local -> parent, float64 (T @ R @ S)."""

        if self._matrix is None:

            matrix = np.identity(4)

            matrix[:3, :3] = (
                self.rotation_matrix[:3, :3]
                * self._scale
            )

            matrix[:3, 3] = self._position

            matrix.flags.writeable = False

            self._matrix = matrix

        return self._matrix

    @property
    def normal_matrix(
        self
    ) -> np.ndarray:
        """
        3x3 matrix for transforming normals. Safe for
        non-uniform and zero scale; see
        math3d.matrices.normal_matrix.
        """

        if self._normal_matrix is None:

            self._normal_matrix = compute_normal_matrix(
                self.matrix
            )

            self._normal_matrix.flags.writeable = False

        return self._normal_matrix

    @property
    def rotation_matrix(
        self
    ) -> np.ndarray:
        """4x4 rotation-only matrix."""

        if self._rotation_matrix is None:

            matrix = np.identity(4)

            matrix[:3, :3] = quaternion.to_matrix3(
                self._orientation
            )

            matrix.flags.writeable = False

            self._rotation_matrix = matrix

        return self._rotation_matrix

    # =====================================================
    # Direction Vectors
    # =====================================================

    @property
    def forward(
        self
    ) -> np.ndarray:

        # OpenGL convention: local forward = -Z

        return -self.rotation_matrix[:3, 2].copy()

    @property
    def right(
        self
    ) -> np.ndarray:

        # Local right = +X

        return self.rotation_matrix[:3, 0].copy()

    @property
    def up(
        self
    ) -> np.ndarray:

        # Local up = +Y

        return self.rotation_matrix[:3, 1].copy()

    # =====================================================
    # Helpers
    # =====================================================

    def _invalidate(self):

        self._matrix = None
        self._rotation_matrix = None
        self._normal_matrix = None

        self._version += 1

    @staticmethod
    def _as_vec3(
        value,
        name: str
    ) -> np.ndarray:

        array = np.array(
            value,
            dtype=np.float64
        )

        engine_assert(
            array.shape == (3,),
            f"Transform {name} must contain three values."
        )

        engine_assert(
            bool(np.all(np.isfinite(array))),
            f"Transform {name} must be finite."
        )

        return array

    @staticmethod
    def _as_quaternion(
        value
    ) -> np.ndarray:

        array = np.array(
            value,
            dtype=np.float64
        )

        engine_assert(
            array.shape == (4,)
            and bool(np.all(np.isfinite(array))),
            "Transform orientation must be four finite values (x, y, z, w)."
        )

        return quaternion.normalize(array)

    @staticmethod
    def _wrap_angles(
        angles: np.ndarray
    ) -> np.ndarray:

        # Wrap to [-180, 180) so angles never grow without
        # bound for objects that spin forever.

        return (
            (np.asarray(angles, dtype=np.float64) + 180.0) % 360.0
            - 180.0
        )

    @staticmethod
    def _read_only(
        array: np.ndarray
    ) -> np.ndarray:

        view = array.view()

        view.flags.writeable = False

        return view

    # =====================================================
    # Representation
    # =====================================================

    def __repr__(self):

        return (
            f"Transform("
            f"position={self._position.tolist()}, "
            f"rotation={self.rotation.tolist()}, "
            f"scale={self._scale.tolist()}"
            f")"
        )
