import numpy as np

from core.assertions import engine_assert

from math3d.matrices import normal_matrix as compute_normal_matrix


class Transform:

    # =====================================================
    # Construction
    # =====================================================
    #
    # position / rotation / scale are exposed as read-only
    # arrays. Change them through the setters or the
    # translate() / rotate() helpers so cached matrices
    # are invalidated:
    #
    #     transform.position = (1.0, 2.0, 3.0)
    #     transform.translate(delta)
    #     transform.rotate((0.0, 30.0 * dt, 0.0))
    #
    # In-place writes such as `transform.position[0] = 1`
    # raise ValueError instead of silently desyncing the
    # cache.

    def __init__(
        self,
        position=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0)
    ):

        self._position = self._as_vec3(
            position,
            "position"
        )

        self._rotation = self._wrap_angles(
            self._as_vec3(
                rotation,
                "rotation"
            )
        )

        self._scale = self._as_vec3(
            scale,
            "scale"
        )

        # Incremented on every change. Lets other systems
        # (e.g. hierarchy) detect changes cheaply.

        self._version = 0

        self._invalidate()

    # =====================================================
    # Components
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

    @property
    def rotation(
        self
    ) -> np.ndarray:
        """
        Euler angles in degrees (X, Y, Z), each wrapped to
        [-180, 180) so they never grow without bound.
        """

        return self._read_only(
            self._rotation
        )

    @rotation.setter
    def rotation(
        self,
        value
    ):

        self._rotation = self._wrap_angles(
            self._as_vec3(
                value,
                "rotation"
            )
        )

        self._invalidate()

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

        self.rotation = (
            self._rotation
            + self._as_vec3(delta_degrees, "rotation delta")
        )

    # =====================================================
    # Model Matrix
    # =====================================================

    @property
    def matrix(
        self
    ) -> np.ndarray:

        if self._matrix is None:

            self._matrix = (
                self._translation_matrix()
                @ self.rotation_matrix
                @ self._scale_matrix()
            )

            self._matrix.flags.writeable = False

        return self._matrix

    # =====================================================
    # Normal Matrix
    # =====================================================

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

    # =====================================================
    # Rotation Matrix
    # =====================================================

    @property
    def rotation_matrix(
        self
    ) -> np.ndarray:

        if self._rotation_matrix is not None:
            return self._rotation_matrix

        x = np.radians(
            float(self._rotation[0])
        )

        y = np.radians(
            float(self._rotation[1])
        )

        z = np.radians(
            float(self._rotation[2])
        )

        # -------------------------------------------------
        # X Rotation
        # -------------------------------------------------

        cos_x = np.cos(x)
        sin_x = np.sin(x)

        rx = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, cos_x, -sin_x, 0.0],
                [0.0, sin_x, cos_x, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ],
            dtype=np.float32
        )

        # -------------------------------------------------
        # Y Rotation
        # -------------------------------------------------

        cos_y = np.cos(y)
        sin_y = np.sin(y)

        ry = np.array(
            [
                [cos_y, 0.0, sin_y, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-sin_y, 0.0, cos_y, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ],
            dtype=np.float32
        )

        # -------------------------------------------------
        # Z Rotation
        # -------------------------------------------------

        cos_z = np.cos(z)
        sin_z = np.sin(z)

        rz = np.array(
            [
                [cos_z, -sin_z, 0.0, 0.0],
                [sin_z, cos_z, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ],
            dtype=np.float32
        )

        # -------------------------------------------------
        # Rotation Order
        # -------------------------------------------------
        #
        # With column vectors:
        #
        #     R = Rz @ Ry @ Rx
        #
        # means X is applied first, then Y, then Z.

        self._rotation_matrix = (
            rz
            @ ry
            @ rx
        )

        self._rotation_matrix.flags.writeable = False

        return self._rotation_matrix

    # =====================================================
    # Direction Vectors
    # =====================================================

    @property
    def forward(
        self
    ) -> np.ndarray:

        # OpenGL convention: local forward = -Z

        return self._direction(
            (0.0, 0.0, -1.0)
        )

    @property
    def right(
        self
    ) -> np.ndarray:

        # Local right = +X

        return self._direction(
            (1.0, 0.0, 0.0)
        )

    @property
    def up(
        self
    ) -> np.ndarray:

        # Local up = +Y

        return self._direction(
            (0.0, 1.0, 0.0)
        )

    def _direction(
        self,
        local
    ) -> np.ndarray:

        world = (
            self.rotation_matrix[:3, :3]
            @ np.asarray(local, dtype=np.float32)
        )

        length = np.linalg.norm(
            world
        )

        engine_assert(
            length > 0.0,
            "Transform direction vector cannot have zero length."
        )

        return (
            world
            / length
        ).astype(
            np.float32
        )

    # =====================================================
    # Translation / Scale Matrices
    # =====================================================

    def _translation_matrix(
        self
    ) -> np.ndarray:

        x, y, z = (
            float(v)
            for v in self._position
        )

        return np.array(
            [
                [1.0, 0.0, 0.0, x],
                [0.0, 1.0, 0.0, y],
                [0.0, 0.0, 1.0, z],
                [0.0, 0.0, 0.0, 1.0]
            ],
            dtype=np.float32
        )

    def _scale_matrix(
        self
    ) -> np.ndarray:

        x, y, z = (
            float(v)
            for v in self._scale
        )

        return np.array(
            [
                [x, 0.0, 0.0, 0.0],
                [0.0, y, 0.0, 0.0],
                [0.0, 0.0, z, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ],
            dtype=np.float32
        )

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
            dtype=np.float32
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
    def _wrap_angles(
        angles: np.ndarray
    ) -> np.ndarray:

        # Wrap to [-180, 180). Keeps float32 precision
        # stable for objects that rotate forever, and
        # leaves pitch-clamped camera angles untouched.

        return (
            (angles + 180.0) % 360.0
            - 180.0
        ).astype(
            np.float32
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
            f"rotation={self._rotation.tolist()}, "
            f"scale={self._scale.tolist()}"
            f")"
        )
