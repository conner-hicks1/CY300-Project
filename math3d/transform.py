import numpy as np

from core.assertions import engine_assert


class Transform:

    # =====================================================
    # Construction
    # =====================================================

    def __init__(
        self,
        position=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0)
    ):

        self.position = np.array(
            position,
            dtype=np.float32
        )

        self.rotation = np.array(
            rotation,
            dtype=np.float32
        )

        self.scale = np.array(
            scale,
            dtype=np.float32
        )

        engine_assert(
            self.position.shape == (3,),
            "Transform position must contain three values."
        )

        engine_assert(
            self.rotation.shape == (3,),
            "Transform rotation must contain three values."
        )

        engine_assert(
            self.scale.shape == (3,),
            "Transform scale must contain three values."
        )

    # =====================================================
    # Model Matrix
    # =====================================================

    @property
    def matrix(
        self
    ) -> np.ndarray:

        translation = (
            self._translation_matrix()
        )

        rotation = (
            self.rotation_matrix
        )

        scale = (
            self._scale_matrix()
        )

        return (
            translation
            @ rotation
            @ scale
        )

    # =====================================================
    # Normal Matrix
    # =====================================================

    @property
    def normal_matrix(
        self
    ) -> np.ndarray:

        # Inverse-transpose of the model matrix's upper
        # 3x3 keeps normals perpendicular to surfaces
        # under non-uniform scale.

        return np.linalg.inv(
            self.matrix[:3, :3]
        ).T.astype(
            np.float32
        )

    # =====================================================
    # Rotation Matrix
    # =====================================================

    @property
    def rotation_matrix(
        self
    ) -> np.ndarray:

        x = np.radians(
            float(self.rotation[0])
        )

        y = np.radians(
            float(self.rotation[1])
        )

        z = np.radians(
            float(self.rotation[2])
        )

        # -------------------------------------------------
        # X Rotation
        # -------------------------------------------------

        cos_x = np.cos(x)
        sin_x = np.sin(x)

        rx = np.array(
            [
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    cos_x,
                    -sin_x,
                    0.0
                ],
                [
                    0.0,
                    sin_x,
                    cos_x,
                    0.0
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

        # -------------------------------------------------
        # Y Rotation
        # -------------------------------------------------

        cos_y = np.cos(y)
        sin_y = np.sin(y)

        ry = np.array(
            [
                [
                    cos_y,
                    0.0,
                    sin_y,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    -sin_y,
                    0.0,
                    cos_y,
                    0.0
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

        # -------------------------------------------------
        # Z Rotation
        # -------------------------------------------------

        cos_z = np.cos(z)
        sin_z = np.sin(z)

        rz = np.array(
            [
                [
                    cos_z,
                    -sin_z,
                    0.0,
                    0.0
                ],
                [
                    sin_z,
                    cos_z,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0
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

        # -------------------------------------------------
        # Rotation Order
        # -------------------------------------------------
        #
        # With column vectors:
        #
        #     R = Rz @ Ry @ Rx
        #
        # means X is applied first, then Y, then Z.

        return (
            rz
            @ ry
            @ rx
        )

    # =====================================================
    # Direction Vectors
    # =====================================================

    @property
    def forward(
        self
    ) -> np.ndarray:

        # OpenGL convention:
        #
        # local forward = -Z

        local_forward = np.array(
            [
                0.0,
                0.0,
                -1.0,
                0.0
            ],
            dtype=np.float32
        )

        world_forward = (
            self.rotation_matrix
            @ local_forward
        )[:3]

        return self._normalize_direction(
            world_forward
        )

    @property
    def right(
        self
    ) -> np.ndarray:

        # Local right = +X

        local_right = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0
            ],
            dtype=np.float32
        )

        world_right = (
            self.rotation_matrix
            @ local_right
        )[:3]

        return self._normalize_direction(
            world_right
        )

    @property
    def up(
        self
    ) -> np.ndarray:

        # Local up = +Y

        local_up = np.array(
            [
                0.0,
                1.0,
                0.0,
                0.0
            ],
            dtype=np.float32
        )

        world_up = (
            self.rotation_matrix
            @ local_up
        )[:3]

        return self._normalize_direction(
            world_up
        )

    # =====================================================
    # Translation Matrix
    # =====================================================

    def _translation_matrix(
        self
    ) -> np.ndarray:

        x = float(
            self.position[0]
        )

        y = float(
            self.position[1]
        )

        z = float(
            self.position[2]
        )

        return np.array(
            [
                [
                    1.0,
                    0.0,
                    0.0,
                    x
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    y
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                    z
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

    # =====================================================
    # Scale Matrix
    # =====================================================

    def _scale_matrix(
        self
    ) -> np.ndarray:

        x = float(
            self.scale[0]
        )

        y = float(
            self.scale[1]
        )

        z = float(
            self.scale[2]
        )

        return np.array(
            [
                [
                    x,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    y,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    z,
                    0.0
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

    # =====================================================
    # Helpers
    # =====================================================

    @staticmethod
    def _normalize_direction(
        direction: np.ndarray
    ) -> np.ndarray:

        length = np.linalg.norm(
            direction
        )

        engine_assert(
            length > 0.0,
            "Transform direction vector cannot have zero length."
        )

        return (
            direction
            / length
        ).astype(
            np.float32
        )