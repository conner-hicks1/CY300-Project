import numpy as np

from core.assertions import engine_assert

from graphics.mesh import Mesh
from graphics.mesh_data import MeshData
from graphics.model_loader import load_model_data


class MeshFactory:

    # =====================================================
    # Conventions
    # =====================================================
    #
    # Every *_data() builder returns CPU-side MeshData and
    # needs no OpenGL context (tests use them directly).
    # create_*() uploads the same data as a Mesh.
    #
    # Triangles are counter-clockwise when viewed from the
    # side their normals face, which face culling relies
    # on. Tangents are derived from UVs by MeshData.

    # =====================================================
    # Triangle
    # =====================================================

    @staticmethod
    def triangle_data() -> MeshData:

        # Lies in the XY plane and faces +Z.

        return MeshData.from_attributes(
            positions=[
                (-0.5, -0.5, 0.0),
                (0.5, -0.5, 0.0),
                (0.0, 0.5, 0.0),
            ],
            normals=[(0.0, 0.0, 1.0)] * 3,
            uvs=[
                (0.0, 0.0),
                (1.0, 0.0),
                (0.5, 1.0),
            ],
            indices=[0, 1, 2]
        )

    @classmethod
    def create_triangle(
        cls
    ) -> Mesh:

        return Mesh.from_data(
            cls.triangle_data()
        )

    # =====================================================
    # Quad
    # =====================================================

    @staticmethod
    def quad_data() -> MeshData:

        # Quad lies in the XY plane and faces +Z.
        #
        # (-0.5, +0.5) -------- (+0.5, +0.5)
        #       3                      2
        #       |                      |
        #       0                      1
        # (-0.5, -0.5) -------- (+0.5, -0.5)

        return MeshData.from_attributes(
            positions=[
                (-0.5, -0.5, 0.0),
                (0.5, -0.5, 0.0),
                (0.5, 0.5, 0.0),
                (-0.5, 0.5, 0.0),
            ],
            normals=[(0.0, 0.0, 1.0)] * 4,
            uvs=[
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
            ],
            indices=[
                0, 1, 2,
                2, 3, 0
            ]
        )

    @classmethod
    def create_quad(
        cls
    ) -> Mesh:

        return Mesh.from_data(
            cls.quad_data()
        )

    # =====================================================
    # Plane
    # =====================================================

    @staticmethod
    def plane_data() -> MeshData:

        # Lies in the XZ plane and faces +Y.

        return MeshData.from_attributes(
            positions=[
                (-0.5, 0.0, 0.5),
                (0.5, 0.0, 0.5),
                (0.5, 0.0, -0.5),
                (-0.5, 0.0, -0.5),
            ],
            normals=[(0.0, 1.0, 0.0)] * 4,
            uvs=[
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
            ],
            indices=[
                0, 1, 2,
                2, 3, 0
            ]
        )

    @classmethod
    def create_plane(
        cls
    ) -> Mesh:

        return Mesh.from_data(
            cls.plane_data()
        )

    # =====================================================
    # Cube
    # =====================================================

    @staticmethod
    def cube_data() -> MeshData:

        # 24 vertices rather than 8 because each face needs
        # independent normals and UV coordinates.
        #
        # Each face lists its corners counter-clockwise as
        # seen from outside: bottom-left, bottom-right,
        # top-right, top-left (in that face's UV space).

        faces = [

            # Normal            Corners

            # Front (+Z)
            (
                (0.0, 0.0, 1.0),
                [
                    (-0.5, -0.5, 0.5),
                    (0.5, -0.5, 0.5),
                    (0.5, 0.5, 0.5),
                    (-0.5, 0.5, 0.5),
                ]
            ),

            # Back (-Z)
            (
                (0.0, 0.0, -1.0),
                [
                    (0.5, -0.5, -0.5),
                    (-0.5, -0.5, -0.5),
                    (-0.5, 0.5, -0.5),
                    (0.5, 0.5, -0.5),
                ]
            ),

            # Left (-X)
            (
                (-1.0, 0.0, 0.0),
                [
                    (-0.5, -0.5, -0.5),
                    (-0.5, -0.5, 0.5),
                    (-0.5, 0.5, 0.5),
                    (-0.5, 0.5, -0.5),
                ]
            ),

            # Right (+X)
            (
                (1.0, 0.0, 0.0),
                [
                    (0.5, -0.5, 0.5),
                    (0.5, -0.5, -0.5),
                    (0.5, 0.5, -0.5),
                    (0.5, 0.5, 0.5),
                ]
            ),

            # Top (+Y)
            (
                (0.0, 1.0, 0.0),
                [
                    (-0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, -0.5),
                    (-0.5, 0.5, -0.5),
                ]
            ),

            # Bottom (-Y)
            (
                (0.0, -1.0, 0.0),
                [
                    (-0.5, -0.5, -0.5),
                    (0.5, -0.5, -0.5),
                    (0.5, -0.5, 0.5),
                    (-0.5, -0.5, 0.5),
                ]
            ),
        ]

        face_uvs = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ]

        positions = []
        normals = []
        uvs = []
        indices = []

        for face_index, (normal, corners) in enumerate(faces):

            base = face_index * 4

            positions.extend(corners)
            normals.extend([normal] * 4)
            uvs.extend(face_uvs)

            indices.extend(
                [
                    base, base + 1, base + 2,
                    base + 2, base + 3, base
                ]
            )

        return MeshData.from_attributes(
            positions=positions,
            normals=normals,
            uvs=uvs,
            indices=indices
        )

    @classmethod
    def create_cube(
        cls
    ) -> Mesh:

        return Mesh.from_data(
            cls.cube_data()
        )

    # =====================================================
    # Sphere
    # =====================================================

    @staticmethod
    def sphere_data(
        segments: int = 48,
        rings: int = 24,
        radius: float = 0.5
    ) -> MeshData:

        # UV sphere. Ring i runs from the north pole
        # (i = 0) to the south pole (i = rings); segment j
        # runs around the Y axis. The seam column is
        # duplicated so UVs can wrap from 1 back to 0.

        engine_assert(
            segments >= 3 and rings >= 2,
            "Sphere needs at least 3 segments and 2 rings."
        )

        engine_assert(
            radius > 0.0,
            "Sphere radius must be positive."
        )

        ring_angles = np.linspace(
            0.0,
            np.pi,
            rings + 1
        )

        segment_angles = np.linspace(
            0.0,
            2.0 * np.pi,
            segments + 1
        )

        phi, theta = np.meshgrid(
            ring_angles,
            segment_angles,
            indexing="ij"
        )

        normals = np.stack(
            (
                np.sin(phi) * np.cos(theta),
                np.cos(phi),
                np.sin(phi) * np.sin(theta)
            ),
            axis=-1
        ).reshape(-1, 3)

        positions = normals * radius

        uvs = np.stack(
            (
                theta / (2.0 * np.pi),
                1.0 - phi / np.pi
            ),
            axis=-1
        ).reshape(-1, 2)

        columns = segments + 1

        indices = []

        for i in range(rings):

            for j in range(segments):

                a = i * columns + j           # (i,   j)
                b = (i + 1) * columns + j     # (i+1, j)
                c = i * columns + j + 1       # (i,   j+1)
                d = (i + 1) * columns + j + 1 # (i+1, j+1)

                # Skip the zero-area triangle at each pole.

                if i != 0:
                    indices.extend((a, c, b))

                if i != rings - 1:
                    indices.extend((c, d, b))

        return MeshData.from_attributes(
            positions=positions,
            normals=normals,
            uvs=uvs,
            indices=indices
        )

    @classmethod
    def create_sphere(
        cls,
        segments: int = 48,
        rings: int = 24,
        radius: float = 0.5
    ) -> Mesh:

        return Mesh.from_data(
            cls.sphere_data(
                segments,
                rings,
                radius
            )
        )

    # =====================================================
    # Models
    # =====================================================

    @staticmethod
    def load_model(
        path
    ) -> Mesh:
        """
        Load an .obj, .gltf or .glb file as a single Mesh.
        """

        return Mesh.from_data(
            load_model_data(path)
        )
