import numpy as np

from OpenGL.GL import GL_TRIANGLES

from core.assertions import engine_assert
from core.logger import Logger

from graphics.buffer import (
    VertexBuffer,
    IndexBuffer
)

from graphics.mesh_data import (
    COLOR_SIZE,
    NORMAL_SIZE,
    POSITION_SIZE,
    TANGENT_SIZE,
    UV_SIZE,
    MeshData
)
from graphics.vertex_array import VertexArray
from graphics.vertex_layout import VertexLayout


def standard_layout() -> VertexLayout:

    # Must match graphics/mesh_data.py and the
    # layout(location = N) inputs in the vertex shaders.

    return (
        VertexLayout()
        .add(POSITION_SIZE)     # 0 Position
        .add(NORMAL_SIZE)       # 1 Normal
        .add(COLOR_SIZE)        # 2 Color
        .add(UV_SIZE)           # 3 UV
        .add(TANGENT_SIZE)      # 4 Tangent (w = handedness)
    )


class Mesh:

    # =====================================================
    # From MeshData
    # =====================================================

    @classmethod
    def from_data(
        cls,
        data: MeshData
    ) -> "Mesh":

        engine_assert(
            data is not None,
            "Mesh.from_data() requires MeshData."
        )

        mesh = cls(
            data.vertices,
            standard_layout(),
            data.indices
        )

        # Kept for CPU-side queries (bounds, picking).

        mesh.data = data

        mesh.bounds = data.bounds

        return mesh

    def __init__(
        self,
        vertices,
        layout,
        indices=None,
        primitive=GL_TRIANGLES
    ):

        engine_assert(
            layout is not None,
            "Mesh requires a VertexLayout."
        )

        engine_assert(
            layout.stride > 0,
            "Mesh layout must have a positive stride."
        )

        self.layout = layout
        self.primitive = primitive

        self.data: MeshData | None = None

        # Local-space (min, max); None for meshes built
        # without MeshData (they cannot be picked).
        self.bounds: tuple[np.ndarray, np.ndarray] | None = None

        # =================================================
        # Vertex Data
        # =================================================

        vertices = np.asarray(
            vertices,
            dtype=np.float32
        )

        engine_assert(
            vertices.size > 0,
            "Mesh must contain vertex data."
        )

        engine_assert(
            vertices.nbytes % layout.stride == 0,
            (
                "Vertex data size must be divisible "
                "by the layout stride."
            )
        )

        self.vertex_count = (
            vertices.nbytes
            // layout.stride
        )

        # =================================================
        # Validate Indices Before GPU Allocation
        # =================================================

        validated_indices = None

        if indices is not None:

            validated_indices = np.asarray(
                indices,
                dtype=np.uint32
            )

            engine_assert(
                validated_indices.size > 0,
                (
                    "Indexed mesh cannot contain "
                    "an empty index array."
                )
            )

            engine_assert(
                np.max(validated_indices)
                < self.vertex_count,
                (
                    "Mesh contains an index that "
                    "references a nonexistent vertex."
                )
            )

        # =================================================
        # VAO / VBO
        # =================================================

        self.vertex_array = VertexArray()

        self.vertex_array.bind()

        self.vertex_buffer = VertexBuffer(
            vertices
        )

        self.vertex_array.add_buffer(
            self.vertex_buffer,
            self.layout
        )

        # =================================================
        # Index Buffer
        # =================================================

        self.index_buffer = None

        if validated_indices is not None:

            # Important:
            # EBO binding is stored inside the VAO.
            self.index_buffer = IndexBuffer(
                validated_indices
            )

        # =================================================
        # Unbind
        # =================================================

        self.vertex_array.unbind()
        self.vertex_buffer.unbind()

        Logger.debug(
            "[Mesh] Created: vertices=%d, indices=%d.",
            self.vertex_count,
            (
                self.index_buffer.count
                if self.index_buffer is not None
                else 0
            )
        )

    # =====================================================
    # Cleanup
    # =====================================================

    def delete(self):

        Logger.debug(
            "[Mesh] Deleting."
        )

        if self.index_buffer is not None:

            self.index_buffer.delete()

            self.index_buffer = None

        self.vertex_buffer.delete()
        self.vertex_array.delete()