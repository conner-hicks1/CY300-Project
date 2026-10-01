import ctypes

from dataclasses import dataclass

import numpy as np

from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_COPY_READ_BUFFER,
    GL_COPY_WRITE_BUFFER,
    GL_DYNAMIC_DRAW,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_FALSE,
    GL_FLOAT,
    GL_STATIC_DRAW,
    GL_UNSIGNED_INT,
    glBindBuffer,
    glBindVertexArray,
    glBufferData,
    glBufferSubData,
    glCopyBufferSubData,
    glDeleteBuffers,
    glDeleteVertexArrays,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glVertexAttribDivisor,
    glVertexAttribIPointer,
    glVertexAttribPointer
)

from core.assertions import engine_assert
from core.logger import Logger

from graphics.mesh_data import (
    COLOR_SIZE,
    FLOATS_PER_VERTEX,
    NORMAL_SIZE,
    POSITION_SIZE,
    TANGENT_SIZE,
    UV_SIZE
)
from graphics.range_allocator import RangeAllocator


# Vertex attribute holding the batched draw's index into the
# per-draw storage buffer (see Renderer.prepare_draws and
# assets/shaders/include/draw_data.glsl).
DRAW_INDEX_LOCATION = 5


@dataclass(frozen=True, slots=True)
class PoolAllocation:

    base_vertex: int
    vertex_count: int

    first_index: int
    index_count: int


class GeometryPool:

    # =====================================================
    # Shared Geometry
    # =====================================================
    #
    # Every mesh lives in one vertex buffer and one index
    # buffer behind a single VAO. Meshes are ranges in those
    # buffers, so any number of them can be drawn with one
    # glMultiDrawElementsIndirect call (each command picks
    # its range via firstIndex / baseVertex) and nothing is
    # rebound between objects.
    #
    # Buffers start small and double (copying on the GPU)
    # when an allocation does not fit; freed ranges are
    # reused (RangeAllocator), which matters once terrain
    # chunks stream in and out.
    #
    # One pool exists per GL context: GeometryPool.instance().

    INITIAL_VERTICES = 64 * 1024
    INITIAL_INDICES = 256 * 1024
    INITIAL_DRAWS = 1024

    _instance: "GeometryPool | None" = None

    @classmethod
    def instance(
        cls
    ) -> "GeometryPool":

        if cls._instance is None:
            cls._instance = cls()

        return cls._instance

    @classmethod
    def shutdown(
        cls
    ):

        if cls._instance is not None:

            cls._instance._delete()

            cls._instance = None

    # =====================================================
    # Construction
    # =====================================================

    def __init__(self):

        self.vertex_array = int(glGenVertexArrays(1))

        self._vertices = RangeAllocator(self.INITIAL_VERTICES)
        self._indices = RangeAllocator(self.INITIAL_INDICES)

        self.vertex_buffer = self._create_buffer(
            GL_ARRAY_BUFFER,
            self.INITIAL_VERTICES * self.vertex_stride
        )

        self.index_buffer = self._create_buffer(
            GL_ELEMENT_ARRAY_BUFFER,
            self.INITIAL_INDICES * 4
        )

        self._draw_index_capacity = 0
        self.draw_index_buffer = int(glGenBuffers(1))

        self._configure_vertex_array()

        self.ensure_draw_capacity(
            self.INITIAL_DRAWS
        )

        Logger.debug(
            "[GeometryPool] Created (%d vertices, %d indices).",
            self.INITIAL_VERTICES,
            self.INITIAL_INDICES
        )

    @property
    def vertex_stride(
        self
    ) -> int:

        return FLOATS_PER_VERTEX * 4

    # =====================================================
    # Allocation
    # =====================================================

    def allocate(
        self,
        vertices: np.ndarray,
        indices: np.ndarray
    ) -> PoolAllocation:

        vertices = np.ascontiguousarray(
            vertices,
            dtype=np.float32
        ).reshape(-1, FLOATS_PER_VERTEX)

        indices = np.ascontiguousarray(
            indices,
            dtype=np.uint32
        ).reshape(-1)

        engine_assert(
            len(vertices) > 0 and len(indices) > 0,
            "GeometryPool allocations need vertices and indices."
        )

        base_vertex = self._allocate_range(
            self._vertices,
            len(vertices),
            self._grow_vertices
        )

        first_index = self._allocate_range(
            self._indices,
            len(indices),
            self._grow_indices
        )

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)

        glBufferSubData(
            GL_ARRAY_BUFFER,
            base_vertex * self.vertex_stride,
            vertices.nbytes,
            vertices
        )

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Indices stay mesh-local; commands add baseVertex.
        # Uploaded through the generic copy target: the
        # element-array binding belongs to a VAO, and touching
        # it with no VAO bound is an error on some drivers.

        glBindBuffer(GL_COPY_WRITE_BUFFER, self.index_buffer)

        glBufferSubData(
            GL_COPY_WRITE_BUFFER,
            first_index * 4,
            indices.nbytes,
            indices
        )

        glBindBuffer(GL_COPY_WRITE_BUFFER, 0)

        return PoolAllocation(
            base_vertex=base_vertex,
            vertex_count=len(vertices),
            first_index=first_index,
            index_count=len(indices)
        )

    def free(
        self,
        allocation: PoolAllocation
    ):

        self._vertices.free(
            allocation.base_vertex,
            allocation.vertex_count
        )

        self._indices.free(
            allocation.first_index,
            allocation.index_count
        )

    @staticmethod
    def _allocate_range(
        allocator: RangeAllocator,
        size: int,
        grow
    ) -> int:

        start = allocator.allocate(size)

        if start is None:

            capacity = max(allocator.capacity, 1)

            while capacity - allocator.used < size or capacity < allocator.capacity + size:
                capacity *= 2

            grow(capacity)

            start = allocator.allocate(size)

        engine_assert(
            start is not None,
            "GeometryPool failed to allocate after growing."
        )

        return start

    # =====================================================
    # Growth
    # =====================================================

    def _grow_vertices(
        self,
        capacity: int
    ):

        self.vertex_buffer = self._grow_buffer(
            self.vertex_buffer,
            GL_ARRAY_BUFFER,
            self._vertices.capacity * self.vertex_stride,
            capacity * self.vertex_stride
        )

        self._vertices.grow(capacity)

        self._configure_vertex_array()

        Logger.info("[GeometryPool] Vertex capacity -> %d.", capacity)

    def _grow_indices(
        self,
        capacity: int
    ):

        self.index_buffer = self._grow_buffer(
            self.index_buffer,
            GL_ELEMENT_ARRAY_BUFFER,
            self._indices.capacity * 4,
            capacity * 4
        )

        self._indices.grow(capacity)

        self._configure_vertex_array()

        Logger.info("[GeometryPool] Index capacity -> %d.", capacity)

    @staticmethod
    def _create_buffer(
        target: int,
        size_bytes: int
    ) -> int:

        # `target` documents the buffer's use; allocation goes
        # through the generic copy target (see allocate()).

        buffer = int(glGenBuffers(1))

        glBindBuffer(GL_COPY_WRITE_BUFFER, buffer)
        glBufferData(GL_COPY_WRITE_BUFFER, size_bytes, None, GL_STATIC_DRAW)
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0)

        return buffer

    def _grow_buffer(
        self,
        old_buffer: int,
        target: int,
        old_bytes: int,
        new_bytes: int
    ) -> int:

        # GPU-side copy; no round trip through Python.

        new_buffer = self._create_buffer(target, new_bytes)

        glBindBuffer(GL_COPY_READ_BUFFER, old_buffer)
        glBindBuffer(GL_COPY_WRITE_BUFFER, new_buffer)

        glCopyBufferSubData(
            GL_COPY_READ_BUFFER,
            GL_COPY_WRITE_BUFFER,
            0,
            0,
            old_bytes
        )

        glBindBuffer(GL_COPY_READ_BUFFER, 0)
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0)

        glDeleteBuffers(1, [old_buffer])

        return new_buffer

    # =====================================================
    # Vertex Layout
    # =====================================================

    def _configure_vertex_array(self):

        # Matches graphics/mesh_data.py and the shaders'
        # layout(location = N) inputs.

        glBindVertexArray(self.vertex_array)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)

        offset = 0

        for location, size in enumerate(
            (POSITION_SIZE, NORMAL_SIZE, COLOR_SIZE, UV_SIZE, TANGENT_SIZE)
        ):

            glEnableVertexAttribArray(location)

            glVertexAttribPointer(
                location,
                size,
                GL_FLOAT,
                GL_FALSE,
                self.vertex_stride,
                ctypes.c_void_p(offset * 4)
            )

            offset += size

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)

        self._configure_draw_index_attribute()

        glBindVertexArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _configure_draw_index_attribute(self):

        # One value per *instance* (divisor 1). Indirect
        # commands set baseInstance = draw index, so instance
        # 0 of command k reads draw_index_buffer[k] = k: each
        # draw finds its own per-draw data without needing
        # gl_DrawID (OpenGL 4.6).

        glBindBuffer(GL_ARRAY_BUFFER, self.draw_index_buffer)

        glEnableVertexAttribArray(DRAW_INDEX_LOCATION)

        glVertexAttribIPointer(
            DRAW_INDEX_LOCATION,
            1,
            GL_UNSIGNED_INT,
            4,
            ctypes.c_void_p(0)
        )

        glVertexAttribDivisor(DRAW_INDEX_LOCATION, 1)

    def ensure_draw_capacity(
        self,
        draws: int
    ):
        """Make the draw-index attribute cover `draws` draws."""

        if draws <= self._draw_index_capacity:
            return

        capacity = max(self.INITIAL_DRAWS, self._draw_index_capacity)

        while capacity < draws:
            capacity *= 2

        values = np.arange(capacity, dtype=np.uint32)

        glBindBuffer(GL_ARRAY_BUFFER, self.draw_index_buffer)
        glBufferData(GL_ARRAY_BUFFER, values.nbytes, values, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        self._draw_index_capacity = capacity

    # =====================================================
    # Binding / Stats
    # =====================================================

    def bind(self):

        glBindVertexArray(self.vertex_array)

    @property
    def stats(
        self
    ) -> dict[str, int]:

        return {
            "vertices_used": self._vertices.used,
            "vertex_capacity": self._vertices.capacity,
            "indices_used": self._indices.used,
            "index_capacity": self._indices.capacity,
        }

    def _delete(self):

        glDeleteBuffers(
            3,
            [self.vertex_buffer, self.index_buffer, self.draw_index_buffer]
        )

        glDeleteVertexArrays(1, [self.vertex_array])

        Logger.debug("[GeometryPool] Deleted.")
