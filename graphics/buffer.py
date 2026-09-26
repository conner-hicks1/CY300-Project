import numpy as np

from OpenGL.GL import (
    glGenBuffers,
    glBindBuffer,
    glBindBufferBase,
    glBufferData,
    glBufferSubData,
    glDeleteBuffers,

    GL_ARRAY_BUFFER,
    GL_DYNAMIC_DRAW,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_STATIC_DRAW,
    GL_UNIFORM_BUFFER
)

from core.assertions import engine_assert
from core.logger import Logger


class VertexBuffer:

    def __init__(
        self,
        data
    ):

        self.data = np.asarray(
            data,
            dtype=np.float32
        )

        engine_assert(
            self.data.size > 0,
            "VertexBuffer cannot contain zero elements."
        )

        self.id = glGenBuffers(
            1
        )

        engine_assert(
            self.id != 0,
            "OpenGL failed to create VertexBuffer."
        )

        self.bind()

        glBufferData(
            GL_ARRAY_BUFFER,
            self.data.nbytes,
            self.data,
            GL_STATIC_DRAW
        )

        Logger.debug(
            "[VertexBuffer] Created ID=%d (%d bytes).",
            self.id,
            self.data.nbytes
        )

    def bind(self):

        engine_assert(
            self.id != 0,
            "Attempted to bind a deleted VertexBuffer."
        )

        glBindBuffer(
            GL_ARRAY_BUFFER,
            self.id
        )

    @staticmethod
    def unbind():

        glBindBuffer(
            GL_ARRAY_BUFFER,
            0
        )

    def delete(self):

        if self.id == 0:
            return

        Logger.debug(
            "[VertexBuffer] Deleting ID=%d.",
            self.id
        )

        glDeleteBuffers(
            1,
            [self.id]
        )

        self.id = 0


class IndexBuffer:

    def __init__(
        self,
        indices
    ):

        self.indices = np.asarray(
            indices,
            dtype=np.uint32
        )

        engine_assert(
            self.indices.size > 0,
            "IndexBuffer cannot contain zero indices."
        )

        self.count = len(
            self.indices
        )

        self.id = glGenBuffers(
            1
        )

        engine_assert(
            self.id != 0,
            "OpenGL failed to create IndexBuffer."
        )

        self.bind()

        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER,
            self.indices.nbytes,
            self.indices,
            GL_STATIC_DRAW
        )

        Logger.debug(
            "[IndexBuffer] Created ID=%d (%d indices).",
            self.id,
            self.count
        )

    def bind(self):

        engine_assert(
            self.id != 0,
            "Attempted to bind a deleted IndexBuffer."
        )

        glBindBuffer(
            GL_ELEMENT_ARRAY_BUFFER,
            self.id
        )

    @staticmethod
    def unbind():

        glBindBuffer(
            GL_ELEMENT_ARRAY_BUFFER,
            0
        )

    def delete(self):

        if self.id == 0:
            return

        Logger.debug(
            "[IndexBuffer] Deleting ID=%d.",
            self.id
        )

        glDeleteBuffers(
            1,
            [self.id]
        )

        self.id = 0


class UniformBuffer:

    # =====================================================
    # Uniform Buffer Object
    # =====================================================
    #
    # Fixed-size GPU buffer attached to a uniform-block
    # binding point. Every shader whose block is bound to
    # the same point (see Shader._bind_uniform_blocks)
    # reads this data.

    def __init__(
        self,
        size: int,
        binding: int
    ):

        engine_assert(
            size > 0,
            "UniformBuffer size must be positive."
        )

        engine_assert(
            binding >= 0,
            "UniformBuffer binding cannot be negative."
        )

        self.size = size
        self.binding = binding

        self.id = glGenBuffers(
            1
        )

        engine_assert(
            self.id != 0,
            "OpenGL failed to create UniformBuffer."
        )

        glBindBuffer(
            GL_UNIFORM_BUFFER,
            self.id
        )

        glBufferData(
            GL_UNIFORM_BUFFER,
            size,
            None,
            GL_DYNAMIC_DRAW
        )

        glBindBufferBase(
            GL_UNIFORM_BUFFER,
            binding,
            self.id
        )

        glBindBuffer(
            GL_UNIFORM_BUFFER,
            0
        )

        Logger.debug(
            "[UniformBuffer] Created ID=%d (%d bytes, binding %d).",
            self.id,
            size,
            binding
        )

    def set_data(
        self,
        data: bytes
    ):

        engine_assert(
            self.id != 0,
            "Cannot write to a deleted UniformBuffer."
        )

        engine_assert(
            len(data) == self.size,
            (
                f"UniformBuffer expects {self.size} bytes, "
                f"got {len(data)}."
            )
        )

        glBindBuffer(
            GL_UNIFORM_BUFFER,
            self.id
        )

        glBufferSubData(
            GL_UNIFORM_BUFFER,
            0,
            self.size,
            data
        )

        glBindBuffer(
            GL_UNIFORM_BUFFER,
            0
        )

    def delete(self):

        if self.id == 0:
            return

        Logger.debug(
            "[UniformBuffer] Deleting ID=%d.",
            self.id
        )

        glDeleteBuffers(
            1,
            [self.id]
        )

        self.id = 0