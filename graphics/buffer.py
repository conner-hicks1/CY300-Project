import numpy as np

from OpenGL.GL import (
    glGenBuffers,
    glBindBuffer,
    glBufferData,
    glDeleteBuffers,

    GL_ARRAY_BUFFER,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_STATIC_DRAW
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