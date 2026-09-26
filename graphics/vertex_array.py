import ctypes

from OpenGL.GL import (
    glGenVertexArrays,
    glBindVertexArray,
    glDeleteVertexArrays,
    glEnableVertexAttribArray,
    glVertexAttribPointer
)

from core.assertions import engine_assert
from core.logger import Logger


class VertexArray:

    def __init__(self):

        self.id = glGenVertexArrays(
            1
        )

        engine_assert(
            self.id != 0,
            "OpenGL failed to create VertexArray."
        )

        Logger.debug(
            "[VertexArray] Created ID=%d.",
            self.id
        )

    def bind(self):

        engine_assert(
            self.id != 0,
            "Attempted to bind a deleted VertexArray."
        )

        glBindVertexArray(
            self.id
        )

    @staticmethod
    def unbind():

        glBindVertexArray(
            0
        )

    def add_buffer(
        self,
        vertex_buffer,
        layout
    ):

        engine_assert(
            vertex_buffer is not None,
            "VertexArray requires a VertexBuffer."
        )

        engine_assert(
            layout is not None,
            "VertexArray requires a VertexLayout."
        )

        engine_assert(
            len(layout.attributes) > 0,
            "VertexLayout must contain attributes."
        )

        self.bind()
        vertex_buffer.bind()

        for index, attribute in enumerate(
            layout.attributes
        ):

            glEnableVertexAttribArray(
                index
            )

            glVertexAttribPointer(
                index,
                attribute.count,
                attribute.gl_type,
                attribute.normalized,
                layout.stride,
                ctypes.c_void_p(
                    attribute.offset
                )
            )

    def delete(self):

        if self.id == 0:
            return

        Logger.debug(
            "[VertexArray] Deleting ID=%d.",
            self.id
        )

        glDeleteVertexArrays(
            1,
            [self.id]
        )

        self.id = 0