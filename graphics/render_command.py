from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_TEXTURE_2D,
    GL_TRIANGLES,
    GL_UNSIGNED_INT,
    glBindTexture,
    glClear,
    glClearColor,
    glDrawArrays,
    glDrawElements,
    glViewport
)

from core.assertions import engine_assert

from graphics.render_state import RenderState


class RenderCommand:

    # =====================================================
    # Clear
    # =====================================================

    @staticmethod
    def set_clear_color(
        color
    ):

        engine_assert(
            len(color) == 4,
            "Clear color must contain four values."
        )

        glClearColor(
            float(color[0]),
            float(color[1]),
            float(color[2]),
            float(color[3])
        )

    @staticmethod
    def clear(
        color: bool = True,
        depth: bool = True
    ):

        mask = 0

        if color:
            mask |= GL_COLOR_BUFFER_BIT

        if depth:
            mask |= GL_DEPTH_BUFFER_BIT

        engine_assert(
            mask != 0,
            "RenderCommand.clear() must clear something."
        )

        glClear(
            mask
        )

    # =====================================================
    # Indexed Drawing
    # =====================================================

    @staticmethod
    def draw_indexed(
        index_count: int
    ):

        engine_assert(
            index_count > 0,
            "Indexed draw count must be positive."
        )

        glDrawElements(
            GL_TRIANGLES,
            index_count,
            GL_UNSIGNED_INT,
            None
        )

    # =====================================================
    # Non-Indexed Drawing
    # =====================================================

    @staticmethod
    def draw_arrays(
        vertex_count: int
    ):

        engine_assert(
            vertex_count > 0,
            "Vertex draw count must be positive."
        )

        glDrawArrays(
            GL_TRIANGLES,
            0,
            vertex_count
        )

    # =====================================================
    # Textures
    # =====================================================

    @staticmethod
    def bind_texture(
        texture_id: int,
        slot: int,
        target: int = GL_TEXTURE_2D
    ):
        """
        Bind a raw GL texture (e.g. a framebuffer
        attachment) to a texture unit. `target` is
        GL_TEXTURE_2D, GL_TEXTURE_CUBE_MAP or
        GL_TEXTURE_2D_ARRAY.
        """

        RenderState.set_active_texture_slot(
            slot
        )

        glBindTexture(
            target,
            texture_id
        )

    # =====================================================
    # Viewport
    # =====================================================

    @staticmethod
    def set_viewport(
        x: int,
        y: int,
        width: int,
        height: int
    ):

        engine_assert(
            width >= 0,
            "Viewport width cannot be negative."
        )

        engine_assert(
            height >= 0,
            "Viewport height cannot be negative."
        )

        glViewport(
            x,
            y,
            width,
            height
        )
