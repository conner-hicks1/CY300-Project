from OpenGL.GL import (
    GL_ONE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    glBlendFunc
)

from core.logger import Logger

from graphics.framebuffer import (
    ColorFormat,
    DepthMode,
    Framebuffer,
    FramebufferSpec
)
from graphics.render_command import RenderCommand
from graphics.render_state import RenderState


class Bloom:

    # =====================================================
    # Physically Based Bloom
    # =====================================================
    #
    # (Jimenez, "Next Generation Post Processing in Call of
    # Duty: Advanced Warfare", 2014.)
    #
    #   1. Downsample the HDR image through a chain of
    #      half-size mips (13-tap filter).
    #   2. Upsample back up the chain (3x3 tent),
    #      additively blending each level onto the next
    #      larger one.
    #
    # The result (mip 0, half resolution) is a wide, smooth
    # blur that the post pass blends in. No brightness
    # threshold: HDR highlights dominate the blur on their
    # own, so glow scales naturally with brightness.
    #
    # The caller disables depth testing and culling around
    # render().

    MAX_LEVELS = 6
    MIN_SIZE = 8

    def __init__(
        self,
        renderer,
        get_downsample_shader,
        get_upsample_shader
    ):
        """
        get_*_shader: callables returning the current Shader
        (looked up each frame so hot reload works).
        """

        self._renderer = renderer

        self._get_downsample = get_downsample_shader
        self._get_upsample = get_upsample_shader

        self._mips: list[Framebuffer] = []

        self._size = (0, 0)

    # =====================================================
    # Mip Chain
    # =====================================================

    def _ensure_mips(
        self,
        width: int,
        height: int
    ):

        if (width, height) == self._size:
            return

        self._delete_mips()

        mip_width = width
        mip_height = height

        for _ in range(self.MAX_LEVELS):

            mip_width //= 2
            mip_height //= 2

            if min(mip_width, mip_height) < self.MIN_SIZE:
                break

            self._mips.append(
                Framebuffer(
                    FramebufferSpec(
                        width=mip_width,
                        height=mip_height,
                        color_format=ColorFormat.RGBA16F,
                        depth_mode=DepthMode.NONE
                    )
                )
            )

        self._size = (width, height)

        Logger.debug(
            "[Bloom] %d mip level(s) for %dx%d.",
            len(self._mips),
            width,
            height
        )

    # =====================================================
    # Render
    # =====================================================

    def render(
        self,
        source_texture_id: int,
        width: int,
        height: int,
        radius: float
    ) -> int | None:
        """
        Blur `source_texture_id` (width x height HDR).
        Returns the texture holding the bloom, or None if
        the target is too small to have any mips.
        """

        self._ensure_mips(
            width,
            height
        )

        if not self._mips:
            return None

        # -------------------------------------------------
        # Downsample
        # -------------------------------------------------

        shader = self._get_downsample()

        shader.bind()
        shader.set_int("uSource", 0)

        source_id = source_texture_id
        source_size = (width, height)

        for level, mip in enumerate(self._mips):

            mip.bind()

            RenderCommand.bind_texture(source_id, 0)

            shader.set_vec2(
                "uSourceTexelSize",
                (1.0 / source_size[0], 1.0 / source_size[1])
            )

            shader.set_bool("uFirstPass", level == 0)

            self._renderer.draw_fullscreen(shader)

            source_id = mip.color_texture_id
            source_size = (mip.width, mip.height)

        # -------------------------------------------------
        # Upsample (additive)
        # -------------------------------------------------

        shader = self._get_upsample()

        shader.bind()
        shader.set_int("uSource", 0)
        shader.set_float("uRadius", radius)
        shader.set_float("uAspectRatio", width / height)

        RenderState.set_blending(True)

        glBlendFunc(GL_ONE, GL_ONE)

        for level in range(len(self._mips) - 1, 0, -1):

            target = self._mips[level - 1]

            target.bind()

            RenderCommand.bind_texture(
                self._mips[level].color_texture_id,
                0
            )

            self._renderer.draw_fullscreen(shader)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        RenderState.set_blending(False)

        return self._mips[0].color_texture_id

    # =====================================================
    # Cleanup
    # =====================================================

    def _delete_mips(self):

        for mip in self._mips:
            mip.delete()

        self._mips = []
        self._size = (0, 0)

    def delete(self):

        self._delete_mips()
