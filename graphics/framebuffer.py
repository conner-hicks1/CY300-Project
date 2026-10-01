from dataclasses import dataclass
from enum import Enum

import numpy as np

from OpenGL.GL import (
    GL_CLAMP_TO_BORDER,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_ATTACHMENT0,
    GL_DEPTH32F_STENCIL8,
    GL_DEPTH_ATTACHMENT,
    GL_DEPTH_COMPONENT,
    GL_DEPTH_COMPONENT24,
    GL_DEPTH_STENCIL_ATTACHMENT,
    GL_FLOAT,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_COMPLETE,
    GL_LINEAR,
    GL_NEAREST,
    GL_NONE,
    GL_RENDERBUFFER,
    GL_RGBA,
    GL_RGBA16F,
    GL_RGBA8,
    GL_TEXTURE_2D,
    GL_TEXTURE_BORDER_COLOR,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNSIGNED_BYTE,
    glBindFramebuffer,
    glBindRenderbuffer,
    glBindTexture,
    glCheckFramebufferStatus,
    glDeleteFramebuffers,
    glDeleteRenderbuffers,
    glDeleteTextures,
    glDrawBuffer,
    glFramebufferRenderbuffer,
    glFramebufferTexture2D,
    glGenFramebuffers,
    glGenRenderbuffers,
    glGenTextures,
    glReadBuffer,
    glRenderbufferStorage,
    glTexImage2D,
    glTexParameterfv,
    glTexParameteri
)

from core.assertions import engine_assert
from core.exceptions import GraphicsError
from core.logger import Logger

from graphics.render_command import RenderCommand


# =========================================================
# Attachment Formats
# =========================================================

class ColorFormat(Enum):

    # 8-bit per channel; values clamp to [0, 1].
    RGBA8 = "rgba8"

    # Half-float; stores HDR values > 1 for tone mapping.
    RGBA16F = "rgba16f"


class DepthMode(Enum):

    NONE = "none"

    # Depth/stencil the renderer can test against but
    # never sample. Cheaper than a texture.
    RENDERBUFFER = "renderbuffer"

    # Sampleable depth texture (shadow maps). Outside the
    # texture reads as depth 1.0 ("fully lit").
    TEXTURE = "texture"


@dataclass(slots=True)
class FramebufferSpec:

    width: int
    height: int

    color_format: ColorFormat | None = ColorFormat.RGBA8
    depth_mode: DepthMode = DepthMode.RENDERBUFFER


# =========================================================
# Framebuffer
# =========================================================

class Framebuffer:

    def __init__(
        self,
        spec: FramebufferSpec
    ):

        engine_assert(
            spec.color_format is not None
            or spec.depth_mode != DepthMode.NONE,
            "Framebuffer needs a color or depth attachment."
        )

        self.spec = spec

        self.id = 0
        self.color_texture_id = 0
        self.depth_texture_id = 0
        self.depth_renderbuffer_id = 0

        self._create()

    # =====================================================
    # Properties
    # =====================================================

    @property
    def width(
        self
    ) -> int:

        return self.spec.width

    @property
    def height(
        self
    ) -> int:

        return self.spec.height

    # =====================================================
    # Binding
    # =====================================================

    def bind(self):
        """
        Bind for rendering and set the viewport to cover
        the whole framebuffer.
        """

        engine_assert(
            self.id != 0,
            "Cannot bind a deleted Framebuffer."
        )

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            self.id
        )

        RenderCommand.set_viewport(
            0,
            0,
            self.spec.width,
            self.spec.height
        )

    @staticmethod
    def bind_default(
        width: int,
        height: int
    ):
        """
        Bind the window's framebuffer.
        """

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            0
        )

        RenderCommand.set_viewport(
            0,
            0,
            width,
            height
        )

    # =====================================================
    # Resize
    # =====================================================

    def resize(
        self,
        width: int,
        height: int
    ):

        engine_assert(
            width > 0 and height > 0,
            "Framebuffer size must be positive."
        )

        if (
            width == self.spec.width
            and height == self.spec.height
        ):
            return

        self._destroy()

        self.spec.width = width
        self.spec.height = height

        self._create()

    # =====================================================
    # Creation
    # =====================================================

    def _create(self):

        width = self.spec.width
        height = self.spec.height

        engine_assert(
            width > 0 and height > 0,
            "Framebuffer size must be positive."
        )

        self.id = glGenFramebuffers(
            1
        )

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            self.id
        )

        try:

            # ---------------------------------------------
            # Color
            # ---------------------------------------------

            if self.spec.color_format is not None:

                self.color_texture_id = self._create_color_texture(
                    width,
                    height,
                    self.spec.color_format
                )

                glFramebufferTexture2D(
                    GL_FRAMEBUFFER,
                    GL_COLOR_ATTACHMENT0,
                    GL_TEXTURE_2D,
                    self.color_texture_id,
                    0
                )

            else:

                # Depth-only (shadow map).

                glDrawBuffer(
                    GL_NONE
                )

                glReadBuffer(
                    GL_NONE
                )

            # ---------------------------------------------
            # Depth
            # ---------------------------------------------

            if self.spec.depth_mode == DepthMode.RENDERBUFFER:

                self.depth_renderbuffer_id = glGenRenderbuffers(
                    1
                )

                glBindRenderbuffer(
                    GL_RENDERBUFFER,
                    self.depth_renderbuffer_id
                )

                glRenderbufferStorage(
                    GL_RENDERBUFFER,
                    GL_DEPTH32F_STENCIL8,
                    width,
                    height
                )

                glBindRenderbuffer(
                    GL_RENDERBUFFER,
                    0
                )

                glFramebufferRenderbuffer(
                    GL_FRAMEBUFFER,
                    GL_DEPTH_STENCIL_ATTACHMENT,
                    GL_RENDERBUFFER,
                    self.depth_renderbuffer_id
                )

            elif self.spec.depth_mode == DepthMode.TEXTURE:

                self.depth_texture_id = self._create_depth_texture(
                    width,
                    height
                )

                glFramebufferTexture2D(
                    GL_FRAMEBUFFER,
                    GL_DEPTH_ATTACHMENT,
                    GL_TEXTURE_2D,
                    self.depth_texture_id,
                    0
                )

            # ---------------------------------------------
            # Validate
            # ---------------------------------------------

            status = glCheckFramebufferStatus(
                GL_FRAMEBUFFER
            )

            if status != GL_FRAMEBUFFER_COMPLETE:

                raise GraphicsError(
                    f"Framebuffer incomplete (status 0x{int(status):X}) "
                    f"for spec {self.spec}."
                )

        except Exception:

            glBindFramebuffer(
                GL_FRAMEBUFFER,
                0
            )

            self._destroy()

            raise

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            0
        )

        Logger.debug(
            "[Framebuffer] Created ID=%d %dx%d (color=%s, depth=%s).",
            self.id,
            width,
            height,
            self.spec.color_format,
            self.spec.depth_mode
        )

    @staticmethod
    def _create_color_texture(
        width: int,
        height: int,
        color_format: ColorFormat
    ) -> int:

        texture = glGenTextures(
            1
        )

        glBindTexture(
            GL_TEXTURE_2D,
            texture
        )

        if color_format == ColorFormat.RGBA16F:

            internal_format = GL_RGBA16F
            data_type = GL_FLOAT

        else:

            internal_format = GL_RGBA8
            data_type = GL_UNSIGNED_BYTE

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            internal_format,
            width,
            height,
            0,
            GL_RGBA,
            data_type,
            None
        )

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        glBindTexture(
            GL_TEXTURE_2D,
            0
        )

        return texture

    @staticmethod
    def _create_depth_texture(
        width: int,
        height: int
    ) -> int:

        texture = glGenTextures(
            1
        )

        glBindTexture(
            GL_TEXTURE_2D,
            texture
        )

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_DEPTH_COMPONENT24,
            width,
            height,
            0,
            GL_DEPTH_COMPONENT,
            GL_FLOAT,
            None
        )

        # Nearest: the shader does its own PCF filtering.

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        # Samples outside the shadow map read as the far
        # plane, i.e. "not in shadow".

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)

        glTexParameterfv(
            GL_TEXTURE_2D,
            GL_TEXTURE_BORDER_COLOR,
            np.array(
                [1.0, 1.0, 1.0, 1.0],
                dtype=np.float32
            )
        )

        glBindTexture(
            GL_TEXTURE_2D,
            0
        )

        return texture

    # =====================================================
    # Destruction
    # =====================================================

    def _destroy(self):

        if self.color_texture_id:

            glDeleteTextures(
                1,
                [self.color_texture_id]
            )

            self.color_texture_id = 0

        if self.depth_texture_id:

            glDeleteTextures(
                1,
                [self.depth_texture_id]
            )

            self.depth_texture_id = 0

        if self.depth_renderbuffer_id:

            glDeleteRenderbuffers(
                1,
                [self.depth_renderbuffer_id]
            )

            self.depth_renderbuffer_id = 0

        if self.id:

            glDeleteFramebuffers(
                1,
                [self.id]
            )

            self.id = 0

    def delete(self):

        if self.id == 0:
            return

        Logger.debug(
            "[Framebuffer] Deleting ID=%d.",
            self.id
        )

        self._destroy()
