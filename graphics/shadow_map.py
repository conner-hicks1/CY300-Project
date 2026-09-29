import numpy as np

from OpenGL.GL import (
    GL_CLAMP_TO_BORDER,
    GL_COMPARE_REF_TO_TEXTURE,
    GL_DEPTH_ATTACHMENT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_COMPONENT,
    GL_DEPTH_COMPONENT24,
    GL_FLOAT,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_COMPLETE,
    GL_LEQUAL,
    GL_LINEAR,
    GL_NONE,
    GL_TEXTURE_2D_ARRAY,
    GL_TEXTURE_BORDER_COLOR,
    GL_TEXTURE_COMPARE_FUNC,
    GL_TEXTURE_COMPARE_MODE,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    glBindFramebuffer,
    glBindTexture,
    glCheckFramebufferStatus,
    glClear,
    glDeleteFramebuffers,
    glDeleteTextures,
    glDrawBuffer,
    glFramebufferTextureLayer,
    glGenFramebuffers,
    glGenTextures,
    glReadBuffer,
    glTexImage3D,
    glTexParameterfv,
    glTexParameteri
)

from core.assertions import engine_assert
from core.exceptions import GraphicsError
from core.logger import Logger

from graphics.render_command import RenderCommand


class ShadowMapArray:

    # =====================================================
    # Layered Shadow Map
    # =====================================================
    #
    # A depth texture array: one layer per cascade (or per
    # spot light), all rendered through one framebuffer by
    # switching the attached layer.
    #
    # Sampled in GLSL as sampler2DArrayShadow: hardware
    # depth comparison with LINEAR filtering returns the
    # fraction of the 2x2 neighbourhood that is lit, i.e.
    # free bilinear PCF on every tap. Outside the map
    # (border) reads as fully lit.

    def __init__(
        self,
        size: int,
        layers: int,
        label: str = "shadow map"
    ):

        self.label = label

        self.size = 0
        self.layers = 0

        self.texture_id = 0
        self.framebuffer_id = glGenFramebuffers(1)

        self.resize(
            size,
            layers
        )

    # =====================================================
    # Storage
    # =====================================================

    def resize(
        self,
        size: int,
        layers: int
    ):

        engine_assert(
            size > 0 and layers > 0,
            "Shadow map size and layer count must be positive."
        )

        if size == self.size and layers == self.layers:
            return

        if self.texture_id:
            glDeleteTextures(1, [self.texture_id])

        self.size = size
        self.layers = layers

        self.texture_id = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture_id)

        glTexImage3D(
            GL_TEXTURE_2D_ARRAY,
            0,
            GL_DEPTH_COMPONENT24,
            size,
            size,
            layers,
            0,
            GL_DEPTH_COMPONENT,
            GL_FLOAT,
            None
        )

        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)

        glTexParameterfv(
            GL_TEXTURE_2D_ARRAY,
            GL_TEXTURE_BORDER_COLOR,
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        )

        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL)

        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

        # Validate with layer 0 attached.

        self.bind_layer(0)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        if status != GL_FRAMEBUFFER_COMPLETE:

            raise GraphicsError(
                f"Shadow map '{self.label}' framebuffer incomplete "
                f"(status 0x{int(status):X})."
            )

        Logger.debug(
            "[ShadowMapArray] '%s': %d^2 x %d layers, ID=%d.",
            self.label,
            size,
            layers,
            self.texture_id
        )

    # =====================================================
    # Rendering
    # =====================================================

    def bind_layer(
        self,
        layer: int
    ):
        """Render into `layer`; sets the viewport."""

        engine_assert(
            0 <= layer < self.layers,
            f"Shadow map layer {layer} out of range."
        )

        glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffer_id)

        glFramebufferTextureLayer(
            GL_FRAMEBUFFER,
            GL_DEPTH_ATTACHMENT,
            self.texture_id,
            0,
            layer
        )

        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)

        RenderCommand.set_viewport(0, 0, self.size, self.size)

    def clear_layer(
        self,
        layer: int
    ):

        self.bind_layer(layer)

        glClear(GL_DEPTH_BUFFER_BIT)

    def bind_texture(
        self,
        slot: int
    ):

        RenderCommand.bind_texture(
            self.texture_id,
            slot,
            GL_TEXTURE_2D_ARRAY
        )

    # =====================================================
    # Cleanup
    # =====================================================

    def delete(self):

        if self.texture_id:

            glDeleteTextures(1, [self.texture_id])

            self.texture_id = 0

        if self.framebuffer_id:

            glDeleteFramebuffers(1, [self.framebuffer_id])

            self.framebuffer_id = 0
