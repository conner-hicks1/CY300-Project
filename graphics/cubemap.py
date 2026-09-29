import numpy as np

from OpenGL.GL import (
    GL_CLAMP_TO_EDGE,
    GL_COLOR_ATTACHMENT0,
    GL_FLOAT,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_COMPLETE,
    GL_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR,
    GL_RGBA,
    GL_RGBA16F,
    GL_TEXTURE_CUBE_MAP,
    GL_TEXTURE_CUBE_MAP_POSITIVE_X,
    GL_TEXTURE_CUBE_MAP_SEAMLESS,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MAX_LEVEL,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_R,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    glBindFramebuffer,
    glBindTexture,
    glCheckFramebufferStatus,
    glDeleteFramebuffers,
    glDeleteTextures,
    glEnable,
    glFramebufferTexture2D,
    glGenFramebuffers,
    glGenTextures,
    glGenerateMipmap,
    glTexImage2D,
    glTexParameteri
)

from core.assertions import engine_assert
from core.exceptions import GraphicsError
from core.logger import Logger

from graphics.render_command import RenderCommand


# =========================================================
# Cube Face Directions
# =========================================================
#
# When rendering into cube face `f` with a fullscreen
# triangle, each fragment must know which world direction
# it represents. For window position (u, v) in [-1, 1]
# (v up, i.e. row 0 at the bottom) the OpenGL spec's face
# selection table (sc, tc, ma) inverts to:
#
#     +X: ( 1, -v, -u)     -X: (-1, -v,  u)
#     +Y: ( u,  1,  v)     -Y: ( u, -1, -v)
#     +Z: ( u, -v,  1)     -Z: (-u, -v, -1)
#
# assets/shaders/include/cubemap.glsl implements the same
# table; tests check this copy against the spec.

def cube_face_direction(
    face: int,
    u: float,
    v: float
) -> np.ndarray:

    table = (
        (1.0, -v, -u),
        (-1.0, -v, u),
        (u, 1.0, v),
        (u, -1.0, -v),
        (u, -v, 1.0),
        (-u, -v, -1.0),
    )

    direction = np.array(
        table[face],
        dtype=np.float64
    )

    return direction / np.linalg.norm(direction)


# =========================================================
# Cubemap
# =========================================================

class Cubemap:

    def __init__(
        self,
        size: int,
        mip_levels: int = 1,
        label: str = "cubemap"
    ):

        engine_assert(
            size > 0 and mip_levels >= 1,
            "Cubemap size and mip levels must be positive."
        )

        self.size = size
        self.mip_levels = mip_levels
        self.label = label

        # Filtering across face edges (core since GL 3.2).
        glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS)

        self.id = glGenTextures(1)

        glBindTexture(GL_TEXTURE_CUBE_MAP, self.id)

        for level in range(mip_levels):

            level_size = max(1, size >> level)

            for face in range(6):

                glTexImage2D(
                    GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
                    level,
                    GL_RGBA16F,
                    level_size,
                    level_size,
                    0,
                    GL_RGBA,
                    GL_FLOAT,
                    None
                )

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

        glTexParameteri(
            GL_TEXTURE_CUBE_MAP,
            GL_TEXTURE_MIN_FILTER,
            GL_LINEAR_MIPMAP_LINEAR if mip_levels > 1 else GL_LINEAR
        )

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, mip_levels - 1)

        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)

        Logger.debug(
            "[Cubemap] Created '%s' %d^2 x6, %d mip(s), ID=%d.",
            label,
            size,
            mip_levels,
            self.id
        )

    def level_size(
        self,
        level: int
    ) -> int:

        return max(1, self.size >> level)

    def generate_mipmaps(self):

        glBindTexture(GL_TEXTURE_CUBE_MAP, self.id)
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)

    def bind(
        self,
        slot: int
    ):

        RenderCommand.bind_texture(
            self.id,
            slot,
            GL_TEXTURE_CUBE_MAP
        )

    def delete(self):

        if self.id:

            glDeleteTextures(1, [self.id])

            self.id = 0


# =========================================================
# Capture Target
# =========================================================

class CubemapCaptureTarget:
    """
    A framebuffer for rendering into one cubemap face/mip
    at a time:

        target.bind_face(cubemap, face, level)
        renderer.draw_fullscreen(shader)    # uFace = face
    """

    def __init__(self):

        self.id = glGenFramebuffers(1)

    def bind_face(
        self,
        cubemap: Cubemap,
        face: int,
        level: int = 0
    ):

        glBindFramebuffer(GL_FRAMEBUFFER, self.id)

        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
            cubemap.id,
            level
        )

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)

        if status != GL_FRAMEBUFFER_COMPLETE:

            raise GraphicsError(
                f"Cubemap capture target incomplete "
                f"(status 0x{int(status):X}) for '{cubemap.label}'."
            )

        size = cubemap.level_size(level)

        RenderCommand.set_viewport(0, 0, size, size)

    def delete(self):

        if self.id:

            glDeleteFramebuffers(1, [self.id])

            self.id = 0
