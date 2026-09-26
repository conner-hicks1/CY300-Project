from pathlib import Path

from PIL import Image

from OpenGL.GL import (
    GL_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR,
    GL_REPEAT,
    GL_RGBA,
    GL_RGBA8,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNPACK_ALIGNMENT,
    GL_UNSIGNED_BYTE,
    glBindTexture,
    glDeleteTextures,
    glGenTextures,
    glGenerateMipmap,
    glPixelStorei,
    glTexImage2D,
    glTexParameteri
)

from core.assertions import engine_assert
from core.logger import Logger

from graphics.render_state import RenderState


class Texture2D:

    # =====================================================
    # Construction
    # =====================================================

    def __init__(
        self,
        filepath: str | Path,
        *,
        generate_mipmaps: bool = True,
        flip_vertical: bool = True
    ):

        self.id = 0

        self.filepath = Path(
            filepath
        ).resolve()

        self.width = 0
        self.height = 0
        self.channels = 4

        self._generate_mipmaps = (
            generate_mipmaps
        )

        Logger.debug(
            "[Texture2D] Loading '%s'.",
            self.filepath
        )

        # -------------------------------------------------
        # Validate File
        # -------------------------------------------------

        engine_assert(
            self.filepath.is_file(),
            (
                "Texture file does not exist: "
                f"{self.filepath}"
            )
        )

        # -------------------------------------------------
        # Load CPU Image
        # -------------------------------------------------

        try:

            with Image.open(
                self.filepath
            ) as image:

                if flip_vertical:

                    image = image.transpose(
                        Image.Transpose.FLIP_TOP_BOTTOM
                    )

                # Normalize all initial textures to RGBA8.

                image = image.convert(
                    "RGBA"
                )

                self.width = image.width
                self.height = image.height

                pixel_data = image.tobytes()

        except Exception:

            Logger.exception(
                "[Texture2D] Failed to load '%s'.",
                self.filepath
            )

            raise

        engine_assert(
            self.width > 0
            and self.height > 0,
            (
                "Texture dimensions must be positive: "
                f"{self.filepath}"
            )
        )

        # -------------------------------------------------
        # Create GPU Texture
        # -------------------------------------------------

        try:

            self.id = glGenTextures(
                1
            )

            engine_assert(
                self.id != 0,
                (
                    "OpenGL failed to create texture "
                    f"for '{self.filepath}'."
                )
            )

            # ---------------------------------------------
            # Bind
            # ---------------------------------------------

            self.bind(
                0
            )

            # ---------------------------------------------
            # Pixel Storage
            # ---------------------------------------------

            glPixelStorei(
                GL_UNPACK_ALIGNMENT,
                1
            )

            # ---------------------------------------------
            # Wrapping
            # ---------------------------------------------

            glTexParameteri(
                GL_TEXTURE_2D,
                GL_TEXTURE_WRAP_S,
                GL_REPEAT
            )

            glTexParameteri(
                GL_TEXTURE_2D,
                GL_TEXTURE_WRAP_T,
                GL_REPEAT
            )

            # ---------------------------------------------
            # Filtering
            # ---------------------------------------------

            min_filter = (
                GL_LINEAR_MIPMAP_LINEAR
                if generate_mipmaps
                else GL_LINEAR
            )

            glTexParameteri(
                GL_TEXTURE_2D,
                GL_TEXTURE_MIN_FILTER,
                min_filter
            )

            glTexParameteri(
                GL_TEXTURE_2D,
                GL_TEXTURE_MAG_FILTER,
                GL_LINEAR
            )

            # ---------------------------------------------
            # Upload
            # ---------------------------------------------

            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA8,
                self.width,
                self.height,
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                pixel_data
            )

            # ---------------------------------------------
            # Mipmaps
            # ---------------------------------------------

            if generate_mipmaps:

                glGenerateMipmap(
                    GL_TEXTURE_2D
                )

            # ---------------------------------------------
            # Unbind
            # ---------------------------------------------

            self.unbind()

        except Exception:

            # If construction fails after glGenTextures(),
            # release the partially constructed GPU object.

            if self.id != 0:

                glDeleteTextures(
                    1,
                    [self.id]
                )

                self.id = 0

            Logger.exception(
                "[Texture2D] GPU creation failed for '%s'.",
                self.filepath
            )

            raise

        Logger.info(
            "[Texture2D] Loaded '%s' (%dx%d, ID=%d).",
            self.filepath.name,
            self.width,
            self.height,
            self.id
        )

    # =====================================================
    # Binding
    # =====================================================

    def bind(
        self,
        slot: int = 0
    ):

        engine_assert(
            self.id != 0,
            "Cannot bind a deleted Texture2D."
        )

        engine_assert(
            isinstance(slot, int),
            "Texture slot must be an integer."
        )

        engine_assert(
            slot >= 0,
            "Texture slot cannot be negative."
        )

        RenderState.set_active_texture_slot(
            slot
        )

        glBindTexture(
            GL_TEXTURE_2D,
            self.id
        )

    # =====================================================
    # Unbinding
    # =====================================================

    @staticmethod
    def unbind():

        glBindTexture(
            GL_TEXTURE_2D,
            0
        )

    # =====================================================
    # Destruction
    # =====================================================

    def delete(self):

        if self.id == 0:
            return

        Logger.debug(
            "[Texture2D] Deleting texture ID %d.",
            self.id
        )

        glDeleteTextures(
            1,
            [self.id]
        )

        self.id = 0

    # =====================================================
    # State
    # =====================================================

    @property
    def is_valid(self) -> bool:

        return self.id != 0

    # =====================================================
    # Representation
    # =====================================================

    def __repr__(self):

        return (
            f"Texture2D("
            f"id={self.id}, "
            f"size={self.width}x{self.height}, "
            f"path='{self.filepath}'"
            f")"
        )