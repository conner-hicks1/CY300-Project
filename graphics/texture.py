import io

from pathlib import Path

from PIL import Image

from OpenGL.GL import (
    GL_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR,
    GL_REPEAT,
    GL_RGBA,
    GL_RGBA8,
    GL_SRGB8_ALPHA8,
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
    #
    # srgb:
    #     True for color data authored by artists (albedo /
    #     base color). The GPU converts sRGB -> linear on
    #     sample, which lighting math requires.
    #
    #     False for non-color data (normal maps, specular /
    #     roughness maps, masks), which are already linear.

    def __init__(
        self,
        filepath: str | Path,
        *,
        srgb: bool = True,
        generate_mipmaps: bool = True,
        flip_vertical: bool = True
    ):

        self.id = 0

        self.filepath: Path | None = Path(
            filepath
        ).resolve()

        self.width = 0
        self.height = 0
        self.channels = 4

        self.srgb = srgb

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

        self._upload(
            pixel_data
        )

        Logger.info(
            "[Texture2D] Loaded '%s' (%dx%d, %s, ID=%d).",
            self.filepath.name,
            self.width,
            self.height,
            "sRGB" if self.srgb else "linear",
            self.id
        )

    # =====================================================
    # From Pixels
    # =====================================================

    @classmethod
    def from_pixels(
        cls,
        width: int,
        height: int,
        rgba_bytes: bytes,
        *,
        srgb: bool,
        generate_mipmaps: bool = False,
        label: str = "<pixels>"
    ) -> "Texture2D":
        """
        Create a texture from raw RGBA8 bytes, e.g. the
        1x1 default textures.
        """

        engine_assert(
            width > 0 and height > 0,
            "Texture dimensions must be positive."
        )

        engine_assert(
            len(rgba_bytes) == width * height * 4,
            "Texture pixel data must be width * height * 4 bytes."
        )

        texture = cls.__new__(
            cls
        )

        texture.id = 0
        texture.filepath = None
        texture.width = width
        texture.height = height
        texture.channels = 4
        texture.srgb = srgb
        texture._generate_mipmaps = generate_mipmaps

        texture._upload(
            bytes(rgba_bytes)
        )

        Logger.debug(
            "[Texture2D] Created '%s' (%dx%d, ID=%d).",
            label,
            width,
            height,
            texture.id
        )

        return texture

    @classmethod
    def from_encoded(
        cls,
        encoded: bytes,
        *,
        srgb: bool,
        label: str = "<encoded>",
        flip_vertical: bool = True
    ) -> "Texture2D":
        """
        Decode a PNG/JPEG held in memory (e.g. an image
        embedded in a .glb file).
        """

        with Image.open(io.BytesIO(encoded)) as image:

            if flip_vertical:

                image = image.transpose(
                    Image.Transpose.FLIP_TOP_BOTTOM
                )

            image = image.convert("RGBA")

            return cls.from_pixels(
                image.width,
                image.height,
                image.tobytes(),
                srgb=srgb,
                generate_mipmaps=True,
                label=label
            )

    @classmethod
    def solid_color(
        cls,
        rgba: tuple[int, int, int, int],
        *,
        srgb: bool
    ) -> "Texture2D":

        engine_assert(
            len(rgba) == 4
            and all(0 <= c <= 255 for c in rgba),
            "Solid color must be four 0-255 values."
        )

        return cls.from_pixels(
            1,
            1,
            bytes(rgba),
            srgb=srgb,
            label=f"solid{tuple(rgba)}"
        )

    # =====================================================
    # GPU Upload
    # =====================================================

    def _upload(
        self,
        pixel_data: bytes
    ):

        engine_assert(
            self.width > 0
            and self.height > 0,
            (
                "Texture dimensions must be positive: "
                f"{self.filepath}"
            )
        )

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
                if self._generate_mipmaps
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
                (
                    GL_SRGB8_ALPHA8
                    if self.srgb
                    else GL_RGBA8
                ),
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

            if self._generate_mipmaps:

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
            f"srgb={self.srgb}, "
            f"path='{self.filepath}'"
            f")"
        )
