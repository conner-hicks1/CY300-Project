from OpenGL.GL import (
    GL_BLEND,
    GL_CULL_FACE,
    GL_DEPTH_TEST,
    GL_LESS,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    GL_TEXTURE0,
    glActiveTexture,
    glBlendFunc,
    glDepthFunc,
    glDisable,
    glEnable
)

from core.assertions import engine_assert
from core.logger import Logger


class RenderState:

    # =====================================================
    # Cached State
    # =====================================================

    _initialized = False

    _depth_test_enabled: bool | None = None
    _blending_enabled: bool | None = None
    _face_culling_enabled: bool | None = None

    _active_texture_slot: int | None = None

    # =====================================================
    # Initialization
    # =====================================================

    @classmethod
    def initialize(cls):

        if cls._initialized:
            return

        Logger.debug(
            "[RenderState] Initializing."
        )

        # -------------------------------------------------
        # Invalidate Cache
        # -------------------------------------------------
        #
        # None means:
        #
        # "We do not currently know what OpenGL's state is."
        #
        # This forces the following calls to actually submit
        # state to OpenGL.

        cls.invalidate()

        # -------------------------------------------------
        # Depth Testing
        # -------------------------------------------------

        cls.set_depth_test(
            True
        )

        glDepthFunc(
            GL_LESS
        )

        # -------------------------------------------------
        # Blending
        # -------------------------------------------------

        cls.set_blending(
            False
        )

        glBlendFunc(
            GL_SRC_ALPHA,
            GL_ONE_MINUS_SRC_ALPHA
        )

        # -------------------------------------------------
        # Face Culling
        # -------------------------------------------------
        #
        # Keep this disabled until all mesh winding is known
        # to be consistently counter-clockwise.

        cls.set_face_culling(
            False
        )

        # -------------------------------------------------
        # Texture Unit
        # -------------------------------------------------

        cls.set_active_texture_slot(
            0
        )

        cls._initialized = True

        Logger.debug(
            "[RenderState] Initialized."
        )

    # =====================================================
    # Depth Testing
    # =====================================================

    @classmethod
    def set_depth_test(
        cls,
        enabled: bool
    ):

        if cls._depth_test_enabled == enabled:
            return

        if enabled:

            glEnable(
                GL_DEPTH_TEST
            )

        else:

            glDisable(
                GL_DEPTH_TEST
            )

        cls._depth_test_enabled = enabled

    # =====================================================
    # Blending
    # =====================================================

    @classmethod
    def set_blending(
        cls,
        enabled: bool
    ):

        if cls._blending_enabled == enabled:
            return

        if enabled:

            glEnable(
                GL_BLEND
            )

        else:

            glDisable(
                GL_BLEND
            )

        cls._blending_enabled = enabled

    # =====================================================
    # Face Culling
    # =====================================================

    @classmethod
    def set_face_culling(
        cls,
        enabled: bool
    ):

        if cls._face_culling_enabled == enabled:
            return

        if enabled:

            glEnable(
                GL_CULL_FACE
            )

        else:

            glDisable(
                GL_CULL_FACE
            )

        cls._face_culling_enabled = enabled

    # =====================================================
    # Texture Unit
    # =====================================================

    @classmethod
    def set_active_texture_slot(
        cls,
        slot: int
    ):

        engine_assert(
            isinstance(slot, int),
            "Texture slot must be an integer."
        )

        engine_assert(
            slot >= 0,
            "Texture slot cannot be negative."
        )

        if cls._active_texture_slot == slot:
            return

        glActiveTexture(
            GL_TEXTURE0 + slot
        )

        cls._active_texture_slot = slot

    # =====================================================
    # Cache
    # =====================================================

    @classmethod
    def invalidate(cls):

        cls._depth_test_enabled = None
        cls._blending_enabled = None
        cls._face_culling_enabled = None

        cls._active_texture_slot = None

    # =====================================================
    # Shutdown
    # =====================================================

    @classmethod
    def shutdown(cls):

        if not cls._initialized:
            return

        Logger.debug(
            "[RenderState] Shutting down."
        )

        cls.invalidate()

        cls._initialized = False