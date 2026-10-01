from OpenGL.GL import (
    GL_BACK,
    GL_BLEND,
    GL_CCW,
    GL_CULL_FACE,
    GL_DEPTH_TEST,
    GL_FILL,
    GL_FRONT,
    GL_FRONT_AND_BACK,
    GL_GREATER,
    GL_LOWER_LEFT,
    GL_ZERO_TO_ONE,
    GL_LINE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POLYGON_OFFSET_LINE,
    GL_SRC_ALPHA,
    GL_TEXTURE0,
    glActiveTexture,
    glBlendFunc,
    glCullFace,
    glClipControl,
    glDepthFunc,
    glDisable,
    glEnable,
    glFrontFace,
    glPolygonMode,
    glPolygonOffset
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
    _cull_front_faces: bool | None = None
    _wireframe: bool | None = None
    _depth_func: int | None = None

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

        # Reversed-Z: clip depth maps to [0, 1] directly
        # (no [-1, 1] -> [0, 1] remap that wastes float
        # precision), near = 1, far = 0, test with GREATER.
        # See math3d.matrices.perspective_reversed_infinite.

        glClipControl(
            GL_LOWER_LEFT,
            GL_ZERO_TO_ONE
        )

        cls.set_depth_func(
            GL_GREATER
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
        # Every MeshFactory / model-loader mesh is wound
        # counter-clockwise when viewed from the side its
        # normals face (verified by tests/test_mesh_data.py),
        # so back faces can be skipped.

        glFrontFace(
            GL_CCW
        )

        cls.set_face_culling(
            True
        )

        cls.set_cull_front_faces(
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

    @classmethod
    def set_cull_front_faces(
        cls,
        cull_front: bool
    ):
        """
        True culls front faces (used by the shadow pass to
        reduce self-shadowing acne); False culls back
        faces (normal rendering).
        """

        if cls._cull_front_faces == cull_front:
            return

        glCullFace(
            GL_FRONT
            if cull_front
            else GL_BACK
        )

        cls._cull_front_faces = cull_front

    @classmethod
    def set_depth_func(
        cls,
        func: int
    ):
        """
        GL_GREATER for the reversed-Z scene, GL_LESS for
        conventional shadow maps, GL_GEQUAL for the sky.
        """

        if cls._depth_func == func:
            return

        glDepthFunc(
            func
        )

        cls._depth_func = func

    # =====================================================
    # Wireframe
    # =====================================================

    @classmethod
    def set_wireframe(
        cls,
        enabled: bool
    ):
        """
        Draw polygons as lines. Lines are pulled slightly
        toward the camera (polygon offset) so an outline
        drawn over a solid object is not lost to depth
        fighting with its own surface.
        """

        if cls._wireframe == enabled:
            return

        glPolygonMode(
            GL_FRONT_AND_BACK,
            GL_LINE if enabled else GL_FILL
        )

        if enabled:

            glEnable(GL_POLYGON_OFFSET_LINE)

            glPolygonOffset(
                -1.0,
                -1.0
            )

        else:

            glDisable(GL_POLYGON_OFFSET_LINE)

        cls._wireframe = enabled

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
        cls._cull_front_faces = None
        cls._wireframe = None
        cls._depth_func = None

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