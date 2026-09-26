from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

from core.assertions import engine_assert
from core.logger import Logger
from core.window import Window

from graphics.render_state import RenderState


class ImGuiLayer:

    # =====================================================
    # Dear ImGui Integration
    # =====================================================
    #
    # Uses imgui_bundle's pure-Python GLFW + OpenGL 3
    # backend. The backend is created with
    # attach_callbacks=False and registered as a raw input
    # listener on Window instead, because installing its
    # own GLFW callbacks would replace the engine's input
    # handling.
    #
    # Per frame:
    #
    #     layer.begin_frame()
    #     ... imgui.* widget calls ...
    #     layer.end_frame()      # after scene rendering

    def __init__(
        self,
        window: Window
    ):

        engine_assert(
            window is not None and window.handle is not None,
            "ImGuiLayer requires an open Window."
        )

        self._window = window

        imgui.create_context()

        io = imgui.get_io()

        # Window layout persistence (gitignored).
        io.set_ini_filename("imgui.ini")

        self._backend = GlfwRenderer(
            window.handle,
            attach_callbacks=False
        )

        window.add_raw_input_listener(
            self._backend
        )

        self._frame_active = False

        Logger.info(
            "[ImGui] Initialized (imgui %s).",
            imgui.get_version()
        )

    # =====================================================
    # Frame
    # =====================================================

    def begin_frame(self):

        engine_assert(
            not self._frame_active,
            "ImGuiLayer.begin_frame() called twice."
        )

        self._backend.process_inputs()

        imgui.new_frame()

        self._frame_active = True

    def end_frame(self):

        engine_assert(
            self._frame_active,
            "ImGuiLayer.end_frame() without begin_frame()."
        )

        imgui.render()

        self._backend.render(
            imgui.get_draw_data()
        )

        # The backend restores the GL state it changes,
        # but resync the cache rather than rely on it.

        RenderState.invalidate()

        self._frame_active = False

    # =====================================================
    # Input Capture
    # =====================================================

    @property
    def wants_mouse(
        self
    ) -> bool:
        """True when the mouse is over (or dragging) a UI element."""

        return imgui.get_io().want_capture_mouse

    @property
    def wants_keyboard(
        self
    ) -> bool:
        """True when a UI text field / widget has keyboard focus."""

        return imgui.get_io().want_capture_keyboard

    def set_mouse_enabled(
        self,
        enabled: bool
    ):
        """
        Stop the UI reacting to the mouse, e.g. while the
        cursor is captured for camera look.
        """

        io = imgui.get_io()

        if enabled:

            io.config_flags &= ~imgui.ConfigFlags_.no_mouse.value

        else:

            io.config_flags |= imgui.ConfigFlags_.no_mouse.value

    # =====================================================
    # Shutdown
    # =====================================================

    def shutdown(self):

        self._window.remove_raw_input_listener(
            self._backend
        )

        self._backend.shutdown()

        imgui.destroy_context()

        Logger.info(
            "[ImGui] Shut down."
        )
