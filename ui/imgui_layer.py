from pathlib import Path

import imgui_bundle

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

        # Docked layout persistence (gitignored). Named apart
        # from the pre-docking floating layout so stale
        # window positions are not reused.
        io.set_ini_filename("editor_layout_docked.ini")

        io.config_flags |= imgui.ConfigFlags_.docking_enable.value

        # Panels only dock when dragged by their title bar
        # (holding Shift is not required).
        io.config_docking_with_shift = False

        self._load_font(io)
        self._apply_style()

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
    # Appearance
    # =====================================================

    FONT_SIZE = 16.0

    @classmethod
    def _load_font(
        cls,
        io
    ):
        """
        Roboto (shipped with imgui_bundle) instead of the
        tiny built-in bitmap font. Falls back silently if
        the file is missing.
        """

        path = (
            Path(imgui_bundle.__file__).parent
            / "assets" / "fonts" / "Roboto" / "Roboto-Regular.ttf"
        )

        if not path.is_file():

            Logger.warning("[ImGui] UI font not found: %s", path)

            return

        io.fonts.add_font_from_file_ttf(str(path), cls.FONT_SIZE)

    @staticmethod
    def _apply_style():

        imgui.style_colors_dark()

        style = imgui.get_style()

        style.window_rounding = 4.0
        style.frame_rounding = 3.0
        style.grab_rounding = 3.0
        style.tab_rounding = 3.0
        style.popup_rounding = 4.0
        style.scrollbar_rounding = 6.0

        style.window_padding = (8.0, 8.0)
        style.frame_padding = (6.0, 4.0)
        style.item_spacing = (8.0, 5.0)

        style.window_border_size = 0.0

        # Slightly cooler, less saturated accents.
        accent = (0.26, 0.47, 0.74, 1.0)
        accent_hover = (0.33, 0.56, 0.86, 1.0)

        style.set_color_(imgui.Col_.header, (0.22, 0.36, 0.56, 0.75))
        style.set_color_(imgui.Col_.header_hovered, accent_hover)
        style.set_color_(imgui.Col_.header_active, accent)
        style.set_color_(imgui.Col_.button, (0.22, 0.36, 0.56, 0.75))
        style.set_color_(imgui.Col_.button_hovered, accent_hover)
        style.set_color_(imgui.Col_.button_active, accent)
        style.set_color_(imgui.Col_.window_bg, (0.10, 0.105, 0.115, 1.0))

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
