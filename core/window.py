import glfw

from OpenGL.GL import (
    GL_RENDERER,
    GL_SHADING_LANGUAGE_VERSION,
    GL_VENDOR,
    GL_VERSION,
    glGetString
)

from core.assertions import engine_assert
from core.events.application_event import (
    WindowCloseEvent,
    WindowFocusEvent,
    WindowLostFocusEvent,
    WindowResizeEvent
)
from core.events.key_event import (
    KeyPressedEvent,
    KeyReleasedEvent,
    KeyTypedEvent
)
from core.events.mouse_event import (
    MouseButtonPressedEvent,
    MouseButtonReleasedEvent,
    MouseMovedEvent,
    MouseScrolledEvent
)
from core.input import Input
from core.key_codes import Key
from core.logger import Logger
from core.mouse_codes import MouseButton


class Window:

    # =====================================================
    # Construction
    # =====================================================

    def __init__(
        self,
        width: int,
        height: int,
        title: str
    ):

        engine_assert(
            width > 0,
            "Window width must be positive."
        )

        engine_assert(
            height > 0,
            "Window height must be positive."
        )

        engine_assert(
            bool(title),
            "Window title cannot be empty."
        )

        self.width = width
        self.height = height
        self.title = title

        self.handle = None

        self._event_callback = None

        self._cursor_captured = False

        self._vsync = False

        # Objects that want raw GLFW callbacks (e.g. the
        # ImGui backend). Window stays the only owner of
        # the GLFW callback slots and forwards to these,
        # so installing a listener cannot silently replace
        # the engine's own input handling.

        self._raw_input_listeners = []

        # -------------------------------------------------
        # GLFW
        # -------------------------------------------------

        if not glfw.init():

            raise RuntimeError(
                "Failed to initialize GLFW."
            )

        try:

            # ---------------------------------------------
            # OpenGL 3.3 Core
            # ---------------------------------------------

            glfw.window_hint(
                glfw.CONTEXT_VERSION_MAJOR,
                3
            )

            glfw.window_hint(
                glfw.CONTEXT_VERSION_MINOR,
                3
            )

            glfw.window_hint(
                glfw.OPENGL_PROFILE,
                glfw.OPENGL_CORE_PROFILE
            )

            # ---------------------------------------------
            # Window
            # ---------------------------------------------

            handle = glfw.create_window(
                width,
                height,
                title,
                None,
                None
            )

            if handle is None:

                raise RuntimeError(
                    "Failed to create GLFW window."
                )

            self.handle = handle

            # ---------------------------------------------
            # Context
            # ---------------------------------------------

            glfw.make_context_current(
                self.handle
            )

            # VSync

            glfw.swap_interval(
                1
            )

            self._vsync = True

            # ---------------------------------------------
            # Callbacks
            # ---------------------------------------------

            self._install_callbacks()

            # ---------------------------------------------
            # OpenGL Information
            # ---------------------------------------------

            self._log_opengl_info()

        except Exception:

            if self.handle is not None:

                glfw.destroy_window(
                    self.handle
                )

                self.handle = None

            glfw.terminate()

            raise

        Logger.info(
            "[Window] Created %dx%d '%s'.",
            width,
            height,
            title
        )

    # =====================================================
    # Callback Registration
    # =====================================================

    def _install_callbacks(self):

        engine_assert(
            self.handle is not None,
            "Cannot install callbacks on a closed Window."
        )

        glfw.set_window_close_callback(
            self.handle,
            self._on_window_close
        )

        glfw.set_framebuffer_size_callback(
            self.handle,
            self._on_framebuffer_resize
        )

        glfw.set_window_focus_callback(
            self.handle,
            self._on_window_focus
        )

        glfw.set_key_callback(
            self.handle,
            self._on_key
        )

        glfw.set_char_callback(
            self.handle,
            self._on_char
        )

        glfw.set_mouse_button_callback(
            self.handle,
            self._on_mouse_button
        )

        glfw.set_cursor_pos_callback(
            self.handle,
            self._on_mouse_move
        )

        glfw.set_scroll_callback(
            self.handle,
            self._on_scroll
        )

    # =====================================================
    # Engine Event Callback
    # =====================================================

    def set_event_callback(
        self,
        callback
    ):

        engine_assert(
            callable(callback),
            "Window event callback must be callable."
        )

        self._event_callback = callback

    def _emit(
        self,
        event
    ):

        if self._event_callback is not None:

            self._event_callback(
                event
            )

    # =====================================================
    # Raw Input Listeners
    # =====================================================
    #
    # A listener may implement any of:
    #
    #     keyboard_callback(window, key, scancode, action, mods)
    #     char_callback(window, codepoint)
    #     mouse_callback(window, x, y)
    #     mouse_button_callback(window, button, action, mods)
    #     scroll_callback(window, x_offset, y_offset)
    #
    # (the same names the imgui_bundle GLFW backend uses).

    def add_raw_input_listener(
        self,
        listener
    ):

        engine_assert(
            listener is not None,
            "Raw input listener cannot be None."
        )

        if listener not in self._raw_input_listeners:

            self._raw_input_listeners.append(
                listener
            )

    def remove_raw_input_listener(
        self,
        listener
    ):

        if listener in self._raw_input_listeners:

            self._raw_input_listeners.remove(
                listener
            )

    def _forward_raw(
        self,
        method_name: str,
        *args
    ):

        for listener in self._raw_input_listeners:

            method = getattr(
                listener,
                method_name,
                None
            )

            if method is not None:

                method(
                    *args
                )

    # =====================================================
    # GLFW Callbacks
    # =====================================================

    def _on_window_close(
        self,
        window
    ):

        self._emit(
            WindowCloseEvent()
        )

    def _on_framebuffer_resize(
        self,
        window,
        width,
        height
    ):

        # These are framebuffer dimensions, which are the
        # dimensions OpenGL's viewport cares about.

        self.width = width
        self.height = height

        self._emit(
            WindowResizeEvent(
                width,
                height
            )
        )

    def _on_window_focus(
        self,
        window,
        focused
    ):

        if focused:

            Input._focus_gained()

            self._emit(
                WindowFocusEvent()
            )

        else:

            Input._focus_lost()

            self._emit(
                WindowLostFocusEvent()
            )
            
    def _on_key(
        self,
        window,
        key,
        scancode,
        action,
        mods
    ):

        self._forward_raw(
            "keyboard_callback",
            window,
            key,
            scancode,
            action,
            mods
        )

        try:
            engine_key = Key(key)

        except ValueError:
            return

        if action == glfw.PRESS:

            Input._press_key(
                engine_key
            )

            self._emit(
                KeyPressedEvent(
                    engine_key,
                    repeat=False
                )
            )

        elif action == glfw.REPEAT:

            # Do NOT call _press_key() here.
            #
            # _keys_pressed represents the transition from
            # up -> down during this frame. GLFW repeat events
            # do not represent a new press.

            self._emit(
                KeyPressedEvent(
                    engine_key,
                    repeat=True
                )
            )

        elif action == glfw.RELEASE:

            Input._release_key(
                engine_key
            )

            self._emit(
                KeyReleasedEvent(
                    engine_key
                )
            )

    def _on_char(
        self,
        window,
        codepoint
    ):

        self._forward_raw(
            "char_callback",
            window,
            codepoint
        )

        self._emit(
            KeyTypedEvent(
                codepoint
            )
        )

    def _on_mouse_button(
        self,
        window,
        button,
        action,
        mods
    ):

        self._forward_raw(
            "mouse_button_callback",
            window,
            button,
            action,
            mods
        )

        try:
            engine_button = MouseButton(
                button
            )

        except ValueError:
            return

        if action == glfw.PRESS:

            Input._press_mouse_button(
                engine_button
            )

            self._emit(
                MouseButtonPressedEvent(
                    engine_button
                )
            )

        elif action == glfw.RELEASE:

            Input._release_mouse_button(
                engine_button
            )

            self._emit(
                MouseButtonReleasedEvent(
                    engine_button
                )
            )

    def _on_mouse_move(
        self,
        window,
        x,
        y
    ):

        self._forward_raw(
            "mouse_callback",
            window,
            x,
            y
        )

        Input._process_mouse_move(
            x,
            y
        )

        self._emit(
            MouseMovedEvent(
                x,
                y
            )
        )

    def _on_scroll(
        self,
        window,
        x_offset,
        y_offset
    ):

        self._forward_raw(
            "scroll_callback",
            window,
            x_offset,
            y_offset
        )

        Input._process_scroll(
            x_offset,
            y_offset
        )

        self._emit(
            MouseScrolledEvent(
                x_offset,
                y_offset
            )
        )

    # =====================================================
    # Window Operations
    # =====================================================

    def poll_events(self):

        engine_assert(
            self.handle is not None,
            "Cannot poll events on a closed Window."
        )

        glfw.poll_events()

    def swap_buffers(self):

        engine_assert(
            self.handle is not None,
            "Cannot swap buffers on a closed Window."
        )

        glfw.swap_buffers(
            self.handle
        )

    def should_close(self) -> bool:

        engine_assert(
            self.handle is not None,
            "Cannot query a closed Window."
        )

        return bool(
            glfw.window_should_close(
                self.handle
            )
        )

    def set_should_close(
        self,
        value: bool
    ):

        engine_assert(
            self.handle is not None,
            "Cannot modify a closed Window."
        )

        glfw.set_window_should_close(
            self.handle,
            value
        )

    # =====================================================
    # Cursor
    # =====================================================
    #
    # Captured: cursor hidden and locked to the window,
    # producing unbounded relative motion (mouse look).

    def set_cursor_captured(
        self,
        captured: bool
    ):

        engine_assert(
            self.handle is not None,
            "Cannot change cursor mode on a closed Window."
        )

        captured = bool(captured)

        if captured == self._cursor_captured:
            return

        glfw.set_input_mode(
            self.handle,
            glfw.CURSOR,

            (
                glfw.CURSOR_DISABLED
                if captured
                else glfw.CURSOR_NORMAL
            )
        )

        # Raw motion avoids OS pointer acceleration while
        # looking around, where supported.

        if glfw.raw_mouse_motion_supported():

            glfw.set_input_mode(
                self.handle,
                glfw.RAW_MOUSE_MOTION,
                captured
            )

        self._cursor_captured = captured

        # Avoid a large artificial mouse delta when
        # switching cursor modes.

        Input._reset_mouse_tracking()

    @property
    def cursor_captured(
        self
    ) -> bool:

        return self._cursor_captured

    # =====================================================
    # VSync
    # =====================================================
    #
    # With VSync on, swap_buffers() waits for the display,
    # capping the frame rate at the refresh rate. Turn it
    # off to measure how fast a frame can actually go.

    def set_vsync(
        self,
        enabled: bool
    ):

        engine_assert(
            self.handle is not None,
            "Cannot change VSync on a closed Window."
        )

        glfw.swap_interval(
            1 if enabled else 0
        )

        self._vsync = bool(enabled)

    @property
    def vsync(
        self
    ) -> bool:

        return self._vsync

    # =====================================================
    # Title
    # =====================================================

    def set_title(
        self,
        title: str
    ):

        engine_assert(
            self.handle is not None,
            "Cannot set the title of a closed Window."
        )

        glfw.set_window_title(
            self.handle,
            title
        )

    # =====================================================
    # Dimensions
    # =====================================================

    @property
    def framebuffer_size(
        self
    ) -> tuple[int, int]:

        engine_assert(
            self.handle is not None,
            "Cannot query a closed Window."
        )

        return glfw.get_framebuffer_size(
            self.handle
        )

    @property
    def aspect_ratio(
        self
    ) -> float:

        width, height = (
            self.framebuffer_size
        )

        if height == 0:
            return 1.0

        return (
            width
            / height
        )

    # =====================================================
    # OpenGL Information
    # =====================================================

    @staticmethod
    def _decode_gl_string(
        value
    ) -> str:

        if value is None:
            return "<unknown>"

        return value.decode(
            "utf-8",
            errors="replace"
        )

    def _log_opengl_info(self):

        vendor = self._decode_gl_string(
            glGetString(
                GL_VENDOR
            )
        )

        renderer = self._decode_gl_string(
            glGetString(
                GL_RENDERER
            )
        )

        version = self._decode_gl_string(
            glGetString(
                GL_VERSION
            )
        )

        glsl = self._decode_gl_string(
            glGetString(
                GL_SHADING_LANGUAGE_VERSION
            )
        )

        Logger.info(
            "[OpenGL] Vendor: %s",
            vendor
        )

        Logger.info(
            "[OpenGL] Renderer: %s",
            renderer
        )

        Logger.info(
            "[OpenGL] Version: %s",
            version
        )

        Logger.info(
            "[OpenGL] GLSL: %s",
            glsl
        )

    # =====================================================
    # Shutdown
    # =====================================================

    def close(self):

        if self.handle is None:
            return

        Logger.info(
            "[Window] Shutting down."
        )

        glfw.destroy_window(
            self.handle
        )

        self.handle = None

        glfw.terminate()

        Logger.info(
            "[Window] GLFW terminated."
        )