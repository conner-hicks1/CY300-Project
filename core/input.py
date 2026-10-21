from core.key_codes import Key
from core.mouse_codes import MouseButton


class Input:

    _keys_down: set[Key] = set()
    _keys_pressed: set[Key] = set()
    _keys_released: set[Key] = set()

    _mouse_buttons_down: set[MouseButton] = set()
    _mouse_buttons_pressed: set[MouseButton] = set()
    _mouse_buttons_released: set[MouseButton] = set()

    _mouse_x = 0.0
    _mouse_y = 0.0

    _mouse_delta_x = 0.0
    _mouse_delta_y = 0.0

    _scroll_x = 0.0
    _scroll_y = 0.0

    _mouse_initialized = False

    _focused = True

    # Rebound keys: the key pressed -> the game's key it
    # stands for (a key not in it stands for itself; None:
    # for nothing, its action moved off it). The game reads
    # its own keys (Key.Z: full throttle); the player may
    # have put them on others.
    _logical: dict[Key, Key | None] = {}

    # Keys as pressed this frame (before rebinding): for
    # choosing a key to bind.
    _physical_pressed: list[Key] = []

    # The mouse's turning: a factor; up and down swapped.
    mouse_sensitivity = 1.0
    invert_y = False

    # =====================================================
    # Frame Lifecycle
    # =====================================================

    @classmethod
    def begin_frame(cls):

        cls._physical_pressed.clear()
        cls._keys_pressed.clear()
        cls._keys_released.clear()

        cls._mouse_buttons_pressed.clear()
        cls._mouse_buttons_released.clear()

        cls._mouse_delta_x = 0.0
        cls._mouse_delta_y = 0.0

        cls._scroll_x = 0.0
        cls._scroll_y = 0.0

    @classmethod
    def clear(cls):

        cls._keys_down.clear()
        cls._keys_pressed.clear()
        cls._keys_released.clear()

        cls._mouse_buttons_down.clear()
        cls._mouse_buttons_pressed.clear()
        cls._mouse_buttons_released.clear()

        cls._mouse_delta_x = 0.0
        cls._mouse_delta_y = 0.0

        cls._scroll_x = 0.0
        cls._scroll_y = 0.0

        cls._mouse_initialized = False

    @classmethod
    def _focus_lost(cls):

        cls._focused = False

        # GLFW may never send RELEASE events for keys/buttons
        # released while the window is unfocused.

        cls._keys_down.clear()
        cls._keys_pressed.clear()
        cls._keys_released.clear()

        cls._mouse_buttons_down.clear()
        cls._mouse_buttons_pressed.clear()
        cls._mouse_buttons_released.clear()

        cls._mouse_delta_x = 0.0
        cls._mouse_delta_y = 0.0

        cls._scroll_x = 0.0
        cls._scroll_y = 0.0

        # The next cursor position becomes the new baseline
        # rather than generating a large mouse delta.

        cls._mouse_initialized = False


    @classmethod
    def _focus_gained(cls):

        cls._focused = True

        cls._reset_mouse_tracking()

        cls._scroll_x = 0.0
        cls._scroll_y = 0.0

    # =====================================================
    # Keyboard Internal
    # =====================================================
   
    @classmethod
    def simulate_press(
        cls,
        key: Key
    ):
        """A scripted press (headless runs, tests): pressed this frame, down until simulate_release."""

        cls._keys_down.add(key)
        cls._keys_pressed.add(key)

    @classmethod
    def simulate_hold(
        cls,
        key: Key
    ):
        """Down (not newly pressed): a key held through frames."""

        cls._keys_down.add(key)

    @classmethod
    def simulate_release(
        cls,
        key: Key
    ):

        cls._keys_down.discard(key)

    # =====================================================
    # Rebinding
    # =====================================================

    @classmethod
    def set_bindings(
        cls,
        bindings: dict[Key, Key]
    ):
        """The game's keys as the player has them: game key -> the key pressed for it (one each)."""

        moved = {logical: physical for logical, physical in bindings.items() if physical != logical}

        # A game key whose action moved, no other put on it:
        # does nothing now.
        cls._logical = {logical: None for logical in moved if logical not in moved.values()}

        cls._logical.update({physical: logical for logical, physical in moved.items()})

        # (A key held as it changes would stick.)
        cls._keys_down.clear()

    @classmethod
    def logical(
        cls,
        key: Key
    ) -> Key | None:
        """The game's key a pressed key stands for (None: nothing)."""

        return cls._logical.get(key, key)

    @classmethod
    def physical_pressed(cls) -> list[Key]:
        """Keys pressed this frame, as on the keyboard (not rebound)."""

        return list(cls._physical_pressed)

    @classmethod
    def _press_key(
        cls,
        key: Key
    ):

        if not cls._focused:
            return

        cls._physical_pressed.append(key)

        key = cls.logical(key)

        if key is None:
            return

        cls._keys_down.add(
            key
        )

        cls._keys_pressed.add(
            key
        )

    @classmethod
    def _release_key(
        cls,
        key: Key
    ):

        key = cls.logical(key)

        if key is None:
            return

        cls._keys_down.discard(
            key
        )

        cls._keys_released.add(
            key
        )

    # =====================================================
    # Keyboard Public
    # =====================================================

    @classmethod
    def is_key_down(
        cls,
        key: Key
    ) -> bool:

        return key in cls._keys_down

    @classmethod
    def is_key_pressed(
        cls,
        key: Key
    ) -> bool:

        return key in cls._keys_pressed

    @classmethod
    def is_key_released(
        cls,
        key: Key
    ) -> bool:

        return key in cls._keys_released

    # =====================================================
    # Mouse Buttons Internal
    # =====================================================

    @classmethod
    def _press_mouse_button(
        cls,
        button: MouseButton
    ):

        if not cls._focused:
            return

        cls._mouse_buttons_down.add(
            button
        )

        cls._mouse_buttons_pressed.add(
            button
        )

    @classmethod
    def _release_mouse_button(
        cls,
        button: MouseButton
    ):

        cls._mouse_buttons_down.discard(
            button
        )

        cls._mouse_buttons_released.add(
            button
        )

    # =====================================================
    # Mouse Buttons Public
    # =====================================================

    @classmethod
    def is_mouse_button_down(
        cls,
        button: MouseButton
    ) -> bool:

        return (
            button
            in cls._mouse_buttons_down
        )

    @classmethod
    def is_mouse_button_pressed(
        cls,
        button: MouseButton
    ) -> bool:

        return (
            button
            in cls._mouse_buttons_pressed
        )

    @classmethod
    def is_mouse_button_released(
        cls,
        button: MouseButton
    ) -> bool:

        return (
            button
            in cls._mouse_buttons_released
        )

    # =====================================================
    # Mouse Movement
    # =====================================================

    @classmethod
    def _process_mouse_move(
        cls,
        x: float,
        y: float
    ):

        if not cls._focused:
            return

        if not cls._mouse_initialized:

            cls._mouse_x = x
            cls._mouse_y = y

            cls._mouse_initialized = True

            return

        cls._mouse_delta_x += (
            x - cls._mouse_x
        )

        cls._mouse_delta_y += (
            y - cls._mouse_y
        )

        cls._mouse_x = x
        cls._mouse_y = y

    @classmethod
    def mouse_position(
        cls
    ) -> tuple[float, float]:

        return (
            cls._mouse_x,
            cls._mouse_y
        )

    @classmethod
    def mouse_delta(
        cls
    ) -> tuple[float, float]:

        # (As the player likes it turning.)
        return (
            cls._mouse_delta_x * cls.mouse_sensitivity,
            cls._mouse_delta_y * cls.mouse_sensitivity * (-1.0 if cls.invert_y else 1.0)
        )

    @classmethod
    def _reset_mouse_tracking(cls):

        cls._mouse_delta_x = 0.0
        cls._mouse_delta_y = 0.0

        cls._mouse_initialized = False

    # =====================================================
    # Scroll
    # =====================================================

    @classmethod
    def _process_scroll(
        cls,
        x_offset: float,
        y_offset: float
    ):

        if not cls._focused:
            return

        cls._scroll_x += x_offset
        cls._scroll_y += y_offset

    @classmethod
    def scroll_delta(
        cls
    ) -> tuple[float, float]:

        return (
            cls._scroll_x,
            cls._scroll_y
        )