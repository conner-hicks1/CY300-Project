from core.input import Input
from core.key_codes import Key

from game.bindings import ControlSettings, key_name


def _press(key):

    Input.begin_frame()
    Input._focused = True
    Input._press_key(key)


def test_a_key_moved_swaps_with_the_action_that_had_it(tmp_path):

    settings = ControlSettings()

    # Full throttle onto K: the science panel takes Z.
    assert settings.bind(Key.Z, Key.K) == Key.K

    assert settings.key_for(Key.Z) == Key.K and settings.key_for(Key.K) == Key.Z

    # Back: nothing moved, nothing kept.
    settings.bind(Key.Z, Key.Z)

    assert settings.bindings == {}

    # Esc stays the menus'.
    assert settings.bind(Key.Z, Key.ESCAPE) is None and settings.bindings == {}


def test_the_input_reads_the_game_keys(tmp_path):

    settings = ControlSettings()

    try:

        settings.bind(Key.Z, Key.K)
        settings.bind(Key.X, Key.NUM_1)

        settings.apply()

        _press(Key.K)

        assert Input.is_key_pressed(Key.Z) and not Input.is_key_pressed(Key.K)

        _press(Key.Z)

        assert Input.is_key_pressed(Key.K)

        # X moved to 1: 1 cuts the throttle, X does nothing.
        _press(Key.NUM_1)

        assert Input.is_key_pressed(Key.X)

        _press(Key.X)

        assert not Input.is_key_pressed(Key.X) and Input.physical_pressed() == [Key.X]

    finally:

        Input.set_bindings({})
        Input.clear()


def test_controls_are_kept(tmp_path):

    path = tmp_path / "controls.json"

    settings = ControlSettings(mouse_sensitivity=1.7, invert_y=True, ui_scale=1.25)

    settings.bind(Key.SPACE, Key.ENTER)

    settings.save(path)

    loaded = ControlSettings.load(path)

    assert loaded.bindings == {Key.SPACE: Key.ENTER}
    assert loaded.mouse_sensitivity == 1.7 and loaded.invert_y and loaded.ui_scale == 1.25

    assert ("Next stage (free camera: up)", "Enter") in loaded.changed()

    assert key_name(Key.LEFT_BRACKET) == "[" and key_name(Key.NUM_5) == "5" and key_name(Key.F5) == "F5"


def test_the_mouse_turns_as_set():

    try:

        Input._mouse_delta_x, Input._mouse_delta_y = 10.0, 4.0

        Input.mouse_sensitivity = 2.0
        Input.invert_y = True

        assert Input.mouse_delta() == (20.0, -8.0)

    finally:

        Input.mouse_sensitivity = 1.0
        Input.invert_y = False
        Input.clear()
