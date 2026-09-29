from imgui_bundle import imgui


def keep_window_on_screen():
    """
    Call right after imgui.begin(). Moves the current
    window back inside the display if it would be (partly)
    off-screen, e.g. when the saved layout came from a
    larger window. Windows larger than the display are
    aligned to its top-left.
    """

    io = imgui.get_io()

    display_width = io.display_size.x
    display_height = io.display_size.y

    if display_width <= 0 or display_height <= 0:
        return

    position = imgui.get_window_pos()
    size = imgui.get_window_size()

    x = min(position.x, display_width - size.x)
    y = min(position.y, display_height - size.y)

    x = max(x, 0.0)
    y = max(y, 0.0)

    if abs(x - position.x) > 0.5 or abs(y - position.y) > 0.5:

        imgui.set_window_pos(
            (x, y)
        )
