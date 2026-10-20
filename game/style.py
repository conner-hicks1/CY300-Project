from contextlib import contextmanager

from imgui_bundle import imgui


# =========================================================
# The Look of the Interface
# =========================================================
#
# One set of colors, sizes and fonts for every panel and the
# HUD, and a few helpers that draw the common pieces the same
# way everywhere:
#
#   panel(...)        a window: placed, padded, its frame
#   heading(text)     a section's title
#   row(label, value) a muted label and its value (fixed-width
#                     digits: numbers do not jitter as they
#                     change)
#   meter(...)        a bar with its text
#   chip(text, tone)  a small colored tag (a state)
#   say(text, tone)   a line in a tone
#
# Tones: "text", "muted", "faint", "accent", "good", "warn",
# "bad", "money", "science".

# ---------------------------------------------------------
# Colors (RGBA, sRGB as ImGui takes them)
# ---------------------------------------------------------

TONES = {
    "text": (0.92, 0.93, 0.95, 1.0),
    "muted": (0.66, 0.69, 0.74, 1.0),
    "faint": (0.46, 0.49, 0.54, 1.0),
    "accent": (0.42, 0.68, 1.0, 1.0),
    "good": (0.46, 0.9, 0.6, 1.0),
    "warn": (1.0, 0.8, 0.32, 1.0),
    "bad": (1.0, 0.44, 0.36, 1.0),
    "money": (1.0, 0.86, 0.42, 1.0),
    "science": (0.55, 0.85, 1.0, 1.0),
}

PANEL = (0.05, 0.056, 0.068, 0.88)
PANEL_SOLID = (0.075, 0.082, 0.098, 0.97)
PANEL_EDGE = (0.32, 0.4, 0.52, 0.35)
FRAME = (0.13, 0.15, 0.19, 1.0)
FRAME_HOVER = (0.19, 0.23, 0.3, 1.0)
ACCENT_DIM = (0.2, 0.36, 0.58, 0.85)
ACCENT_HOVER = (0.3, 0.5, 0.8, 1.0)

# Situations, each a tone.
SITUATIONS = {
    "Landed": "good",
    "Flying": "accent",
    "Sub-orbital": "warn",
    "Orbiting": "good",
    "Escaping": "science",
    "Flyby": "science",
    "Destroyed": "bad",
}

# ---------------------------------------------------------
# Sizes (px)
# ---------------------------------------------------------

MARGIN = 12.0                       # panels from the screen's edges
GAP = 8.0                           # between panels
PADDING = (12.0, 10.0)
ROUNDING = 6.0

FONT_SIZE = 16.0
LARGE_SIZE = 22.0

# The fonts, (font, size) each (ui/imgui_layer.py fills
# these in; until then the default font).
FONTS: dict = {}


def tone(
    name: str | None
) -> tuple:

    return TONES.get(name or "text", TONES["text"])


def apply():
    """The ImGui style: every window and widget from these tokens."""

    imgui.style_colors_dark()

    style = imgui.get_style()

    style.window_rounding = ROUNDING
    style.child_rounding = ROUNDING
    style.frame_rounding = 4.0
    style.grab_rounding = 4.0
    style.tab_rounding = 4.0
    style.popup_rounding = ROUNDING
    style.scrollbar_rounding = 6.0

    style.window_padding = PADDING
    style.frame_padding = (8.0, 4.0)
    style.item_spacing = (8.0, 5.0)
    style.item_inner_spacing = (6.0, 4.0)

    style.window_border_size = 1.0
    style.frame_border_size = 0.0

    style.window_title_align = (0.0, 0.5)

    colors = {
        imgui.Col_.text: TONES["text"],
        imgui.Col_.text_disabled: TONES["faint"],
        imgui.Col_.window_bg: PANEL_SOLID,
        imgui.Col_.popup_bg: PANEL_SOLID,
        imgui.Col_.border: PANEL_EDGE,
        imgui.Col_.frame_bg: FRAME,
        imgui.Col_.frame_bg_hovered: FRAME_HOVER,
        imgui.Col_.frame_bg_active: FRAME_HOVER,
        imgui.Col_.title_bg: (0.07, 0.08, 0.1, 1.0),
        imgui.Col_.title_bg_active: (0.1, 0.13, 0.19, 1.0),
        imgui.Col_.header: ACCENT_DIM,
        imgui.Col_.header_hovered: ACCENT_HOVER,
        imgui.Col_.header_active: TONES["accent"],
        imgui.Col_.button: ACCENT_DIM,
        imgui.Col_.button_hovered: ACCENT_HOVER,
        imgui.Col_.button_active: TONES["accent"],
        imgui.Col_.check_mark: TONES["accent"],
        imgui.Col_.slider_grab: TONES["accent"],
        imgui.Col_.slider_grab_active: ACCENT_HOVER,
        imgui.Col_.plot_histogram: TONES["accent"],
        imgui.Col_.separator: (0.3, 0.34, 0.4, 0.5),
        imgui.Col_.table_header_bg: (0.1, 0.12, 0.16, 1.0),
        imgui.Col_.table_row_bg_alt: (1.0, 1.0, 1.0, 0.03),
    }

    for slot, color in colors.items():
        style.set_color_(slot, color)


# ---------------------------------------------------------
# Pieces
# ---------------------------------------------------------

# A HUD panel: placed by the game, not moved or focused,
# sized to what is in it.
HUD_FLAGS = (
    imgui.WindowFlags_.no_decoration.value
    | imgui.WindowFlags_.no_inputs.value
    | imgui.WindowFlags_.no_docking.value
    | imgui.WindowFlags_.no_saved_settings.value
    | imgui.WindowFlags_.no_focus_on_appearing.value
    | imgui.WindowFlags_.no_nav.value
    | imgui.WindowFlags_.always_auto_resize.value
)


@contextmanager
def font(
    name: str,
    size: float | None = None
):
    """Draw in one of the fonts (regular when not loaded), at its size or another."""

    chosen = FONTS.get(name)

    if chosen is not None:
        imgui.push_font(chosen[0], chosen[1] if size is None else size)

    try:
        yield
    finally:
        if chosen is not None:
            imgui.pop_font()


@contextmanager
def panel(
    name: str,
    position,
    pivot=(0.0, 0.0),
    flags: int = HUD_FLAGS,
    alpha: float | None = None
):
    """A HUD panel at a screen position (its pivot there)."""

    imgui.set_next_window_pos(position, imgui.Cond_.always, pivot)

    imgui.push_style_color(imgui.Col_.window_bg, PANEL if alpha is None else (*PANEL[:3], alpha))

    imgui.begin(name, None, flags)

    try:
        yield
    finally:
        imgui.end()
        imgui.pop_style_color()


def heading(
    text: str,
    tone_name: str = "muted"
):
    """A section's title: small, bold, muted, a rule under it."""

    with font("bold"):
        imgui.text_colored(tone(tone_name), text.upper())


def say(
    text: str,
    tone_name: str | None = None
):

    if tone_name is None or tone_name == "text":
        imgui.text(text)
    else:
        imgui.text_colored(tone(tone_name), text)


def row(
    label: str,
    value: str,
    tone_name: str | None = None,
    label_width: float = 78.0
):
    """A muted label, then its value in fixed-width digits."""

    imgui.text_colored(tone("muted"), label)

    imgui.same_line(label_width)

    with font("mono"):
        say(value, tone_name)


def meter(
    label: str,
    fraction: float,
    text: str,
    tone_name: str = "accent",
    width: float = 120.0,
    label_width: float = 78.0
):
    """A labelled bar."""

    imgui.text_colored(tone("muted"), label)

    imgui.same_line(label_width)

    # The fill a little dim: the text over it reads.
    color = tone(tone_name)

    imgui.push_style_color(imgui.Col_.plot_histogram, (color[0], color[1], color[2], 0.6))
    imgui.push_style_color(imgui.Col_.frame_bg, FRAME)

    imgui.progress_bar(min(max(fraction, 0.0), 1.0), (width, 0.0), text)

    imgui.pop_style_color(2)


def chip(
    text: str,
    tone_name: str = "accent",
    same_line: bool = False
):
    """A small colored tag, a state in a word or two."""

    if same_line:
        imgui.same_line()

    color = tone(tone_name)

    pad_x, pad_y = 6.0, 1.0

    size = imgui.calc_text_size(text)

    start = imgui.get_cursor_screen_pos()

    end = (start.x + size.x + 2.0 * pad_x, start.y + size.y + 2.0 * pad_y)

    draw = imgui.get_window_draw_list()

    draw.add_rect_filled(start, end, imgui.get_color_u32((color[0], color[1], color[2], 0.18)), 4.0)
    draw.add_text((start.x + pad_x, start.y + pad_y), imgui.get_color_u32(color), text)

    imgui.dummy((end[0] - start.x, end[1] - start.y))
