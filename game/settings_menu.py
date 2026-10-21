from collections.abc import Callable
from dataclasses import dataclass

from imgui_bundle import imgui

from core.input import Input
from core.key_codes import Key

from game import style
from game.bindings import ACTIONS, ControlSettings, key_name, what_does
from game.quality import QUALITY_NAMES


# =========================================================
# Settings
# =========================================================
#
# From the pause menu and the title screen, in tabs:
#
#   Graphics   the quality preset
#   Sound      volumes (master, effects, alerts, music),
#              mute, engines through the hull in vacuum
#   Controls   every key, each rebound by clicking it and
#              pressing another (game/bindings.py); the
#              mouse's sensitivity, up / down swapped; hints
#   Interface  its size (UI scale)
#
# It only draws and changes the settings it is given; the
# application keeps them (SettingsContext's callbacks).

@dataclass(slots=True)
class SettingsContext:

    quality: str | None
    set_quality: Callable[[str], None]
    sound: object                       # audio/sound.py SoundEngine (or None)
    hints: object                       # game/guide.py Hints (or None)
    controls: ControlSettings
    controls_changed: Callable[[], None]
    ui_fits: float = 1.75               # the biggest size the window holds


WIDTH = 640.0

FLAGS = (
    imgui.WindowFlags_.no_collapse.value
    | imgui.WindowFlags_.no_docking.value
    | imgui.WindowFlags_.no_saved_settings.value
)


class SettingsMenu:

    def __init__(self):

        self.open = False

        # The action waiting for its new key (a game key), or
        # None.
        self.capturing: Key | None = None

        # The last key change, to say (what moved where).
        self._said = ""

        # A tab to open on (its label), once; or None.
        self._tab: str | None = None

    def show(
        self,
        tab: str | None = None
    ):
        """tab: "Graphics", "Sound", "Controls" or "Interface" (None: as last left)."""

        self.open = True
        self.capturing = None
        self._said = ""
        self._tab = tab

    def close(self):

        self.open = False
        self.capturing = None

    def back(self):
        """Esc: stop choosing a key, or close."""

        if self.capturing is not None:
            self.capturing = None
        else:
            self.close()

    # -----------------------------------------------------
    # Draw
    # -----------------------------------------------------

    def draw(
        self,
        context: SettingsContext
    ):

        if not self.open:
            return

        self._capture(context)

        viewport = imgui.get_main_viewport()

        imgui.set_next_window_pos(
            (viewport.work_pos.x + viewport.work_size.x * 0.5, viewport.work_pos.y + viewport.work_size.y * 0.5),
            imgui.Cond_.always,
            (0.5, 0.5)
        )

        imgui.set_next_window_size((WIDTH, min(600.0, viewport.work_size.y * 0.9)), imgui.Cond_.always)

        opened, keep = imgui.begin("Settings###settings", True, FLAGS)

        if not keep:
            self.close()

        if opened and imgui.begin_tab_bar("settings tabs"):

            for label, draw in (
                ("Graphics", self._draw_graphics),
                ("Sound", self._draw_sound),
                ("Controls", self._draw_controls),
                ("Interface", self._draw_interface),
            ):

                flags = imgui.TabItemFlags_.set_selected.value if label == self._tab else 0

                if imgui.begin_tab_item(label, None, flags)[0]:

                    draw(context)

                    imgui.end_tab_item()

            imgui.end_tab_bar()

            self._tab = None

        imgui.end()

    def _capture(
        self,
        context: SettingsContext
    ):
        """Choosing a key: the first one pressed (Esc: never mind)."""

        if self.capturing is None:
            return

        for key in Input.physical_pressed():

            if key == Key.ESCAPE:
                return

            action = self.capturing

            other = context.controls.bind(action, key)

            self.capturing = None

            self._said = (
                f"{key_name(key)}: {_what(action)}; {_what(other)} moved to {key_name(context.controls.key_for(other))}"
                if other is not None
                else f"{key_name(key)}: {_what(action)}"
            )

            context.controls_changed()

            return

    @staticmethod
    def _draw_graphics(
        context: SettingsContext
    ):

        style.heading("Quality")

        for name in QUALITY_NAMES:

            if imgui.radio_button(name, context.quality == name):
                context.set_quality(name)

            imgui.same_line()

        imgui.new_line()

        if context.quality is None:
            imgui.text_disabled("Custom (changed in the developer tools).")

        imgui.push_text_wrap_pos(WIDTH - 30.0)

        imgui.text_disabled(
            "Low: no rocks, trees or grass, short shadows, fewer atmosphere samples, the scene at 70% of the "
            "window's size. Medium: 85%. High: full size, mountains' shadows too."
        )

        imgui.pop_text_wrap_pos()

    @staticmethod
    def _draw_sound(
        context: SettingsContext
    ):

        sound = context.sound

        if sound is None:

            imgui.text_disabled("No sound.")

            return

        settings = sound.settings

        if not sound.playing:
            imgui.text_disabled("No sound device (or not Windows): silent.")

        changed = False

        moved, settings.muted = imgui.checkbox("Mute", settings.muted)
        changed |= moved

        for label, name, about in (
            ("Master", "master", "Everything."),
            ("Effects", "effects", "Engines, wind, staging, landings."),
            ("Alerts", "alerts", "Notices' chimes and alarms, the interface's clicks."),
            ("Music", "music", "On the title screen and out in space."),
        ):

            imgui.set_next_item_width(300.0)

            moved, value = imgui.slider_float(label, getattr(settings, name), 0.0, 1.0, "%.2f")

            if moved:
                setattr(settings, name, value)

            imgui.set_item_tooltip(about)

            changed |= imgui.is_item_deactivated_after_edit()

        moved, settings.structure = imgui.checkbox("Engines through the hull in vacuum", settings.structure)
        changed |= moved

        imgui.set_item_tooltip("Space carries no sound: off, engines are silent out of the air. On: a muffled rumble through the structure, with the camera on the vessel.")

        if changed:
            sound.save()

        if imgui.button("Test", (120.0, 0.0)):
            sound.play("chime", alert=True)

    def _draw_controls(
        self,
        context: SettingsContext
    ):

        controls = context.controls

        changed = False

        imgui.set_next_item_width(300.0)

        moved, controls.mouse_sensitivity = imgui.slider_float("Mouse sensitivity", controls.mouse_sensitivity, 0.2, 3.0, "%.2fx")

        changed |= imgui.is_item_deactivated_after_edit()

        imgui.set_item_tooltip("How fast the camera and the map turn with the mouse.")

        moved, controls.invert_y = imgui.checkbox("Mouse up / down swapped", controls.invert_y)
        changed |= moved

        hints = context.hints

        if hints is not None:

            moved, enabled = imgui.checkbox("Hints", hints.enabled)

            if moved:
                hints.set_enabled(enabled)

            imgui.set_item_tooltip("A word the first time something comes up (on the pad, a stage run dry, reentry heat...).")

            imgui.same_line()

            if imgui.small_button("Show them all again"):
                hints.reset()

        if changed:
            context.controls_changed()

        imgui.separator_text("Keys")

        if self.capturing is not None:
            style.say(f"Press a key for: {_what(self.capturing)}  (Esc: never mind)", "warn")
        elif self._said:
            style.say(self._said, "good")
        else:
            style.say("Click a key to put it on another; a key taken swaps with the action that had it. Esc stays the menus'.", "muted")

        if imgui.begin_child("keys", (0.0, -34.0), imgui.ChildFlags_.borders.value):

            for group, actions in ACTIONS:

                imgui.separator_text(group)

                if imgui.begin_table(f"keys {group}", 2, imgui.TableFlags_.row_bg.value):

                    imgui.table_setup_column("what", imgui.TableColumnFlags_.width_stretch.value)
                    imgui.table_setup_column("key", imgui.TableColumnFlags_.width_fixed.value, style.px(150.0))

                    for action, what in actions:

                        imgui.table_next_row()
                        imgui.table_next_column()

                        moved_off = controls.key_for(action) != action

                        style.say(what, "accent" if moved_off else None)

                        imgui.table_next_column()

                        label = "..." if self.capturing == action else key_name(controls.key_for(action))

                        if imgui.button(f"{label}##{action.name}", (style.px(140.0), 0.0)):

                            self.capturing = action
                            self._said = ""

                    imgui.end_table()

        imgui.end_child()

        if imgui.button("Every key back to its own", (240.0, 0.0)):

            controls.reset_keys()

            self.capturing = None
            self._said = "Every key back to its own"

            context.controls_changed()

    @staticmethod
    def _draw_interface(
        context: SettingsContext
    ):

        controls = context.controls

        imgui.set_next_item_width(300.0)

        _, controls.ui_scale = imgui.slider_float("Size", controls.ui_scale, 0.75, 1.75, "%.2fx")

        # (Changed as it is let go: the layout follows.)
        if imgui.is_item_deactivated_after_edit():
            context.controls_changed()

        imgui.set_item_tooltip("The HUD's, panels' and menus' size (text, buttons, spacing).")

        if controls.ui_scale > context.ui_fits + 1e-3:
            style.say(f"This window holds up to {context.ui_fits:.2f}x: shown at that (a bigger window, or full screen, for more).", "warn")

        if imgui.button("Back to 1x", (160.0, 0.0)):

            controls.ui_scale = 1.0

            context.controls_changed()


def _what(
    action: Key | None
) -> str:

    return what_does(action) if action is not None else ""
