from collections.abc import Callable
from dataclasses import dataclass

from imgui_bundle import imgui

from game.quality import QUALITY_NAMES


# =========================================================
# Pause Menu
# =========================================================
#
# Esc opens it (and pauses the clock); Esc or Resume closes
# it. Pages:
#
#   Revert    back to the launch, the vessel as new
#   Build     the part editor (game/part_editor.py)
#   Travel    go to any body of the solar system: a fresh
#             start on its surface (or in orbit over a giant
#             or a small body)
#   Camera    follow the vessel or fly free; jump to the
#             surface, low orbit, or the whole body in view
#   Graphics  quality preset
#   Controls  the keys; hints on / off
#
# It only draws and reports; the application acts through
# MenuActions.

@dataclass(slots=True)
class MenuActions:

    resume: Callable[[], None]
    revert: Callable[[], None]
    build: Callable[[], None]
    toggle_camera: Callable[[], None]
    travel: Callable[[str], None]
    to_surface: Callable[[], None]
    to_orbit: Callable[[], None]
    whole_body: Callable[[], None]
    set_quality: Callable[[str], None]
    quit: Callable[[], None]
    title: Callable[[], None] = lambda: None


@dataclass(slots=True)
class Destination:

    id: str
    name: str
    description: str = ""


CONTROLS: tuple[tuple[str, str], ...] = (
    ("Shift / Ctrl", "Throttle up / down"),
    ("Z / X", "Full throttle / cut"),
    ("Space", "Next stage (lights engines, separates, arms the parachute)"),
    ("W / S", "Pitch: nose forward / back (forward is east at launch)"),
    ("A / D", "Yaw: nose left / right"),
    ("Q / E", "Roll"),
    ("T", "Stability assist on / off"),
    ("F", "Its mode: hold attitude, prograde, retrograde, maneuver (along the planned burn: map, Maneuver)"),
    ("M", "Map: orbits, apoapsis / periapsis, encounters (right mouse turns it, scroll zooms)"),
    ("Right mouse + move", "Swing the camera round the vessel; scroll: closer / farther"),
    ("V", "Free camera (hold right mouse: W A S D Space Shift fly it)"),
    ("R", "Run the experiments aboard (science)"),
    ("K", "Science panel: what can be learned here, results aboard, mapping"),
    ("C", "Flight computer: write a sequence and send it (a probe flies it on its own)"),
    ("P", "Space programme: funds, missions and their deadlines, launch windows"),
    ("L", "Vessels in flight: fly one, target one, end a mission, move propellant alongside"),
    ("[ / ]", "Fly the previous / next vessel"),
    ("B", "Dock with the targeted vessel automatically (ports at the ends of both stacks)"),
    ("Arrows, H / N", "Translate with the thrusters (RCS): forward / back, left / right, fore / aft"),
    (". / ,", "Time warp faster / slower (5x at most under thrust or in the air)"),
    ("/", "Back to real time"),
    ("Esc", "This menu (pauses)"),
    ("F5 / F9", "Quicksave / go back to the quicksave"),
    ("F2", "Hide the HUD"),
    ("F3", "Developer tools (stats, render settings, profiler)"),
)

MENU_FLAGS = (
    imgui.WindowFlags_.no_collapse.value
    | imgui.WindowFlags_.no_docking.value
    | imgui.WindowFlags_.no_saved_settings.value
    | imgui.WindowFlags_.always_auto_resize.value
    | imgui.WindowFlags_.no_move.value
)

BUTTON_WIDTH = 260.0


class PauseMenu:

    MAIN = "main"
    TRAVEL = "travel"
    CAMERA = "camera"
    GRAPHICS = "graphics"
    SOUND = "sound"
    CONTROLS = "controls"

    def __init__(
        self,
        destinations: dict[str, list[Destination]]
    ):
        """destinations: group (Planets, Moons of Jupiter...) -> bodies."""

        self.open = False

        self.page = self.MAIN

        self._destinations = destinations

        # Travel open (a sandbox).
        self._travel = True

    def toggle(self):

        if self.open:
            self.close()
        else:
            self.show()

    def show(self):

        self.open = True
        self.page = self.MAIN

    def close(self):

        self.open = False

    def back(self):
        """Esc: one page back, or closed from the main page."""

        if self.page == self.MAIN:
            self.close()
        else:
            self.page = self.MAIN

    # -----------------------------------------------------
    # Draw
    # -----------------------------------------------------

    def draw(
        self,
        actions: MenuActions,
        body: str,
        quality: str | None,
        free_camera: bool = False,
        vessel: bool = False,
        travel: bool = True,
        sound=None,
        hints=None
    ):
        """travel: whether Travel is open (a sandbox; a career flies there). sound: audio/sound.py SoundEngine (its settings). hints: game/guide.py Hints."""

        self._sound = sound
        self._hints = hints

        if not self.open:
            return

        self._travel = travel

        viewport = imgui.get_main_viewport()

        imgui.set_next_window_pos(
            (viewport.work_pos.x + viewport.work_size.x * 0.5, viewport.work_pos.y + viewport.work_size.y * 0.5),
            imgui.Cond_.always,
            (0.5, 0.5)
        )

        imgui.set_next_window_size_constraints((BUTTON_WIDTH + 30.0, 0.0), (900.0, viewport.work_size.y * 0.85))

        titles = {
            self.MAIN: "Paused",
            self.TRAVEL: "Travel",
            self.CAMERA: "Camera",
            self.GRAPHICS: "Graphics",
            self.SOUND: "Sound",
            self.CONTROLS: "Controls",
        }

        imgui.begin(f"{titles[self.page]}###pause menu", None, MENU_FLAGS)

        if self.page == self.MAIN:
            self._draw_main(actions, body, vessel)
        elif self.page == self.TRAVEL:
            self._draw_travel(actions, body)
        elif self.page == self.CAMERA:
            self._draw_camera(actions, body, free_camera, vessel)
        elif self.page == self.GRAPHICS:
            self._draw_graphics(actions, quality)
        elif self.page == self.SOUND:
            self._draw_sound()
        else:
            self._draw_controls()

        if self.page != self.MAIN:

            imgui.separator()

            if imgui.button("Back", (BUTTON_WIDTH, 0)):
                self.page = self.MAIN

        imgui.end()

    def _draw_main(
        self,
        actions: MenuActions,
        body: str,
        vessel: bool
    ):

        imgui.text_disabled(f"Near {body}")

        if imgui.button("Resume", (BUTTON_WIDTH, 0)):
            actions.resume()

        if vessel and imgui.button("Revert to launch", (BUTTON_WIDTH, 0)):
            actions.revert()

        if imgui.button("Build vessel...", (BUTTON_WIDTH, 0)):
            actions.build()

        imgui.set_item_tooltip("The part editor, on the launch pad.")

        for label, page in (
            ("Travel...", self.TRAVEL),
            ("Camera...", self.CAMERA),
            ("Graphics...", self.GRAPHICS),
            ("Sound...", self.SOUND),
            ("Controls", self.CONTROLS),
        ):

            closed = page == self.TRAVEL and not self._travel

            imgui.begin_disabled(closed)

            if imgui.button(label, (BUTTON_WIDTH, 0)):
                self.page = page

            imgui.end_disabled()

            if closed:
                imgui.set_item_tooltip("A career flies there; travel anywhere in a sandbox game.")

        imgui.separator()

        if imgui.button("Save and quit to title", (BUTTON_WIDTH, 0)):
            actions.title()

        imgui.set_item_tooltip("The title screen: another game, or a new one. (F5 quicksaves, F9 goes back to it.)")

        if imgui.button("Quit to desktop", (BUTTON_WIDTH, 0)):
            actions.quit()

        imgui.set_item_tooltip("The game is saved as you quit.")

    def _draw_travel(
        self,
        actions: MenuActions,
        body: str
    ):

        imgui.text_disabled("Start over at another body.")

        for group, members in self._destinations.items():

            imgui.separator_text(group)

            for index, destination in enumerate(members):

                if index % 3:
                    imgui.same_line()

                current = destination.name == body

                imgui.begin_disabled(current)

                if imgui.button(destination.name, (BUTTON_WIDTH / 2.0 + 20.0, 0)):
                    actions.travel(destination.id)

                imgui.end_disabled()

                if destination.description:
                    imgui.set_item_tooltip(destination.description)

    @staticmethod
    def _draw_camera(
        actions: MenuActions,
        body: str,
        free_camera: bool,
        vessel: bool
    ):

        if vessel:

            if imgui.button("Follow the vessel" if free_camera else "Free camera", (BUTTON_WIDTH, 0)):
                actions.toggle_camera()

            imgui.set_item_tooltip("V switches too.")

            imgui.separator_text("Free camera to")

        if imgui.button("Surface", (BUTTON_WIDTH, 0)):
            actions.to_surface()

        imgui.set_item_tooltip(f"A scenic spot on {body}, 150 m up.")

        if imgui.button("Low orbit", (BUTTON_WIDTH, 0)):
            actions.to_orbit()

        imgui.set_item_tooltip("Straight up to 400 km, looking at the horizon.")

        if imgui.button(f"All of {body}", (BUTTON_WIDTH, 0)):
            actions.whole_body()

        imgui.set_item_tooltip("Back off until the whole body is in view.")

    @staticmethod
    def _draw_graphics(
        actions: MenuActions,
        quality: str | None
    ):

        imgui.text("Quality")

        for name in QUALITY_NAMES:

            if imgui.radio_button(name, quality == name):
                actions.set_quality(name)

            imgui.same_line()

        imgui.new_line()

        if quality is None:
            imgui.text_disabled("Custom (changed in the developer tools).")

        imgui.text_disabled(
            "Low: no rocks, trees or grass, short shadows, fewer\n"
            "atmosphere samples. High: mountains' shadows too."
        )

    def _draw_sound(self):

        sound = self._sound

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
            ("Alerts", "alerts", "Notices' chimes and alarms."),
        ):

            imgui.set_next_item_width(BUTTON_WIDTH - 70.0)

            moved, value = imgui.slider_float(label, getattr(settings, name), 0.0, 1.0, "%.2f")

            if moved:
                setattr(settings, name, value)

            imgui.set_item_tooltip(about)

            changed |= imgui.is_item_deactivated_after_edit()

        if changed:
            sound.save()

        if imgui.button("Test", (BUTTON_WIDTH, 0)):
            sound.play("chime", alert=True)

    def _draw_controls(self):

        hints = self._hints

        if hints is not None:

            changed, enabled = imgui.checkbox("Hints", hints.enabled)

            if changed:
                hints.set_enabled(enabled)

            imgui.set_item_tooltip("A word the first time something comes up (on the pad, a stage run dry, reentry heat...).")

            imgui.same_line()

            if imgui.small_button("Show them all again"):
                hints.reset()

            imgui.separator()

        if imgui.begin_table("controls", 2, imgui.TableFlags_.row_bg.value):

            for key, action in CONTROLS:

                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(key)
                imgui.table_next_column()
                imgui.text_disabled(action)

            imgui.end_table()
