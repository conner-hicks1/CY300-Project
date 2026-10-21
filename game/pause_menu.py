from collections.abc import Callable
from dataclasses import dataclass

from imgui_bundle import imgui

from game.bindings import keys


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
#   Settings  graphics, sound, controls, the interface
#             (game/settings_menu.py)
#   Encyclopaedia  how it works, the parts, the bodies
#             (game/encyclopaedia.py)
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
    quit: Callable[[], None]
    title: Callable[[], None] = lambda: None
    settings: Callable[[], None] = lambda: None
    encyclopaedia: Callable[[], None] = lambda: None


@dataclass(slots=True)
class Destination:

    id: str
    name: str
    description: str = ""


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
        free_camera: bool = False,
        vessel: bool = False,
        travel: bool = True
    ):
        """travel: whether Travel is open (a sandbox; a career flies there)."""

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
        }

        imgui.begin(f"{titles[self.page]}###pause menu", None, MENU_FLAGS)

        if self.page == self.MAIN:
            self._draw_main(actions, body, vessel)
        elif self.page == self.TRAVEL:
            self._draw_travel(actions, body)
        else:
            self._draw_camera(actions, body, free_camera, vessel)

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
        ):

            closed = page == self.TRAVEL and not self._travel

            imgui.begin_disabled(closed)

            if imgui.button(label, (BUTTON_WIDTH, 0)):
                self.page = page

            imgui.end_disabled()

            if closed:
                imgui.set_item_tooltip("A career flies there; travel anywhere in a sandbox game.")

        if imgui.button("Settings...", (BUTTON_WIDTH, 0)):
            actions.settings()

        imgui.set_item_tooltip("Graphics, sound, controls (every key, the mouse), the interface's size.")

        if imgui.button("Encyclopaedia", (BUTTON_WIDTH, 0)):
            actions.encyclopaedia()

        imgui.set_item_tooltip(keys("How it works, every part, every body ({J})."))

        imgui.separator()

        if imgui.button("Save and quit to title", (BUTTON_WIDTH, 0)):
            actions.title()

        imgui.set_item_tooltip(keys("The title screen: another game, or a new one. ({F5} quicksaves, {F9} goes back to it.)"))

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

            imgui.set_item_tooltip(keys("{V} switches too."))

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
