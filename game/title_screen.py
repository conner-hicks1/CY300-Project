import time

from collections.abc import Callable
from dataclasses import dataclass

from imgui_bundle import imgui

from game import style
from game.bindings import keys
from game.style import tone

from planet.orbits import date_of

from game.lessons import LESSONS

from programme import saves
from programme.saves import Slot


# =========================================================
# Title Screen
# =========================================================
#
# Over the game as it starts (the last game played behind
# it, paused), and again from the pause menu (Save and quit
# to title). Pages:
#
#   main   Continue (the game behind), New game, Flight
#          school, Load game, Settings, Encyclopaedia, Quit
#   school flight school's lessons (game/lessons.py), each
#          a short flight step by step
#   new    a name, career or sandbox
#   load   the saved games, the last played first: load one
#          (or its quicksave / autosave), delete one
#
# It only draws and reports; the application acts through
# TitleActions.

@dataclass(slots=True)
class TitleActions:

    resume: Callable[[], None]                      # the game behind
    play: Callable[[Slot, str | None], None]        # a game (as saved, or its quicksave / autosave)
    new: Callable[[str, bool], None]                # name, sandbox
    quit: Callable[[], None]
    school: Callable[[str], None] = lambda lesson: None     # flight school: a lesson
    settings: Callable[[], None] = lambda: None
    encyclopaedia: Callable[[], None] = lambda: None


def ago(
    then: float,
    now: float | None = None
) -> str:
    """'just now', '5 minutes ago', '3 days ago'."""

    seconds = max((time.time() if now is None else now) - then, 0.0)

    for size, unit in ((86_400.0 * 365.0, "year"), (86_400.0 * 30.0, "month"), (86_400.0, "day"), (3_600.0, "hour"), (60.0, "minute")):

        if seconds >= size:

            count = int(seconds // size)

            return f"{count} {unit}{'' if count == 1 else 's'} ago"

    return "just now"


def describe(
    slot: Slot
) -> str:
    """A game in a line: its date, funds, vessels, science."""

    summary = slot.summary

    parts = []

    if "clock" in summary:
        parts.append(date_of(summary["clock"]).strftime("%d %b %Y"))

    if not slot.sandbox and summary.get("funds") is not None:
        parts.append(f"${summary['funds']:,.1f}M")

    vessels = int(summary.get("vessels", 0))

    parts.append(f"{vessels} vessel{'' if vessels == 1 else 's'} in flight")

    if summary.get("science"):
        parts.append(f"{summary['science']:,.0f} science")

    return "  ·  ".join(parts)


WIDTH = 440.0
ROW = 76.0                          # a saved game's row in the list

FLAGS = (
    imgui.WindowFlags_.no_decoration.value
    | imgui.WindowFlags_.no_background.value
    | imgui.WindowFlags_.no_saved_settings.value
    | imgui.WindowFlags_.no_docking.value
    | imgui.WindowFlags_.no_move.value
)


class TitleScreen:

    MAIN = "main"
    NEW = "new"
    LOAD = "load"
    SCHOOL = "school"

    def __init__(
        self,
        directory=saves.SAVES_DIRECTORY
    ):

        self.directory = directory

        self.open = False

        self.page = self.MAIN

        self.slots: list[Slot] = []

        self._selected = 0

        self._name = ""
        self._sandbox = False

        self._confirm_delete = False

    def show(self):

        self.open = True
        self.page = self.MAIN

        self.refresh()

    def close(self):

        self.open = False

    def refresh(self):

        self.slots = saves.slots(self.directory)

        self._selected = min(self._selected, max(len(self.slots) - 1, 0))

    def back(self):
        """Esc: back to the main page."""

        self.page = self.MAIN
        self._confirm_delete = False

    def _start_new(self):

        self.page = self.NEW

        self._sandbox = False
        self._name = saves.unused_name(False, self.directory)

    # -----------------------------------------------------
    # Draw
    # -----------------------------------------------------

    def draw(
        self,
        actions: TitleActions,
        current: Slot | None
    ):
        """current: the game behind (Continue goes back to it), or None."""

        if not self.open:
            return

        viewport = imgui.get_main_viewport()

        left, top = viewport.work_pos.x, viewport.work_pos.y
        width, height = viewport.work_size.x, viewport.work_size.y

        # A shade over the left of the view, for the menu to
        # read against (the game to the right).
        shade = imgui.get_background_draw_list()

        dark = imgui.get_color_u32((0.02, 0.025, 0.035, 0.9))
        clear = imgui.get_color_u32((0.02, 0.025, 0.035, 0.0))

        shade.add_rect_filled_multi_color((left, top), (left + width * 0.62, top + height), dark, clear, clear, dark)

        imgui.set_next_window_pos((left, top), imgui.Cond_.always)
        imgui.set_next_window_size((width, height), imgui.Cond_.always)

        imgui.begin("###title screen", None, FLAGS)

        margin = max(48.0, width * 0.06)

        imgui.set_cursor_pos((margin, max(height * 0.14, 40.0)))

        imgui.begin_group()

        with style.font("bold", 18.0):
            imgui.text_colored(tone("accent"), "C Y 3 0 0")

        with style.font("bold", 58.0):
            imgui.text("SPACEFLIGHT")

        style.say("The real solar system, from the launch pad out.", "muted")

        imgui.dummy((0.0, 26.0))

        if self.page == self.NEW:
            self._draw_new(actions)
        elif self.page == self.SCHOOL:
            self._draw_school(actions)
        elif self.page == self.LOAD:
            self._draw_load(actions, current)
        else:
            self._draw_main(actions, current)

        imgui.end_group()

        imgui.set_cursor_pos((margin, height - 34.0))

        style.say(keys("{F5} quicksave  ·  {F9} quickload  ·  the game saves itself as you play"), "faint")

        imgui.end()

    def _button(
        self,
        label: str,
        enabled: bool = True
    ) -> bool:

        imgui.begin_disabled(not enabled)

        with style.font("bold"):
            clicked = imgui.button(label, (WIDTH, 38.0))

        imgui.end_disabled()

        return clicked and enabled

    def _draw_main(
        self,
        actions: TitleActions,
        current: Slot | None
    ):

        if current is not None:

            if self._button(f"Continue: {current.name}"):
                actions.resume()

            style.say(describe(current), "muted")

            imgui.dummy((0.0, 6.0))

        if self._button("New game"):
            self._start_new()

        if self._button("Flight school"):
            self.page = self.SCHOOL

        imgui.set_item_tooltip("Short lessons, step by step: a first orbit, maneuver nodes, docking, landing, power, light delay (nothing is saved).")

        if self._button(f"Load game ({len(self.slots)})", bool(self.slots)):

            self.page = self.LOAD

            self._confirm_delete = False

        imgui.dummy((0.0, 6.0))

        half = WIDTH * 0.5 - 4.0

        with style.font("bold"):

            if imgui.button("Settings", (half, 38.0)):
                actions.settings()

            imgui.same_line()

            if imgui.button("Encyclopaedia", (half, 38.0)):
                actions.encyclopaedia()

        if self._button("Quit"):
            actions.quit()

    def _draw_new(
        self,
        actions: TitleActions
    ):

        style.heading("New game")

        imgui.set_next_item_width(WIDTH)

        _, self._name = imgui.input_text("##game name", self._name)

        imgui.dummy((0.0, 4.0))

        for sandbox, label, about in (
            (False, "Career", "Run a space programme from 2026: a budget, missions on the real calendar, research, crews. Starts on Earth."),
            (True, "Sandbox", "Every part, no budget or deadlines; travel to any body from the pause menu."),
        ):

            if imgui.radio_button(label, self._sandbox == sandbox):

                self._sandbox = sandbox

                # (The suggested name follows the kind of game.)
                if self._name.strip() in ("", saves.unused_name(not sandbox, self.directory)):
                    self._name = saves.unused_name(sandbox, self.directory)

            imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + WIDTH)
            style.say(about, "muted")
            imgui.pop_text_wrap_pos()

        imgui.dummy((0.0, 8.0))

        name = self._name.strip()

        taken = any(slot.name.casefold() == name.casefold() for slot in self.slots)

        if taken:
            style.say(f"A game named {name} is saved already.", "warn")

        if self._button("Start", bool(name) and not taken):
            actions.new(name, self._sandbox)

        if self._button("Back"):
            self.back()

    def _draw_school(
        self,
        actions: TitleActions
    ):

        style.heading("Flight school")

        imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + WIDTH)

        style.say("Each a short flight, set up where it starts, done step by step. A sandbox: nothing is saved.", "muted")

        imgui.pop_text_wrap_pos()

        imgui.dummy((0.0, 4.0))

        # (As many as fit, the rest scrolled to: the Back
        # button and the footer below.)
        room = max(imgui.get_main_viewport().work_size.y - imgui.get_cursor_pos_y() - 100.0, 120.0)

        if imgui.begin_child("lessons", (WIDTH + 16.0, room)):

            for item in LESSONS:

                with style.font("bold"):
                    clicked = imgui.button(f"{item.title}##lesson {item.id}", (WIDTH, 32.0))

                if clicked:
                    actions.school(item.id)

                imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + WIDTH)
                style.say(item.about, "muted")
                imgui.pop_text_wrap_pos()

                imgui.dummy((0.0, 2.0))

        imgui.end_child()

        if self._button("Back"):
            self.back()

    def _draw_load(
        self,
        actions: TitleActions,
        current: Slot | None
    ):

        style.heading("Saved games")

        if not self.slots:

            self.back()

            return

        rows = min(len(self.slots), 6)

        if imgui.begin_child("saved games", (WIDTH, rows * ROW + 18.0), imgui.ChildFlags_.borders.value):

            for index, slot in enumerate(self.slots):

                start = imgui.get_cursor_pos()

                clicked, _ = imgui.selectable(f"##slot{index}", index == self._selected, 0, (0.0, ROW - 6.0))

                if clicked:

                    self._selected = index

                    self._confirm_delete = False

                if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(imgui.MouseButton_.left):
                    actions.play(slot, None)

                imgui.set_cursor_pos((start.x + 6.0, start.y + 4.0))

                with style.font("bold"):
                    imgui.text(slot.name)

                style.chip("Sandbox" if slot.sandbox else "Career", "science" if slot.sandbox else "money", same_line=True)

                imgui.set_cursor_pos_x(start.x + 6.0)
                style.say(describe(slot), "muted")

                imgui.set_cursor_pos_x(start.x + 6.0)
                style.say(f"Played {ago(slot.played)}", "faint")

                imgui.set_cursor_pos((start.x, start.y + ROW))

                # (An item there: what the row takes up.)
                imgui.dummy((0.0, 0.0))

        imgui.end_child()

        slot = self.slots[self._selected]

        imgui.dummy((0.0, 4.0))

        if self._button(f"Load {slot.name}"):
            actions.play(slot, None)

        for kind, label in ((saves.QUICKSAVE, "Quicksave"), (saves.AUTOSAVE, "Autosave")):

            when = slot.kept(kind)

            if when is not None and self._button(f"Load the {label.lower()} ({ago(when)})"):
                actions.play(slot, kind)

        if self._confirm_delete:

            style.say(f"Delete {slot.name} and its quicksave and autosave? It cannot be undone.", "bad")

            imgui.push_style_color(imgui.Col_.button, (0.6, 0.18, 0.15, 0.9))
            imgui.push_style_color(imgui.Col_.button_hovered, (0.8, 0.26, 0.2, 1.0))

            if imgui.button("Delete", (WIDTH * 0.5 - 4.0, 30.0)):

                saves.delete(slot)

                self._confirm_delete = False

                self.refresh()

            imgui.pop_style_color(2)

            imgui.same_line()

            if imgui.button("Keep it", (WIDTH * 0.5 - 4.0, 30.0)):
                self._confirm_delete = False

        else:

            behind = current is not None and current.folder == slot.folder

            if self._button("Delete...", not behind):
                self._confirm_delete = True

            if behind:
                imgui.set_item_tooltip("The game behind this menu: load another first.")

        if self._button("Back"):
            self.back()
