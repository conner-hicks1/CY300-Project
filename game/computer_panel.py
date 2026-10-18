from dataclasses import dataclass, field

from imgui_bundle import imgui

from game import format, style
from game.map_view import PANEL_FLAGS

from vessel.sequence import SequenceError, list_programs, parse


# =========================================================
# Flight Computer Panel
# =========================================================
#
# C: write, check and send the vessel a sequence
# (vessel/sequence.py), and watch it run.
#
#   the link: the station, the light time each way, or why
#             there is none (crewed: no need)
#   a program: open one (built-in examples, or the
#             player's), edit it, save it, check it
#   Send: it starts when it arrives (now, crewed); Stop
#             likewise
#   running: the line it is on, its log
#
# Drawing only: what was clicked comes back as actions.

GOOD = style.tone("good")
NOTE = style.tone("muted")
WARN = style.tone("warn")
BAD = style.tone("bad")


@dataclass(slots=True)
class ComputerInfo:

    # The link: (connected, station or why not, one-way
    # delay s), crewed.
    connected: bool = False
    link: str = ""
    delay: float = 0.0
    crewed: bool = False

    # Running: the current line's text and number, waiting
    # until (s from now, or None), its log [(clock, text)],
    # done.
    running: bool = False
    line: str = ""
    line_number: int = 0
    wait: float | None = None
    log: list = field(default_factory=list)
    done: bool = False

    # Sent and still on the way: [(what, s until it arrives)].
    in_flight: list = field(default_factory=list)


class ComputerPanel:

    def __init__(self):

        self.open = False

        self.text = (
            "# A flight computer sequence: one statement a line.\n"
            "# Open an example below, or write your own.\n"
            "sas prograde\n"
            "wait 10 s\n"
            "log Hello from the probe\n"
        )

        self.name = "my_program"

        # The last check: "" fine, else the error.
        self.error = ""
        self.checked = False

    def toggle(self):

        self.open = not self.open

    def draw(
        self,
        info: ComputerInfo
    ) -> list[tuple]:

        actions: list[tuple] = []

        if not self.open:
            return actions

        viewport = imgui.get_main_viewport()

        imgui.set_next_window_pos(
            (viewport.work_pos.x + viewport.work_size.x * 0.5, viewport.work_pos.y + 60.0),
            imgui.Cond_.first_use_ever,
            (0.5, 0.0)
        )

        opened, self.open = imgui.begin("Flight computer (C)", self.open, PANEL_FLAGS)

        if not opened:

            imgui.end()

            return actions

        self._link(info)

        imgui.separator()

        self._editor(actions)

        imgui.separator()

        self._status(info, actions)

        imgui.end()

        return actions

    @staticmethod
    def _link(
        info: ComputerInfo
    ):

        if info.crewed:

            imgui.text_colored(GOOD, "Crewed: the crew flies it and runs the computer (no delay).")

            if info.connected:
                imgui.text_colored(NOTE, f"Link: {info.link}, {format.countdown(info.delay)} each way")
            else:
                imgui.text_colored(NOTE, f"No link home ({info.link}): results cannot be sent.")

        elif info.connected:

            imgui.text_colored(GOOD, f"Link: {info.link}")
            imgui.same_line()
            imgui.text_colored(WARN if info.delay > 5.0 else NOTE, f"{format.countdown(info.delay)} each way")

            if info.delay > 5.0:
                imgui.text_colored(NOTE, "Commands (keys too) arrive that much later.")

        else:

            imgui.text_colored(BAD, f"No link: {info.link}")
            imgui.text_colored(NOTE, "The probe flies only what it was sent.")

    def _editor(
        self,
        actions: list
    ):

        programs = list_programs()

        if imgui.begin_combo("##open", "Open a program..."):

            for name, path, own in programs:

                if imgui.selectable(f"{name}{'' if own else '  (example)'}##{path}", False)[0]:

                    try:
                        with open(path, encoding="utf-8") as handle:
                            self.text = handle.read()
                    except OSError as error:
                        self.error = str(error)
                    else:
                        self.name = name
                        self.error = ""
                        self.checked = False

            imgui.end_combo()

        changed, self.text = imgui.input_text_multiline("##program", self.text, (520.0, 260.0))

        if changed:
            self.checked = False

        imgui.set_next_item_width(160.0)

        _, self.name = imgui.input_text("##name", self.name)

        imgui.same_line()

        if imgui.button("Save"):
            actions.append(("save", self.name, self.text))

        imgui.same_line()

        if imgui.button("Check"):
            self.check()

        imgui.same_line()

        if imgui.button("Send"):

            program = self.check()

            if program is not None:
                actions.append(("send", program))

        if self.checked:

            if self.error:
                imgui.text_colored(BAD, self.error)
            else:
                imgui.text_colored(GOOD, "No errors.")

    def check(self):
        """Parse the text: the Program, or None (the error shows)."""

        self.checked = True

        try:
            program = parse(self.text)
        except SequenceError as error:
            self.error = str(error)
            return None

        if not program.statements:

            self.error = "Nothing to run."

            return None

        self.error = ""

        return program

    @staticmethod
    def _status(
        info: ComputerInfo,
        actions: list
    ):

        for what, left in info.in_flight:
            imgui.text_colored(WARN, f"On the way: {what}, arrives in {format.countdown(left)}")

        if info.running and not info.done:

            text = f"Running line {info.line_number}: {info.line}"

            if info.wait is not None:
                text += f"  ({format.countdown(info.wait)})"

            imgui.text_colored(GOOD, text)

            if imgui.button("Stop"):
                actions.append(("stop",))

        elif info.running:

            imgui.text_colored(NOTE, "Finished.")

        else:

            imgui.text_disabled("Not running.")

        for when, text in info.log[-8:]:
            imgui.text_colored(NOTE, f"  {text}")
