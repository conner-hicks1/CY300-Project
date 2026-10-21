import json

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from imgui_bundle import imgui

from core.logger import Logger

from game import style
from game.style import tone


# =========================================================
# Flight School and Hints
# =========================================================
#
# Flight school (the title screen starts it): lessons
# (game/lessons.py), step by step. Each step says what to do
# (the keys) and is done when the flight shows it (full
# throttle, the engines lit, climbing, tipped east, apoapsis
# out of the air, the circularizing burn planned...). Lost
# the vessel: the lesson starts over.
#
# Hints: a word, once, the first time something comes up
# (on the pad, a stage run dry, the apoapsis out of the air,
# reentry heat, coming down with a parachute unarmed...).
# Seen ones are not shown again; off in the pause menu
# (Controls), or shown again from the start. Kept in
# data/settings/hints.json.
#
# Both read a Moment: the flight as they need it, made by
# the application each frame (controllers/guide.py).

HINTS_PATH = Path("data/settings/hints.json")

# s a hint stays (unless dismissed).
HINT_SECONDS = 14.0


@dataclass(frozen=True, slots=True)
class Moment:
    """The flight now, for flight school and the hints."""

    situation: str = "Landed"       # Landed, Flying, Sub-orbital, Orbiting, Escaping, Flyby, Destroyed
    body: str = "Earth"
    height: float = 0.0             # m above the ground (or sea level, high up)
    throttle: float = 0.0
    staged: bool = False            # a stage fired (launched)
    thrust: float = 0.0             # N
    sas: bool = False
    sas_mode: str = "hold"
    tilt: float = 0.0               # deg, the nose from straight up
    vertical_speed: float = 0.0     # m/s
    surface_speed: float = 0.0      # m/s
    apoapsis: float | None = None   # m above sea level
    periapsis: float | None = None
    air_top: float = 0.0            # m: the atmosphere's top (0: none)
    air: bool = False               # in the atmosphere now
    stage_dry: bool = False         # the burning stage is out of propellant, more stages left
    heat: float = 0.0               # nearest part to its limit, 0..1
    parachute: str = ""             # "", "armed", "open"; no parachute: "none"
    legs: str = ""                  # "" none, "up", "lowering", "down", "raising"
    map_open: bool = False
    warp: float = 1.0
    node: bool = False              # a maneuver node planned
    node_in: float | None = None    # s until its burn starts
    node_dv: float | None = None    # m/s it asks (left, as flown)
    career: bool = False

    # The target (a vessel), and m to it; translating on the
    # thrusters; docked since the lesson began.
    target: bool = False
    target_distance: float | None = None
    translating: bool = False
    docked: bool = False

    # Power: the battery's share full (no battery: 1); the
    # wings ("", "folded", "unfolding", "out", "folding").
    charge: float = 1.0
    panels: str = ""

    # The link: s of light delay one way; commands on their
    # way; a flight computer program aboard; its panel open.
    delay: float = 0.0
    uplink: bool = False
    program: bool = False
    computer_open: bool = False

    @property
    def destroyed(self) -> bool:

        return self.situation == "Destroyed"

    @property
    def on_pad(self) -> bool:

        return self.situation == "Landed" and not self.staged


# ---------------------------------------------------------
# Flight school
# ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Step:

    title: str
    text: str
    done: Callable[[Moment], bool] | None     # None: read, then Next


# The height Earth's air ends at for the lesson (m); the
# orbit is good above it.
ORBIT_APOAPSIS = 80_000.0

FIRST_ORBIT: tuple[Step, ...] = (
    Step(
        "Full throttle",
        "Your rocket stands on the pad. Press Z for full throttle (Shift and Ctrl move it up and down).",
        lambda m: m.throttle >= 0.95
    ),
    Step(
        "Light the engines",
        "Press Space: the first stage's engines light and the rocket lifts off.",
        lambda m: m.staged and m.thrust > 0.0
    ),
    Step(
        "Hold it steady",
        "Press T for stability assist: it holds the rocket where it points.",
        lambda m: m.sas
    ),
    Step(
        "Climb",
        "Go straight up to 1 km, out of the thickest air.",
        lambda m: m.height >= 1_000.0
    ),
    Step(
        "Gravity turn",
        "Hold W a moment to tip the nose about 10 degrees east (the way the Earth turns: it gives you speed).",
        lambda m: m.tilt >= 8.0
    ),
    Step(
        "Follow prograde",
        "Press F until stability assist says PROGRADE: the rocket follows its motion, and gravity bends the climb over.",
        lambda m: m.sas and m.sas_mode == "prograde"
    ),
    Step(
        "Raise the apoapsis",
        "Keep burning until the apoapsis (the top of the arc, bottom of the screen) is above 80 km. When a stage runs dry, Space drops it and lights the next.",
        lambda m: m.apoapsis is not None and m.apoapsis >= ORBIT_APOAPSIS
    ),
    Step(
        "Coast",
        "Press X to cut the engines. You will coast up to the apoapsis.",
        lambda m: m.throttle <= 0.01 and m.apoapsis is not None and m.apoapsis >= ORBIT_APOAPSIS * 0.9
    ),
    Step(
        "The map",
        "Press M for the map: your path round the Earth. It falls back in: at the top it needs more speed sideways.",
        lambda m: m.map_open
    ),
    Step(
        "Plan the burn",
        "In the map's Maneuver panel, click Circularize: a node at the apoapsis, the burn that makes the orbit round.",
        lambda m: m.node
    ),
    Step(
        "Point at the burn",
        "Press F until stability assist says MANEUVER: the rocket turns to the burn's direction.",
        lambda m: m.sas and m.sas_mode == "maneuver"
    ),
    Step(
        "Wait for it",
        "Time warp (.) towards the node; it slows on its own as the burn nears. Back to real time with /.",
        lambda m: m.node_in is not None and m.node_in <= 15.0
    ),
    Step(
        "Circularize",
        "When the HUD's countdown reaches zero, Z for full throttle; X when the burn is done (the HUD's m/s left near 0).",
        lambda m: m.situation == "Orbiting"
    ),
)

class FlightSchool:
    """A lesson's steps (game/lessons.py), done as the flight shows them."""

    def __init__(self):

        self.lesson = None

        self.active = False

        self.index = 0

        self.lost = False

        # px tall as last drawn (0: not drawn).
        self.height = 0.0

    @property
    def steps(self) -> tuple[Step, ...]:

        return self.lesson.steps if self.lesson is not None else ()

    def start(
        self,
        lesson
    ):
        """lesson: game/lessons.py Lesson (its steps, its title, where it starts, what is said at the end)."""

        self.lesson = lesson

        self.active = True
        self.index = 0
        self.lost = False

    def stop(self):

        self.active = False

    @property
    def finished(self) -> bool:

        return self.index >= len(self.steps)

    @property
    def step(self) -> Step | None:

        return None if self.finished else self.steps[self.index]

    def update(
        self,
        moment: Moment
    ):
        """Steps done as the flight shows them (several at once, if done already); a step to read waits for Next."""

        if not self.active:
            return

        if moment.destroyed:

            self.lost = True

            return

        # From the pad, reverted: from the start.
        if self.lesson.setup == "pad" and moment.on_pad and (self.lost or self.index > 1):
            self.index = 0

        self.lost = False

        while not self.finished and self.step.done is not None and self.step.done(moment):
            self.index += 1

    def skip(self):

        if not self.finished:
            self.index += 1

    def draw(
        self,
        leave: Callable[[], None],
        again: Callable[[], None] = lambda: None
    ):
        """leave: out of flight school; again: the lesson from its start (set up afresh)."""

        self.height = 0.0

        if not self.active:
            return

        viewport = imgui.get_main_viewport()

        lesson = self.lesson

        with style.panel("Flight school", (viewport.work_pos.x + viewport.work_size.x * 0.5, viewport.work_pos.y + 54.0), (0.5, 0.0), style.HUD_FLAGS & ~imgui.WindowFlags_.no_inputs.value, alpha=0.94):

            imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + 430.0)

            if self.finished:

                style.heading(f"Flight school: {lesson.title}, done", "good")

                with style.font("bold", 20.0):
                    style.say("Well flown!", "good")

                style.say(lesson.done)

            elif self.lost:

                style.heading(f"Flight school: {lesson.title}", "bad")

                if lesson.setup == "pad":
                    style.say("The vessel was lost. Esc, then Revert to launch, and try again (the lesson starts over).")
                else:
                    style.say("The vessel was lost. Try again: the lesson starts over where it began.")

            else:

                style.heading(f"Flight school: {lesson.title}  {self.index + 1} / {len(self.steps)}", "accent")

                with style.font("bold", 20.0):
                    style.say(self.step.title)

                style.say(self.step.text)

            imgui.pop_text_wrap_pos()

            imgui.dummy((0.0, 2.0))

            if not self.finished and not self.lost:

                if self.step.done is None:

                    if imgui.small_button("Next"):
                        self.skip()

                elif imgui.small_button("Skip this step"):
                    self.skip()

                imgui.same_line()

            if lesson.setup != "pad" and (self.lost or not self.finished):

                if imgui.small_button("Start over"):
                    again()

                imgui.same_line()

            if imgui.small_button("Leave flight school"):
                leave()

            self.height = imgui.get_window_height()


# ---------------------------------------------------------
# Hints
# ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Hint:

    id: str
    text: str
    when: Callable[[Moment], bool]


HINTS: tuple[Hint, ...] = (
    Hint(
        "pad",
        "On the pad: Z for full throttle, Space to light the engines, T to hold steady. F1 lists every key; Esc pauses.",
        lambda m: m.on_pad
    ),
    Hint(
        "gravity turn",
        "Climbing straight up wastes fuel: tip the nose a little east (W) early on, then let F (prograde) follow the arc over.",
        lambda m: m.staged and m.air and m.height > 4_000.0 and m.tilt < 3.0 and m.thrust > 0.0
    ),
    Hint(
        "stage dry",
        "This stage is out of propellant: Space drops it and lights the next.",
        lambda m: m.stage_dry and not m.on_pad
    ),
    Hint(
        "apoapsis out of the air",
        "The apoapsis is out of the air: X cuts the engines; coast up and burn sideways at the top to make the orbit round (the map, M, plans it: Circularize).",
        lambda m: m.situation == "Sub-orbital" and m.air_top > 0.0 and m.apoapsis is not None and m.apoapsis > m.air_top + 5_000.0 and m.thrust > 0.0
    ),
    Hint(
        "orbit",
        "In orbit. P: the programme's missions; K: the science you can do here (R runs the experiments); the map's Navigation panel finds the way to the Moon and planets.",
        lambda m: m.situation == "Orbiting"
    ),
    Hint(
        "map",
        "The map: right mouse turns it, scroll zooms. Maneuver plans burns (drag the handles); Navigation picks a target and its launch window.",
        lambda m: m.map_open
    ),
    Hint(
        "warp",
        "Time warp: . faster, , slower, / back to real time. In the air or under thrust it holds to 5x.",
        lambda m: m.warp >= 10.0
    ),
    Hint(
        "node",
        "A burn planned: F until MANEUVER points the rocket along it; the HUD counts down to its start (warp there and it slows as it nears).",
        lambda m: m.node
    ),
    Hint(
        "wings",
        "Power is running low and the solar wings are folded: U unfolds them (fold them again before a burn through the air).",
        lambda m: m.panels == "folded" and m.charge < 0.3 and not m.on_pad
    ),
    Hint(
        "power out",
        "Out of power: no reaction wheels, no link, no computer. Sunlight on unfolded wings (U) charges the battery again.",
        lambda m: m.charge <= 0.001 and m.panels != "" and not m.on_pad
    ),
    Hint(
        "heat",
        "Getting hot: come in shallower, and keep the heat shield first (F: retrograde points it into the flow).",
        lambda m: m.heat >= 0.7
    ),
    Hint(
        "parachute",
        "Coming down through the air: Space arms the parachute; it opens when it is low and slow enough.",
        lambda m: m.air and m.staged and m.vertical_speed < -30.0 and m.parachute == "" and m.height < 30_000.0
    ),
    Hint(
        "legs",
        "Landing: G lowers the legs. Come down slower than a few m/s, upright, on level ground.",
        lambda m: m.legs == "up" and m.height < 3_000.0 and m.vertical_speed < -2.0
    ),
    Hint(
        "lost",
        "Lost. Esc > Revert to launch tries the flight again; F9 goes back to a quicksave (F5 makes one).",
        lambda m: m.destroyed
    ),
)


class Hints:
    """One hint at a time, each only once (as long as hints are on)."""

    def __init__(
        self,
        path: Path | None = HINTS_PATH,
        hints: tuple[Hint, ...] = HINTS
    ):
        """path: where the seen ones are kept (None: not kept)."""

        self.path = Path(path) if path is not None else None

        self.hints = hints

        self.enabled = True

        self.seen: set[str] = set()

        # (Hint, s left) on screen, or None.
        self.showing: tuple[Hint, float] | None = None

        # px tall as last drawn (0: not drawn).
        self.height = 0.0

        self._load()

    def update(
        self,
        moment: Moment,
        dt: float
    ):

        if self.showing is not None:

            hint, left = self.showing

            left -= dt

            self.showing = (hint, left) if left > 0.0 else None

        if not self.enabled or self.showing is not None:
            return

        for hint in self.hints:

            if hint.id in self.seen or not hint.when(moment):
                continue

            self.seen.add(hint.id)
            self.showing = (hint, HINT_SECONDS)

            self.save()

            return

    def dismiss(self):

        self.showing = None

    def reset(self):
        """Every hint shown again (once more)."""

        self.seen.clear()
        self.showing = None

        self.save()

    def set_enabled(
        self,
        enabled: bool
    ):

        self.enabled = enabled

        if not enabled:
            self.showing = None

        self.save()

    def save(self):

        if self.path is None:
            return

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"enabled": self.enabled, "seen": sorted(self.seen)}, indent=1), encoding="utf-8")
        except OSError as error:
            Logger.warning("[Hints] Could not save: %s", error)

    def _load(self):

        if self.path is None:
            return

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return

        self.enabled = bool(data.get("enabled", True))
        self.seen = {str(name) for name in data.get("seen", [])}

    def draw(self):

        self.height = 0.0

        if self.showing is None:
            return

        hint, left = self.showing

        viewport = imgui.get_main_viewport()

        # Under the date at the top, centred (where flight
        # school's steps go: no hints over a lesson).
        position = (viewport.work_pos.x + viewport.work_size.x * 0.5, viewport.work_pos.y + 54.0)

        fade = min(left / 1.0, 1.0)

        with style.panel("Hint", position, (0.5, 0.0), style.HUD_FLAGS & ~imgui.WindowFlags_.no_inputs.value, alpha=0.92 * fade):

            imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + 460.0)

            imgui.text_colored(tone("accent"), "HINT")

            imgui.same_line()

            style.say(hint.text)

            imgui.pop_text_wrap_pos()

            if imgui.small_button("Got it"):
                self.dismiss()

            imgui.same_line()

            if imgui.small_button("No more hints"):
                self.set_enabled(False)

            self.height = imgui.get_window_height()
