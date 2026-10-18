from dataclasses import dataclass, field

from imgui_bundle import imgui

from game import format, style
from game.map_view import PANEL_FLAGS


# =========================================================
# Maneuver Panel
# =========================================================
#
# On the map, right side: plan a burn.
#
#   no node    put one at the apoapsis, the periapsis or a
#              little ahead; or take a suggested burn
#              (circularize, capture, come down, the window,
#              a course correction)
#   a node     when (move it by steps or to an apsis), how
#              much along prograde / normal / radial (drag,
#              or step), what it costs (m/s, burn time), what
#              the orbit becomes; warp to it, or delete it
#
# Drawing only: what was clicked comes back as actions for
# the application to carry out (it owns the clock and the
# vessel).

GOOD = style.tone("good")
NOTE = style.tone("muted")
WARN = style.tone("warn")

TIME_STEPS = ((-600.0, "-10m"), (-60.0, "-1m"), (-10.0, "-10s"), (10.0, "+10s"), (60.0, "+1m"), (600.0, "+10m"))

DV_STEPS = (-10.0, -1.0, 1.0, 10.0)

COMPONENTS = (
    ("prograde", "Prograde", (0.85, 0.85, 0.3, 1.0)),
    ("normal", "Normal", (0.85, 0.4, 0.95, 1.0)),
    ("radial", "Radial", (0.4, 0.85, 1.0, 1.0)),
)


@dataclass(slots=True)
class NodeInfo:
    """What the panel shows (the application fills it in each frame)."""

    # The vessel can have a node (flying, not landed).
    can_plan: bool = False

    # The node: s from now, its components (m/s), what is
    # left to burn (m/s), full-throttle burn time (s; inf:
    # not enough delta-v), the vessel's delta-v.
    exists: bool = False
    time_to: float = 0.0
    components: dict = field(default_factory=dict)
    remaining: float = 0.0
    burn_time: float = 0.0
    delta_v: float = 0.0

    # The orbit after it: (text, color or None) lines.
    after: list = field(default_factory=list)

    # Where a new node can go: "apoapsis", "periapsis".
    apsides: tuple = ()

    # Suggested burns' labels (and a line of why, or None).
    suggestions: list = field(default_factory=list)

    # Over the conic's end (another sphere): a note.
    note: str | None = None


class ManeuverPanel:

    def draw(
        self,
        info: NodeInfo
    ) -> list[tuple]:
        """The panel (top right of the map); returns the actions clicked."""

        actions: list[tuple] = []

        viewport = imgui.get_main_viewport()

        imgui.set_next_window_pos(
            (viewport.work_pos.x + viewport.work_size.x - 10.0, viewport.work_pos.y + 100.0),
            imgui.Cond_.always,
            (1.0, 0.0)
        )

        imgui.begin("Maneuver", None, PANEL_FLAGS)

        if not info.can_plan:

            imgui.text_colored(NOTE, "Plan burns once flying (not landed).")

            imgui.end()

            return actions

        if not info.exists:
            self._new(info, actions)
        else:
            self._node(info, actions)

        if info.note:
            imgui.text_colored(WARN, info.note)

        imgui.end()

        return actions

    @staticmethod
    def _new(
        info: NodeInfo,
        actions: list
    ):

        imgui.text("New node:")

        for apsis in info.apsides:

            imgui.same_line()

            if imgui.small_button("At Ap" if apsis == "apoapsis" else "At Pe"):
                actions.append(("new", apsis))

        imgui.same_line()

        if imgui.small_button("In 5 min"):
            actions.append(("new", "soon"))

        if info.suggestions:

            imgui.separator()

            imgui.text_colored(NOTE, "Suggested:")

            for index, (label, why) in enumerate(info.suggestions):

                if imgui.button(f"{label}##suggest{index}"):
                    actions.append(("suggest", index))

                if why:
                    imgui.same_line()
                    imgui.text_colored(NOTE, why)

    @staticmethod
    def _node(
        info: NodeInfo,
        actions: list
    ):

        time_to = info.time_to

        if time_to >= 0.0:
            imgui.text(f"Node in {format.countdown(time_to)}")
        else:
            imgui.text_colored(WARN, f"Node {format.countdown(-time_to)} ago")

        imgui.same_line()

        if imgui.small_button("Delete"):
            actions.append(("delete",))

        imgui.same_line()

        if imgui.small_button("Warp to it"):
            actions.append(("warp",))

        for seconds, label in TIME_STEPS:

            if imgui.small_button(f"{label}##time"):
                actions.append(("time", seconds))

            imgui.same_line()

        for apsis in info.apsides:

            if imgui.small_button("To Ap" if apsis == "apoapsis" else "To Pe"):
                actions.append(("time_to", apsis))

            imgui.same_line()

        imgui.new_line()

        imgui.separator()

        for key, label, color in COMPONENTS:

            value = info.components.get(key, 0.0)

            imgui.text_colored(color, f"{label:<9}")

            imgui.same_line(80.0)

            imgui.set_next_item_width(110.0)

            changed, new = imgui.drag_float(f"##{key}", value, 1.0, -50_000.0, 50_000.0, "%.1f m/s")

            if changed:
                actions.append(("set", key, new))

            for step in DV_STEPS:

                imgui.same_line()

                if imgui.small_button(f"{step:+.0f}##{key}{step}"):
                    actions.append(("set", key, value + step))

            imgui.same_line()

            if imgui.small_button(f"0##{key}zero"):
                actions.append(("set", key, 0.0))

        imgui.separator()

        enough = info.remaining <= info.delta_v

        imgui.text(f"Delta-v  {info.remaining:,.1f} m/s")
        imgui.same_line()
        imgui.text_colored(GOOD if enough else WARN, f"(of {info.delta_v:,.0f} left)")

        if info.burn_time == float("inf"):
            imgui.text_colored(WARN, "Not enough delta-v to finish it")
        else:
            imgui.text(f"Burn     {format.countdown(info.burn_time)} at full throttle")
            imgui.text_colored(NOTE, f"Start {format.countdown(info.burn_time / 2.0)} before the node")

        imgui.text_colored(NOTE, "SAS maneuver mode (F) points along it")

        if info.after:

            imgui.separator()

            for text, color in info.after:

                if color is None:
                    imgui.text(text)
                else:
                    imgui.text_colored(color, text)
