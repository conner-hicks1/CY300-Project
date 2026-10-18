from dataclasses import dataclass, field

from imgui_bundle import imgui

from game import format, style
from game.map_view import PANEL_FLAGS


# =========================================================
# Science Panel
# =========================================================
#
# K: what the vessel can find out here and what it carries.
#
#   the science banked so far
#   here: the situation, the region or site below (and why
#         the site matters), the nearest site
#   experiments aboard: whether each works here, what a new
#         result would be worth; Run all (R)
#   results aboard: worth recovered (landing back home) or
#         transmitted now (half)
#   mapping: how much of the body is mapped, and whether the
#         camera is mapping now
#
# Drawing only: clicks come back as actions.

GOOD = style.tone("good")
NOTE = style.tone("muted")
WARN = style.tone("warn")
SITE = style.tone("money")


@dataclass(slots=True)
class ScienceInfo:

    points: float = 0.0

    # Where: "landed at Gale crater, Mars"; the site's story.
    here: str = ""
    about: str = ""

    # The nearest site: (name, distance m), or None.
    nearest: tuple | None = None

    # science_system.ExperimentStatus each.
    experiments: list = field(default_factory=list)

    # (title, worth recovered, worth transmitted) each.
    results: list = field(default_factory=list)

    # Where results are recovered (the launch body's name).
    home: str = ""

    # Share of the body mapped (0..1), its name, the camera's
    # status line.
    mapped: float | None = None
    body: str = ""
    mapping: str = ""


class SciencePanel:

    def __init__(self):

        self.open = False

    def toggle(self):

        self.open = not self.open

    def draw(
        self,
        info: ScienceInfo
    ) -> list[tuple]:

        actions: list[tuple] = []

        if not self.open:
            return actions

        viewport = imgui.get_main_viewport()

        imgui.set_next_window_pos(
            (viewport.work_pos.x + 10.0, viewport.work_pos.y + viewport.work_size.y * 0.25),
            imgui.Cond_.first_use_ever
        )

        imgui.set_next_window_size_constraints((360.0, 0.0), (520.0, viewport.work_size.y * 0.7))

        opened, self.open = imgui.begin("Science (K)", self.open, PANEL_FLAGS)

        if not opened:
            imgui.end()
            return actions

        imgui.text_colored(GOOD, f"{info.points:,.1f} science")

        imgui.separator()

        if info.here:
            imgui.text_wrapped(f"Here: {info.here}")

        if info.about:
            imgui.text_colored(SITE, info.about)

        if info.nearest is not None:

            name, distance = info.nearest

            imgui.text_colored(NOTE, f"Nearest site: {name}, {format.distance(distance)} away")

        imgui.separator()

        if info.experiments:

            imgui.text("Experiments aboard")

            for status in info.experiments:

                if not status.works:
                    imgui.text_disabled(f"  {status.name}: {status.why_not}")
                elif status.held:
                    imgui.text_colored(NOTE, f"  {status.name}: done here (aboard)")
                elif status.worth < 0.05:
                    imgui.text_disabled(f"  {status.name}: nothing new here")
                else:
                    imgui.text_colored(GOOD, f"  {status.name}: ~{status.worth:,.1f} science")

            if imgui.button("Run experiments (R)"):
                actions.append(("run",))

        else:

            imgui.text_disabled("No experiments aboard (add Science parts, or a capsule)")

        if info.results:

            imgui.separator()

            imgui.text(f"Results aboard (recovered landing on {info.home}):")

            for index, (title, recovered, transmitted) in enumerate(info.results):

                imgui.text_wrapped(f"{title}")

                imgui.text_colored(NOTE, f"    {recovered:,.1f} recovered")

                imgui.same_line()

                if imgui.small_button(f"Transmit ({transmitted:,.1f})##send{index}"):
                    actions.append(("transmit", index))

        if info.mapped is not None:

            imgui.separator()

            imgui.text(f"{info.body} mapped: {info.mapped * 100.0:,.1f}%")

            imgui.progress_bar(info.mapped, (-1.0, 0.0), "")

            if info.mapping:
                imgui.text_colored(GOOD if info.mapping.startswith("Mapping") else NOTE, info.mapping)

            imgui.text_disabled("Mapped ground shows its sharpest heights.")

        imgui.end()

        return actions
