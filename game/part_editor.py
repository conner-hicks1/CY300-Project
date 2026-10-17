from collections.abc import Callable
from dataclasses import dataclass

from imgui_bundle import imgui

from vessel.design import (
    DesignDraft,
    SavedDesign,
    list_designs,
    open_design,
    save_draft
)
from vessel.parts import PartDefinition, PartError
from vessel.vessel import Vessel


# =========================================================
# Part Editor
# =========================================================
#
# Building a vessel on the launch pad, where it will fly
# from; the rocket stands there and changes as you edit:
#
#   Parts (left)    the catalog by kind; a click adds the part
#                   below the selected one (or at the bottom)
#   Vessel (right)  the stack from the top down: select a
#                   part to move it, remove it or change its
#                   stage; automatic staging; the numbers that
#                   matter per stage (delta-v in vacuum and at
#                   this body's surface, thrust to weight here,
#                   burn time); what stops it flying; save,
#                   open, launch
#
# Vessels are single stacks (the physics flies one). Opened
# from the pause menu; Launch puts the design on the pad,
# Cancel puts back the one before.

KIND_TITLES = (
    ("command", "Command"),
    ("tank", "Propellant tanks"),
    ("engine", "Engines"),
    ("decoupler", "Stage separators"),
    ("parachute", "Parachutes"),
    ("structure", "Structure"),
)

WINDOW_FLAGS = (
    imgui.WindowFlags_.no_collapse.value
    | imgui.WindowFlags_.no_docking.value
    | imgui.WindowFlags_.no_saved_settings.value
    | imgui.WindowFlags_.no_move.value
    | imgui.WindowFlags_.no_resize.value
)

PROBLEM_COLOR = (1.0, 0.45, 0.35, 1.0)
WARNING_COLOR = (1.0, 0.82, 0.3, 1.0)


@dataclass(slots=True)
class LaunchBody:
    """Where it will launch: for thrust to weight and surface delta-v."""

    name: str
    surface_pressure: float         # bar
    gravity: float                  # m/s^2


@dataclass(slots=True)
class EditorActions:

    launch: Callable[[object], None]    # a VesselDesign
    cancel: Callable[[], None]


def part_summary(
    part: PartDefinition
) -> list[str]:
    """Lines for a part's tooltip."""

    lines = [part.name, f"Mass {part.mass:,.0f} kg", f"{part.radius * 2:.2f} m wide, {part.height:.2f} m tall"]

    if part.propellant:
        lines.append(f"Propellant {part.propellant:,.0f} kg")

    if part.engine is not None:
        engine = part.engine
        lines.append(f"Thrust {engine.thrust_vacuum / 1000:,.0f} kN (vacuum)")
        lines.append(f"Isp {engine.isp_sea_level:.0f} s at sea level, {engine.isp_vacuum:.0f} s in vacuum")

    if part.parachute is not None:
        lines.append(f"Canopy {part.parachute.area:,.0f} m2, opens below {part.parachute.max_speed:.0f} m/s")

    if part.reaction_wheel_torque:
        lines.append(f"Reaction wheels {part.reaction_wheel_torque / 1000:,.0f} kN m")

    lines.append(f"Survives {part.crash_speed:.0f} m/s")

    return lines


class PartEditor:

    def __init__(
        self,
        parts: dict[str, PartDefinition]
    ):

        self.parts = parts

        self.open = False

        self.draft = DesignDraft()

        # The stack row selected (None: none).
        self.selected: int | None = None

        self.body = LaunchBody("Earth", 1.01325, 9.80665)

        # Set by edits; the application rebuilds the rocket
        # on the pad and clears it.
        self.changed = False

        self._message = ""
        self._designs: list[SavedDesign] = []

    # -----------------------------------------------------
    # Opening
    # -----------------------------------------------------

    def start(
        self,
        draft: DesignDraft,
        body: LaunchBody
    ):

        self.open = True
        self.draft = draft
        self.body = body
        self.selected = None
        self.changed = True
        self._message = ""

    def _edited(
        self
    ):

        self.changed = True
        self._message = ""

    # -----------------------------------------------------
    # Drawing
    # -----------------------------------------------------

    def draw(
        self,
        actions: EditorActions
    ):

        if not self.open:
            return

        viewport = imgui.get_main_viewport()

        x, y = viewport.work_pos.x, viewport.work_pos.y
        width, height = viewport.work_size.x, viewport.work_size.y

        self._draw_catalog((x + 10.0, y + 10.0), (300.0, height - 20.0))
        self._draw_vessel(actions, (x + width - 430.0, y + 10.0), (420.0, height - 20.0))

        draw = imgui.get_foreground_draw_list()

        draw.add_text(
            (x + width * 0.5 - 150.0, y + height - 24.0),
            imgui.get_color_u32((0.85, 0.85, 0.85, 0.9)),
            "Right mouse: turn the view   |   scroll: closer / farther"
        )

    def _draw_catalog(
        self,
        position,
        size
    ):

        imgui.set_next_window_pos(position)
        imgui.set_next_window_size(size)

        imgui.begin("Parts", None, WINDOW_FLAGS)

        imgui.text_disabled(
            "Click to add below the selected part."
            if self.selected is not None
            else "Click to add at the bottom."
        )

        for kind, title in KIND_TITLES:

            members = [part for part in self.parts.values() if part.kind == kind]

            if not members:
                continue

            if imgui.collapsing_header(title, imgui.TreeNodeFlags_.default_open.value):

                for part in members:

                    if imgui.button(f"{part.name}##add {part.id}", (-1.0, 0.0)):
                        self._add(part)

                    imgui.set_item_tooltip("\n".join(part_summary(part)))

        imgui.end()

    def _add(
        self,
        part: PartDefinition
    ):

        index = self.selected + 1 if self.selected is not None else len(self.draft.entries)

        self.selected = self.draft.insert(index, part)

        self.draft.auto_stage()

        self._edited()

    def _draw_vessel(
        self,
        actions: EditorActions,
        position,
        size
    ):

        imgui.set_next_window_pos(position)
        imgui.set_next_window_size(size)

        imgui.begin("Vessel", None, WINDOW_FLAGS)

        changed, name = imgui.input_text("Name", self.draft.name)

        if changed:
            self.draft.name = name

        self._draw_stack()

        imgui.separator()

        self._draw_stats()

        imgui.separator()

        self._draw_files(actions)

        imgui.end()

    def _draw_stack(self):

        draft = self.draft

        imgui.separator_text("Stack, top to bottom")

        if imgui.begin_child("stack", (0.0, 230.0), imgui.ChildFlags_.borders.value):

            for index, entry in enumerate(draft.entries):

                stage = f"stage {entry.stage}" if entry.stage is not None else ""

                clicked, _ = imgui.selectable(
                    f"{index + 1:2d}. {entry.part.name}##row{index}",
                    self.selected == index
                )

                if clicked:
                    self.selected = None if self.selected == index else index
                    self.changed = True

                if stage:
                    imgui.same_line(330.0)
                    imgui.text_disabled(stage)

            if not draft.entries:
                imgui.text_disabled("Empty: add parts from the left.")

        imgui.end_child()

        selected = self.selected if self.selected is not None and self.selected < len(draft.entries) else None

        imgui.begin_disabled(selected is None)

        if imgui.button("Up") and selected is not None:
            self.selected = draft.move(selected, -1)
            self._edited()

        imgui.same_line()

        if imgui.button("Down") and selected is not None:
            self.selected = draft.move(selected, 1)
            self._edited()

        imgui.same_line()

        if imgui.button("Remove") and selected is not None:

            draft.remove(selected)

            self.selected = min(selected, len(draft.entries) - 1) if draft.entries else None

            self._edited()

        staged = selected is not None and draft.entries[selected].stage is not None

        imgui.same_line()

        imgui.begin_disabled(not staged)

        if imgui.button("Stage -") and staged:
            draft.set_stage(selected, draft.entries[selected].stage - 1)
            self._edited()

        imgui.same_line()

        if imgui.button("Stage +") and staged:
            draft.set_stage(selected, draft.entries[selected].stage + 1)
            self._edited()

        imgui.end_disabled()

        imgui.end_disabled()

        imgui.same_line()

        if imgui.button("Auto stage"):
            draft.auto_stage()
            self._edited()

        imgui.set_item_tooltip("Bottom engines first, then each separator with the engines above it, parachutes last.")

    def _draw_stats(self):

        draft = self.draft

        body = self.body

        if not draft.entries:
            return

        vessel = Vessel.from_design(draft.preview())

        stats = vessel.stage_stats(body.surface_pressure, body.gravity)

        imgui.text(f"{vessel.mass / 1000:,.1f} t   {vessel.height:.1f} m tall   {vessel.radius * 2:.2f} m wide")

        total = sum(stage.delta_v_vacuum for stage in stats)

        imgui.text(f"Delta-v {total:,.0f} m/s in vacuum")

        imgui.text_disabled(f"TWR and surface delta-v: on {body.name}")

        imgui.set_item_tooltip(f"{body.name}: gravity {body.gravity:.2f} m/s2, surface pressure {body.surface_pressure:.3g} bar.")

        flags = imgui.TableFlags_.row_bg.value | imgui.TableFlags_.borders_inner_h.value

        if stats and imgui.begin_table("stages", 5, flags):

            for heading in ("Stage", "Delta-v", "Surface", "TWR", "Burn"):
                imgui.table_setup_column(heading)

            imgui.table_headers_row()

            for stage in stats:

                imgui.table_next_row()

                imgui.table_next_column()
                imgui.text(str(stage.stage))

                if stage.engines:

                    imgui.table_next_column()
                    imgui.text(f"{stage.delta_v_vacuum:,.0f} m/s")

                    imgui.table_next_column()
                    imgui.text(f"{stage.delta_v_surface:,.0f}")

                    imgui.table_next_column()

                    if stage.twr_surface < 1.0 and stage.stage == stats[0].stage:
                        imgui.text_colored(WARNING_COLOR, f"{stage.twr_surface:.2f}")
                    else:
                        imgui.text(f"{stage.twr_surface:.2f}")

                    imgui.table_next_column()
                    imgui.text(f"{stage.burn_time:,.0f} s")

                else:

                    imgui.table_next_column()
                    imgui.text_disabled("-")

            imgui.end_table()

        if stats and stats[0].engines and stats[0].twr_surface < 1.0:
            imgui.text_colored(WARNING_COLOR, f"The first stage cannot lift off {body.name} (TWR below 1).")

        for problem in draft.problems():
            imgui.text_colored(PROBLEM_COLOR, problem)

        for warning in draft.warnings():
            imgui.text_colored(WARNING_COLOR, warning)

    def _draw_files(
        self,
        actions: EditorActions
    ):

        draft = self.draft

        if imgui.button("New"):

            self.draft = DesignDraft("Untitled")
            self.selected = None

            self._edited()

        imgui.same_line()

        if imgui.button("Save"):

            try:
                path = save_draft(draft)
                self._message = f"Saved to {path.as_posix()}."
            except OSError as error:
                self._message = f"Not saved: {error}"

        imgui.same_line()

        if imgui.button("Open..."):

            self._designs = list_designs()

            imgui.open_popup("open design")

        if imgui.begin_popup("open design"):

            for saved in self._designs:

                label = f"{saved.name}{'  (built in)' if saved.built_in else ''}##{saved.path}"

                if imgui.selectable(label, False)[0]:

                    try:

                        self.draft = open_design(saved, self.parts)
                        self.selected = None

                        self._edited()

                        self._message = f"Opened {saved.name}."

                    except (OSError, ValueError, PartError) as error:

                        self._message = f"Could not open {saved.name}: {error}"

            if not self._designs:
                imgui.text_disabled("No saved designs.")

            imgui.end_popup()

        if self._message:
            imgui.text_disabled(self._message)

        imgui.separator()

        can_fly = not draft.problems()

        imgui.begin_disabled(not can_fly)

        if imgui.button("Launch", (200.0, 0.0)) and can_fly:

            try:
                actions.launch(draft.to_design(self.parts))
                self.open = False
            except PartError as error:
                self._message = str(error)

        imgui.end_disabled()

        imgui.same_line()

        if imgui.button("Cancel", (-1.0, 0.0)):

            self.open = False

            actions.cancel()
