from dataclasses import dataclass, field

from imgui_bundle import imgui

from game import style
from game.map_view import PANEL_FLAGS


# =========================================================
# Research Panel
# =========================================================
#
# The tech tree (programme/research.py) as boxes in columns,
# each column a step further from the start, lines from
# what a node needs to it:
#
#   known      green: its parts can be built
#   open       blue, its cost: click to research it (amber:
#              not enough science yet)
#   locked     grey: something before it is not known
#
# Hovering one tells what it is for and which parts it
# brings. Opened from the Programme panel (P) and the part
# editor.
#
# Drawing only: a click comes back as ("research", node id).

BOX = (204.0, 62.0)
GAP = (34.0, 14.0)


@dataclass(slots=True)
class ResearchInfo:

    # Science to spend.
    science: float = 0.0
    sandbox: bool = False

    # (node, "known" / "open" / "locked") each.
    nodes: list = field(default_factory=list)

    # Node id -> its column.
    depth: dict = field(default_factory=dict)

    # Part id -> its name.
    part_names: dict = field(default_factory=dict)

    # Node id -> its name.
    node_names: dict = field(default_factory=dict)


def layout(
    info: ResearchInfo
) -> dict[str, tuple[int, float]]:
    """Node id -> (column, row): rows ordered by the rows of what each needs (fewer lines crossing)."""

    columns: dict[int, list] = {}

    for node, _ in info.nodes:
        columns.setdefault(info.depth.get(node.id, 0), []).append(node)

    places: dict[str, tuple[int, float]] = {}

    for column in sorted(columns):

        members = columns[column]

        def parents_row(node):

            rows = [places[other][1] for other in node.requires if other in places]

            return sum(rows) / len(rows) if rows else 0.0

        members.sort(key=parents_row)

        for row, node in enumerate(members):
            places[node.id] = (column, float(row))

    return places


class ResearchPanel:

    def __init__(self):

        self.open = False

    def toggle(self):

        self.open = not self.open

    def draw(
        self,
        info: ResearchInfo
    ) -> list[tuple]:

        actions: list[tuple] = []

        if not self.open:
            return actions

        viewport = imgui.get_main_viewport()

        imgui.set_next_window_pos(
            (viewport.work_pos.x + viewport.work_size.x * 0.5, viewport.work_pos.y + 70.0),
            imgui.Cond_.first_use_ever,
            (0.5, 0.0)
        )

        opened, self.open = imgui.begin("Research", self.open, PANEL_FLAGS)

        if not opened:

            imgui.end()

            return actions

        if info.sandbox:
            style.say("A sandbox knows everything: every part can be built.", "muted")
        else:
            imgui.text_colored(style.tone("science"), f"{info.science:,.1f} science to spend")
            imgui.same_line()
            style.say("  earned by experiments and missions; click an open node to research it", "faint")

        imgui.spacing()

        self._tree(info, actions)

        imgui.end()

        return actions

    @staticmethod
    def _tree(
        info: ResearchInfo,
        actions: list
    ):

        places = layout(info)

        if not places:
            return

        width, height = BOX
        gap_x, gap_y = GAP

        columns = max(column for column, _ in places.values()) + 1
        rows = max(row for _, row in places.values()) + 1

        origin = imgui.get_cursor_screen_pos()

        draw = imgui.get_window_draw_list()

        def corner(node_id):

            column, row = places[node_id]

            return origin.x + column * (width + gap_x), origin.y + row * (height + gap_y)

        states = {node.id: status for node, status in info.nodes}

        # The lines first, under the boxes.
        for node, status in info.nodes:

            x, y = corner(node.id)

            for other in node.requires:

                if other not in places:
                    continue

                ox, oy = corner(other)

                start = (ox + width, oy + height * 0.5)
                end = (x, y + height * 0.5)

                bend = gap_x * 0.6

                known = states.get(other) == "known"

                color = style.tone("good" if known and status != "locked" else "faint")

                draw.add_bezier_cubic(
                    start,
                    (start[0] + bend, start[1]),
                    (end[0] - bend, end[1]),
                    end,
                    imgui.get_color_u32((color[0], color[1], color[2], 0.7)),
                    2.0
                )

        bold = style.FONTS.get("bold")

        for node, status in info.nodes:

            x, y = corner(node.id)

            affordable = info.science >= node.cost - 1e-9

            if status == "known":
                tone, label = "good", "known"
            elif status == "open":
                tone, label = ("accent", f"{node.cost:,.0f} science") if affordable else ("warn", f"{node.cost:,.0f} science")
            else:
                tone, label = "faint", f"{node.cost:,.0f} science"

            color = style.tone(tone)

            imgui.set_cursor_screen_pos((x, y))

            clicked = imgui.invisible_button(f"##tech {node.id}", (width, height))

            hovered = imgui.is_item_hovered()

            fill = (color[0], color[1], color[2], 0.22 if hovered else 0.12)

            # Solid under its tint: lines passing behind it do
            # not show through.
            draw.add_rect_filled((x, y), (x + width, y + height), imgui.get_color_u32(style.PANEL_SOLID), 6.0)
            draw.add_rect_filled((x, y), (x + width, y + height), imgui.get_color_u32(fill), 6.0)
            draw.add_rect((x, y), (x + width, y + height), imgui.get_color_u32((color[0], color[1], color[2], 0.9)), 6.0, 1.5 if status != "locked" else 1.0)

            name_color = imgui.get_color_u32(style.tone("text" if status != "locked" else "muted"))

            if bold is not None:
                draw.add_text(bold[0], bold[1], (x + 10.0, y + 7.0), name_color, node.name)
            else:
                draw.add_text((x + 10.0, y + 7.0), name_color, node.name)

            draw.add_text((x + 10.0, y + 26.0), imgui.get_color_u32(color), label)
            draw.add_text((x + 10.0, y + 42.0), imgui.get_color_u32(style.tone("faint")), f"{len(node.parts)} part" + ("" if len(node.parts) == 1 else "s"))

            if hovered:

                imgui.begin_tooltip()

                with style.font("bold"):
                    imgui.text(node.name)

                imgui.push_text_wrap_pos(320.0)
                imgui.text_colored(style.tone("muted"), node.about)
                imgui.pop_text_wrap_pos()

                for part_id in node.parts:
                    imgui.bullet_text(info.part_names.get(part_id, part_id))

                if node.requires:
                    imgui.text_colored(style.tone("faint"), "Needs " + ", ".join(info.node_names.get(other, other) for other in node.requires))

                if status == "open" and not info.sandbox:

                    if affordable:
                        imgui.text_colored(style.tone("accent"), "Click to research it")
                    else:
                        imgui.text_colored(style.tone("warn"), f"{node.cost - info.science:,.1f} more science needed")

                imgui.end_tooltip()

            if clicked and status == "open" and affordable and not info.sandbox:
                actions.append(("research", node.id))

        # The tree's size (the window fits it).
        imgui.set_cursor_screen_pos(origin)
        imgui.dummy((columns * (width + gap_x) - gap_x, rows * (height + gap_y) - gap_y))
