from dataclasses import dataclass

from imgui_bundle import imgui

from ui.panels import DockSlot, PanelRegistry


@dataclass(frozen=True, slots=True)
class ScreenRect:

    # ImGui screen coordinates (logical pixels, top-left
    # origin).
    x: float
    y: float
    width: float
    height: float

    def contains(
        self,
        px: float,
        py: float
    ) -> bool:

        return (
            self.x <= px < self.x + self.width
            and self.y <= py < self.y + self.height
        )


class EditorLayout:

    # =====================================================
    # Docked Editor Layout
    # =====================================================
    #
    # A dockspace covers the window below the menu bar. Its
    # central node is left empty and transparent: that is
    # the 3D viewport, and the renderer draws the scene into
    # exactly that rectangle (see Application.render), so
    # panels never cover the view.
    #
    #     +------------+----------------------+-----------+
    #     | Hierarchy  |                      |           |
    #     +------------+       viewport       | Inspector |
    #     | Planet     |                      |           |
    #     |            +----------------------+           |
    #     |            | Render|Stats|Profiler |           |
    #     +------------+----------------------+-----------+
    #
    # The default arrangement is built on first run and by
    # View > Reset Layout; after that ImGui saves whatever
    # the user drags around (editor_layout_docked.ini).

    DOCKSPACE_NAME = "EditorDockSpace"

    # Fractions of the window for the default layout.
    LEFT_WIDTH = 0.25
    RIGHT_WIDTH = 0.22
    BOTTOM_HEIGHT = 0.27
    LEFT_SPLIT = 0.30           # hierarchy share of the left column

    def __init__(
        self,
        panels: PanelRegistry
    ):

        self._panels = panels

        self._reset_requested = False

        # Viewport rectangle from the last begin(); None
        # before the first frame.
        self.viewport_rect: ScreenRect | None = None

    def request_reset(self):

        self._reset_requested = True

    def begin(self) -> ScreenRect:
        """
        Call once per frame before any panel. Returns the
        viewport rectangle (also kept in viewport_rect).
        """

        viewport = imgui.get_main_viewport()

        dock_id = imgui.get_id(self.DOCKSPACE_NAME)

        node = imgui.internal.dock_builder_get_node(dock_id)

        if node is None or self._reset_requested or node.is_leaf_node():

            self._build_default(dock_id, viewport)

            self._reset_requested = False

        imgui.dock_space_over_viewport(
            dock_id,
            viewport,
            imgui.DockNodeFlags_.passthru_central_node.value
        )

        central = imgui.internal.dock_builder_get_central_node(dock_id)

        if central is not None:

            rect = ScreenRect(
                central.pos.x,
                central.pos.y,
                central.size.x,
                central.size.y
            )

        else:

            rect = ScreenRect(
                viewport.work_pos.x,
                viewport.work_pos.y,
                viewport.work_size.x,
                viewport.work_size.y
            )

        self.viewport_rect = rect

        return rect

    def _build_default(
        self,
        dock_id: int,
        viewport
    ):

        internal = imgui.internal

        internal.dock_builder_remove_node(dock_id)

        internal.dock_builder_add_node(
            dock_id,
            internal.DockNodeFlagsPrivate_.dock_space.value
        )

        internal.dock_builder_set_node_pos(dock_id, viewport.work_pos)
        internal.dock_builder_set_node_size(dock_id, viewport.work_size)

        def split(node, direction, ratio):

            # -> (node toward `direction`, the rest)
            _, toward, rest = internal.dock_builder_split_node_py(node, direction, ratio)

            return toward, rest

        left, rest = split(dock_id, imgui.Dir.left, self.LEFT_WIDTH)

        right_width = self.RIGHT_WIDTH / (1.0 - self.LEFT_WIDTH)

        right, center = split(rest, imgui.Dir.right, right_width)

        bottom, center = split(center, imgui.Dir.down, self.BOTTOM_HEIGHT)

        left_top, left_bottom = split(left, imgui.Dir.up, self.LEFT_SPLIT)

        nodes = {
            DockSlot.LEFT_TOP: left_top,
            DockSlot.LEFT_BOTTOM: left_bottom,
            DockSlot.RIGHT: right,
            DockSlot.BOTTOM: bottom,
        }

        first_in_slot = {}

        for panel in self._panels:

            internal.dock_builder_dock_window(panel.name, nodes[panel.slot])

            first_in_slot.setdefault(panel.slot, panel.name)

        internal.dock_builder_finish(dock_id)

        # Each area opens on its first panel's tab.
        for name in first_in_slot.values():
            self._panels.request_focus(name)
