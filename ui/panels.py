from dataclasses import dataclass

from imgui_bundle import imgui


# =========================================================
# Panel Registry
# =========================================================
#
# Every tool window of the editor is registered here with
# its default dock slot. The registry
#
#   * remembers which panels are open (closing a panel's
#     tab hides it; View menu shows it again),
#   * drives the View menu,
#   * tells EditorLayout where each panel docks by default.
#
# Adding a tool window (e.g. a tectonics or climate panel)
# is one register() call plus:
#
#     if panels.is_visible("Tectonics"):
#         if panels.begin("Tectonics"):
#             ...widgets...
#         panels.end()

class DockSlot:

    LEFT_TOP = "left_top"
    LEFT_BOTTOM = "left_bottom"
    RIGHT = "right"
    BOTTOM = "bottom"


@dataclass(slots=True)
class Panel:

    name: str
    slot: str
    visible: bool = True

    # One line for the View menu tooltip.
    description: str = ""


class PanelRegistry:

    def __init__(self):

        self._panels: dict[str, Panel] = {}

        # Panel -> frames to wait before bringing it to the
        # front (a freshly built layout docks its windows
        # during the first frame; focusing then is lost).
        self._focus: dict[str, int] = {}

    def register(
        self,
        name: str,
        slot: str,
        visible: bool = True,
        description: str = ""
    ) -> Panel:

        panel = Panel(name, slot, visible, description)

        self._panels[name] = panel

        return panel

    def __iter__(self):

        return iter(self._panels.values())

    def get(
        self,
        name: str
    ) -> Panel | None:

        return self._panels.get(name)

    def is_visible(
        self,
        name: str
    ) -> bool:

        panel = self._panels.get(name)

        return panel is not None and panel.visible

    def show(
        self,
        name: str,
        visible: bool = True
    ):

        panel = self._panels.get(name)

        if panel is not None:
            panel.visible = visible

    # -----------------------------------------------------
    # Window
    # -----------------------------------------------------

    def begin(
        self,
        name: str
    ) -> bool:
        """
        imgui.begin() with a close button wired to the
        panel's visibility. Returns False when the window is
        collapsed or its tab is hidden (skip the widgets).
        Always pair with end().
        """

        if name in self._focus:

            self._focus[name] -= 1

            if self._focus[name] <= 0:

                del self._focus[name]

                imgui.set_next_window_focus()

        expanded, keep_open = imgui.begin(name, True)

        if not keep_open:
            self.show(name, False)

        return expanded

    def request_focus(
        self,
        name: str
    ):
        """Select the panel's tab (and focus it) next frame."""

        self._focus[name] = 2

    @staticmethod
    def end():

        imgui.end()

    # -----------------------------------------------------
    # Menu
    # -----------------------------------------------------

    def draw_menu_items(self):
        """Checkable item per panel (inside a View menu)."""

        for panel in self._panels.values():

            clicked, _ = imgui.menu_item(panel.name, "", panel.visible)

            if clicked:
                panel.visible = not panel.visible

            if panel.description:
                imgui.set_item_tooltip(panel.description)
