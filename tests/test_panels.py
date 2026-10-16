from ui.panels import PanelRegistry


# =========================================================
# Panel Registry (developer tools)
# =========================================================

def test_registry_keeps_order_and_visibility():

    panels = PanelRegistry()

    panels.register("A")
    panels.register("B", visible=False)

    assert [panel.name for panel in panels] == ["A", "B"]

    assert panels.is_visible("A")
    assert not panels.is_visible("B")
    assert not panels.is_visible("missing")

    panels.show("A", False)
    panels.show("B")

    assert not panels.is_visible("A")
    assert panels.is_visible("B")
