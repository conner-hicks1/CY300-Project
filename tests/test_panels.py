import numpy as np
import pytest

from editor.planet_panel import TERRAIN_PRESETS, _distance, sun_orientation
from ecs.components import PlanetComponent
from math3d import quaternion
from systems.planet_system import TERRAIN_VIEWS
from ui.editor_layout import ScreenRect
from ui.panels import DockSlot, PanelRegistry


# =========================================================
# Panel Registry
# =========================================================

def test_registry_keeps_order_and_visibility():

    panels = PanelRegistry()

    panels.register("A", DockSlot.LEFT_TOP)
    panels.register("B", DockSlot.BOTTOM, visible=False)

    assert [panel.name for panel in panels] == ["A", "B"]

    assert panels.is_visible("A")
    assert not panels.is_visible("B")
    assert not panels.is_visible("missing")

    panels.show("A", False)
    panels.show("B")

    assert not panels.is_visible("A")
    assert panels.is_visible("B")


def test_every_slot_is_known_to_the_layout():

    slots = {DockSlot.LEFT_TOP, DockSlot.LEFT_BOTTOM, DockSlot.RIGHT, DockSlot.BOTTOM}

    panels = PanelRegistry()

    for slot in slots:
        panels.register(slot, slot)

    assert {panel.slot for panel in panels} == slots


def test_screen_rect_contains():

    rect = ScreenRect(10.0, 20.0, 100.0, 50.0)

    assert rect.contains(10.0, 20.0)
    assert rect.contains(109.0, 69.0)
    assert not rect.contains(110.0, 30.0)
    assert not rect.contains(50.0, 19.0)


# =========================================================
# Planet Panel Helpers
# =========================================================

@pytest.mark.parametrize("meters, text", [
    (12.4, "12 m"),
    (999.0, "999 m"),
    (2_950.0, "3.0 km"),
    (400_000.0, "400 km"),
    (9_555_000.0, "9,555 km"),
])
def test_distance_formatting(meters, text):

    assert _distance(meters) == text


def test_sun_orientation_points_light_away_from_sun():

    for sun in ([0.3, 0.8, -0.5], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]):

        sun = np.array(sun) / np.linalg.norm(sun)

        matrix = quaternion.to_matrix3(sun_orientation(sun))

        light_forward = -matrix[:, 2]

        np.testing.assert_allclose(light_forward, -sun, atol=1e-9)


def test_presets_only_set_terrain_fields():

    fields = set(PlanetComponent.__dataclass_fields__)

    for preset in TERRAIN_PRESETS.values():
        assert set(preset) <= fields


def test_terrain_views_start_with_natural():

    assert TERRAIN_VIEWS[0][0] == "Natural"
    assert len({name for name, _ in TERRAIN_VIEWS}) == len(TERRAIN_VIEWS)
