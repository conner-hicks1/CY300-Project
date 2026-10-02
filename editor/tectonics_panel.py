import math
import random

from typing import TYPE_CHECKING

from imgui_bundle import imgui

from ecs.components import (
    PlanetComponent,
    TectonicsComponent,
    TransformComponent
)
from ecs.entity import Entity

from systems.planet_system import TERRAIN_VIEWS, PlanetSystem
from systems.tectonics_system import TectonicsSystem

from ui.panels import DockSlot

if TYPE_CHECKING:
    from editor.scene_editor import SceneEditor


# Data views most useful while watching plates move.
_TECTONIC_VIEWS = ("Natural", "Plates", "Crust age", "Boundaries", "Elevation")


class TectonicsPanel:

    # =====================================================
    # Tectonics Panel
    # =====================================================
    #
    # Plays the plate simulation of the selected (or first)
    # planet: play / pause / step / reset, speed, plate
    # settings, and what the plates are doing right now.
    # Docks as a tab next to the Planet panel.

    def __init__(
        self,
        editor: "SceneEditor",
        tectonics: TectonicsSystem,
        planets: PlanetSystem
    ):

        self._editor = editor
        self._tectonics = tectonics
        self._planets = planets

        editor.panels.register(
            "Tectonics",
            DockSlot.LEFT_BOTTOM,
            description="Plate tectonics: continents, ocean basins and mountain belts over time."
        )

    def draw(self):

        editor = self._editor

        if not editor.visible or not editor.panels.is_visible("Tectonics"):
            return

        if editor.panels.begin("Tectonics"):

            imgui.push_item_width(-125.0)

            try:
                self._draw_contents()
            finally:
                imgui.pop_item_width()

        editor.panels.end()

    # =====================================================
    # Contents
    # =====================================================

    def _planet(
        self
    ) -> Entity | None:

        scene = self._editor.scene

        selected = self._editor.selected

        if selected is not None and scene.try_get_component(selected, PlanetComponent) is not None:
            return selected

        for entity, _, _ in scene.registry.view_with(TransformComponent, PlanetComponent):
            return entity

        return None

    def _draw_contents(self):

        editor = self._editor
        scene = editor.scene

        planet = self._planet()

        if planet is None:

            imgui.text_wrapped("Add a planet first (Planet panel or File > New Demo Scene).")

            return

        component = scene.try_get_component(planet, TectonicsComponent)

        if component is None:

            imgui.text_wrapped(
                "Plate tectonics replaces the planet's noise continents with "
                "a simulation: plates drift, collide into mountain ranges, "
                "and pull apart into new oceans."
            )

            if imgui.button("Enable plate tectonics"):

                scene.add_component(planet, TectonicsComponent())

                editor.record("Enable plate tectonics")

            return

        status = self._tectonics.status(planet, component)

        # -------------------------------------------------
        # Playback
        # -------------------------------------------------

        if not status.ready:

            imgui.text_colored((0.95, 0.8, 0.35, 1.0), "Setting up plates...")

        elif status.catching_up:

            imgui.text_colored(
                (0.95, 0.8, 0.35, 1.0),
                f"Re-simulating: {status.time:,.0f} / {status.target_time:,.0f} Myr"
            )

        imgui.text(f"Planet age: {status.time:,.0f} million years")

        width = (imgui.get_content_region_avail().x - 2 * imgui.get_style().item_spacing.x) / 3.0

        if status.playing:

            imgui.push_style_color(imgui.Col_.button, (0.75, 0.2, 0.15, 1.0))

            if imgui.button("Pause", (width, 0)):
                self._tectonics.play(planet, False)

            imgui.pop_style_color()

        else:

            imgui.begin_disabled(not status.ready)

            if imgui.button("Play", (width, 0)):
                self._tectonics.play(planet, True)

            imgui.end_disabled()

        imgui.set_item_tooltip("Run the simulation. Watch from orbit (Planet > Whole planet).")

        imgui.same_line()

        imgui.begin_disabled(not status.ready or status.playing)

        if imgui.button(f"Step {component.time_step:g} Myr", (width, 0)):
            self._tectonics.step(planet)

        imgui.end_disabled()

        imgui.same_line()

        if imgui.button("Reset", (width, 0)):

            self._tectonics.reset(planet, component)

            editor.record("Reset tectonics")

        imgui.set_item_tooltip("Back to the starting continents (same seed).")

        changed, rate = imgui.slider_float(
            "Speed",
            self._tectonics.rate,
            5.0,
            500.0,
            "%.0f Myr/s",
            imgui.SliderFlags_.logarithmic.value
        )

        if changed:
            self._tectonics.rate = rate

        imgui.set_item_tooltip(
            "Upper limit; each step takes about "
            f"{max(status.step_seconds, 0.01) * 1000:.0f} ms to compute."
        )

        # -------------------------------------------------
        # View shortcuts
        # -------------------------------------------------

        imgui.separator_text("Show")

        names = [name for name, _ in TERRAIN_VIEWS]

        for index, label in enumerate(_TECTONIC_VIEWS):

            if index:
                imgui.same_line()

            mode = names.index(label)

            active = self._planets.view_mode == mode

            if active:
                imgui.push_style_color(imgui.Col_.button, (0.33, 0.56, 0.86, 1.0))

            if imgui.small_button(label):
                self._planets.view_mode = mode

            if active:
                imgui.pop_style_color()

            imgui.set_item_tooltip(TERRAIN_VIEWS[mode][1])

        # -------------------------------------------------
        # Now
        # -------------------------------------------------

        if status.ready:

            imgui.separator_text("Now")

            imgui.text(
                f"{len(status.plates)} plates, "
                f"{status.continental_fraction * 100.0:.0f}% continental crust"
            )

            if imgui.begin_table(
                "plates",
                3,
                imgui.TableFlags_.row_bg.value | imgui.TableFlags_.sizing_stretch_prop.value
            ):

                imgui.table_setup_column("Plate")
                imgui.table_setup_column("Area")
                imgui.table_setup_column("Speed")
                imgui.table_headers_row()

                for info in sorted(status.plates, key=lambda plate: -plate.cell_fraction)[:12]:

                    imgui.table_next_row()

                    imgui.table_next_column()

                    color = _plate_color(info.index)

                    imgui.color_button(f"##plate{info.index}", (*color, 1.0), 0, (12, 12))
                    imgui.same_line()
                    imgui.text(
                        f"#{info.index}"
                        + (" (continental)" if info.continental_fraction > 0.3 else "")
                    )

                    imgui.table_next_column()
                    imgui.text(f"{info.cell_fraction * 100.0:.0f}%")

                    imgui.table_next_column()
                    imgui.text(f"{info.speed_cm_per_year:.1f} cm/yr")

                imgui.end_table()

        # -------------------------------------------------
        # Settings (restart the simulation)
        # -------------------------------------------------

        if imgui.collapsing_header("Settings"):

            imgui.text_disabled("Changing these restarts from 0 Myr.")

            changed, seed = imgui.input_int("Seed", component.seed)

            if changed:
                component.seed = seed
                editor.record("Tectonics seed")

            imgui.same_line()

            if imgui.button("Random"):
                component.seed = random.randint(1, 999_999)
                editor.record("Random tectonics seed")

            changed, count = imgui.slider_int("Plates", component.plate_count, 3, 30)

            if changed:
                component.plate_count = count

            self._record_on_release("Plate count")

            self._slider(component, "land_fraction", "Continents", 0.05, 0.7, "%.2f",
                         "Fraction of the surface that starts as continent.")

            self._slider(component, "plate_speed", "Plate speed", 1.0, 20.0, "%.1f cm/yr",
                         "Earth's plates move 2-10 cm per year.")

            self._slider(component, "time_step", "Time step", 1.0, 20.0, "%.0f Myr",
                         "Simulated time per step. Larger = faster but coarser.")

            changed, resolution = imgui.slider_int("Grid", component.resolution, 48, 192)

            if changed:
                component.resolution = resolution

            self._record_on_release("Tectonics grid")

            spacing = (math.pi * 0.5 * scene.get_component(planet, PlanetComponent).radius) / max(component.resolution, 1)

            imgui.set_item_tooltip(
                f"Cells per cube face edge: {6 * component.resolution ** 2:,} cells, "
                f"~{spacing / 1000.0:.0f} km apart. Higher is sharper but slower."
            )

        if imgui.button("Disable plate tectonics"):

            scene.remove_component(planet, TectonicsComponent)

            editor.record("Disable plate tectonics")

    # -----------------------------------------------------
    # Widgets
    # -----------------------------------------------------

    def _slider(
        self,
        component,
        field: str,
        label: str,
        low: float,
        high: float,
        fmt: str,
        tooltip: str
    ):

        changed, value = imgui.slider_float(label, float(getattr(component, field)), low, high, fmt)

        if changed:
            setattr(component, field, value)

        self._record_on_release(label)

        imgui.set_item_tooltip(tooltip)

    def _record_on_release(
        self,
        label: str
    ):

        if imgui.is_item_deactivated_after_edit():
            self._editor.record(label)


def _plate_color(
    index: int
) -> tuple[float, float, float]:
    """Same colors as the Plates view (include/terrain.glsl plateColor)."""

    hue = (index * 0.618034) % 1.0

    return tuple(
        0.55 + 0.45 * math.cos(6.2831853 * (hue + offset))
        for offset in (0.0, 0.33, 0.67)
    )
