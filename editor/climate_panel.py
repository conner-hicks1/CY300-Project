from typing import TYPE_CHECKING

from imgui_bundle import imgui

from ecs.components import (
    BodyComponent,
    ClimateComponent,
    PlanetComponent,
    TransformComponent
)
from ecs.entity import Entity

from planet.bodies import climate_physics
from planet.climate import BIOMES, KELVIN, STEFAN_BOLTZMANN
from planet.phases import liquid_range

from systems.climate_system import ClimateSystem
from systems.planet_system import TERRAIN_VIEWS, PlanetSystem

from ui.panels import DockSlot

if TYPE_CHECKING:
    from editor.scene_editor import SceneEditor


_CLIMATE_VIEWS = ("Natural", "Temperature", "Rainfall", "Biomes")

# Same colors as biomeColor() in include/terrain.glsl.
_BIOME_COLORS = (
    (0.92, 0.95, 1.00),
    (0.62, 0.60, 0.48),
    (0.18, 0.40, 0.35),
    (0.15, 0.55, 0.18),
    (0.72, 0.80, 0.35),
    (0.95, 0.80, 0.45),
    (0.85, 0.65, 0.25),
    (0.02, 0.38, 0.10),
)


class ClimatePanel:

    # =====================================================
    # Climate Panel
    # =====================================================
    #
    # Climate of the selected (or first) planet: global
    # temperature, humidity and axial tilt, a summary of
    # what the land looks like (biome shares), and quick
    # access to the temperature / rainfall / biome views.
    # Docks as a tab next to the Planet panel.

    def __init__(
        self,
        editor: "SceneEditor",
        climate: ClimateSystem,
        planets: PlanetSystem
    ):

        self._editor = editor
        self._climate = climate
        self._planets = planets

        editor.panels.register(
            "Climate",
            DockSlot.LEFT_BOTTOM,
            description="Temperature, winds, rainfall and biomes."
        )

    def draw(self):

        editor = self._editor

        if not editor.visible or not editor.panels.is_visible("Climate"):
            return

        if editor.panels.begin("Climate"):

            imgui.push_item_width(-125.0)

            try:
                self._draw_contents()
            finally:
                imgui.pop_item_width()

        editor.panels.end()

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

        component = scene.try_get_component(planet, ClimateComponent)

        if component is None:

            imgui.text_wrapped(
                "The climate model works out temperature, winds and "
                "rainfall from the planet's land and oceans, and grows "
                "biomes from them: rainforests, deserts in the rain "
                "shadow of mountains, taiga, tundra and ice."
            )

            if imgui.button("Enable climate"):

                scene.add_component(planet, ClimateComponent())

                editor.record("Enable climate")

            return

        status = self._climate.status(planet)

        if not status.ready or status.computing:

            imgui.text_colored((0.95, 0.8, 0.35, 1.0), "Computing climate...")

        # -------------------------------------------------
        # Controls
        # -------------------------------------------------

        self._slider(
            component, "temperature_offset", "Temperature", -25.0, 25.0, "%+.1f C",
            "Warms or cools the whole planet (ice age vs hothouse)."
        )

        self._slider(
            component, "humidity", "Humidity", 0.1, 4.0, "%.2f x Earth",
            "How much water the air carries: overall rainfall.",
            logarithmic=True
        )

        self._slider(
            component, "axial_tilt", "Axial tilt", 0.0, 60.0, "%.1f deg",
            "Earth: 23.4. Low tilt: hot equator, frozen poles. "
            "High tilt: the year's sunlight evens out."
        )

        self._draw_physics(planet, component)

        # -------------------------------------------------
        # Views
        # -------------------------------------------------

        imgui.separator_text("Show")

        names = [name for name, _ in TERRAIN_VIEWS]

        for index, label in enumerate(_CLIMATE_VIEWS):

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
        # Summary
        # -------------------------------------------------

        field = status.field

        self._draw_liquid(planet, field)

        if field is not None:

            imgui.separator_text("Whole planet")

            imgui.text(
                f"Mean {field.global_mean_temperature:+.1f} C   "
                f"range {field.min_temperature:+.0f} .. {field.max_temperature:+.0f} C"
            )

            imgui.separator_text("Land")

            imgui.text(f"Average temperature  {field.mean_temperature:+.1f} C")
            imgui.text(f"Average rainfall     {field.mean_precipitation:,.0f} mm / year")

            for index, fraction in sorted(
                enumerate(field.biome_fractions),
                key=lambda item: -item[1]
            ):

                if fraction < 0.005:
                    continue

                imgui.color_button(
                    f"##biome{index}",
                    (*_BIOME_COLORS[index], 1.0),
                    0,
                    (12, 12)
                )

                imgui.same_line()

                imgui.progress_bar(
                    fraction,
                    (imgui.get_content_region_avail().x * 0.45, 0),
                    f"{fraction * 100.0:.0f}%"
                )

                imgui.same_line()

                imgui.text(BIOMES[index])

            imgui.text_disabled(
                f"Computed in {status.compute_seconds:.2f} s; "
                "updates when the land changes."
            )

        if imgui.button("Disable climate"):

            scene.remove_component(planet, ClimateComponent)

            editor.record("Disable climate")

    # -----------------------------------------------------
    # Liquid and Ice
    # -----------------------------------------------------

    def _draw_liquid(
        self,
        planet: Entity,
        field
    ):
        """
        The seas' phase at this pressure and temperature
        (planet/phases.py), the ice caps, and life.
        """

        scene = self._editor.scene

        component = scene.try_get_component(planet, PlanetComponent)

        if component is None:
            return

        body = scene.try_get_component(planet, BodyComponent)

        pressure = body.surface_pressure_bar if body is not None else 1.014

        imgui.separator_text("Liquid and ice")

        phase = liquid_range(component.liquid, pressure)

        if component.liquid == "none":

            imgui.text("No liquid: basins stay dry.")

        elif phase is None:

            imgui.text(f"Liquid  {component.liquid} (molten by volcanic heat)")

        elif not phase.can_be_liquid:

            imgui.text_wrapped(
                f"No liquid {component.liquid} at {pressure:.3g} bar (below its "
                f"triple point): seas are ice, sublimating above {phase.boiling:+.0f} C."
            )

        else:

            imgui.text(
                f"Liquid  {component.liquid}: freezes {phase.freezing:+.0f} C, "
                f"boils {phase.boiling:+.0f} C"
            )
            imgui.set_item_tooltip(f"At the surface pressure ({pressure:.3g} bar).")

        if field is not None and field.liquid_state != "none":

            state = field.liquid_state

            color = (
                (0.95, 0.55, 0.3, 1.0) if state == "boiled away"
                else (0.7, 0.85, 1.0, 1.0) if "frozen" in state
                else (0.5, 0.8, 0.95, 1.0)
            )

            detail = f" ({field.frozen_fraction * 100.0:.0f}% iced over)" if state == "partly frozen" else ""

            imgui.text_colored(color, f"Seas    {state}{detail}")

            if state == "boiled away":
                imgui.set_item_tooltip(
                    "Hotter than boiling on average: the oceans evaporated "
                    "and the basins are dry."
                )

        if component.ice == "none":

            imgui.text_disabled("No ice caps.")

        else:

            imgui.text(f"Ice     {component.ice.replace('_', ' ')} below {component.frost_point:+.0f} C")
            imgui.set_item_tooltip("Where the annual mean is colder, the ground ices over.")

        if component.palette == "biomes":

            changed, life = imgui.checkbox("Life", component.life)

            if changed:

                component.life = life

                self._editor.record("Life")

            imgui.set_item_tooltip("Vegetation; without it the same climate zones are bare ground.")

    # -----------------------------------------------------
    # Physics
    # -----------------------------------------------------

    def _draw_physics(
        self,
        planet: Entity,
        component: ClimateComponent
    ):
        """
        What-if controls on the body itself: move it closer
        to or farther from its star, brighten or darken it,
        thicken or thin its greenhouse. The climate follows
        from the energy balance.
        """

        scene = self._editor.scene

        imgui.separator_text("Physics")

        body = scene.try_get_component(planet, BodyComponent)

        if body is None:

            imgui.text_disabled("Earth's sunlight and air (no body profile).")

            return

        self._slider(
            body, "orbit_distance_au", "Star distance", 0.1, 50.0, "%.3f AU",
            "Distance from the star. Sunlight falls with its square.",
            logarithmic=True
        )

        self._slider(
            body, "bond_albedo", "Albedo", 0.0, 0.95, "%.3f",
            "Fraction of sunlight reflected back to space "
            "(clouds, ice, bright ground). Earth 0.31, Venus 0.76."
        )

        self._slider(
            body, "greenhouse_depth", "Greenhouse", 0.0, 300.0, "%.2f",
            "Infrared optical depth of the air: how well it "
            "traps heat. Earth ~1, Venus ~150, airless 0.",
            logarithmic=True
        )

        planet_component = scene.try_get_component(planet, PlanetComponent)

        physics = climate_physics(
            body,
            planet_component.liquid if planet_component is not None else "water",
            component.humidity
        )

        absorbed = physics["stellar_flux"] / 4.0 * (1.0 - physics["bond_albedo"])

        equilibrium = (max(absorbed, 0.0) / STEFAN_BOLTZMANN) ** 0.25 - KELVIN

        greenhouse = (1.0 + 0.75 * physics["greenhouse_depth"]) ** 0.25

        imgui.text(f"Sunlight      {physics['stellar_flux']:,.0f} W/m^2")
        imgui.set_item_tooltip("Starlight arriving at the body (Earth 1361).")

        imgui.text(f"Equilibrium   {equilibrium:+.0f} C  (no greenhouse)")

        imgui.text(
            f"With greenhouse  {(equilibrium + KELVIN) * greenhouse - KELVIN:+.0f} C"
        )
        imgui.set_item_tooltip("Uniform-planet estimate; the map adds latitude and transport.")

        imgui.text(f"Lapse rate    {physics['lapse_rate'] * 1000.0:.1f} C per km")
        imgui.set_item_tooltip("From gravity and the air's heat capacity (g / cp), less in moist or thin air.")

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
        tooltip: str,
        logarithmic: bool = False
    ):

        flags = imgui.SliderFlags_.logarithmic.value if logarithmic else 0

        changed, value = imgui.slider_float(label, float(getattr(component, field)), low, high, fmt, flags)

        if changed:
            setattr(component, field, value)

        if imgui.is_item_deactivated_after_edit():
            self._editor.record(label)

        imgui.set_item_tooltip(tooltip)
