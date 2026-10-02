from typing import TYPE_CHECKING

from imgui_bundle import imgui

from ecs.components import (
    AtmosphereComponent,
    BodyComponent,
    CameraComponent,
    CameraControllerComponent,
    ClimateComponent,
    DirectionalLightComponent,
    HierarchyComponent,
    MeshRendererComponent,
    NameComponent,
    PlanetComponent,
    PointLightComponent,
    RotatorComponent,
    SpotLightComponent,
    TectonicsComponent,
    TransformComponent
)
from ecs.entity import Entity

from planet.phases import ICES

from ui.window_utils import keep_window_on_screen

from scene.scene_serializer import (
    COMPONENT_CODECS,
    SceneFormatError
)

if TYPE_CHECKING:
    from editor.scene_editor import SceneEditor


# =========================================================
# Inspector Panel
# =========================================================
#
# Edits the selected entity's components.
#
# Undo granularity: continuous widgets (drags, sliders,
# color pickers) record one step when the user lets go
# (is_item_deactivated_after_edit); discrete widgets
# (checkboxes, combos) record as soon as they change.


def draw_inspector_panel(
    editor: "SceneEditor"
):

    io = imgui.get_io()

    imgui.set_next_window_pos(
        (io.display_size.x - 340.0, imgui.get_frame_height() + 314.0),
        imgui.Cond_.first_use_ever
    )

    imgui.set_next_window_size(
        (330.0, max(200.0, io.display_size.y - imgui.get_frame_height() - 330.0)),
        imgui.Cond_.first_use_ever
    )

    editor.panels.begin("Inspector")

    keep_window_on_screen()

    entity = editor.selected

    if entity is None:

        imgui.text_disabled(
            "Nothing selected.\n\n"
            "Click an object in the viewport or the\n"
            "Hierarchy, or use the Create menu."
        )

        imgui.end()

        return

    # Leave room for the field labels on the right.
    imgui.push_item_width(-125.0)

    try:
        _Inspector(editor, entity).draw()
    finally:
        imgui.pop_item_width()

    imgui.end()


class _Inspector:

    def __init__(
        self,
        editor: "SceneEditor",
        entity: Entity
    ):

        self.editor = editor
        self.scene = editor.scene
        self.entity = entity

        # Structural changes run after drawing.
        self.actions = []

    # =====================================================
    # Undo Helpers
    # =====================================================

    def continuous(
        self,
        label: str
    ):
        """Call right after a drag/slider/color widget."""

        if imgui.is_item_deactivated_after_edit():
            self.editor.record(f"{label} ({self.editor.entity_name(self.entity)})")

    def discrete(
        self,
        changed: bool,
        label: str
    ):
        """Call with the `changed` flag of a checkbox/combo."""

        if changed:
            self.editor.record(f"{label} ({self.editor.entity_name(self.entity)})")

    # =====================================================
    # Draw
    # =====================================================

    def draw(self):

        self._draw_name()

        imgui.separator()

        for codec in COMPONENT_CODECS:

            component = self.scene.try_get_component(
                self.entity,
                codec.component_type
            )

            if component is None or codec.name == "Name":
                continue

            self._draw_component(
                codec,
                component
            )

        imgui.spacing()

        self._draw_add_component()

        for action in self.actions:
            action()

    def _draw_name(self):

        name = self.scene.try_get_component(
            self.entity,
            NameComponent
        )

        if name is None:
            return

        changed, value = imgui.input_text(
            "Name",
            name.name
        )

        if changed and value.strip():
            name.name = value

        self.continuous("Rename")

        imgui.text_disabled(
            f"Entity {self.entity.index} (generation {self.entity.generation})"
        )

    def _draw_component(
        self,
        codec,
        component
    ):

        imgui.push_id(codec.name)

        flags = imgui.TreeNodeFlags_.default_open.value

        if codec.removable:

            opened, keep = imgui.collapsing_header(
                codec.name,
                True,
                flags
            )

            if not keep:

                self.actions.append(
                    lambda: self._remove(codec)
                )

        else:

            opened = imgui.collapsing_header(
                codec.name,
                flags
            )

        if opened:

            editor_function = _COMPONENT_EDITORS.get(
                type(component)
            )

            if editor_function is not None:
                editor_function(self, component)
            else:
                imgui.text_disabled("(no editable fields)")

        imgui.pop_id()

    def _remove(
        self,
        codec
    ):

        self.scene.remove_component(
            self.entity,
            codec.component_type
        )

        self.editor.record(
            f"Remove {codec.name} ({self.editor.entity_name(self.entity)})"
        )

    def _draw_add_component(self):

        width = imgui.get_content_region_avail().x

        if imgui.button("Add Component", (width, 0)):
            imgui.open_popup("add_component")

        if not imgui.begin_popup("add_component"):
            return

        any_available = False

        for codec in COMPONENT_CODECS:

            if (
                codec.create_default is None
                or self.scene.has_component(self.entity, codec.component_type)
            ):
                continue

            any_available = True

            allowed, reason = self.editor.can_add_component(
                codec.component_type
            )

            if imgui.menu_item(codec.name, "", False, allowed)[0]:
                self.actions.append(lambda c=codec: self._add(c))

            if not allowed:
                imgui.set_item_tooltip(reason)

        if not any_available:
            imgui.text_disabled("All components added.")

        imgui.end_popup()

    def _add(
        self,
        codec
    ):

        try:
            component = codec.create_default(self.editor.resources)

        except SceneFormatError as error:

            self.editor.set_status(str(error), True)

            return

        self.scene.add_component(
            self.entity,
            component
        )

        self.editor.record(
            f"Add {codec.name} ({self.editor.entity_name(self.entity)})"
        )

    # =====================================================
    # Shared Widgets
    # =====================================================

    def color(
        self,
        component,
        field: str = "color",
        label: str = "Color"
    ):

        changed, value = imgui.color_edit3(
            label,
            list(getattr(component, field))
        )

        if changed:
            setattr(component, field, tuple(value))

        self.continuous(label)

    def slider(
        self,
        component,
        field: str,
        label: str,
        minimum: float,
        maximum: float,
        fmt: str = "%.2f",
        logarithmic: bool = False
    ):

        flags = (
            imgui.SliderFlags_.logarithmic.value
            if logarithmic
            else 0
        )

        changed, value = imgui.slider_float(
            label,
            float(getattr(component, field)),
            minimum,
            maximum,
            fmt,
            flags
        )

        if changed:
            setattr(component, field, value)

        self.continuous(label)

    def checkbox(
        self,
        component,
        field: str,
        label: str
    ):

        changed, value = imgui.checkbox(
            label,
            bool(getattr(component, field))
        )

        if changed:
            setattr(component, field, value)

        self.discrete(changed, label)

    def vec3(
        self,
        component,
        field: str,
        label: str,
        speed: float
    ):

        changed, value = imgui.drag_float3(
            label,
            list(getattr(component, field)),
            speed
        )

        if changed:
            setattr(component, field, tuple(value))

        self.continuous(label)


# =========================================================
# Component Editors
# =========================================================

def _edit_transform(
    inspector: _Inspector,
    component: TransformComponent
):

    transform = component.transform

    for label, attribute, speed in (
        ("Position", "position", 0.02),
        ("Rotation", "rotation", 0.5),
        ("Scale", "scale", 0.01),
    ):

        changed, value = imgui.drag_float3(
            label,
            getattr(transform, attribute).tolist(),
            speed
        )

        if changed:
            setattr(transform, attribute, value)

        inspector.continuous(label)

    if imgui.small_button("Reset"):

        transform.position = (0.0, 0.0, 0.0)
        transform.rotation = (0.0, 0.0, 0.0)
        transform.scale = (1.0, 1.0, 1.0)

        inspector.discrete(True, "Reset transform")


def _edit_hierarchy(
    inspector: _Inspector,
    component: HierarchyComponent
):

    editor = inspector.editor

    imgui.text(
        f"Parent: {editor.entity_name(component.parent)}"
    )

    if imgui.small_button("Select parent"):
        inspector.actions.append(lambda: editor.select(component.parent))

    imgui.same_line()

    if imgui.small_button("Unparent"):

        # Keeps the world transform; recorded for undo.
        inspector.actions.append(
            lambda: editor.reparent(inspector.entity, None)
        )

    imgui.text_disabled("Drag in the Hierarchy to re-parent.")


def _edit_mesh_renderer(
    inspector: _Inspector,
    component: MeshRendererComponent
):

    resources = inspector.editor.resources

    for label, manager, attribute in (
        ("Mesh", resources.meshes, "mesh"),
        ("Material", resources.materials, "material"),
    ):

        current = getattr(component, attribute)

        current_key = manager.key_of(current) or "?"

        if imgui.begin_combo(label, current_key):

            for key, handle in manager.handle_items():

                if key.startswith("engine/"):
                    continue

                clicked, _ = imgui.selectable(
                    key,
                    key == current_key
                )

                if clicked and handle != current:

                    setattr(component, attribute, handle)

                    inspector.discrete(True, f"Change {label.lower()}")

            imgui.end_combo()

    inspector.checkbox(component, "casts_shadows", "Casts shadows")


def _edit_camera(
    inspector: _Inspector,
    component: CameraComponent
):

    inspector.slider(component, "fov", "FOV", 10.0, 120.0, "%.1f")
    inspector.slider(component, "near", "Near", 0.01, 10.0, "%.3f", logarithmic=True)
    inspector.slider(component, "far", "Far", 1.0, 1000.0, "%.1f", logarithmic=True)

    # Keep the planes ordered; the projection needs it.
    if component.far <= component.near:
        component.far = component.near * 2.0

    inspector.checkbox(component, "primary", "Primary")

    if component.primary:
        imgui.text_disabled("The view is rendered through this camera.")


def _edit_camera_controller(
    inspector: _Inspector,
    component: CameraControllerComponent
):

    inspector.slider(component, "movement_speed", "Move speed", 0.1, 50.0, "%.1f", logarithmic=True)
    inspector.slider(component, "mouse_sensitivity", "Look sensitivity", 0.01, 1.0, "%.3f")

    inspector.checkbox(component, "planet_mode", "Planet mode")

    if component.planet_mode:
        inspector.vec3(component, "planet_center", "Planet center", 1.0)
        inspector.slider(component, "planet_radius", "Planet radius", 0.0, 1.0e7, "%.0f", logarithmic=True)
        inspector.slider(component, "altitude_speed", "Altitude speed", 0.0, 5.0, "%.2f")
        inspector.slider(component, "min_altitude", "Min altitude", 0.0, 1000.0, "%.1f", logarithmic=True)

        imgui.text_disabled(
            "With a planet in the scene, center and radius\n"
            "follow the nearest planet and the ground below."
        )

    imgui.text_disabled("Hold the right mouse button to fly.")


def _edit_planet(
    inspector: _Inspector,
    component: PlanetComponent
):

    changed, seed = imgui.input_int("Seed", component.seed)

    if changed:
        component.seed = seed

    inspector.discrete(changed, "Seed")

    inspector.slider(component, "radius", "Radius (m)", 1_000.0, 10_000_000.0, "%.0f", logarithmic=True)

    imgui.separator_text("Terrain")

    inspector.slider(component, "continent_frequency", "Continent freq.", 0.2, 6.0)
    inspector.slider(component, "continent_height", "Continent height", 0.0, 10_000.0, "%.0f")
    inspector.slider(component, "land_bias", "Land bias", -0.5, 0.5, "%.3f")
    inspector.slider(component, "mountain_frequency", "Mountain freq.", 5.0, 1000.0, "%.0f", logarithmic=True)
    inspector.slider(component, "mountain_height", "Mountain height", 0.0, 20_000.0, "%.0f")
    inspector.slider(component, "detail_height", "Detail height", 0.0, 2_000.0, "%.0f")

    imgui.separator_text("Craters")

    inspector.slider(component, "crater_density", "Impact rate", 0.0, 4.0, "%.2f x Moon")
    imgui.set_item_tooltip("0 = no craters. How many show also depends on the surface's age.")
    inspector.slider(component, "surface_age", "Surface age (Myr)", 0.0, 4_500.0, "%.0f")
    imgui.set_item_tooltip("Used where no tectonic simulation gives the crust's age.")
    inspector.slider(component, "crater_erosion", "Erosion (Myr)", 0.0, 5_000.0, "%.0f")
    imgui.set_item_tooltip("Craters older than about this are worn away (0 = never).")
    inspector.slider(component, "crater_min_diameter", "Smallest (m)", 0.0, 5_000.0, "%.0f")
    imgui.set_item_tooltip("The atmosphere burns up smaller impactors.")
    inspector.slider(component, "crater_transition", "Complex above (m)", 500.0, 40_000.0, "%.0f")
    imgui.set_item_tooltip("Larger craters have flat floors and central peaks.")
    inspector.checkbox(component, "crater_rays", "Bright rays")

    imgui.separator_text("Level of detail")

    for field, label, low, high in (
        ("resolution", "Chunk vertices", 5, 65),
        ("max_depth", "Max depth", 0, 20),
    ):

        changed, value = imgui.slider_int(label, getattr(component, field), low, high)

        if changed:
            setattr(component, field, value)

        inspector.continuous(label)

    inspector.slider(component, "split_factor", "Split distance", 0.5, 4.0)

    imgui.separator_text("Surface")

    for field, label, options in (
        ("liquid", "Liquid", ("water", "methane", "lava", "none")),
        ("palette", "Palette", ("biomes", "mineral", "bands")),
        ("ice", "Ice caps", ICES),
    ):

        current = getattr(component, field)

        index = options.index(current) if current in options else 0

        changed, index = imgui.combo(label, index, list(options))

        if changed:
            setattr(component, field, options[index])

        inspector.discrete(changed, label)

    if component.palette != "biomes":

        inspector.color(component, "color_low", "Low / belts")
        inspector.color(component, "color_high", "High / zones")

    if component.palette == "mineral":

        inspector.color(component, "color_steep", "Cliffs")
        inspector.color(component, "color_ice", "Frost / ice")

    if component.palette != "bands":

        inspector.slider(component, "frost_point", "Frost point (C)", -250.0, 50.0, "%.1f")
        imgui.set_item_tooltip("Below this annual mean temperature the ground ices over.")

    if component.palette == "biomes":

        inspector.checkbox(component, "life", "Life (vegetation)")

    if component.palette == "bands":

        changed, bands = imgui.slider_int("Bands", component.bands, 1, 40)

        if changed:
            component.bands = bands

        inspector.continuous("Bands")

    imgui.text_disabled(
        "Terrain changes rebuild the planet.\n"
        "Chunk counts: Stats panel."
    )


def _edit_atmosphere(
    inspector: _Inspector,
    component: AtmosphereComponent
):

    inspector.slider(component, "height", "Height (m)", 1_000.0, 1_000_000.0, "%.0f", logarithmic=True)

    imgui.separator_text("Rayleigh (air)")

    inspector.vec3(component, "rayleigh_scattering", "Scattering /Mm", 0.1)
    inspector.slider(component, "rayleigh_scale_height", "Scale height##r", 100.0, 50_000.0, "%.0f", logarithmic=True)

    imgui.separator_text("Mie (haze)")

    inspector.slider(component, "mie_scattering", "Scattering##m", 0.0, 100.0, "%.2f", logarithmic=True)
    inspector.slider(component, "mie_absorption", "Absorption##m", 0.0, 100.0, "%.2f", logarithmic=True)
    inspector.slider(component, "mie_scale_height", "Scale height##m", 100.0, 20_000.0, "%.0f", logarithmic=True)
    inspector.slider(component, "mie_anisotropy", "Anisotropy (g)", 0.0, 0.99)
    inspector.vec3(component, "mie_scattering_tint", "Scattering tint", 0.01)
    inspector.vec3(component, "mie_absorption_tint", "Absorption tint", 0.01)

    imgui.separator_text("Ozone")

    inspector.vec3(component, "ozone_absorption", "Absorption /Mm", 0.01)
    inspector.slider(component, "ozone_altitude", "Altitude (m)", 0.0, 100_000.0, "%.0f")
    inspector.slider(component, "ozone_thickness", "Thickness (m)", 1_000.0, 100_000.0, "%.0f")

    inspector.slider(component, "ground_albedo", "Ground albedo", 0.0, 1.0)

    imgui.text_disabled(
        "Needs a Planet on the same entity.\n"
        "Coefficients per megameter (1e-6 / m).\n"
        "Defaults: Earth."
    )


def _edit_tectonics(
    inspector: _Inspector,
    component: TectonicsComponent
):

    imgui.text(f"Simulated: {component.simulated_time:,.0f} million years")

    imgui.text_disabled(
        "Play, step and settings: Tectonics panel.\n"
        "Saved scenes re-simulate to this time\n"
        "on load (from the seed)."
    )


def _edit_climate(
    inspector: _Inspector,
    component: ClimateComponent
):

    inspector.slider(component, "temperature_offset", "Temperature (C)", -25.0, 25.0, "%+.1f")
    inspector.slider(component, "humidity", "Humidity", 0.1, 4.0, "%.2f", logarithmic=True)
    inspector.slider(component, "axial_tilt", "Axial tilt", 0.0, 60.0, "%.1f")

    imgui.text_disabled("Summary and views: Climate panel.")


def _edit_body(
    inspector: _Inspector,
    component: BodyComponent
):

    imgui.text(f"{component.name or 'Unnamed body'}  ({component.kind.replace('_', ' ')})")

    imgui.text_disabled(f"Orbits {component.orbits}; profile '{component.profile}'")

    if imgui.begin_table("body facts", 2, imgui.TableFlags_.row_bg.value):

        for label, value in body_facts(component):

            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_disabled(label)
            imgui.table_next_column()
            imgui.text(value)

        imgui.end_table()

    imgui.text_disabled(
        "Physical data from the body profile\n"
        "(assets/bodies). Planet panel > Body\n"
        "applies another preset."
    )


def body_facts(
    component: BodyComponent
) -> list[tuple[str, str]]:
    """Readable facts about a body, for the inspector and Planet panel."""

    def duration(hours):

        hours = abs(hours)

        return f"{hours:.1f} h" if hours < 48.0 else f"{hours / 24.0:.1f} days"

    year = component.year_days

    facts = [
        ("Gravity", f"{component.surface_gravity:.2f} m/s2  ({component.surface_gravity / 9.80665:.2f} g)"),
        ("Day (solar)", duration(component.solar_day_hours)),
        ("Rotation", duration(component.rotation_hours) + (" (retrograde)" if component.rotation_hours < 0 else "")),
        ("Year", f"{year:,.1f} days" if year < 1000 else f"{year / 365.256:,.1f} Earth years"),
        ("Distance", f"{component.orbit_distance_au:.3f} AU from the star"),
        ("Sunlight", f"{component.star_luminosity / max(component.orbit_distance_au, 1e-9) ** 2 * 100.0:.2f}% of Earth's"),
        ("Mean temperature", f"{component.mean_temperature:+.0f} C"),
        ("Surface pressure", (
            f"{component.surface_pressure_bar:,.3g} bar" if component.surface_pressure_bar > 0 else "none (vacuum)"
        )),
        ("Mass", f"{component.mass:.3e} kg"),
    ]

    return facts


def _edit_directional_light(
    inspector: _Inspector,
    component: DirectionalLightComponent
):

    inspector.color(component)
    inspector.slider(component, "intensity", "Intensity", 0.0, 20.0)
    inspector.checkbox(component, "casts_shadows", "Casts shadows")

    imgui.text_disabled(
        "Direction = the entity's forward (-Z).\n"
        "Also places the sun in the sky; ambient light\n"
        "comes from the sky (Render > Sky & IBL)."
    )


def _edit_point_light(
    inspector: _Inspector,
    component: PointLightComponent
):

    inspector.color(component)
    inspector.slider(component, "intensity", "Intensity", 0.0, 200.0, logarithmic=True)
    inspector.slider(component, "range", "Range", 0.1, 50.0)

    imgui.text_disabled("Point lights do not cast shadows.")


def _edit_spot_light(
    inspector: _Inspector,
    component: SpotLightComponent
):

    inspector.color(component)
    inspector.slider(component, "intensity", "Intensity", 0.0, 200.0, logarithmic=True)
    inspector.slider(component, "range", "Range", 0.1, 50.0)
    inspector.checkbox(component, "casts_shadows", "Casts shadows")
    inspector.slider(component, "outer_angle", "Outer angle", 0.0, 89.0, "%.1f")

    # Inner can never exceed outer.
    component.inner_angle = min(component.inner_angle, component.outer_angle)

    inspector.slider(component, "inner_angle", "Inner angle", 0.0, max(component.outer_angle, 0.01), "%.1f")


def _edit_rotator(
    inspector: _Inspector,
    component: RotatorComponent
):

    inspector.vec3(component, "degrees_per_second", "Degrees / sec", 0.5)

    if not inspector.editor.playing:
        imgui.text_disabled("Spins while playing (Play / Ctrl+P).")


_COMPONENT_EDITORS = {
    TransformComponent: _edit_transform,
    HierarchyComponent: _edit_hierarchy,
    MeshRendererComponent: _edit_mesh_renderer,
    CameraComponent: _edit_camera,
    CameraControllerComponent: _edit_camera_controller,
    PlanetComponent: _edit_planet,
    AtmosphereComponent: _edit_atmosphere,
    TectonicsComponent: _edit_tectonics,
    ClimateComponent: _edit_climate,
    BodyComponent: _edit_body,
    DirectionalLightComponent: _edit_directional_light,
    PointLightComponent: _edit_point_light,
    SpotLightComponent: _edit_spot_light,
    RotatorComponent: _edit_rotator,
}
