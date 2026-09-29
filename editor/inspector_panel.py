from typing import TYPE_CHECKING

from imgui_bundle import imgui

from ecs.components import (
    CameraComponent,
    CameraControllerComponent,
    DirectionalLightComponent,
    HierarchyComponent,
    MeshRendererComponent,
    NameComponent,
    PointLightComponent,
    RotatorComponent,
    SpotLightComponent,
    TransformComponent
)
from ecs.entity import Entity

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

    imgui.begin("Inspector")

    entity = editor.selected

    if entity is None:

        imgui.text_disabled(
            "Nothing selected.\n\n"
            "Click an object in the viewport or the\n"
            "Hierarchy, or use the Create menu."
        )

        imgui.end()

        return

    _Inspector(editor, entity).draw()

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

    imgui.text_disabled("Hold the right mouse button to fly.")


def _edit_directional_light(
    inspector: _Inspector,
    component: DirectionalLightComponent
):

    inspector.color(component)
    inspector.slider(component, "intensity", "Intensity", 0.0, 10.0)
    inspector.slider(component, "ambient", "Ambient", 0.0, 1.0, "%.3f")
    inspector.checkbox(component, "casts_shadows", "Casts shadows")

    imgui.text_disabled("Direction = the entity's forward (-Z).")


def _edit_point_light(
    inspector: _Inspector,
    component: PointLightComponent
):

    inspector.color(component)
    inspector.slider(component, "intensity", "Intensity", 0.0, 50.0)
    inspector.slider(component, "range", "Range", 0.1, 50.0)


def _edit_spot_light(
    inspector: _Inspector,
    component: SpotLightComponent
):

    inspector.color(component)
    inspector.slider(component, "intensity", "Intensity", 0.0, 50.0)
    inspector.slider(component, "range", "Range", 0.1, 50.0)
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
    DirectionalLightComponent: _edit_directional_light,
    PointLightComponent: _edit_point_light,
    SpotLightComponent: _edit_spot_light,
    RotatorComponent: _edit_rotator,
}
