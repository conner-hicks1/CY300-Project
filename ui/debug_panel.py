from collections.abc import Callable
from dataclasses import dataclass

from imgui_bundle import imgui

from core.timer import Timer

from ecs.components import (
    CameraComponent,
    DirectionalLightComponent,
    MeshRendererComponent,
    NameComponent,
    PointLightComponent,
    RotatorComponent,
    SpotLightComponent,
    TransformComponent
)
from ecs.entity import Entity

from graphics.render_settings import (
    RenderSettings,
    Tonemapper
)
from graphics.renderer import (
    Renderer,
    RenderStats
)

from resources.resources import Resources

from scene.scene import Scene


# =========================================================
# Debug Context
# =========================================================
#
# Everything the panel can inspect or edit, gathered by
# the Application each frame.

@dataclass(slots=True)
class DebugContext:

    scene: Scene
    resources: Resources
    settings: RenderSettings
    stats: RenderStats
    timer: Timer

    # Forces a rebuild of every shader; returns how many
    # were reloaded.
    reload_shaders: Callable[[], int]


class DebugPanel:

    # =====================================================
    # Construction
    # =====================================================

    def __init__(self):

        self.visible = True

        self._last_reload_message = ""

    # =====================================================
    # Draw
    # =====================================================

    def draw(
        self,
        context: DebugContext
    ):

        if not self.visible:
            return

        self._draw_engine_window(
            context
        )

        self._draw_scene_window(
            context
        )

        self._draw_materials_window(
            context
        )

    @staticmethod
    def _right_column_x() -> float:

        # Scene/Materials dock to the right edge; Engine
        # takes the left, keeping the middle clear.

        return max(
            350.0,
            imgui.get_io().display_size.x - 340.0
        )

    # =====================================================
    # Engine Window
    # =====================================================

    def _draw_engine_window(
        self,
        context: DebugContext
    ):

        imgui.set_next_window_pos(
            (10, 10),
            imgui.Cond_.first_use_ever
        )

        imgui.set_next_window_size(
            (330, 0),
            imgui.Cond_.first_use_ever
        )

        imgui.begin("Engine")

        timer = context.timer
        stats = context.stats
        settings = context.settings

        fps = timer.fps

        imgui.text(
            f"FPS: {fps:6.1f}   "
            f"({1000.0 / fps if fps > 0 else 0.0:5.2f} ms)"
        )

        imgui.text(
            f"Draw calls: {stats.draw_calls}   "
            f"Triangles: {stats.triangles}"
        )

        imgui.text_disabled(
            "Hold RMB: look  |  WASD/QE: move  |  "
            "F1: UI  |  F5: reload shaders"
        )

        # -------------------------------------------------
        # Output
        # -------------------------------------------------

        if imgui.collapsing_header(
            "Output",
            imgui.TreeNodeFlags_.default_open
        ):

            _, settings.exposure = imgui.slider_float(
                "Exposure",
                settings.exposure,
                0.05,
                8.0,
                "%.2f"
            )

            names = [
                tonemapper.name.title()
                for tonemapper in Tonemapper
            ]

            changed, index = imgui.combo(
                "Tone mapper",
                int(settings.tonemapper),
                names
            )

            if changed:
                settings.tonemapper = Tonemapper(index)

            _, settings.gamma = imgui.slider_float(
                "Gamma",
                settings.gamma,
                1.0,
                3.0,
                "%.2f"
            )

            changed, color = imgui.color_edit3(
                "Background",
                list(settings.clear_color)
            )

            if changed:
                settings.clear_color = tuple(color)

        # -------------------------------------------------
        # Shadows
        # -------------------------------------------------

        if imgui.collapsing_header(
            "Shadows",
            imgui.TreeNodeFlags_.default_open
        ):

            _, settings.shadows_enabled = imgui.checkbox(
                "Enabled",
                settings.shadows_enabled
            )

            sizes = [512, 1024, 2048, 4096]

            current = (
                sizes.index(settings.shadow_map_size)
                if settings.shadow_map_size in sizes
                else 2
            )

            changed, index = imgui.combo(
                "Map size",
                current,
                [str(size) for size in sizes]
            )

            if changed:
                settings.shadow_map_size = sizes[index]

            _, settings.shadow_extent = imgui.slider_float(
                "Extent",
                settings.shadow_extent,
                1.0,
                50.0,
                "%.1f"
            )

            changed, center = imgui.drag_float3(
                "Center",
                list(settings.shadow_center),
                0.05
            )

            if changed:
                settings.shadow_center = tuple(center)

            _, settings.shadow_bias_min = imgui.slider_float(
                "Bias min",
                settings.shadow_bias_min,
                0.0,
                0.01,
                "%.5f"
            )

            _, settings.shadow_bias_max = imgui.slider_float(
                "Bias max",
                settings.shadow_bias_max,
                0.0,
                0.05,
                "%.4f"
            )

        # -------------------------------------------------
        # Debug
        # -------------------------------------------------

        if imgui.collapsing_header(
            "Debug",
            imgui.TreeNodeFlags_.default_open
        ):

            _, settings.show_light_gizmos = imgui.checkbox(
                "Light gizmos",
                settings.show_light_gizmos
            )

            _, settings.gizmo_scale = imgui.slider_float(
                "Gizmo scale",
                settings.gizmo_scale,
                0.02,
                0.5,
                "%.2f"
            )

            if imgui.button("Reload shaders (F5)"):

                count = context.reload_shaders()

                self._last_reload_message = (
                    f"Reloaded {count} shader(s)."
                )

            if self._last_reload_message:

                imgui.same_line()

                imgui.text_disabled(
                    self._last_reload_message
                )

        imgui.end()

    # =====================================================
    # Scene Window
    # =====================================================

    def _draw_scene_window(
        self,
        context: DebugContext
    ):

        imgui.set_next_window_pos(
            (self._right_column_x(), 10),
            imgui.Cond_.first_use_ever
        )

        imgui.set_next_window_size(
            (330, 400),
            imgui.Cond_.first_use_ever
        )

        imgui.begin("Scene")

        scene = context.scene

        for entity in scene.entities():

            imgui.push_id(
                f"entity{entity.index}"
            )

            if imgui.tree_node(
                self._entity_label(scene, entity)
            ):

                self._draw_entity(
                    scene,
                    entity
                )

                imgui.tree_pop()

            imgui.pop_id()

        imgui.end()

    @staticmethod
    def _entity_label(
        scene: Scene,
        entity: Entity
    ) -> str:

        name = scene.try_get_component(
            entity,
            NameComponent
        )

        tags = []

        for component_type, tag in (
            (CameraComponent, "camera"),
            (DirectionalLightComponent, "sun"),
            (PointLightComponent, "point"),
            (SpotLightComponent, "spot"),
            (MeshRendererComponent, "mesh"),
        ):

            if scene.has_component(entity, component_type):
                tags.append(tag)

        label = (
            name.name
            if name is not None
            else f"Entity {entity.index}"
        )

        return (
            f"{label}  [{', '.join(tags)}]"
            if tags
            else label
        )

    def _draw_entity(
        self,
        scene: Scene,
        entity: Entity
    ):

        # -------------------------------------------------
        # Transform
        # -------------------------------------------------

        transform_component = scene.try_get_component(
            entity,
            TransformComponent
        )

        if transform_component is not None:

            transform = transform_component.transform

            changed, value = imgui.drag_float3(
                "Position",
                transform.position.tolist(),
                0.02
            )

            if changed:
                transform.position = value

            changed, value = imgui.drag_float3(
                "Rotation",
                transform.rotation.tolist(),
                0.5
            )

            if changed:
                transform.rotation = value

            changed, value = imgui.drag_float3(
                "Scale",
                transform.scale.tolist(),
                0.01
            )

            if changed:
                transform.scale = value

        # -------------------------------------------------
        # Rotator
        # -------------------------------------------------

        rotator = scene.try_get_component(
            entity,
            RotatorComponent
        )

        if rotator is not None:

            changed, value = imgui.drag_float3(
                "Spin (deg/s)",
                list(rotator.degrees_per_second),
                0.5
            )

            if changed:
                rotator.degrees_per_second = tuple(value)

        # -------------------------------------------------
        # Mesh Renderer
        # -------------------------------------------------

        mesh_renderer = scene.try_get_component(
            entity,
            MeshRendererComponent
        )

        if mesh_renderer is not None:

            _, mesh_renderer.casts_shadows = imgui.checkbox(
                "Casts shadows",
                mesh_renderer.casts_shadows
            )

        # -------------------------------------------------
        # Lights
        # -------------------------------------------------

        directional = scene.try_get_component(
            entity,
            DirectionalLightComponent
        )

        if directional is not None:

            imgui.separator_text("Directional light")

            self._edit_color_intensity(directional, 0.0, 10.0)

            _, directional.ambient = imgui.slider_float(
                "Ambient",
                directional.ambient,
                0.0,
                1.0,
                "%.3f"
            )

            _, directional.casts_shadows = imgui.checkbox(
                "Casts shadows##light",
                directional.casts_shadows
            )

        point = scene.try_get_component(
            entity,
            PointLightComponent
        )

        if point is not None:

            imgui.separator_text("Point light")

            self._edit_color_intensity(point, 0.0, 50.0)

            _, point.range = imgui.slider_float(
                "Range",
                point.range,
                0.1,
                50.0,
                "%.2f"
            )

        spot = scene.try_get_component(
            entity,
            SpotLightComponent
        )

        if spot is not None:

            imgui.separator_text("Spot light")

            self._edit_color_intensity(spot, 0.0, 50.0)

            _, spot.range = imgui.slider_float(
                "Range",
                spot.range,
                0.1,
                50.0,
                "%.2f"
            )

            _, spot.outer_angle = imgui.slider_float(
                "Outer angle",
                spot.outer_angle,
                0.0,
                89.0,
                "%.1f"
            )

            _, spot.inner_angle = imgui.slider_float(
                "Inner angle",
                min(spot.inner_angle, spot.outer_angle),
                0.0,
                spot.outer_angle,
                "%.1f"
            )

        # -------------------------------------------------
        # Camera
        # -------------------------------------------------

        camera = scene.try_get_component(
            entity,
            CameraComponent
        )

        if camera is not None:

            imgui.separator_text("Camera")

            _, camera.fov = imgui.slider_float(
                "FOV",
                camera.fov,
                10.0,
                120.0,
                "%.1f"
            )

    @staticmethod
    def _edit_color_intensity(
        light,
        min_intensity: float,
        max_intensity: float
    ):

        changed, color = imgui.color_edit3(
            "Color",
            list(light.color)
        )

        if changed:
            light.color = tuple(color)

        _, light.intensity = imgui.slider_float(
            "Intensity",
            light.intensity,
            min_intensity,
            max_intensity,
            "%.2f"
        )

    # =====================================================
    # Materials Window
    # =====================================================

    def _draw_materials_window(
        self,
        context: DebugContext
    ):

        imgui.set_next_window_pos(
            (self._right_column_x(), 420),
            imgui.Cond_.first_use_ever
        )

        imgui.set_next_window_size(
            (330, 0),
            imgui.Cond_.first_use_ever
        )

        imgui.begin("Materials")

        defaults = Renderer.DEFAULT_MATERIAL_VALUES

        for key, material in context.resources.materials.items():

            imgui.push_id(
                key
            )

            if imgui.tree_node(
                key
            ):

                changed, color = imgui.color_edit3(
                    "Base color",
                    list(material.get_value("uBaseColor", defaults["uBaseColor"]))
                )

                if changed:
                    material.set_vec3("uBaseColor", color)

                changed, value = imgui.slider_float(
                    "Specular",
                    material.get_value("uSpecularStrength", defaults["uSpecularStrength"]),
                    0.0,
                    2.0,
                    "%.2f"
                )

                if changed:
                    material.set_float("uSpecularStrength", value)

                changed, value = imgui.slider_float(
                    "Shininess",
                    material.get_value("uShininess", defaults["uShininess"]),
                    1.0,
                    512.0,
                    "%.0f",
                    imgui.SliderFlags_.logarithmic
                )

                if changed:
                    material.set_float("uShininess", value)

                changed, value = imgui.slider_float(
                    "Normal strength",
                    material.get_value("uNormalStrength", defaults["uNormalStrength"]),
                    0.0,
                    3.0,
                    "%.2f"
                )

                if changed:
                    material.set_float("uNormalStrength", value)

                uv_scale = material.get_value(
                    "uUVScale",
                    defaults["uUVScale"]
                )

                changed, value = imgui.slider_float(
                    "UV scale",
                    uv_scale[0],
                    0.1,
                    20.0,
                    "%.2f"
                )

                if changed:
                    material.set_vec2("uUVScale", (value, value))

                texture_names = ", ".join(
                    name
                    for name, _ in material.textures
                )

                imgui.text_disabled(
                    f"Textures: {texture_names or 'defaults'}"
                )

                imgui.tree_pop()

            imgui.pop_id()

        imgui.end()
