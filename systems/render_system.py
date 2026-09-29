import numpy as np

from core.assertions import engine_assert
from core.logger import Logger
from core.profiler import Profiler

from ecs.components import (
    CameraComponent,
    DirectionalLightComponent,
    MeshRendererComponent,
    PointLightComponent,
    SpotLightComponent,
    TransformComponent
)
from ecs.entity import Entity
from ecs.registry import Registry

from graphics.framebuffer import (
    ColorFormat,
    DepthMode,
    Framebuffer,
    FramebufferSpec
)
from graphics.lighting import (
    MAX_POINT_LIGHTS,
    MAX_SPOT_LIGHTS,
    DirectionalLight,
    LightEnvironment,
    PointLight,
    SpotLight
)
from graphics.mesh_factory import MeshFactory
from graphics.render_command import RenderCommand
from graphics.render_settings import RenderSettings
from graphics.render_state import RenderState
from graphics.renderer import Renderer
from graphics.shader import Shader
from graphics.texture import Texture2D
from graphics.uniform_blocks import ShadowParameters

from math3d.camera import Camera
from math3d.matrices import (
    look_at,
    orthographic
)

from resources.resources import Resources

from scene.scene import Scene


SHADER_DIRECTORY = "assets/shaders"


class RenderSystem:

    # =====================================================
    # Frame Pipeline
    # =====================================================
    #
    #   1. Gather camera + lights from the scene (world
    #      space; TransformSystem must have run).
    #   2. Upload per-frame UBOs (Renderer.begin_scene).
    #   3. Shadow pass: scene depth from the directional
    #      light into the shadow map.
    #   4. Scene pass: lit geometry + light gizmos into an
    #      HDR (RGBA16F) framebuffer.
    #   5. Post pass: exposure, tone mapping and gamma
    #      correction onto the window.
    #
    # The debug UI (if any) draws after this, directly on
    # the window.

    # =====================================================
    # Construction
    # =====================================================

    def __init__(
        self,
        renderer: Renderer,
        resources: Resources,
        settings: RenderSettings | None = None,
        profiler: Profiler | None = None
    ):

        engine_assert(
            renderer is not None,
            "RenderSystem requires a Renderer."
        )

        engine_assert(
            resources is not None,
            "RenderSystem requires Resources."
        )

        self._renderer = renderer
        self._resources = resources

        self.settings = settings or RenderSettings()

        # A disabled profiler makes every scope a no-op.

        if profiler is None:

            profiler = Profiler()
            profiler.enabled = False

        self._profiler = profiler

        # Camera used for the most recent frame (editor
        # picking / gizmos); None before the first render.
        self.last_camera: Camera | None = None

        # Messages already logged by _warn_once().
        self._warnings: set[str] = set()

        # -------------------------------------------------
        # Engine Shaders
        # -------------------------------------------------

        self._shadow_shader_handle = self._load_engine_shader(
            "engine/shadow_depth",
            "shadow_depth"
        )

        self._unlit_shader_handle = self._load_engine_shader(
            "engine/unlit",
            "unlit"
        )

        self._post_shader_handle = self._load_engine_shader(
            "engine/post",
            "post"
        )

        # -------------------------------------------------
        # Default Textures
        # -------------------------------------------------
        #
        # Materials without these maps still sample valid
        # data: white albedo, "straight up" normal map,
        # full-strength specular map.

        white_srgb = resources.textures.load(
            "engine/white",
            lambda: Texture2D.solid_color(
                (255, 255, 255, 255),
                srgb=True
            )
        )

        flat_normal = resources.textures.load(
            "engine/flat_normal",
            lambda: Texture2D.solid_color(
                (128, 128, 255, 255),
                srgb=False
            )
        )

        white_linear = resources.textures.load(
            "engine/white_linear",
            lambda: Texture2D.solid_color(
                (255, 255, 255, 255),
                srgb=False
            )
        )

        renderer.set_default_texture(
            "uTexture",
            resources.textures.get(white_srgb)
        )

        renderer.set_default_texture(
            "uNormalMap",
            resources.textures.get(flat_normal)
        )

        renderer.set_default_texture(
            "uSpecularMap",
            resources.textures.get(white_linear)
        )

        # -------------------------------------------------
        # Gizmo Mesh
        # -------------------------------------------------

        self._gizmo_mesh_handle = resources.meshes.load(
            "engine/gizmo_cube",
            MeshFactory.create_cube
        )

        # -------------------------------------------------
        # Framebuffers (created on first use)
        # -------------------------------------------------

        self._hdr_framebuffer: Framebuffer | None = None
        self._shadow_framebuffer: Framebuffer | None = None

    def _load_engine_shader(
        self,
        key: str,
        name: str
    ):

        return self._resources.shaders.load(
            key,
            lambda: Shader(
                f"{SHADER_DIRECTORY}/{name}.vert.glsl",
                f"{SHADER_DIRECTORY}/{name}.frag.glsl"
            )
        )

    # =====================================================
    # Render
    # =====================================================

    def render(
        self,
        scene: Scene,
        width: int,
        height: int,
        selected: Entity | None = None
    ):
        """
        selected: entity to outline (editor selection).
        """

        engine_assert(
            scene is not None,
            "RenderSystem requires a Scene."
        )

        engine_assert(
            not scene.is_shutdown,
            "RenderSystem cannot render a shutdown Scene."
        )

        engine_assert(
            width > 0 and height > 0,
            "RenderSystem framebuffer size must be positive."
        )

        registry = scene.registry

        settings = self.settings

        profiler = self._profiler

        # -------------------------------------------------
        # 1-2. Camera + Lights, Per-Frame Data
        # -------------------------------------------------

        with profiler.scope("Gather"):

            camera = self._build_primary_camera(
                registry,
                width / height
            )

            # For editor picking and gizmos, which work in
            # the same space as the frame on screen.
            self.last_camera = camera

            lighting = self._build_light_environment(
                registry
            )

            shadows = self._build_shadow_parameters(
                lighting
            )

            self._renderer.begin_scene(
                camera,
                lighting,
                shadows
            )

        # -------------------------------------------------
        # 3. Shadow Pass
        # -------------------------------------------------

        with profiler.scope("Shadow pass", gpu=True):

            shadow_framebuffer = self._ensure_shadow_framebuffer()

            self._render_shadow_pass(
                registry,
                shadow_framebuffer,
                shadows.enabled
            )

        self._renderer.set_shadow_map(
            shadow_framebuffer.depth_texture_id
        )

        # -------------------------------------------------
        # 4. Scene Pass (HDR)
        # -------------------------------------------------

        with profiler.scope("Scene pass", gpu=True):

            hdr_framebuffer = self._ensure_hdr_framebuffer(
                width,
                height
            )

            hdr_framebuffer.bind()

            RenderCommand.set_clear_color(
                (*settings.clear_color, 1.0)
            )

            RenderCommand.clear()

            self._render_meshes(
                registry
            )

            if settings.show_light_gizmos:

                with profiler.scope("Gizmos"):

                    self._render_light_gizmos(
                        lighting
                    )

            if (
                selected is not None
                and registry.is_alive(selected)
            ):

                self._render_selection(
                    registry,
                    selected
                )

        # -------------------------------------------------
        # 5. Post Pass (to window)
        # -------------------------------------------------

        with profiler.scope("Post", gpu=True):

            Framebuffer.bind_default(
                width,
                height
            )

            self._render_post(
                hdr_framebuffer
            )

        self._renderer.end_scene()

    # =====================================================
    # Shadow Pass
    # =====================================================

    def _build_shadow_parameters(
        self,
        lighting: LightEnvironment
    ) -> ShadowParameters:

        settings = self.settings

        directional = lighting.directional

        parameters = ShadowParameters(
            enabled=False,
            bias_min=settings.shadow_bias_min,
            bias_max=settings.shadow_bias_max,
            texel_size=1.0 / settings.shadow_map_size
        )

        if (
            not settings.shadows_enabled
            or directional is None
            or not directional.casts_shadows
        ):
            return parameters

        parameters.enabled = True

        parameters.light_space_matrix = self.light_space_matrix(
            directional.direction,
            settings.shadow_center,
            settings.shadow_extent
        )

        return parameters

    @staticmethod
    def light_space_matrix(
        direction,
        center,
        extent: float
    ) -> np.ndarray:
        """
        Orthographic view-projection for a directional
        light covering a (2 * extent)^2 area around center.
        """

        engine_assert(
            extent > 0.0,
            "Shadow extent must be positive."
        )

        direction = np.asarray(
            direction,
            dtype=np.float32
        )

        center = np.asarray(
            center,
            dtype=np.float32
        )

        # Up must not be parallel to the light direction.

        up = (
            (0.0, 0.0, 1.0)
            if abs(float(direction[1])) > 0.99
            else (0.0, 1.0, 0.0)
        )

        distance = extent * 2.0

        view = look_at(
            center - direction * distance,
            center,
            up
        )

        projection = orthographic(
            -extent,
            extent,
            -extent,
            extent,
            0.1,
            distance * 2.0
        )

        return (
            projection
            @ view
        ).astype(
            np.float32
        )

    def _ensure_shadow_framebuffer(
        self
    ) -> Framebuffer:

        size = int(
            self.settings.shadow_map_size
        )

        if self._shadow_framebuffer is None:

            self._shadow_framebuffer = Framebuffer(
                FramebufferSpec(
                    width=size,
                    height=size,
                    color_format=None,
                    depth_mode=DepthMode.TEXTURE
                )
            )

        else:

            self._shadow_framebuffer.resize(
                size,
                size
            )

        return self._shadow_framebuffer

    def _render_shadow_pass(
        self,
        registry: Registry,
        framebuffer: Framebuffer,
        enabled: bool
    ):

        framebuffer.bind()

        # Always clear, so a disabled pass leaves a map
        # that reads as "fully lit".

        RenderCommand.clear(
            color=False,
            depth=True
        )

        if not enabled:
            return

        shader = self._resources.shaders.get(
            self._shadow_shader_handle
        )

        shader.bind()

        # Culling front faces writes the back faces' depth,
        # which pushes the stored depth away from lit
        # surfaces and reduces acne. Open meshes (the
        # floor plane) are receivers only, so losing them
        # from the map costs nothing.

        RenderState.set_cull_front_faces(
            True
        )

        for _, transform, mesh_renderer in registry.view_with(
            TransformComponent,
            MeshRendererComponent
        ):

            if not mesh_renderer.casts_shadows:
                continue

            self._renderer.draw_depth(
                self._resources.meshes.get(
                    mesh_renderer.mesh
                ),
                shader,
                transform.world_matrix
            )

        RenderState.set_cull_front_faces(
            False
        )

    # =====================================================
    # Scene Pass
    # =====================================================

    def _ensure_hdr_framebuffer(
        self,
        width: int,
        height: int
    ) -> Framebuffer:

        if self._hdr_framebuffer is None:

            self._hdr_framebuffer = Framebuffer(
                FramebufferSpec(
                    width=width,
                    height=height,
                    color_format=ColorFormat.RGBA16F,
                    depth_mode=DepthMode.RENDERBUFFER
                )
            )

        else:

            self._hdr_framebuffer.resize(
                width,
                height
            )

        return self._hdr_framebuffer

    def _render_meshes(
        self,
        registry: Registry
    ):

        for _, transform, mesh_renderer in registry.view_with(
            TransformComponent,
            MeshRendererComponent
        ):

            self._renderer.draw(
                self._resources.meshes.get(
                    mesh_renderer.mesh
                ),
                self._resources.materials.get(
                    mesh_renderer.material
                ),
                self._resources,
                transform.world_matrix
            )

    def _render_light_gizmos(
        self,
        lighting: LightEnvironment
    ):

        shader = self._resources.shaders.get(
            self._unlit_shader_handle
        )

        mesh = self._resources.meshes.get(
            self._gizmo_mesh_handle
        )

        scale = self.settings.gizmo_scale

        lights = [
            (point.position, point.color, point.intensity)
            for point in lighting.point_lights
        ] + [
            (spot.position, spot.color, spot.intensity)
            for spot in lighting.spot_lights
        ]

        for position, color, intensity in lights:

            model = np.diag(
                [scale, scale, scale, 1.0]
            ).astype(np.float32)

            model[:3, 3] = position

            # Bright enough (> 1) to read as a glowing
            # source after tone mapping.

            brightness = max(
                1.0,
                intensity
            )

            self._renderer.draw_unlit(
                mesh,
                shader,
                model,
                tuple(
                    float(c) * brightness
                    for c in color
                )
            )

    # =====================================================
    # Selection Outline
    # =====================================================

    # Bright (HDR) orange so it stays saturated after tone
    # mapping.
    SELECTION_COLOR = (4.0, 1.4, 0.15)

    def _render_selection(
        self,
        registry: Registry,
        entity: Entity
    ):

        transform = registry.try_get(
            entity,
            TransformComponent
        )

        if transform is None:
            return

        mesh_renderer = registry.try_get(
            entity,
            MeshRendererComponent
        )

        shader = self._resources.shaders.get(
            self._unlit_shader_handle
        )

        if mesh_renderer is not None:

            mesh = self._resources.meshes.get(
                mesh_renderer.mesh
            )

            model = transform.world_matrix

        else:

            # Lights, cameras and empty entities have no
            # geometry: outline a marker cube at their
            # position, a bit larger than a light gizmo.

            mesh = self._resources.meshes.get(
                self._gizmo_mesh_handle
            )

            scale = self.settings.gizmo_scale * 1.6

            model = np.diag(
                [scale, scale, scale, 1.0]
            ).astype(np.float32)

            model[:3, 3] = transform.world_position

        RenderState.set_wireframe(True)

        self._renderer.draw_unlit(
            mesh,
            shader,
            model,
            self.SELECTION_COLOR
        )

        RenderState.set_wireframe(False)

    # =====================================================
    # Post Pass
    # =====================================================

    def _render_post(
        self,
        hdr_framebuffer: Framebuffer
    ):

        settings = self.settings

        shader = self._resources.shaders.get(
            self._post_shader_handle
        )

        shader.bind()

        RenderCommand.bind_texture(
            hdr_framebuffer.color_texture_id,
            0
        )

        shader.set_int("uHdrBuffer", 0)
        shader.set_float("uExposure", settings.exposure)
        shader.set_float("uGamma", settings.gamma)
        shader.set_int("uTonemapper", int(settings.tonemapper))

        # A fullscreen triangle needs neither depth testing
        # nor culling.

        RenderState.set_depth_test(False)
        RenderState.set_face_culling(False)

        self._renderer.draw_fullscreen(
            shader
        )

        RenderState.set_depth_test(True)
        RenderState.set_face_culling(True)

    # =====================================================
    # Primary Camera
    # =====================================================

    def _build_primary_camera(
        self,
        registry: Registry,
        aspect_ratio: float
    ) -> Camera:

        primary_transform = None
        primary_camera = None

        primary_count = 0

        for _, transform, camera_component in registry.view_with(
            TransformComponent,
            CameraComponent
        ):

            if not camera_component.primary:
                continue

            primary_count += 1

            # With several primaries the first one wins.
            # (An editor user can easily tick "primary" on
            # a second camera; that must not crash.)

            if primary_camera is None:

                primary_transform = transform
                primary_camera = camera_component

        if primary_count > 1:

            self._warn_once(
                "Scene has more than one primary camera; "
                "using the first."
            )

        if primary_camera is None:

            self._warn_once(
                "Scene has no primary camera; using a default "
                "view."
            )

            return Camera(
                position=(0.0, 3.0, 8.0),
                target=(0.0, 0.0, 0.0),
                aspect_ratio=aspect_ratio
            )

        position = primary_transform.world_position

        return Camera(
            position=position,
            target=position + primary_transform.world_forward,
            up=primary_transform.world_up,
            fov=primary_camera.fov,
            aspect_ratio=aspect_ratio,
            near=primary_camera.near,
            far=primary_camera.far
        )

    # =====================================================
    # Light Environment
    # =====================================================

    def _build_light_environment(
        self,
        registry: Registry
    ) -> LightEnvironment:

        lighting = LightEnvironment()

        # Scenes are edited live, so bad values (too many
        # lights, a zero range, crossed spot angles) are
        # clamped or skipped with a one-time warning rather
        # than crashing the frame. Scene files are validated
        # strictly on load (scene/scene_serializer.py).

        # -------------------------------------------------
        # Directional (optional, at most one)
        # -------------------------------------------------
        #
        # Scene ambient comes from the directional light.
        # With no directional light, ambient is zero.

        for _, transform, component in registry.view_with(
            TransformComponent,
            DirectionalLightComponent
        ):

            if lighting.directional is not None:

                self._warn_once(
                    "Scene has more than one directional light; "
                    "using the first."
                )

                continue

            lighting.directional = DirectionalLight(
                direction=transform.world_forward,
                color=component.color,
                intensity=component.intensity,
                casts_shadows=component.casts_shadows
            )

            lighting.ambient = component.ambient

        # -------------------------------------------------
        # Point Lights
        # -------------------------------------------------

        for _, transform, component in registry.view_with(
            TransformComponent,
            PointLightComponent
        ):

            if len(lighting.point_lights) >= MAX_POINT_LIGHTS:

                self._warn_once(
                    f"Scene has more than {MAX_POINT_LIGHTS} point "
                    "lights; extra ones are ignored."
                )

                break

            lighting.point_lights.append(
                PointLight(
                    position=transform.world_position,
                    color=component.color,
                    intensity=component.intensity,
                    range=self._valid_range(component.range)
                )
            )

        # -------------------------------------------------
        # Spot Lights
        # -------------------------------------------------

        for _, transform, component in registry.view_with(
            TransformComponent,
            SpotLightComponent
        ):

            if len(lighting.spot_lights) >= MAX_SPOT_LIGHTS:

                self._warn_once(
                    f"Scene has more than {MAX_SPOT_LIGHTS} spot "
                    "lights; extra ones are ignored."
                )

                break

            # 0 <= inner <= outer < 90

            outer_angle = min(
                max(float(component.outer_angle), 0.0),
                89.0
            )

            inner_angle = min(
                max(float(component.inner_angle), 0.0),
                outer_angle
            )

            lighting.spot_lights.append(
                SpotLight(
                    position=transform.world_position,
                    direction=transform.world_forward,
                    color=component.color,
                    intensity=component.intensity,
                    range=self._valid_range(component.range),
                    inner_cutoff=float(
                        np.cos(np.radians(inner_angle))
                    ),
                    outer_cutoff=float(
                        np.cos(np.radians(outer_angle))
                    )
                )
            )

        return lighting

    @staticmethod
    def _valid_range(
        value: float
    ) -> float:

        # The shader divides by range.
        return max(
            float(value),
            0.01
        )

    def _warn_once(
        self,
        message: str
    ):

        if message in self._warnings:
            return

        self._warnings.add(
            message
        )

        Logger.warning(
            "[RenderSystem] %s",
            message
        )

    # =====================================================
    # Shutdown
    # =====================================================

    def shutdown(self):

        if self._hdr_framebuffer is not None:

            self._hdr_framebuffer.delete()

            self._hdr_framebuffer = None

        if self._shadow_framebuffer is not None:

            self._shadow_framebuffer.delete()

            self._shadow_framebuffer = None
