import numpy as np

from core.assertions import engine_assert

from ecs.components import (
    CameraComponent,
    DirectionalLightComponent,
    MeshRendererComponent,
    PointLightComponent,
    SpotLightComponent,
    TransformComponent
)
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
        settings: RenderSettings | None = None
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
        height: int
    ):

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

        # -------------------------------------------------
        # 1. Camera + Lights
        # -------------------------------------------------

        camera = self._build_primary_camera(
            registry,
            width / height
        )

        lighting = self._build_light_environment(
            registry
        )

        shadows = self._build_shadow_parameters(
            lighting
        )

        # -------------------------------------------------
        # 2. Per-Frame Data
        # -------------------------------------------------

        self._renderer.begin_scene(
            camera,
            lighting,
            shadows
        )

        # -------------------------------------------------
        # 3. Shadow Pass
        # -------------------------------------------------

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

            self._render_light_gizmos(
                lighting
            )

        # -------------------------------------------------
        # 5. Post Pass (to window)
        # -------------------------------------------------

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

        for entity in registry.view(
            TransformComponent,
            MeshRendererComponent
        ):

            mesh_renderer = registry.get(
                entity,
                MeshRendererComponent
            )

            if not mesh_renderer.casts_shadows:
                continue

            self._renderer.draw_depth(
                self._resources.meshes.get(
                    mesh_renderer.mesh
                ),
                shader,
                registry.get(
                    entity,
                    TransformComponent
                ).world_matrix
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

        for entity in registry.view(
            TransformComponent,
            MeshRendererComponent
        ):

            mesh_renderer = registry.get(
                entity,
                MeshRendererComponent
            )

            self._renderer.draw(
                self._resources.meshes.get(
                    mesh_renderer.mesh
                ),
                self._resources.materials.get(
                    mesh_renderer.material
                ),
                self._resources,
                registry.get(
                    entity,
                    TransformComponent
                ).world_matrix
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

        for entity in registry.view(
            TransformComponent,
            CameraComponent
        ):

            camera_component = registry.get(
                entity,
                CameraComponent
            )

            if not camera_component.primary:
                continue

            engine_assert(
                primary_camera is None,
                "Scene contains more than one primary camera."
            )

            primary_transform = registry.get(
                entity,
                TransformComponent
            )

            primary_camera = camera_component

        engine_assert(
            primary_transform is not None,
            "Scene has no primary camera TransformComponent."
        )

        engine_assert(
            primary_camera is not None,
            "Scene has no primary CameraComponent."
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

        # -------------------------------------------------
        # Directional (optional, at most one)
        # -------------------------------------------------
        #
        # Scene ambient comes from the directional light.
        # With no directional light, ambient is zero.

        for entity in registry.view(
            TransformComponent,
            DirectionalLightComponent
        ):

            engine_assert(
                lighting.directional is None,
                "Scene contains more than one directional light."
            )

            transform = registry.get(
                entity,
                TransformComponent
            )

            component = registry.get(
                entity,
                DirectionalLightComponent
            )

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

        for entity in registry.view(
            TransformComponent,
            PointLightComponent
        ):

            transform = registry.get(
                entity,
                TransformComponent
            )

            component = registry.get(
                entity,
                PointLightComponent
            )

            engine_assert(
                component.range > 0.0,
                "PointLightComponent range must be positive."
            )

            lighting.point_lights.append(
                PointLight(
                    position=transform.world_position,
                    color=component.color,
                    intensity=component.intensity,
                    range=component.range
                )
            )

        engine_assert(
            len(lighting.point_lights) <= MAX_POINT_LIGHTS,
            (
                "Scene exceeds the maximum of "
                f"{MAX_POINT_LIGHTS} point lights."
            )
        )

        # -------------------------------------------------
        # Spot Lights
        # -------------------------------------------------

        for entity in registry.view(
            TransformComponent,
            SpotLightComponent
        ):

            transform = registry.get(
                entity,
                TransformComponent
            )

            component = registry.get(
                entity,
                SpotLightComponent
            )

            engine_assert(
                component.range > 0.0,
                "SpotLightComponent range must be positive."
            )

            engine_assert(
                0.0
                <= component.inner_angle
                <= component.outer_angle
                < 90.0,
                (
                    "SpotLightComponent angles must satisfy "
                    "0 <= inner_angle <= outer_angle < 90."
                )
            )

            lighting.spot_lights.append(
                SpotLight(
                    position=transform.world_position,
                    direction=transform.world_forward,
                    color=component.color,
                    intensity=component.intensity,
                    range=component.range,
                    inner_cutoff=float(
                        np.cos(np.radians(component.inner_angle))
                    ),
                    outer_cutoff=float(
                        np.cos(np.radians(component.outer_angle))
                    )
                )
            )

        engine_assert(
            len(lighting.spot_lights) <= MAX_SPOT_LIGHTS,
            (
                "Scene exceeds the maximum of "
                f"{MAX_SPOT_LIGHTS} spot lights."
            )
        )

        return lighting

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
