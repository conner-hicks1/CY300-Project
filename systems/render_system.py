import math
import time

import numpy as np

from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_CONSTANT_ALPHA,
    GL_GEQUAL,
    GL_GREATER,
    GL_LESS,
    GL_ONE,
    GL_ONE_MINUS_CONSTANT_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC1_COLOR,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D_ARRAY,
    glBlendColor,
    glBlendFunc,
    glClear,
    glClearColor
)

from core.assertions import engine_assert
from core.logger import Logger
from core.profiler import Profiler

from ecs.components import (
    AtmosphereComponent,
    BodyComponent,
    CameraComponent,
    DirectionalLightComponent,
    MeshRendererComponent,
    PlanetComponent,
    PointLightComponent,
    SpotLightComponent,
    TransformComponent
)
from ecs.entity import Entity
from ecs.registry import Registry

from graphics.atmosphere import (
    SUN_ANGULAR_RADIUS,
    AtmosphereLuts,
    AtmosphereParameters,
    AtmosphereSky,
    pack_atmosphere_block
)
from graphics.bloom import Bloom
from graphics.draw_list import (
    DrawItem,
    frustum_planes,
    spheres_in_frustum
)
from graphics.environment import (
    Environment,
    SkyParameters
)
from graphics.clouds import CloudParameters, cloud_cover_map
from graphics.framebuffer import (
    ColorFormat,
    ColorTarget,
    DepthMode,
    Framebuffer,
    FramebufferSpec
)
from graphics.lighting import (
    MAX_CASCADES,
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
from graphics.shadow_map import ShadowMapArray
from graphics.shadows import (
    compute_cascades,
    spot_shadow,
    to_render_space
)
from graphics.texture import Texture2D
from graphics.uniform_blocks import LightingFrame

from math3d import quaternion
from math3d.camera import Camera

from planet.bodies import AU_M, SOLAR_RADIUS_M

from resources.resources import Resources

from scene.scene import Scene


SHADER_DIRECTORY = "assets/shaders"


class RenderSystem:

    # =====================================================
    # Frame Pipeline
    # =====================================================
    #
    #   1. Gather: camera + lights from the scene (world
    #      space; TransformSystem must have run), shadow
    #      cascades / spot shadow matrices, sky parameters.
    #      Re-bake image-based lighting if the sky changed.
    #      Upload per-frame UBOs (Renderer.begin_scene).
    #   2. Shadows: depth from the directional light into
    #      each cascade, and from each shadowed spot light.
    #   3. Scene (HDR, RGBA16F): PBR geometry, sky
    #      background, light gizmos, selection outline.
    #   4. Bloom: blur chain of the HDR image.
    #   5. Post: bloom mix, exposure, tone mapping, gamma
    #      -> LDR buffer (or the window if FXAA is off).
    #   6. FXAA -> window.
    #
    # The debug UI draws after this, directly on the window.

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

        # Set per frame by the application: no haze over
        # geometry (planet data views).
        self.suppress_haze = False

        # Messages already logged by _warn_once().
        self._warnings: set[str] = set()

        # -------------------------------------------------
        # Engine Shaders
        # -------------------------------------------------
        #
        # key -> handle. Fullscreen passes share one vertex
        # shader (fullscreen triangle from gl_VertexID).

        self._shaders = {
            "shadow_depth": self._load_engine_shader("shadow_depth"),
            "unlit": self._load_engine_shader("unlit"),
            "sky": self._load_engine_shader("sky"),
            "post": self._load_engine_shader("post", "fullscreen"),
            "exposure": self._load_engine_shader("exposure", "fullscreen"),
            "fxaa": self._load_engine_shader("fxaa", "fullscreen"),
            "bloom_downsample": self._load_engine_shader("bloom_downsample", "fullscreen"),
            "bloom_upsample": self._load_engine_shader("bloom_upsample", "fullscreen"),
            "ibl_sky": self._load_engine_shader("ibl_sky", "fullscreen"),
            "ibl_irradiance": self._load_engine_shader("ibl_irradiance", "fullscreen"),
            "ibl_prefilter": self._load_engine_shader("ibl_prefilter", "fullscreen"),
            "ibl_brdf": self._load_engine_shader("ibl_brdf", "fullscreen"),
            "ibl_atmosphere": self._load_engine_shader("ibl_atmosphere", "fullscreen"),
            "atmosphere": self._load_engine_shader("atmosphere", "fullscreen"),
            "atmosphere_transmittance": self._load_engine_shader("atmosphere_transmittance", "fullscreen"),
            "atmosphere_multiscatter": self._load_engine_shader("atmosphere_multiscatter", "fullscreen"),
            "atmosphere_diffuse": self._load_engine_shader("atmosphere_diffuse", "fullscreen"),
        }

        # -------------------------------------------------
        # Default Textures
        # -------------------------------------------------
        #
        # Materials without these maps still sample valid
        # data. White makes a map neutral (the factor alone
        # decides); the flat normal points straight out.

        white_srgb = self._solid_texture("engine/white", (255, 255, 255, 255), srgb=True)
        white_linear = self._solid_texture("engine/white_linear", (255, 255, 255, 255), srgb=False)
        flat_normal = self._solid_texture("engine/flat_normal", (128, 128, 255, 255), srgb=False)

        for sampler_name, texture in (
            ("uBaseColorMap", white_srgb),
            ("uMetallicRoughnessMap", white_linear),
            ("uNormalMap", flat_normal),
            ("uOcclusionMap", white_linear),
            ("uEmissiveMap", white_srgb),
        ):

            renderer.set_default_texture(
                sampler_name,
                texture
            )

        # -------------------------------------------------
        # Gizmo Mesh
        # -------------------------------------------------

        self._gizmo_mesh_handle = resources.meshes.load(
            "engine/gizmo_cube",
            MeshFactory.create_cube
        )

        # -------------------------------------------------
        # GPU Resources
        # -------------------------------------------------

        # Every layer array exists from the start so the
        # lit shader's shadow samplers are always valid.

        self._cascade_maps = ShadowMapArray(
            self.settings.shadow_map_size,
            MAX_CASCADES,
            "cascades"
        )

        self._spot_maps = ShadowMapArray(
            self.settings.spot_shadow_map_size,
            MAX_SPOT_LIGHTS,
            "spot shadows"
        )

        self._environment = Environment(
            renderer,
            self._shader
        )

        self._atmosphere_luts = AtmosphereLuts(
            renderer,
            self._shader
        )

        self._bloom = Bloom(
            renderer,
            lambda: self._shader("bloom_downsample"),
            lambda: self._shader("bloom_upsample")
        )

        # Created at the window size on first use.

        self._hdr_framebuffer: Framebuffer | None = None
        self._ldr_framebuffer: Framebuffer | None = None

        # Auto exposure: a 1x1 target that follows the scene
        # (see _update_exposure), and when it last did.
        self._exposure_framebuffer: Framebuffer | None = None

        # entity -> climate field (for where clouds form);
        # set by the application.
        self.climate_provider = lambda entity: None

        self._clock_start = time.perf_counter()
        self._exposure_time: float | None = None

        # The HDR color without its depth, for the
        # atmosphere pass (which samples that depth).
        self._hdr_color_target: ColorTarget | None = None

    def _load_engine_shader(
        self,
        name: str,
        vertex: str | None = None
    ):

        vertex = vertex or name

        return self._resources.shaders.load(
            f"engine/{name}",
            lambda: Shader(
                f"{SHADER_DIRECTORY}/{vertex}.vert.glsl",
                f"{SHADER_DIRECTORY}/{name}.frag.glsl"
            )
        )

    def _shader(
        self,
        name: str
    ) -> Shader:

        return self._resources.shaders.get(
            self._shaders[name]
        )

    def _solid_texture(
        self,
        key: str,
        rgba,
        srgb: bool
    ) -> Texture2D:

        handle = self._resources.textures.load(
            key,
            lambda: Texture2D.solid_color(rgba, srgb=srgb)
        )

        return self._resources.textures.get(handle)

    # =====================================================
    # Render
    # =====================================================

    def render(
        self,
        scene: Scene,
        width: int,
        height: int,
        selected: Entity | None = None,
        extra_items: list[DrawItem] | None = None,
        output_origin: tuple[int, int] = (0, 0)
    ):
        """
        width, height: size of the image (the editor's 3D
            viewport, or the whole window).
        output_origin: where the image goes in the window's
            framebuffer (pixels, bottom-left origin).
        selected: entity to outline (editor selection).
        extra_items: draws that are not entities (planet
            terrain chunks), rendered and shadowed like
            scene meshes.
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

        # The debug UI may have changed GL state last frame.
        RenderState.set_depth_test(True)
        RenderState.set_face_culling(True)

        # -------------------------------------------------
        # 1. Gather
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

            frame = self._build_lighting_frame(
                camera,
                lighting
            )

            sky = self._sky_parameters(
                lighting
            )

            atmosphere = self._prepare_atmosphere(
                registry,
                camera,
                lighting
            )

        with profiler.scope("Environment", gpu=True):

            self._fullscreen_state(True)

            if atmosphere is not None:

                self._atmosphere_luts.update(
                    atmosphere.parameters
                )

                sky = atmosphere

            self._environment.update(sky)

            self._fullscreen_state(False)

        self._renderer.begin_scene(
            camera,
            lighting,
            frame
        )

        # Every renderable's matrices, computed and uploaded
        # once for all passes (shadow layers + scene).

        with profiler.scope("Prepare draws"):

            prepared = self._renderer.prepare_draws(
                self._collect_draw_items(registry)
                + (extra_items or [])
            )

        # Render-space camera clip matrix, for culling.
        camera_clip = (
            camera.projection_matrix
            @ camera.view_rotation_matrix
        )

        # -------------------------------------------------
        # 2. Shadows
        # -------------------------------------------------

        with profiler.scope("Shadow pass", gpu=True):

            self._render_shadows(
                registry,
                lighting,
                frame
            )

        self._renderer.set_frame_textures(
            {
                "uCascadeShadowMaps": (
                    self._cascade_maps.texture_id,
                    GL_TEXTURE_2D_ARRAY
                ),
                "uSpotShadowMaps": (
                    self._spot_maps.texture_id,
                    GL_TEXTURE_2D_ARRAY
                ),
                **self._environment.textures(),
                **self._atmosphere_luts.textures(),
            }
        )

        # -------------------------------------------------
        # 3. Scene (HDR)
        # -------------------------------------------------

        with profiler.scope("Scene pass", gpu=True):

            hdr = self._ensure_framebuffer(
                "_hdr_framebuffer",
                width,
                height,
                ColorFormat.RGBA16F,
                DepthMode.TEXTURE
            )

            hdr.bind()

            RenderCommand.set_clear_color(
                (*settings.clear_color, 1.0)
            )

            RenderCommand.clear()

            visible = np.flatnonzero(
                spheres_in_frustum(
                    frustum_planes(camera_clip),
                    prepared.centers,
                    prepared.radii
                )
            )

            self._renderer.draw_batch(
                self._resources,
                visible
            )

            if atmosphere is not None:

                with profiler.scope("Atmosphere"):

                    self._render_atmosphere(hdr)

            elif settings.show_sky:

                with profiler.scope("Sky"):

                    self._render_sky(sky)

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
        # 4-6. Bloom, Post, FXAA
        # -------------------------------------------------

        self._fullscreen_state(True)

        bloom_texture = None

        if settings.bloom_enabled and settings.bloom_intensity > 0.0:

            with profiler.scope("Bloom", gpu=True):

                bloom_texture = self._bloom.render(
                    hdr.color_texture_id,
                    width,
                    height,
                    settings.bloom_radius
                )

        with profiler.scope("Post", gpu=True):

            exposure_texture = self._update_exposure(
                bloom_texture if bloom_texture is not None else hdr.color_texture_id
            )

            if settings.fxaa_enabled:

                ldr = self._ensure_framebuffer(
                    "_ldr_framebuffer",
                    width,
                    height,
                    ColorFormat.RGBA8,
                    DepthMode.NONE
                )

                ldr.bind()

            else:

                Framebuffer.bind_default(width, height, *output_origin)

            self._render_post(
                hdr,
                bloom_texture,
                exposure_texture
            )

            if settings.fxaa_enabled:

                Framebuffer.bind_default(width, height, *output_origin)

                self._render_fxaa(
                    ldr,
                    width,
                    height
                )

        self._fullscreen_state(False)

        self._renderer.end_scene()

    @staticmethod
    def _fullscreen_state(
        entering: bool
    ):
        """Fullscreen passes need neither depth nor culling."""

        RenderState.set_depth_test(not entering)
        RenderState.set_face_culling(not entering)

    def _ensure_framebuffer(
        self,
        attribute: str,
        width: int,
        height: int,
        color_format: ColorFormat,
        depth_mode: DepthMode
    ) -> Framebuffer:

        framebuffer = getattr(self, attribute)

        if framebuffer is None:

            framebuffer = Framebuffer(
                FramebufferSpec(
                    width=width,
                    height=height,
                    color_format=color_format,
                    depth_mode=depth_mode
                )
            )

            setattr(self, attribute, framebuffer)

        else:

            framebuffer.resize(width, height)

        return framebuffer

    # =====================================================
    # Lighting Frame (shadow matrices)
    # =====================================================

    def _build_lighting_frame(
        self,
        camera: Camera,
        lighting: LightEnvironment
    ) -> LightingFrame:

        settings = self.settings

        frame = LightingFrame(
            ibl_intensity=settings.ibl_intensity,
            shadow_depth_bias=settings.shadow_depth_bias,
            shadow_normal_offset=settings.shadow_normal_offset,
            visualize_cascades=settings.visualize_cascades
        )

        directional = lighting.directional

        if (
            settings.shadows_enabled
            and directional is not None
            and directional.casts_shadows
            and settings.shadow_distance > camera.near
        ):

            frame.cascades = compute_cascades(
                camera,
                directional.direction,
                count=min(max(int(settings.cascade_count), 1), MAX_CASCADES),
                distance=settings.shadow_distance,
                split_lambda=settings.cascade_split_lambda,
                map_size=int(settings.shadow_map_size)
            )

        frame.spot_shadows = [
            spot_shadow(
                spot.position,
                spot.direction,
                spot.outer_angle,
                spot.range,
                int(settings.spot_shadow_map_size)
            )
            if settings.shadows_enabled and spot.casts_shadows
            else None
            for spot in lighting.spot_lights
        ]

        # Shadow matrices are fitted in world space (float64)
        # and then moved into the camera-relative render
        # space the GPU works in.

        origin = camera.position

        for cascade in frame.cascades:
            cascade.matrix = to_render_space(cascade.matrix, origin)

        for shadow in frame.spot_shadows:

            if shadow is not None:
                shadow.matrix = to_render_space(shadow.matrix, origin)

        return frame

    # =====================================================
    # Shadow Pass
    # =====================================================

    def _render_shadows(
        self,
        registry: Registry,
        lighting: LightEnvironment,
        frame: LightingFrame
    ):

        settings = self.settings

        self._cascade_maps.resize(
            int(settings.shadow_map_size),
            MAX_CASCADES
        )

        self._spot_maps.resize(
            int(settings.spot_shadow_map_size),
            MAX_SPOT_LIGHTS
        )

        passes = []

        for layer, cascade in enumerate(frame.cascades):
            passes.append((self._cascade_maps, layer, cascade.matrix))

        for layer, shadow in enumerate(frame.spot_shadows):

            if shadow is not None:
                passes.append((self._spot_maps, layer, shadow.matrix))

        if not passes:
            return

        prepared = self._renderer.prepared

        casters = np.flatnonzero(
            prepared.casts_shadows
        )

        shader = self._shader("shadow_depth")

        shader.bind()

        # Culling front faces writes the back faces' depth,
        # moving stored depth away from lit surfaces.
        # Open meshes (the floor plane) are receivers only.

        RenderState.set_cull_front_faces(True)

        # Shadow maps use conventional depth (0 near, 1 far;
        # cleared to 1 in clear_layer), unlike the
        # reversed-Z scene.
        RenderState.set_depth_func(GL_LESS)

        for shadow_map, layer, matrix in passes:

            shadow_map.clear_layer(layer)

            shader.set_mat4(
                "uLightMatrix",
                matrix
            )

            # Only casters inside this layer's light volume.
            inside = spheres_in_frustum(
                frustum_planes(matrix),
                prepared.centers[casters],
                prepared.radii[casters]
            )

            self._renderer.draw_depth_batch(
                shader,
                casters[inside]
            )

        RenderState.set_cull_front_faces(False)

        RenderState.set_depth_func(GL_GREATER)

    # =====================================================
    # Scene Items
    # =====================================================

    def _collect_draw_items(
        self,
        registry: Registry
    ) -> list[DrawItem]:

        meshes = self._resources.meshes
        materials = self._resources.materials

        return [
            DrawItem(
                mesh=meshes.get(mesh_renderer.mesh),
                material=materials.get(mesh_renderer.material),
                world_matrix=transform.world_matrix,
                casts_shadows=mesh_renderer.casts_shadows
            )
            for _, transform, mesh_renderer in registry.view_with(
                TransformComponent,
                MeshRendererComponent
            )
        ]

    # =====================================================
    # Sky
    # =====================================================

    def _sky_parameters(
        self,
        lighting: LightEnvironment
    ) -> SkyParameters:

        directional = lighting.directional

        return SkyParameters.from_scene(
            self.settings,
            light_direction=(
                directional.direction
                if directional is not None
                else None
            ),
            light_color=(
                directional.color
                if directional is not None
                else (1.0, 1.0, 1.0)
            ),
            light_intensity=(
                directional.intensity
                if directional is not None
                else 0.0
            )
        )

    # =====================================================
    # Atmosphere
    # =====================================================

    def _prepare_atmosphere(
        self,
        registry: Registry,
        camera: Camera,
        lighting: LightEnvironment
    ) -> AtmosphereSky | None:
        """
        Find the planet atmosphere (first one) and upload
        the AtmosphereBlock. Returns the sky description
        for the environment bake, or None (the block is
        then uploaded disabled).
        """

        found = None

        if self.settings.atmosphere_enabled:

            # The first planet; one without air gets a vacuum
            # (black sky) rather than the procedural sky.
            for entity, transform, planet in registry.view_with(
                TransformComponent,
                PlanetComponent
            ):

                found = (
                    transform,
                    planet,
                    registry.try_get(entity, AtmosphereComponent),
                    registry.try_get(entity, BodyComponent),
                    entity
                )

                break

        if found is None:

            self._renderer.set_atmosphere(pack_atmosphere_block(None))

            return None

        transform, planet, component, body, planet_entity = found

        if component is not None:

            parameters = AtmosphereParameters.from_components(
                planet.radius,
                component
            )

        else:

            parameters = AtmosphereParameters.vacuum(planet.radius)

        # The sun's apparent size: from the body's distance to
        # its star (Earth's sky without a body profile).
        sun_angular_radius = SUN_ANGULAR_RADIUS

        if body is not None and body.orbit_distance_au > 0.0:

            sun_angular_radius = math.degrees(
                math.atan(
                    body.star_radius * SOLAR_RADIUS_M
                    / (body.orbit_distance_au * AU_M)
                )
            )

        center = np.asarray(transform.world_matrix, dtype=np.float64)[:3, 3]

        camera_offset = np.asarray(camera.position, dtype=np.float64) - center

        clouds, cloud_key = self._prepare_clouds(planet_entity, planet, component)

        cloud_drift = 0.0

        if clouds is not None:

            elapsed = time.perf_counter() - self._clock_start

            cloud_drift = (elapsed * clouds.speed / max(planet.radius, 1.0)) % (2.0 * math.pi)

        directional = lighting.directional

        if directional is not None:

            toward = -np.asarray(directional.direction, dtype=np.float64)

            sun_direction = tuple(float(v) for v in toward / np.linalg.norm(toward))

            sun_illuminance = tuple(
                float(c) * float(directional.intensity)
                for c in directional.color
            )

        else:

            sun_direction = None
            sun_illuminance = (0.0, 0.0, 0.0)

        self._renderer.set_atmosphere(
            pack_atmosphere_block(
                parameters,
                planet_center_relative=-camera_offset,
                sun_direction=sun_direction,
                sun_illuminance=sun_illuminance,
                steps=self.settings.atmosphere_samples,
                aerial_perspective=(
                    self.settings.aerial_perspective
                    and not self.suppress_haze
                ),
                sun_angular_radius=sun_angular_radius,
                clouds=clouds,
                cloud_drift=cloud_drift,
                planet_frame=_inverse_rotation(transform.world_matrix)
            )
        )

        distance = float(np.linalg.norm(camera_offset))

        up = camera_offset / distance if distance > 0.0 else np.array([0.0, 1.0, 0.0])

        return AtmosphereSky(
            parameters=parameters,
            sun_direction=sun_direction,
            sun_illuminance=sun_illuminance,
            camera_up=tuple(float(v) for v in up),
            altitude_km=max(distance / 1000.0 - parameters.ground_radius, 0.0),
            textures=tuple(
                (name, texture_id)
                for name, (texture_id, _) in self._atmosphere_luts.textures().items()
            ),
            clouds=cloud_key
        )

    def _prepare_clouds(
        self,
        entity,
        planet,
        component
    ):
        """
        The planet's cloud layer (or None) and a key for it;
        rebuilds the cover map when the climate or coverage
        changes.
        """

        cloud_map = self._atmosphere_luts.cloud_map

        if component is None or component.cloud_coverage <= 0.0:

            cloud_map.clear()

            return None, None

        clouds = CloudParameters(
            coverage=float(np.clip(component.cloud_coverage, 0.0, 1.0)),
            altitude=max(float(component.cloud_altitude), 1.0) / 1000.0,
            optical_depth=max(float(component.cloud_optical_depth), 0.0),
            scale=max(float(component.cloud_scale), 1_000.0) / 1000.0,
            color=tuple(float(c) for c in component.cloud_color),
            speed=float(component.cloud_speed)
        )

        climate = self.climate_provider(entity)

        version = climate.version if climate is not None else None

        cloud_map.update(
            (version, round(clouds.coverage, 3)),
            lambda: cloud_cover_map(clouds.coverage, climate)
        )

        key = (
            version,
            round(clouds.coverage, 3),
            round(clouds.altitude, 2),
            round(clouds.optical_depth, 2),
            round(clouds.scale, 1),
        )

        return clouds, key

    def _render_atmosphere(
        self,
        hdr: Framebuffer
    ):
        """
        Sky + aerial perspective over the opaque scene
        (assets/shaders/atmosphere.frag.glsl), composited
        with dual-source blending. Leaves the HDR
        framebuffer bound.
        """

        target = self._hdr_color_target

        if (
            target is None
            or target.texture_id != hdr.color_texture_id
            or target.width != hdr.width
            or target.height != hdr.height
        ):

            if target is not None:
                target.delete()

            target = ColorTarget(hdr.color_texture_id, hdr.width, hdr.height)

            self._hdr_color_target = target

        target.bind()

        shader = self._shader("atmosphere")

        shader.bind()

        luts = self._atmosphere_luts.textures()

        for unit, (name, (texture_id, _)) in enumerate(luts.items()):

            RenderCommand.bind_texture(texture_id, unit)

            shader.set_int(name, unit)

        depth_unit = len(luts)

        RenderCommand.bind_texture(hdr.depth_texture_id, depth_unit)

        shader.set_int("uSceneDepth", depth_unit)

        self._fullscreen_state(True)

        # result = inScattered + scene * transmittance
        RenderState.set_blending(True)

        glBlendFunc(GL_ONE, GL_SRC1_COLOR)

        self._renderer.draw_fullscreen(shader)

        # Back to the engine default (see RenderState).
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        RenderState.set_blending(False)

        self._fullscreen_state(False)

        hdr.bind()

    def _render_sky(
        self,
        sky: SkyParameters
    ):

        shader = self._shader("sky")

        shader.bind()

        sky.apply(shader)

        # Drawn at depth 0 (the far plane under reversed-Z):
        # fills only pixels no geometry covered, since the
        # depth buffer was cleared to 0.

        RenderState.set_depth_func(GL_GEQUAL)

        RenderState.set_face_culling(False)

        self._renderer.draw_fullscreen(shader)

        RenderState.set_face_culling(True)

        RenderState.set_depth_func(GL_GREATER)

    # =====================================================
    # Light Gizmos
    # =====================================================

    def _render_light_gizmos(
        self,
        lighting: LightEnvironment
    ):

        shader = self._shader("unlit")

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
            )

            # float64: a world position at planet scale does
            # not fit float32 (the renderer narrows after
            # making it camera-relative).
            model[:3, 3] = position

            # Bright enough (> 1) to read as a glowing
            # source after tone mapping, and to bloom.

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

        shader = self._shader("unlit")

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
            )

            # float64: a world position at planet scale does
            # not fit float32 (the renderer narrows after
            # making it camera-relative).
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
    # Post / FXAA
    # =====================================================

    # Average scene luminance that needs no adjustment
    # (Earth daylight at the default exposure), the limits of
    # the adjustment, and how fast the eye follows (s).
    EXPOSURE_REFERENCE = 0.65
    EXPOSURE_LIMITS = (0.1, 12.0)
    EXPOSURE_TIME = 0.5

    # Brightening stops before the brightest surfaces clip.
    EXPOSURE_HIGHLIGHT = 1.6

    def _update_exposure(
        self,
        scene_texture: int
    ) -> int | None:
        """
        Eye adaptation: measure the scene into a 1x1 target,
        blended with its previous value so the exposure
        eases toward the new one. Returns the target's
        texture (None: auto exposure off).
        """

        settings = self.settings

        if not settings.auto_exposure:

            self._exposure_time = None

            return None

        target = self._exposure_framebuffer

        fresh = target is None

        if fresh:

            target = Framebuffer(
                FramebufferSpec(
                    width=1,
                    height=1,
                    color_format=ColorFormat.RGBA16F,
                    depth_mode=DepthMode.NONE
                )
            )

            self._exposure_framebuffer = target

        now = time.perf_counter()

        elapsed = 1.0 if self._exposure_time is None else now - self._exposure_time

        self._exposure_time = now

        target.bind()

        if fresh:

            glClearColor(1.0, 1.0, 1.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)

        shader = self._shader("exposure")

        shader.bind()

        RenderCommand.bind_texture(scene_texture, 0)

        shader.set_int("uScene", 0)
        shader.set_float("uReference", self.EXPOSURE_REFERENCE)
        shader.set_float("uAdaptation", float(np.clip(settings.exposure_adaptation, 0.0, 1.0)))
        shader.set_vec2("uLimits", self.EXPOSURE_LIMITS)
        shader.set_float("uHighlight", self.EXPOSURE_HIGHLIGHT)

        # new = measured * a + previous * (1 - a)
        blend = 1.0 - math.exp(-max(elapsed, 0.0) / self.EXPOSURE_TIME)

        RenderState.set_blending(True)

        glBlendColor(0.0, 0.0, 0.0, float(blend))
        glBlendFunc(GL_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA)

        self._renderer.draw_fullscreen(shader)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        RenderState.set_blending(False)

        return target.color_texture_id

    def _render_post(
        self,
        hdr: Framebuffer,
        bloom_texture: int | None,
        exposure_texture: int | None = None
    ):

        settings = self.settings

        shader = self._shader("post")

        shader.bind()

        RenderCommand.bind_texture(hdr.color_texture_id, 0)

        shader.set_int("uHdrBuffer", 0)

        # The bloom sampler always points at a valid 2D
        # texture, even when bloom is off.
        RenderCommand.bind_texture(
            bloom_texture if bloom_texture is not None else hdr.color_texture_id,
            1
        )

        shader.set_int("uBloomTexture", 1)
        shader.set_bool("uBloomEnabled", bloom_texture is not None)
        shader.set_float("uBloomIntensity", settings.bloom_intensity)

        shader.set_float("uExposure", settings.exposure)

        # Eye adaptation (1x1; unit 2).
        RenderCommand.bind_texture(
            exposure_texture if exposure_texture is not None else hdr.color_texture_id,
            2
        )

        shader.set_int("uAutoExposureTexture", 2)
        shader.set_bool("uAutoExposure", exposure_texture is not None)
        shader.set_float("uGamma", settings.gamma)
        shader.set_int("uTonemapper", int(settings.tonemapper))

        self._renderer.draw_fullscreen(shader)

    def _render_fxaa(
        self,
        ldr: Framebuffer,
        width: int,
        height: int
    ):

        shader = self._shader("fxaa")

        shader.bind()

        RenderCommand.bind_texture(ldr.color_texture_id, 0)

        shader.set_int("uInput", 0)
        shader.set_vec2("uTexelSize", (1.0 / width, 1.0 / height))

        self._renderer.draw_fullscreen(shader)

    # =====================================================
    # Primary Camera
    # =====================================================

    def camera_position(
        self,
        scene: Scene
    ) -> np.ndarray:
        """World position of the camera that renders the scene."""

        return np.asarray(
            self._build_primary_camera(scene.registry, 1.0).position,
            dtype=np.float64
        )

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
                    ),
                    outer_angle=outer_angle,
                    casts_shadows=component.casts_shadows
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

        for attribute in ("_hdr_framebuffer", "_ldr_framebuffer", "_exposure_framebuffer"):

            framebuffer = getattr(self, attribute)

            if framebuffer is not None:

                framebuffer.delete()

                setattr(self, attribute, None)

        self._cascade_maps.delete()
        self._spot_maps.delete()

        self._environment.delete()
        self._atmosphere_luts.delete()
        self._bloom.delete()

        if self._hdr_color_target is not None:

            self._hdr_color_target.delete()

            self._hdr_color_target = None


def _inverse_rotation(
    world_matrix
) -> tuple[float, float, float, float]:
    """Quaternion (xyzw) taking world directions into an entity's own frame."""

    rotation = np.asarray(world_matrix, dtype=np.float64)[:3, :3]

    # Remove any scale from the columns.
    rotation = rotation / np.maximum(np.linalg.norm(rotation, axis=0, keepdims=True), 1e-12)

    q = quaternion.conjugate(quaternion.from_matrix3(rotation))

    return tuple(float(v) for v in q)
