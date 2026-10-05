import math
import time

from dataclasses import dataclass

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
    GL_TEXTURE_2D,
    GL_TEXTURE_2D_ARRAY,
    glBlendColor,
    glBlendFunc,
    glClear,
    glClearColor,
    glDepthMask
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
    OrbitComponent,
    PlanetComponent,
    RingsComponent,
    PointLightComponent,
    SpotLightComponent,
    StarComponent,
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
from graphics.comets import MAX_COMETS, comet_view, pack_comet_uniforms
from graphics.bodies_block import (
    BodySphere,
    RingSystem,
    eclipsing as shadow_reaches,
    pack_bodies_block,
    umbra_glow
)
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
from graphics.mesh import Mesh
from graphics.rings import RingProfileTexture, disc_mesh_data, ring_profile, unflat_bands
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
from graphics.star_field import TWINKLE, BodyPoint, StarField, reflected_illuminance
from graphics.scatter import ScatterRenderer
from graphics.material import Material
from graphics.shadows import (
    Cascade,
    compute_cascades,
    spot_shadow,
    terrain_shadow,
    terrain_shadow_radius,
    to_render_space
)
from graphics.texture import Texture2D
from graphics.uniform_blocks import MAX_BODIES, TERRAIN_SHADOW_LAYER, LightingFrame

from math3d import quaternion
from math3d.camera import Camera

from planet.bodies import AU_M, SOLAR_RADIUS_M
from planet.orbits import ecliptic_to_engine
from planet.stars import load_catalog

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
    #   3. Scene (HDR, RGBA32F): PBR geometry, sky, stars
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
            "rings": self._load_engine_shader("rings"),
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
            "stars": self._load_engine_shader("stars"),
            "milky_way": self._load_engine_shader("milky_way", "fullscreen"),
            "comets": self._load_engine_shader("comets", "fullscreen"),
        }

        # The night sky (created on first use: it reads the
        # star catalog), and what it shows this frame.
        self._star_field: StarField | None = None
        self._sky_view: dict | None = None

        # Rocks, trees and grass: their meshes and buffers
        # (created on first use), lit like everything else
        # (lit.frag with an instancing vertex shader).
        self._scatter: ScatterRenderer | None = None
        self._scatter_frame = None
        self._scatter_commands: dict = {}
        self._scatter_shadow_commands: dict = {}

        self._scatter_shader = self._resources.shaders.load(
            "engine/scatter",
            lambda: Shader(f"{SHADER_DIRECTORY}/scatter.vert.glsl", f"{SHADER_DIRECTORY}/lit.frag.glsl")
        )

        self._scatter_depth_shader = self._resources.shaders.load(
            "engine/scatter_depth",
            lambda: Shader(f"{SHADER_DIRECTORY}/scatter_depth.vert.glsl", f"{SHADER_DIRECTORY}/shadow_depth.frag.glsl")
        )

        self._scatter_material = Material(self._scatter_shader)

        for name, value in (
            ("uTerrainShading", 0.0),
            ("uTerrainView", 0.0),
            ("uColorMapStrength", 0.0),
            ("uRoughness", 0.85),
            ("uMetallic", 0.0),
            ("uOcclusionStrength", 0.0),
            ("uScatterDetail", 1.0),
        ):
            self._scatter_material.set_float(name, value)

        self._scatter_material.set_vec3("uBaseColor", (1.0, 1.0, 1.0))
        self._scatter_material.set_vec3("uEmissive", (0.0, 0.0, 0.0))

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
            ("uColorMap", white_srgb),
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

        # Rings: a unit disc scaled to each ring system.
        self._ring_mesh_handle = resources.meshes.load(
            "engine/ring_disc",
            lambda: Mesh.from_data(disc_mesh_data())
        )

        # (world matrix, outer radius m, color) of the ring
        # system to draw this frame, if any, and its radial
        # profile.
        self._rings_to_draw = None
        self._ring_system: RingSystem | None = None
        self._ring_profile = RingProfileTexture()

        # The camera's atmosphere (packed block, body), other
        # bodies' air to draw before it (block, parameters,
        # lookup tables, bodies block), and their lookup
        # tables by entity.
        self._primary_atmosphere = None
        self._distant_atmospheres: list = []
        self._distant_luts: dict[Entity, AtmosphereLuts] = {}

        self._bodies_block = b""
        self._sun_angular_radius = 0.0

        # entity -> (distance to its star m, star radius m),
        # live from the orbits; set by the application.
        self.star_provider = lambda entity: None

        # Where the star is in the world (m), or None (no
        # orbits: the sunlight's direction is all there is).
        self.star_locator = lambda: None

        # The rocks, trees and grass to draw this frame
        # (graphics/scatter.py ScatterFrame), or None.
        self.scatter_provider = lambda: None

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
                lighting,
                registry
            )

            sky = self._sky_parameters(
                lighting
            )

            atmosphere = self._prepare_atmosphere(
                registry,
                camera,
                lighting
            )

            self._prepare_sky_view(registry, camera, lighting)

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

        with profiler.scope("Prepare scatter"):
            self._prepare_scatter(camera_clip)

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
                "uRingProfile": (self._ring_profile.texture_id, GL_TEXTURE_2D),
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
                ColorFormat.RGBA32F,
                DepthMode.TEXTURE
            )

            hdr.bind()

            # Under an atmosphere the background is space: the
            # sky goes over it (through it, in a thin one).
            RenderCommand.set_clear_color(
                (0.0, 0.0, 0.0, 1.0)
                if atmosphere is not None
                else (*settings.clear_color, 1.0)
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

            if self._scatter_frame is not None:

                with profiler.scope("Scatter", gpu=True):

                    self._render_scatter()

            # The stars behind everything, before the air
            # (which dims them, or drowns them by day).
            if atmosphere is not None and settings.show_stars:

                with profiler.scope("Stars", gpu=True):

                    self._render_stars(hdr, width, height)

                    self._render_comets(hdr)

            if atmosphere is not None:

                with profiler.scope("Atmosphere"):

                    self._render_atmospheres(hdr)

            elif settings.show_sky:

                with profiler.scope("Sky"):

                    self._render_sky(sky)

            if self._rings_to_draw is not None:

                with profiler.scope("Rings"):

                    self._render_rings()

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

            # (Measured on the full-float scene: the night's
            # faint light is below the bloom chain's half
            # floats.)
            exposure_texture = self._update_exposure(hdr.color_texture_id, bloom_texture)

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

    # Near a solid surface the cascades reach further than a
    # prop scene's (rocks, trees and hills nearby cast
    # shadows): this far plus this much per meter of height,
    # up to the limit (m).
    GROUND_SHADOW_DISTANCE = (300.0, 3.0, 3_000.0)

    def _build_lighting_frame(
        self,
        camera: Camera,
        lighting: LightEnvironment,
        registry: Registry | None = None
    ) -> LightingFrame:

        settings = self.settings

        # The solid body the camera is at (its ground shadows).
        ground = None

        if registry is not None:

            solid = [
                b for b in self._gather_bodies(registry, camera)
                if not (b.body is not None and b.body.kind in ("gas_giant", "ice_giant"))
            ]

            if solid:
                ground = min(solid, key=lambda b: b.surface_distance)

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

            distance = settings.shadow_distance

            if ground is not None:

                base, per_meter, limit = self.GROUND_SHADOW_DISTANCE

                distance = max(
                    distance,
                    min(base + per_meter * max(ground.surface_distance, 0.0), limit)
                )

            frame.cascades = compute_cascades(
                camera,
                directional.direction,
                count=min(max(int(settings.cascade_count), 1), MAX_CASCADES),
                distance=distance,
                split_lambda=settings.cascade_split_lambda,
                map_size=int(settings.shadow_map_size)
            )

            # Mountains' shadows over the land in view.
            if ground is not None and settings.terrain_shadows:

                up = (np.asarray(camera.position, dtype=np.float64) - ground.center) / max(ground.distance, 1.0)

                frame.terrain_shadow = self._cached_terrain_shadow(
                    ground.center + up * ground.planet.radius,
                    np.asarray(directional.direction, dtype=np.float64),
                    terrain_shadow_radius(ground.surface_distance, ground.planet.radius),
                    int(settings.shadow_map_size)
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

        if frame.terrain_shadow is not None:
            frame.terrain_shadow.matrix = to_render_space(frame.terrain_shadow.matrix, origin)

        for shadow in frame.spot_shadows:

            if shadow is not None:
                shadow.matrix = to_render_space(shadow.matrix, origin)

        return frame

    # The terrain shadow is redrawn only when its view has
    # changed this much (share of its width; sun angle, rad),
    # or this long (s) has passed (terrain streamed in):
    # mountains do not move, and the sun only slowly.
    TERRAIN_SHADOW_REUSE = (0.03, 2e-4, 1.0)

    def _cached_terrain_shadow(
        self,
        center: np.ndarray,
        direction: np.ndarray,
        radius: float,
        map_size: int
    ):
        """
        This frame's terrain shadow (world space): last
        frame's while still good (then its layer is not
        redrawn), else a new one.
        """

        cached = getattr(self, "_terrain_shadow_cache", None)

        now = time.perf_counter()

        moved, turned, age = self.TERRAIN_SHADOW_REUSE

        direction = direction / max(float(np.linalg.norm(direction)), 1e-12)

        if (
            cached is not None
            and abs(cached["radius"] - radius) < moved * radius
            and float(np.linalg.norm(cached["center"] - center)) < moved * radius
            and float(cached["direction"] @ direction) > math.cos(turned)
            and now - cached["time"] < age
            and cached["size"] == map_size
        ):

            shadow = cached["shadow"]

            self._terrain_shadow_fresh = False

            return Cascade(matrix=shadow.matrix.copy(), split_far=shadow.split_far, texel_world_size=shadow.texel_world_size)

        shadow = terrain_shadow(center, direction, radius, map_size)

        self._terrain_shadow_cache = {
            "center": np.array(center, dtype=np.float64),
            "direction": direction.copy(),
            "radius": radius,
            "time": now,
            "size": map_size,
            "shadow": Cascade(matrix=shadow.matrix.copy(), split_far=shadow.split_far, texel_world_size=shadow.texel_world_size),
        }

        self._terrain_shadow_fresh = True

        return shadow

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

        # (One more layer: the terrain shadow.)
        self._cascade_maps.resize(
            int(settings.shadow_map_size),
            MAX_CASCADES + 1
        )

        self._spot_maps.resize(
            int(settings.spot_shadow_map_size),
            MAX_SPOT_LIGHTS
        )

        passes = []

        for layer, cascade in enumerate(frame.cascades):
            passes.append((self._cascade_maps, layer, cascade.matrix, None))

        # (Redrawn only when it changed: _cached_terrain_shadow.)
        if frame.terrain_shadow is not None and getattr(self, "_terrain_shadow_fresh", True):
            passes.append((self._cascade_maps, TERRAIN_SHADOW_LAYER, frame.terrain_shadow.matrix, "terrain"))

        for layer, shadow in enumerate(frame.spot_shadows):

            if shadow is not None:
                passes.append((self._spot_maps, layer, shadow.matrix, None))

        if not passes:
            return

        prepared = self._renderer.prepared

        casters = np.flatnonzero(
            prepared.casts_shadows
        )

        # Planet terrain (the terrain shadow's only casters).
        terrain_casters = casters[
            np.array(
                [
                    float(prepared.items[i].material.get_value("uTerrainShading", 0.0)) > 0.5
                    for i in casters
                ],
                dtype=bool
            )
        ] if len(casters) else casters

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

        for shadow_map, layer, matrix, only in passes:

            shadow_map.clear_layer(layer)

            members = terrain_casters if only == "terrain" else casters

            shader.set_mat4(
                "uLightMatrix",
                matrix
            )

            # Only casters inside this layer's light volume.
            inside = spheres_in_frustum(
                frustum_planes(matrix),
                prepared.centers[members],
                prepared.radii[members]
            )

            self._renderer.draw_depth_batch(
                shader,
                members[inside]
            )

            # Rocks and trees in the near cascades.
            if only is None and shadow_map is self._cascade_maps and self._scatter_frame is not None:

                depth = self._resources.shaders.get(self._scatter_depth_shader)

                depth.bind()
                depth.set_mat4("uLightMatrix", matrix)

                self._scatter.draw(depth, self._scatter_frame, self._scatter_shadow_commands, 0.0)

                shader.bind()

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
    #
    # Every planet and moon is gathered once per frame:
    #
    # - the one the camera is at (nearest surface) gives the
    #   atmosphere block: the sky, haze, sunset light and the
    #   image-based lighting;
    # - other bodies with air are drawn as seen from afar
    #   (Earth's blue limb and clouds from the Moon), each
    #   with lookup tables of its own, before the camera's
    #   sky goes over them;
    # - all of them are spheres in the bodies block, for
    #   eclipses, with the ring system of the nearest ringed
    #   planet.

    # Other bodies' air drawn per frame, nearest first.
    MAX_DISTANT_ATMOSPHERES = 2

    def _gather_bodies(
        self,
        registry: Registry,
        camera: Camera
    ) -> list["_Body"]:

        camera_position = np.asarray(camera.position, dtype=np.float64)

        bodies = []

        for entity, transform, planet in registry.view_with(
            TransformComponent,
            PlanetComponent
        ):

            world = np.asarray(transform.world_matrix, dtype=np.float64)

            center = world[:3, 3]

            distance = float(np.linalg.norm(center - camera_position))

            bodies.append(
                _Body(
                    entity=entity,
                    transform=transform,
                    planet=planet,
                    atmosphere=registry.try_get(entity, AtmosphereComponent),
                    body=registry.try_get(entity, BodyComponent),
                    rings=registry.try_get(entity, RingsComponent),
                    center=center,
                    distance=distance,
                    surface_distance=distance - planet.radius
                )
            )

        return bodies

    def _prepare_atmosphere(
        self,
        registry: Registry,
        camera: Camera,
        lighting: LightEnvironment
    ) -> AtmosphereSky | None:
        """
        Upload the atmosphere block of the body the camera is
        at, the bodies block, and plan the other bodies' air.
        Returns the sky description for the environment bake,
        or None (the block is then uploaded disabled).
        """

        bodies = self._gather_bodies(registry, camera)

        camera_position = np.asarray(camera.position, dtype=np.float64)

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

        self._prepare_rings(bodies, camera_position)

        self._distant_atmospheres = []

        if not bodies or not self.settings.atmosphere_enabled:

            self._renderer.set_atmosphere(pack_atmosphere_block(None))

            self._sun_angular_radius = math.radians(SUN_ANGULAR_RADIUS)

            self._upload_bodies(bodies, None, sun_direction, camera_position)

            return None

        # The body the camera is at; one without air gets a
        # vacuum (black sky) rather than the procedural sky.
        primary = min(bodies, key=lambda b: b.surface_distance)

        self._sun_angular_radius = self._sun_radius(primary)

        parameters, block, cloud_key = self._atmosphere_block(
            primary,
            self._atmosphere_luts,
            camera_position,
            sun_direction,
            sun_illuminance,
            steps=self.settings.atmosphere_samples
        )

        self._renderer.set_atmosphere(block)

        self._primary_atmosphere = (block, primary)

        self._upload_bodies(bodies, primary, sun_direction, camera_position)

        # Other bodies' air, as seen from here (farthest drawn
        # first, under nearer ones).
        distant = sorted(
            (
                b for b in bodies
                if b is not primary and b.atmosphere is not None
            ),
            key=lambda b: b.distance
        )[:self.MAX_DISTANT_ATMOSPHERES]

        for body in reversed(distant):

            luts = self._distant_luts.get(body.entity)

            if luts is None:

                luts = AtmosphereLuts(self._renderer, self._shader)

                self._distant_luts[body.entity] = luts

            distant_parameters, distant_block, _ = self._atmosphere_block(
                body,
                luts,
                camera_position,
                sun_direction,
                sun_illuminance,
                steps=max(8, self.settings.atmosphere_samples // 2)
            )

            self._distant_atmospheres.append(
                (
                    distant_block,
                    distant_parameters,
                    luts,
                    self._pack_bodies(bodies, body, sun_direction, camera_position)
                )
            )

        # Bodies whose air is no longer drawn free their
        # lookup tables.
        drawn = {b.entity for b in distant}

        for entity in [e for e in self._distant_luts if e not in drawn]:
            self._distant_luts.pop(entity).delete()

        distance = primary.distance

        offset = camera_position - primary.center

        up = offset / distance if distance > 0.0 else np.array([0.0, 1.0, 0.0])

        return AtmosphereSky(
            parameters=parameters,
            sun_direction=sun_direction,
            sun_illuminance=sun_illuminance,
            camera_up=tuple(float(v) for v in up),
            altitude_km=max(distance / 1000.0 - parameters.ground_radius, 0.0),
            textures=tuple(
                (name, texture_id)
                for name, (texture_id, _) in self._atmosphere_luts.textures().items()
            ) + (("uRingProfile", self._ring_profile.texture_id),),
            clouds=cloud_key
        )

    def _atmosphere_block(
        self,
        body: "_Body",
        luts: AtmosphereLuts,
        camera_position: np.ndarray,
        sun_direction,
        sun_illuminance,
        steps: int
    ):
        """(parameters, packed atmosphere block, cloud key) of a body's air."""

        planet = body.planet

        if body.atmosphere is not None:

            parameters = AtmosphereParameters.from_components(
                planet.radius,
                body.atmosphere
            )

        else:

            parameters = AtmosphereParameters.vacuum(planet.radius)

        clouds, cloud_key = self._prepare_clouds(body.entity, planet, body.atmosphere, luts)

        cloud_drift = 0.0

        if clouds is not None:

            elapsed = time.perf_counter() - self._clock_start

            cloud_drift = (elapsed * clouds.speed / max(planet.radius, 1.0)) % (2.0 * math.pi)

        block = pack_atmosphere_block(
            parameters,
            planet_center_relative=body.center - camera_position,
            sun_direction=sun_direction,
            sun_illuminance=sun_illuminance,
            steps=steps,
            aerial_perspective=(
                self.settings.aerial_perspective
                and not self.suppress_haze
            ),
            sun_angular_radius=math.degrees(self._sun_angular_radius),
            clouds=clouds,
            cloud_drift=cloud_drift,
            planet_frame=_inverse_rotation(body.transform.world_matrix),
            flattening=planet.oblateness,
            no_surface=planet.palette == "bands",
            lightning_time=time.perf_counter() - self._clock_start
        )

        return parameters, block, cloud_key

    def _sun_radius(
        self,
        body: "_Body"
    ) -> float:
        """
        The sun's apparent radius (rad) at a body: from its
        distance to the star (live, with orbits), else its
        profile's (Earth's sky without one).
        """

        star = self.star_provider(body.entity)

        if star is not None:

            distance, star_radius = star

            return math.atan(star_radius / max(distance, 1.0))

        info = body.body

        if info is not None and info.orbit_distance_au > 0.0:

            return math.atan(
                info.star_radius * SOLAR_RADIUS_M
                / (info.orbit_distance_au * AU_M)
            )

        return math.radians(SUN_ANGULAR_RADIUS)

    def _pack_bodies(
        self,
        bodies: list["_Body"],
        focus: "_Body | None",
        sun_direction,
        camera_position: np.ndarray
    ) -> bytes:
        """
        The bodies block, with the bodies whose shadow can
        fall on `focus` (and its air) first.
        """

        spheres = []

        for b in bodies:

            pressure = (
                b.body.surface_pressure_bar
                if b.body is not None
                else (1.0 if b.atmosphere is not None else 0.0)
            )

            spheres.append(
                BodySphere(
                    center=tuple(float(v) for v in b.center),
                    radius=float(b.planet.radius),
                    glow=umbra_glow(pressure) if b.atmosphere is not None else (0.0, 0.0, 0.0),
                    light=_body_light(b.body)
                )
            )

        # Nearest first if there are too many.
        order = sorted(range(len(bodies)), key=lambda i: bodies[i].distance)[:MAX_BODIES]

        eclipsing = []

        if focus is not None and sun_direction is not None:

            top = focus.planet.radius + (
                focus.atmosphere.height if focus.atmosphere is not None else 0.0
            )

            eclipsing = [
                i for i in order
                if bodies[i] is not focus
                and shadow_reaches(
                    spheres[i],
                    focus.center,
                    top,
                    sun_direction,
                    self._sun_angular_radius
                )
            ]

        rest = [i for i in order if i not in eclipsing]

        return pack_bodies_block(
            [spheres[i] for i in eclipsing + rest],
            camera_position,
            self._sun_angular_radius,
            eclipsing_count=len(eclipsing),
            rings=self._ring_system
        )

    def _upload_bodies(
        self,
        bodies,
        focus,
        sun_direction,
        camera_position
    ):

        self._bodies_block = self._pack_bodies(bodies, focus, sun_direction, camera_position)

        self._renderer.set_bodies(self._bodies_block)

    def _prepare_rings(
        self,
        bodies: list["_Body"],
        camera_position: np.ndarray
    ):
        """
        The ring system of the nearest ringed planet: rebuild
        its profile texture when it changed, and remember it
        for the bodies block and for drawing.
        """

        profile = self._ring_profile

        self._rings_to_draw = None
        self._ring_system = None

        ringed = [
            b for b in bodies
            if b.rings is not None and b.rings.outer_radius > b.rings.inner_radius
        ]

        if not ringed:

            profile.clear()

            return

        body = min(ringed, key=lambda b: b.distance)

        component = body.rings

        bands = unflat_bands(component.bands)

        profile.update(
            (component.inner_radius, component.outer_radius, tuple(component.bands), component.seed),
            lambda: ring_profile(component.inner_radius, component.outer_radius, bands, component.seed)
        )

        world = np.asarray(body.transform.world_matrix, dtype=np.float64)

        self._rings_to_draw = (
            world,
            float(component.outer_radius),
            tuple(float(c) for c in component.color)
        )

        self._ring_system = RingSystem(
            center=tuple(float(v) for v in body.center),
            frame=_inverse_rotation(world),
            planet_radius=float(body.planet.radius),
            flattening=float(body.planet.oblateness),
            inner=float(component.inner_radius),
            outer=float(component.outer_radius),
            opacity=float(component.opacity)
        )

    # =====================================================
    # Night Sky
    # =====================================================

    def _prepare_sky_view(
        self,
        registry: Registry,
        camera: Camera,
        lighting: LightEnvironment
    ):
        """
        What the star pass needs this frame: the planets and
        moons as points of light (their brightness from the
        sunlight on them, their albedo, size, distance and
        phase), the camera's planet (it hides the stars
        behind it) and how much air there is to make the
        stars twinkle.
        """

        self._sky_view = None

        if not self.settings.show_stars:
            return

        bodies = self._gather_bodies(registry, camera)

        if not bodies:
            return

        camera_position = np.asarray(camera.position, dtype=np.float64)

        star_component = next((s for _, s in registry.view_with(StarComponent)), None)

        star_position = self.star_locator()

        star = (
            (np.asarray(star_position, dtype=np.float64), star_component)
            if star_position is not None and star_component is not None
            else None
        )

        directional = lighting.directional

        toward_sun = (
            -np.asarray(directional.direction, dtype=np.float64)
            if directional is not None
            else np.array([0.0, 1.0, 0.0])
        )

        toward_sun /= max(np.linalg.norm(toward_sun), 1e-12)

        sun_color = (
            np.asarray(directional.color, dtype=np.float64)
            if directional is not None
            else np.ones(3)
        )

        points = []

        for b in bodies:

            info = b.body

            if info is None or b.distance <= b.planet.radius:
                continue

            # Sunlight on it (physical: 5 at 1 AU from the Sun).
            if star is not None:

                offset = star[0] - b.center

                to_sun = offset / max(np.linalg.norm(offset), 1.0)

                sunlight = 5.0 * star[1].luminosity / max(float(np.linalg.norm(offset)) / AU_M, 1e-6) ** 2

            else:

                to_sun = toward_sun

                sunlight = 5.0 * info.star_luminosity / max(info.orbit_distance_au, 1e-6) ** 2

            to_camera = (camera_position - b.center) / b.distance

            phase = math.acos(float(np.clip(to_camera @ to_sun, -1.0, 1.0)))

            illuminance = reflected_illuminance(
                sunlight,
                info.geometric_albedo,
                b.planet.radius,
                b.distance,
                phase
            )

            if illuminance <= 0.0:
                continue

            color = np.asarray(info.disc_color, dtype=np.float64) * sun_color

            color /= max(float(color @ np.array([0.2126, 0.7152, 0.0722])), 1e-6)

            points.append(
                BodyPoint(
                    direction=tuple(float(v) for v in -to_camera),
                    illuminance=float(illuminance),
                    color=tuple(float(v) for v in color),
                    angular_radius=math.asin(min(b.planet.radius / b.distance, 1.0))
                )
            )

        # The camera's planet: it hides what is behind it, and
        # its air makes stars twinkle.
        primary = min(bodies, key=lambda b: b.surface_distance)

        offset = camera_position - primary.center

        distance = max(float(np.linalg.norm(offset)), 1.0)

        twinkle = 0.0

        if primary.atmosphere is not None and primary.body is not None:

            altitude = max(distance - primary.planet.radius, 0.0)

            density = primary.body.surface_pressure_bar * math.exp(
                -altitude / max(primary.atmosphere.rayleigh_scale_height, 1.0)
            )

            twinkle = TWINKLE * min(density, 2.0)

        # Active comets: coma and tails.
        comets = []

        if star is not None:

            for b in bodies:

                info = b.body

                if info is None or info.comet_afrho <= 0.0:
                    continue

                orbit = registry.try_get(b.entity, OrbitComponent)

                if orbit is None:
                    continue

                i = math.radians(orbit.inclination)
                node = math.radians(orbit.ascending_node)

                normal = ecliptic_to_engine((math.sin(i) * math.sin(node), -math.sin(i) * math.cos(node), math.cos(i)))

                view = comet_view(
                    b.center,
                    star[0],
                    normal,
                    camera_position,
                    info.comet_afrho,
                    info.comet_gas,
                    star[1].luminosity
                )

                if view is not None:
                    comets.append(view)

        comets.sort(key=lambda c: sum(v * v for v in c.center))

        self._sky_view = {
            "comets": comets[:MAX_COMETS],
            "points": points,
            "twinkle": twinkle,
            "up": offset / distance,
            "planet": (*((primary.center - camera_position) / 1000.0), primary.planet.radius / 1000.0),
        }

    def _render_stars(
        self,
        hdr: Framebuffer,
        width: int,
        height: int
    ):
        """
        The Milky Way and the stars (graphics/star_field.py),
        added where nothing has been drawn: depth-tested at
        infinity, not written.
        """

        view = self._sky_view

        if view is None:
            return

        if self._star_field is None:

            catalog = load_catalog()

            if not catalog.real:
                Logger.warning(
                    "[Stars] Catalog not found: random stars instead (python tools/fetch_stars.py)."
                )

            self._star_field = StarField(catalog)

        field = self._star_field

        if not field.baked:

            RenderState.set_depth_test(False)
            RenderState.set_blending(False)

            field.bake_milky_way(self._shader("milky_way"), self._renderer.draw_fullscreen)

            hdr.bind()

        field.set_bodies(view["points"])

        RenderState.set_depth_test(True)
        RenderState.set_depth_func(GL_GEQUAL)
        RenderState.set_blending(True)

        glDepthMask(False)
        glBlendFunc(GL_ONE, GL_ONE)

        field.draw_milky_way(self._shader("milky_way"), self._renderer.draw_fullscreen)

        field.draw_points(
            self._shader("stars"),
            (width, height),
            view["twinkle"],
            view["up"],
            time.perf_counter() - self._clock_start,
            view["planet"]
        )

        glDepthMask(True)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        RenderState.set_blending(False)
        RenderState.set_depth_func(GL_GREATER)

    def _render_comets(
        self,
        hdr: Framebuffer
    ):
        """
        Active comets' coma and tails (comets.frag.glsl),
        added in front of the stars and behind what is drawn
        (it reads the depth: a color-only target). Leaves the
        HDR framebuffer bound.
        """

        view = self._sky_view

        if view is None or not view["comets"]:
            return

        self._bind_hdr_color(hdr)

        shader = self._shader("comets")

        shader.bind()

        for name, values in pack_comet_uniforms(view["comets"]).items():
            for i, value in enumerate(values):
                shader.set_vec4(f"{name}[{i}]", tuple(float(v) for v in value))

        shader.set_int("uCometCount", len(view["comets"]))

        RenderCommand.bind_texture(hdr.depth_texture_id, 0)

        shader.set_int("uSceneDepth", 0)

        RenderState.set_depth_test(False)
        RenderState.set_blending(True)

        glBlendFunc(GL_ONE, GL_ONE)

        self._renderer.draw_fullscreen(shader)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        RenderState.set_blending(False)
        RenderState.set_depth_test(True)

        hdr.bind()

    def _bind_hdr_color(
        self,
        hdr: Framebuffer
    ) -> ColorTarget:
        """Bind the HDR color alone (passes that sample its depth)."""

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

        return target

    # =====================================================
    # Rocks, Trees, Grass
    # =====================================================

    def _prepare_scatter(
        self,
        camera_clip: np.ndarray
    ):
        """This frame's scatter (planet/scatter.py): upload when it changed, the blocks to draw."""

        frame = self.scatter_provider()

        self._scatter_frame = None

        if frame is None or not frame.layers:
            return

        if self._scatter is None:
            self._scatter = ScatterRenderer()

        self._scatter.update(frame)

        self._scatter_frame = frame

        self._scatter_commands = self._scatter.commands(frame, camera_clip)

        # Shadows: everything within reach (casters beside
        # the view still shade it).
        self._scatter_shadow_commands = self._scatter.commands(frame, None, shadows=True)

    def _render_scatter(self):

        frame = self._scatter_frame

        material = self._scatter_material

        for name, value in frame.body_values.items():
            if value is not None:
                material.set_vec4(name, value)

        shader = self._resources.shaders.get(self._scatter_shader)

        self._renderer.bind_material(shader, material, self._resources)

        self._scatter.draw(
            shader,
            frame,
            self._scatter_commands,
            time.perf_counter() - self._clock_start
        )

    def _render_rings(self):
        """
        The ring system over the scene (premultiplied alpha,
        depth-tested against the planet, not written).
        """

        world, outer, color = self._rings_to_draw

        shader = self._shader("rings")

        shader.bind()

        RenderCommand.bind_texture(self._ring_profile.texture_id, 0)

        shader.set_int("uRingProfile", 0)

        shader.set_vec3("uRingColor", color)

        # Unit disc -> the rings' outer radius, in the
        # planet's equatorial plane.
        rotation = world[:3, :3] / np.maximum(np.linalg.norm(world[:3, :3], axis=0, keepdims=True), 1e-12)

        model = np.eye(4)
        model[:3, :3] = rotation * outer
        model[:3, 3] = world[:3, 3]

        RenderState.set_depth_test(True)
        RenderState.set_depth_func(GL_GREATER)
        RenderState.set_face_culling(False)
        RenderState.set_blending(True)

        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(False)

        self._renderer.draw_mesh(
            self._resources.meshes.get(self._ring_mesh_handle),
            shader,
            model
        )

        glDepthMask(True)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        RenderState.set_blending(False)
        RenderState.set_face_culling(True)

    def _prepare_clouds(
        self,
        entity,
        planet,
        component,
        luts: AtmosphereLuts
    ):
        """
        The planet's cloud layer (or None) and a key for it;
        rebuilds the cover map when the climate or coverage
        changes.
        """

        cloud_map = luts.cloud_map

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

    def _render_atmospheres(
        self,
        hdr: Framebuffer
    ):
        """
        Other bodies' air (as seen from afar, farthest
        first), then the camera's own sky and haze over
        everything. Leaves the HDR framebuffer bound.
        """

        primary_block, _ = self._primary_atmosphere

        for block, parameters, luts, bodies_block in self._distant_atmospheres:

            self._renderer.set_atmosphere(block)
            self._renderer.set_bodies(bodies_block)

            self._fullscreen_state(True)

            luts.update(parameters)

            self._render_atmosphere(hdr, luts, distant=True)

        if self._distant_atmospheres:

            self._renderer.set_atmosphere(primary_block)
            self._renderer.set_bodies(self._bodies_block)

        self._render_atmosphere(hdr, self._atmosphere_luts, distant=False)

    def _render_atmosphere(
        self,
        hdr: Framebuffer,
        luts: AtmosphereLuts,
        distant: bool
    ):
        """
        Sky + aerial perspective over the opaque scene
        (assets/shaders/atmosphere.frag.glsl), composited
        with dual-source blending. Leaves the HDR
        framebuffer bound.
        """

        self._bind_hdr_color(hdr)

        shader = self._shader("atmosphere")

        shader.bind()

        textures = luts.textures()

        for unit, (name, (texture_id, _)) in enumerate(textures.items()):

            RenderCommand.bind_texture(texture_id, unit)

            shader.set_int(name, unit)

        ring_unit = len(textures)

        RenderCommand.bind_texture(self._ring_profile.texture_id, ring_unit)

        shader.set_int("uRingProfile", ring_unit)

        depth_unit = ring_unit + 1

        RenderCommand.bind_texture(hdr.depth_texture_id, depth_unit)

        shader.set_int("uSceneDepth", depth_unit)

        shader.set_float("uDistantAtmosphere", 1.0 if distant else 0.0)

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
    # the adjustment in lit scenes, and how fast the eye
    # follows (s).
    EXPOSURE_REFERENCE = 0.65
    EXPOSURE_LIMITS = (0.1, 12.0)
    EXPOSURE_TIME = 0.5

    # How far the eye brightens in the dark (empty space, a
    # night side): enough for starlight (~1e-10 of sunlight
    # per star) and the Milky Way (exposure.frag.glsl).
    EXPOSURE_DARK_LIMIT = 1.0e6

    # Brightening stops before the brightest surfaces clip;
    # in the dark, well before (a moonlit landscape looks
    # dim).
    EXPOSURE_HIGHLIGHT = 1.6
    EXPOSURE_NIGHT_HIGHLIGHT = 0.15

    def _update_exposure(
        self,
        scene_texture: int,
        blurred_texture: int | None = None
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

            # (The target holds log(exposure): 0 = x1.)
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)

        shader = self._shader("exposure")

        shader.bind()

        RenderCommand.bind_texture(scene_texture, 0)

        shader.set_int("uScene", 0)

        RenderCommand.bind_texture(blurred_texture if blurred_texture is not None else scene_texture, 1)

        shader.set_int("uPeak", 1)
        shader.set_bool("uHasPeak", blurred_texture is not None)
        shader.set_float("uReference", self.EXPOSURE_REFERENCE)
        shader.set_float("uAdaptation", float(np.clip(settings.exposure_adaptation, 0.0, 1.0)))
        shader.set_vec2("uLimits", self.EXPOSURE_LIMITS)
        shader.set_float("uDarkLimit", self.EXPOSURE_DARK_LIMIT)
        shader.set_float("uHighlight", self.EXPOSURE_HIGHLIGHT)
        shader.set_float("uNightHighlight", self.EXPOSURE_NIGHT_HIGHLIGHT)

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

        if self._star_field is not None:

            self._star_field.delete()

            self._star_field = None

        if self._scatter is not None:

            self._scatter.delete()

            self._scatter = None

        for attribute in ("_hdr_framebuffer", "_ldr_framebuffer", "_exposure_framebuffer"):

            framebuffer = getattr(self, attribute)

            if framebuffer is not None:

                framebuffer.delete()

                setattr(self, attribute, None)

        self._cascade_maps.delete()
        self._spot_maps.delete()

        self._environment.delete()
        self._atmosphere_luts.delete()

        for luts in self._distant_luts.values():
            luts.delete()

        self._distant_luts.clear()

        self._ring_profile.delete()
        self._bloom.delete()

        if self._hdr_color_target is not None:

            self._hdr_color_target.delete()

            self._hdr_color_target = None


@dataclass(slots=True)
class _Body:

    # A planet or moon as the frame sees it.

    entity: Entity
    transform: TransformComponent
    planet: PlanetComponent
    atmosphere: AtmosphereComponent | None
    body: BodyComponent | None
    rings: RingsComponent | None

    center: np.ndarray          # world (m)
    distance: float             # from the camera to its center
    surface_distance: float     # ... to its (mean) surface


def _body_light(
    body: BodyComponent | None
) -> tuple[float, float, float]:
    """Geometric albedo x disc color (luminance 1): what a body reflects onto its neighbors."""

    if body is None:
        return (0.0, 0.0, 0.0)

    color = np.asarray(body.disc_color, dtype=np.float64)

    color = color / max(float(color @ np.array([0.2126, 0.7152, 0.0722])), 1e-6)

    return tuple(float(c) * float(body.geometric_albedo) for c in color)


def _inverse_rotation(
    world_matrix
) -> tuple[float, float, float, float]:
    """Quaternion (xyzw) taking world directions into an entity's own frame."""

    rotation = np.asarray(world_matrix, dtype=np.float64)[:3, :3]

    # Remove any scale from the columns.
    rotation = rotation / np.maximum(np.linalg.norm(rotation, axis=0, keepdims=True), 1e-12)

    q = quaternion.conjugate(quaternion.from_matrix3(rotation))

    return tuple(float(v) for v in q)
