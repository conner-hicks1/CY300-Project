import math

from pathlib import Path

import numpy as np

from imgui_bundle import imgui

from core.assertions import engine_assert
from core.cprofile_capture import CProfileCapture
from core.events import (
    EventDispatcher,
    MouseButtonPressedEvent,
    MouseButtonReleasedEvent,
    WindowCloseEvent,
    WindowLostFocusEvent,
    WindowResizeEvent
)
from core.handle import Handle
from core.input import Input
from core.jobs import JobSystem
from core.key_codes import Key
from core.logger import Logger
from core.mouse_codes import MouseButton
from core.profiler import Profiler
from core.timer import Timer
from core.window import Window

from ecs.components import (
    AtmosphereComponent,
    BodyComponent,
    CameraComponent,
    ClimateComponent,
    PlanetComponent,
    TectonicsComponent,
    CameraControllerComponent,
    DirectionalLightComponent,
    HierarchyComponent,
    NameComponent,
    TransformComponent
)
from ecs.entity import Entity

from graphics.framebuffer import Framebuffer
from graphics.gpu_timer import GpuTimer
from graphics.material import Material
from graphics.mesh_factory import MeshFactory
from graphics.model_loader import load_gltf_material
from graphics.render_command import RenderCommand
from graphics.renderer import Renderer
from graphics.shader import Shader
from graphics.texture import Texture2D

from math3d import quaternion
from math3d.transform import Transform

from planet import solar
from planet.bodies import ProfileError, components_for, load_presets, preset_groups
from planet.spawn import find_spawn
from planet.tectonics import TectonicField, TectonicSimulation
from planet.terrain import Terrain

from resources.resources import Resources

from systems.camera_controller_system import (
    CameraControllerSystem
)
from systems.planet_system import PlanetSystem, terrain_settings_for
from systems.tectonics_system import TectonicsSystem, tectonic_settings_for
from systems.climate_system import ClimateSystem, climate_settings_for, compute_climate
from systems.render_system import RenderSystem
from systems.rotator_system import RotatorSystem
from systems.transform_system import TransformSystem

from scene.scene import Scene
from scene.scene_serializer import (
    SceneSerializer,
    model_material_key
)

from editor.planet_panel import PlanetPanel, sun_orientation
from editor.tectonics_panel import TectonicsPanel
from editor.climate_panel import ClimatePanel
from editor.scene_editor import SceneEditor

from ui.debug_panel import DebugContext, DebugPanel
from ui.imgui_layer import ImGuiLayer
from ui.panels import PanelRegistry


class Application:

    # =====================================================
    # Timing
    # =====================================================
    #
    # Simulation (RotatorSystem, future physics) runs on a
    # fixed timestep so results do not depend on frame
    # rate. Input/camera run once per rendered frame.
    #
    # MAX_FIXED_STEPS_PER_FRAME stops a slow frame from
    # triggering ever more catch-up steps ("spiral of
    # death"); excess time is dropped.

    FIXED_DELTA_TIME = 1.0 / 60.0
    MAX_FIXED_STEPS_PER_FRAME = 5

    # How often to check shader files for edits.
    SHADER_POLL_INTERVAL = 0.5

    # Main-thread time per frame for job completions.
    JOB_BUDGET_SECONDS = 0.002

    # =====================================================
    # Construction
    # =====================================================

    def __init__(self):

        # -------------------------------------------------
        # Lifecycle
        # -------------------------------------------------

        self._initialized = False
        self._running = False
        self._shutdown = False

        # -------------------------------------------------
        # Core
        # -------------------------------------------------

        self.window: Window | None = None
        self.timer: Timer | None = None

        self._fixed_accumulator = 0.0
        self._shader_poll_timer = 0.0

        # True while RMB is held and the cursor is captured.
        self._looking = False

        # -------------------------------------------------
        # Profiling
        # -------------------------------------------------

        self.profiler = Profiler()
        self.cprofile_capture = CProfileCapture()
        self.gpu_timer: GpuTimer | None = None

        # -------------------------------------------------
        # Graphics / Systems
        # -------------------------------------------------

        self.renderer: Renderer | None = None
        self.render_system: RenderSystem | None = None
        self.transform_system: TransformSystem | None = None
        self.rotator_system: RotatorSystem | None = None
        self.camera_controller_system: (
            CameraControllerSystem | None
        ) = None

        # Background work (mesh generation, loading);
        # results are handed back in update().
        self.jobs: JobSystem | None = None

        self.planet_system: PlanetSystem | None = None
        self.tectonics_system: TectonicsSystem | None = None
        self.tectonics_panel: TectonicsPanel | None = None
        self.climate_system: ClimateSystem | None = None
        self.climate_panel: ClimatePanel | None = None

        # -------------------------------------------------
        # UI
        # -------------------------------------------------

        self.imgui_layer: ImGuiLayer | None = None
        self.debug_panel: DebugPanel | None = None
        self.planet_panel: PlanetPanel | None = None

        # -------------------------------------------------
        # Resources
        # -------------------------------------------------

        self.resources: Resources | None = None

        self.handles: dict[str, Handle] = {}

        # -------------------------------------------------
        # Scene
        # -------------------------------------------------

        self.scene: Scene | None = None

        # -------------------------------------------------
        # Editor
        # -------------------------------------------------

        self.serializer: SceneSerializer | None = None
        self.editor: SceneEditor | None = None

    # =====================================================
    # Initialization
    # =====================================================

    def initialize(
        self,
        scene_path=None
    ):

        engine_assert(
            not self._initialized,
            "Application is already initialized."
        )

        engine_assert(
            not self._shutdown,
            "Cannot initialize an application after shutdown."
        )

        Logger.info(
            "[Application] Initializing."
        )

        # -------------------------------------------------
        # Window
        # -------------------------------------------------

        self.window = Window(
            1280,
            720,
            "OpenGL Engine"
        )

        self.window.set_event_callback(
            self.on_event
        )

        # -------------------------------------------------
        # Core
        # -------------------------------------------------

        self.timer = Timer()

        # -------------------------------------------------
        # Graphics
        # -------------------------------------------------

        self.renderer = Renderer()

        if GpuTimer.is_supported():

            self.gpu_timer = GpuTimer()

            self.profiler.set_gpu_backend(
                self.gpu_timer
            )

        else:

            Logger.warning(
                "[Application] GPU timestamp queries unavailable; "
                "profiler will report CPU time only."
            )

        framebuffer_width, framebuffer_height = (
            self.window.framebuffer_size
        )

        RenderCommand.set_viewport(
            0,
            0,
            framebuffer_width,
            framebuffer_height
        )

        # -------------------------------------------------
        # Resources
        # -------------------------------------------------

        self.resources = Resources()

        # -------------------------------------------------
        # ECS
        # -------------------------------------------------

        self.scene = Scene(
            "Test Scene"
        )

        # -------------------------------------------------
        # Systems
        # -------------------------------------------------

        self.transform_system = TransformSystem()
        self.rotator_system = RotatorSystem()
        self.camera_controller_system = CameraControllerSystem()

        self.jobs = JobSystem()

        self.render_system = RenderSystem(
            self.renderer,
            self.resources,
            profiler=self.profiler
        )

        # -------------------------------------------------
        # UI
        # -------------------------------------------------

        self.imgui_layer = ImGuiLayer(
            self.window
        )

        # -------------------------------------------------
        # Content
        # -------------------------------------------------

        self._load_resources()

        self.tectonics_system = TectonicsSystem(
            self.jobs
        )

        self.climate_system = ClimateSystem(
            self.jobs,
            tectonic_field=self.tectonics_system.field
        )

        self.planet_system = PlanetSystem(
            self.resources,
            self.jobs,
            self.handles["planet_material"],
            field_provider=self.tectonics_system.field,
            climate_provider=self.climate_system.field
        )

        self.serializer = SceneSerializer(
            self.resources,
            load_material=self._create_model_material
        )

        # Body profiles (assets/bodies/*.json) for File >
        # New Planet.
        self.body_presets = self._load_body_presets()

        self.editor = SceneEditor(
            self.scene,
            self.resources,
            self.render_system,
            self.window,
            self.serializer,
            populate_demo_scene=self._populate_demo_scene,
            load_model=MeshFactory.load_model,
            load_model_material=self._load_model_material_handle,
            panels=PanelRegistry(),
            body_presets={
                group: [(profile.id, profile.name) for profile in members]
                for group, members in preset_groups(self.body_presets).items()
            },
            populate_body_scene=self._populate_body_scene
        )

        # Tool windows, in View-menu order after the
        # editor's Hierarchy / Inspector.
        self.planet_panel = PlanetPanel(
            self.editor,
            self.planet_system,
            apply_preset=self.apply_body_preset,
            descriptions={
                profile.id: profile.description
                for profile in self.body_presets.values()
            }
        )

        self.tectonics_panel = TectonicsPanel(
            self.editor,
            self.tectonics_system,
            self.planet_system
        )

        self.climate_panel = ClimatePanel(
            self.editor,
            self.climate_system,
            self.planet_system
        )

        self.debug_panel = DebugPanel(
            self.editor.panels
        )

        self._load_initial_scene(
            scene_path
        )

        # Resolve world matrices before the first frame, so
        # the editor has valid data even if update() is
        # skipped.
        self.transform_system.update(
            self.scene
        )

        self._initialized = True

        Logger.info(
            "[Application] Initialization complete."
        )

    # =====================================================
    # Initial Scene
    # =====================================================

    DEFAULT_SCENE_PATH = Path("assets/scenes/planet.scene.json")

    def _load_initial_scene(
        self,
        scene_path
    ):
        """
        --scene PATH if given, else the default scene file if
        it exists, else the built-in demo. A scene file that
        fails to load falls back to the demo (the error is
        logged and shown in the editor status bar).
        """

        path = (
            Path(scene_path)
            if scene_path is not None
            else self.DEFAULT_SCENE_PATH
        )

        if path.is_file() and self.editor.load_initial(path):
            return

        if scene_path is not None and not path.is_file():

            Logger.error(
                "[Application] Scene file not found: %s",
                path
            )

        self.scene.clear()

        self._populate_demo_scene(
            self.scene
        )

        self.editor.new_scene_loaded()

    # =====================================================
    # Resource Loading
    # =====================================================

    def _load_resources(self):

        engine_assert(
            self.resources is not None,
            "Application has no Resources."
        )

        resources = self.resources
        handles = self.handles

        # -------------------------------------------------
        # Shader
        # -------------------------------------------------

        handles["lit"] = resources.shaders.load(
            "lit",
            lambda: Shader(
                "assets/shaders/lit.vert.glsl",
                "assets/shaders/lit.frag.glsl"
            )
        )

        # -------------------------------------------------
        # Textures
        # -------------------------------------------------
        #
        # Color maps are sRGB; data maps are linear.

        handles["uv_checker"] = resources.textures.load(
            "uv_checker",
            lambda: Texture2D(
                "assets/textures/uv_checker.png",
                srgb=True
            )
        )

        handles["tiles_albedo"] = resources.textures.load(
            "tiles_albedo",
            lambda: Texture2D(
                "assets/textures/tiles_albedo.png",
                srgb=True
            )
        )

        handles["tiles_normal"] = resources.textures.load(
            "tiles_normal",
            lambda: Texture2D(
                "assets/textures/tiles_normal.png",
                srgb=False
            )
        )

        # Occlusion (R), roughness (G), metallic (B).
        handles["tiles_orm"] = resources.textures.load(
            "tiles_orm",
            lambda: Texture2D(
                "assets/textures/tiles_orm.png",
                srgb=False
            )
        )

        # -------------------------------------------------
        # Materials
        # -------------------------------------------------

        # Metallic-roughness PBR (see assets/shaders/lit.frag.glsl).
        # Former Blinn-Phong shininess s maps to roughness
        # ~ sqrt(2 / (s + 2)): 32 -> 0.24, 128 -> 0.12.

        lit = handles["lit"]

        def cube_material():

            material = Material(lit)

            material.set_texture("uBaseColorMap", handles["uv_checker"])
            material.set_float("uMetallic", 0.0)
            material.set_float("uRoughness", 0.55)

            return material

        def floor_material():

            material = Material(lit)

            material.set_texture("uBaseColorMap", handles["tiles_albedo"])
            material.set_texture("uNormalMap", handles["tiles_normal"])

            # One texture carries occlusion, roughness and
            # metallic; the factors let the map decide.
            material.set_texture("uMetallicRoughnessMap", handles["tiles_orm"])
            material.set_texture("uOcclusionMap", handles["tiles_orm"])
            material.set_float("uMetallic", 1.0)
            material.set_float("uRoughness", 1.0)

            material.set_vec2("uUVScale", (3.0, 3.0))

            return material

        def pbr_material(color, metallic, roughness, emissive=(0.0, 0.0, 0.0)):

            def create():

                material = Material(lit)

                material.set_vec3("uBaseColor", color)
                material.set_float("uMetallic", metallic)
                material.set_float("uRoughness", roughness)
                material.set_vec3("uEmissive", emissive)

                return material

            return create

        handles["cube_material"] = resources.materials.load(
            "cube",
            cube_material
        )

        handles["floor_material"] = resources.materials.load(
            "floor",
            floor_material
        )

        # Keys kept from the Blinn-Phong era so existing
        # scene files still resolve.

        handles["red_material"] = resources.materials.load(
            "glossy_red",
            pbr_material((0.8, 0.08, 0.06), metallic=0.0, roughness=0.25)
        )

        # Real gold: a metal whose base color is its
        # reflectance.
        handles["gold_material"] = resources.materials.load(
            "glossy_gold",
            pbr_material((1.0, 0.77, 0.34), metallic=1.0, roughness=0.22)
        )

        handles["teal_material"] = resources.materials.load(
            "glossy_teal",
            pbr_material((0.1, 0.6, 0.55), metallic=0.0, roughness=0.4)
        )

        # Emissive: glows (and blooms) without lighting
        # anything else.
        handles["glow_material"] = resources.materials.load(
            "glow",
            pbr_material((0.05, 0.05, 0.05), metallic=0.0, roughness=0.5, emissive=(0.6, 3.5, 4.0))
        )

        # -------------------------------------------------
        # Meshes
        # -------------------------------------------------

        handles["cube"] = resources.meshes.load(
            "cube",
            MeshFactory.create_cube
        )

        handles["plane"] = resources.meshes.load(
            "plane",
            MeshFactory.create_plane
        )

        handles["sphere"] = resources.meshes.load(
            "sphere",
            MeshFactory.create_sphere
        )

        # Planet terrain: albedo and roughness are computed
        # per pixel from per-vertex terrain inputs
        # (assets/shaders/include/terrain.glsl).

        def planet_material():

            material = Material(lit)

            material.set_float("uTerrainShading", 1.0)

            return material

        handles["planet_material"] = resources.materials.load(
            PlanetSystem.MATERIAL_KEY,
            planet_material
        )

    # =====================================================
    # Body Profiles
    # =====================================================

    def _load_body_presets(
        self
    ):

        try:

            presets = load_presets()

        except ProfileError as error:

            # A broken profile must not stop the editor; the
            # demo needs Earth though.
            Logger.error("[Application] Body profiles: %s", error)

            presets = {}

        Logger.info(
            "[Application] %d body profile(s): %s.",
            len(presets),
            ", ".join(profile.name for profile in presets.values())
        )

        engine_assert(
            self.DEMO_BODY in presets,
            f"assets/bodies/{self.DEMO_BODY}.json is missing or invalid."
        )

        return presets

    def apply_body_preset(
        self,
        entity: Entity,
        profile_id: str
    ):
        """
        Turn an existing planet into another body: its
        planet, body, atmosphere, climate and tectonics
        components are replaced by the profile's (the
        transform stays). The camera backs off to show the
        whole new planet. One undo step.
        """

        profile = self.body_presets[profile_id]

        scene = self.scene

        seed = scene.get_component(entity, PlanetComponent).seed

        for component_type in (
            PlanetComponent,
            BodyComponent,
            AtmosphereComponent,
            ClimateComponent,
            TectonicsComponent,
        ):

            if scene.has_component(entity, component_type):
                scene.remove_component(entity, component_type)

        parts = components_for(profile, seed=seed)

        for component in parts.all():
            scene.add_component(entity, component)

        name = scene.try_get_component(entity, NameComponent)

        if name is not None:
            name.name = profile.name

        camera = next(
            (
                camera_entity
                for camera_entity, camera in scene.registry.view_with(CameraComponent)
                if camera.primary
            ),
            None
        )

        if camera is not None:

            center = np.asarray(
                scene.get_component(entity, TransformComponent).world_matrix,
                dtype=np.float64
            )[:3, 3]

            self.editor.frame_planet(camera, center, parts.planet.radius)

        self.editor.record(f"Become {profile.name}")

        self.editor.set_status(f"The planet is now {profile.name}.")

    # =====================================================
    # Model Materials
    # =====================================================

    # glTF texture slot -> (sampler, is color data)
    _MODEL_TEXTURE_SLOTS = {
        "base_color": ("uBaseColorMap", True),
        "metallic_roughness": ("uMetallicRoughnessMap", False),
        "normal": ("uNormalMap", False),
        "occlusion": ("uOcclusionMap", False),
        "emissive": ("uEmissiveMap", True),
    }

    def _create_model_material(
        self,
        key: str
    ) -> Material:
        """
        Build the Material for "<model path>#material" from
        the model file (glTF materials; other formats get a
        neutral default).
        """

        model_path, _, _ = key.partition("#")

        description = load_gltf_material(
            model_path
        )

        material = Material(
            self.handles["lit"]
        )

        if description is None:

            material.set_float("uRoughness", 0.6)

            return material

        material.set_vec3("uBaseColor", description.base_color)
        material.set_float("uMetallic", description.metallic)
        material.set_float("uRoughness", description.roughness)
        material.set_vec3("uEmissive", description.emissive)
        material.set_float("uNormalStrength", description.normal_scale)
        material.set_float("uOcclusionStrength", description.occlusion_strength)

        for slot, source in description.textures.items():

            sampler, srgb = self._MODEL_TEXTURE_SLOTS[slot]

            texture_key = f"{model_path}#{source.label}:{'srgb' if srgb else 'linear'}"

            def load(source=source, srgb=srgb, texture_key=texture_key):

                if source.path is not None:
                    return Texture2D(source.path, srgb=srgb)

                return Texture2D.from_encoded(
                    source.data,
                    srgb=srgb,
                    label=texture_key
                )

            material.set_texture(
                sampler,
                self.resources.textures.load(texture_key, load)
            )

        Logger.info(
            "[Application] Imported material '%s' from %s.",
            description.name,
            model_path
        )

        return material

    def _load_model_material_handle(
        self,
        model_path: str
    ):
        """Handle of a model's material (loading it once)."""

        key = model_material_key(
            model_path
        )

        return self.resources.materials.load(
            key,
            lambda: self._create_model_material(key)
        )

    # =====================================================
    # Scene Creation
    # =====================================================

    def _create_entity(
        self,
        name: str,
        transform: Transform | None = None,
        *components,
        parent: Entity | None = None
    ) -> Entity:

        entity = self.scene.create_entity()

        self.scene.add_component(
            entity,
            NameComponent(name)
        )

        self.scene.add_component(
            entity,
            TransformComponent(
                transform=transform or Transform()
            )
        )

        if parent is not None:

            self.scene.add_component(
                entity,
                HierarchyComponent(parent)
            )

        for component in components:

            self.scene.add_component(
                entity,
                component
            )

        return entity

    # The built-in demo and File > New Demo Scene.
    DEMO_BODY = "earth"

    def _populate_demo_scene(
        self,
        scene: Scene
    ):
        """Earth (with tectonics, climate and atmosphere)."""

        self._populate_body_scene(scene, self.DEMO_BODY)

    def _populate_body_scene(
        self,
        scene: Scene,
        body_id: str
    ):
        """
        Fill `scene` with a planet built from a body profile
        (assets/bodies/<body_id>.json; File > New Planet), a
        sun, and a camera.

        Solid bodies: the planet is turned so a scenic spot
        (planet/spawn.py) is on top, and placed so that spot
        sits at the world origin (the sky and editor grid
        assume +Y is up there); the camera stands above it.
        Gas giants: the camera starts in orbit.

        Tectonics and climate, when the body has them, are
        computed up front (~1 s) so the spawn point is
        chosen on the real terrain and the first frame
        already shows it.
        """

        engine_assert(
            scene is self.scene,
            "The planet builder fills the application's scene."
        )

        profile = self.body_presets[body_id]

        parts = components_for(profile)

        planet = parts.planet

        terrain_settings = terrain_settings_for(planet)

        # -------------------------------------------------
        # Simulations
        # -------------------------------------------------

        initial = None
        tectonic_field = None

        if parts.tectonics is not None:

            simulation = TectonicSimulation(
                tectonic_settings_for(planet, parts.tectonics)
            )

            initial = simulation.initial_state()

            tectonic_field = TectonicField.from_state(simulation.grid, initial, version=0)

        climate_field = None

        if parts.climate is not None:

            climate_field, _ = compute_climate(
                climate_settings_for(parts.climate),
                Terrain(terrain_settings, tectonic_field)
            )

        terrain = Terrain(terrain_settings, tectonic_field, climate_field)

        # -------------------------------------------------
        # Where to stand
        # -------------------------------------------------

        up = np.array([0.0, 1.0, 0.0])

        if profile.has_solid_surface:

            spawn = find_spawn(terrain)

            spawn_direction = spawn.direction
            view_direction = spawn.view_direction

            ground_height = (
                max(spawn.elevation, 0.0)
                if terrain_settings.has_liquid
                else spawn.elevation
            )

        else:

            # Over the bands at mid latitude.
            spawn_direction = np.array([0.0, np.sin(np.radians(20.0)), np.cos(np.radians(20.0))])
            view_direction = np.array([1.0, 0.0, 0.0])
            ground_height = 0.0

        # Rotation taking the spawn direction to +Y.
        axis = np.cross(spawn_direction, up)

        axis_length = float(np.linalg.norm(axis))

        angle = math.degrees(
            math.atan2(axis_length, float(np.dot(spawn_direction, up)))
        )

        orientation = (
            quaternion.from_axis_angle(axis / axis_length, angle)
            if axis_length > 1e-9
            else quaternion.identity()
        )

        ground = planet.radius + ground_height

        center = (0.0, -ground, 0.0)

        view = quaternion.rotate_vector(orientation, view_direction)

        # -------------------------------------------------
        # Camera
        # -------------------------------------------------

        fov = 60.0

        if profile.has_solid_surface:

            camera_position = (0.0, 500.0, 0.0)

            camera_orientation = quaternion.multiply(
                quaternion.look_rotation(view, up),
                quaternion.from_euler((-6.0, 0.0, 0.0))
            )

        else:

            # Back far enough for the whole planet to fit.
            distance = planet.radius / math.sin(math.radians(fov) * 0.5) * 1.25

            camera_position = (0.0, distance - ground, 0.0)

            camera_orientation = quaternion.look_rotation(
                np.array([0.0, -1.0, 0.0]),
                np.array([0.0, 0.0, -1.0])
            )

        self._create_entity(
            "Camera",
            Transform(
                position=camera_position,
                orientation=camera_orientation
            ),
            CameraComponent(
                fov=fov,
                near=0.5,
                far=2000.0,
                primary=True
            ),
            CameraControllerComponent(
                movement_speed=30.0,
                mouse_sensitivity=0.1,
                planet_mode=True,
                planet_center=center,
                planet_radius=ground,
                altitude_speed=1.0,
                min_altitude=2.0
            )
        )

        # -------------------------------------------------
        # Sun: mid-morning in the spawn point's spring
        # -------------------------------------------------

        tilt = min(profile.axial_tilt_deg, 180.0 - profile.axial_tilt_deg)

        sun_world = quaternion.rotate_vector(
            orientation,
            solar.sun_direction(
                spawn_direction,
                hour=10.5,
                declination=math.radians(min(15.0, tilt))
            )
        )

        self._create_entity(
            "Sun",
            Transform(
                orientation=sun_orientation(sun_world)
            ),
            DirectionalLightComponent(
                color=(1.0, 0.96, 0.9),
                intensity=5.0
            )
        )

        # -------------------------------------------------
        # Planet
        # -------------------------------------------------

        planet_entity = self._create_entity(
            profile.name,
            Transform(
                position=center,
                orientation=orientation
            ),
            *parts.all()
        )

        if parts.tectonics is not None:
            self.tectonics_system.prime(planet_entity, planet, parts.tectonics, initial)

        if parts.climate is not None:

            self.climate_system.prime(
                planet_entity,
                planet,
                parts.climate,
                climate_field,
                self.tectonics_system.field(planet_entity)
            )

    # =====================================================
    # Main Loop
    # =====================================================

    def run(
        self,
        exit_after: float | None = None,
        screenshot_path=None
    ):
        """
        exit_after: stop after this many seconds of the
            main loop (loading time excluded), for smoke
            tests and profiling runs.
        screenshot_path: with exit_after, save the last
            frame here before exiting.
        """

        engine_assert(
            self._initialized,
            "Application must be initialized before run()."
        )

        engine_assert(
            not self._running,
            "Application is already running."
        )

        engine_assert(
            not self._shutdown,
            "Cannot run an application after shutdown."
        )

        engine_assert(
            self.window is not None,
            "Application has no Window."
        )

        engine_assert(
            self.timer is not None,
            "Application has no Timer."
        )

        self._running = True

        Logger.info(
            "[Application] Entering main loop."
        )

        # Loading time must not count as the first frame.

        self.timer.reset()

        loop_start_ns = self.timer.time_ns

        profiler = self.profiler

        try:

            while not self.window.should_close():

                self.cprofile_capture.begin_frame()

                profiler.begin_frame()

                with profiler.scope("Events"):

                    Input.begin_frame()

                    self.window.poll_events()

                    self.timer.update()

                with profiler.scope("Update"):

                    self.update()

                with profiler.scope("Render"):

                    self.render()

                if (
                    exit_after is not None
                    and (self.timer.time_ns - loop_start_ns) / 1e9 >= exit_after
                ):

                    if screenshot_path is not None:

                        self.save_screenshot(
                            screenshot_path
                        )

                    self.window.set_should_close(
                        True
                    )

                # With VSync on, this blocks until the
                # display is ready, so it absorbs whatever
                # is left of the refresh interval.

                with profiler.scope("Swap"):

                    self.window.swap_buffers()

                profiler.end_frame()

                if self.cprofile_capture.end_frame():

                    # Frames under cProfile run ~2x slower
                    # and the capture's file write is one
                    # huge frame; drop them from the stats.

                    profiler.reset()

        finally:

            self._running = False

            self.cprofile_capture.cancel()

            Logger.info(
                "[Application] Leaving main loop."
            )

            if profiler.frame_times_ms:

                Logger.info(
                    "[Profiler] Final statistics:\n%s",
                    profiler.format_report()
                )

    # =====================================================
    # Update
    # =====================================================

    def set_ui_visible(
        self,
        visible: bool
    ):

        self.debug_panel.visible = visible
        self.editor.visible = visible

    def update(self):

        engine_assert(
            self.timer is not None
            and self.window is not None
            and self.scene is not None,
            "Application is not initialized."
        )

        dt = self.timer.delta_time

        # -------------------------------------------------
        # Application Input
        # -------------------------------------------------

        # Esc no longer quits: in the editor it clears the
        # selection, and a stray keypress must not throw
        # away unsaved work. Close the window or use
        # File > Exit.

        ui_wants_keyboard = (
            self.imgui_layer.wants_keyboard
            and not self._looking
        )

        if self.editor.ui_hidden_requested:

            self.editor.ui_hidden_requested = False

            self.set_ui_visible(False)

        if not ui_wants_keyboard:

            if Input.is_key_pressed(Key.F1):

                self.set_ui_visible(
                    not self.debug_panel.visible
                )

            if Input.is_key_pressed(Key.F5):

                self.resources.reload_changed_shaders(
                    force=True
                )

        profiler = self.profiler

        # -------------------------------------------------
        # Shader Hot Reload
        # -------------------------------------------------

        self._shader_poll_timer += dt

        if self._shader_poll_timer >= self.SHADER_POLL_INTERVAL:

            self._shader_poll_timer = 0.0

            with profiler.scope("Shader poll"):

                self.resources.reload_changed_shaders()

        # -------------------------------------------------
        # Fixed-Step Simulation
        # -------------------------------------------------
        #
        # Only while the editor is in Play mode; in Edit
        # mode the scene holds still.

        if self.editor.playing:
            self._fixed_accumulator += dt
        else:
            self._fixed_accumulator = 0.0

        steps = 0

        with profiler.scope("Fixed update"):

            while (
                self._fixed_accumulator >= self.FIXED_DELTA_TIME
                and steps < self.MAX_FIXED_STEPS_PER_FRAME
            ):

                self.fixed_update(
                    self.FIXED_DELTA_TIME
                )

                self._fixed_accumulator -= self.FIXED_DELTA_TIME

                steps += 1

        if steps == self.MAX_FIXED_STEPS_PER_FRAME:

            # Drop time we could not catch up on.

            self._fixed_accumulator = 0.0

        # -------------------------------------------------
        # Background Job Results
        # -------------------------------------------------
        #
        # Finished jobs' callbacks (usually GPU uploads)
        # run here, on the GL thread, within a time budget.

        with profiler.scope("Jobs"):

            self.jobs.process_completions(
                self.JOB_BUDGET_SECONDS
            )

        # -------------------------------------------------
        # Camera
        # -------------------------------------------------

        with profiler.scope("Camera"):

            # Planet-mode cameras fly relative to the
            # ground under them.
            self.planet_system.update_camera_ground(
                self.scene
            )

            # Editor-style flying: WASD / Space / Shift only
            # while the right mouse button is held, so the
            # same keys can switch gizmo modes the rest of
            # the time.

            self.camera_controller_system.update(
                self.scene,
                dt,
                look_enabled=self._looking,
                move_enabled=self._looking
            )

        # -------------------------------------------------
        # World Transforms
        # -------------------------------------------------
        #
        # Last, so rendering sees this frame's movement.

        with profiler.scope("Transforms"):

            self.transform_system.update(
                self.scene
            )

        # -------------------------------------------------
        # Planet Streaming
        # -------------------------------------------------

        # Data views show the terrain without haze.
        self.render_system.suppress_haze = (
            self.planet_system.view_mode != 0
        )

        with profiler.scope("Tectonics"):

            self.tectonics_system.update(
                self.scene
            )

        with profiler.scope("Climate"):

            self.climate_system.update(
                self.scene
            )

        with profiler.scope("Planet"):

            self.planet_system.update(
                self.scene,
                self.render_system.camera_position(self.scene)
            )

    def fixed_update(
        self,
        fixed_delta_time: float
    ):

        self.rotator_system.fixed_update(
            self.scene,
            fixed_delta_time
        )

    # =====================================================
    # Render
    # =====================================================

    def render(self):

        engine_assert(
            self.render_system is not None
            and self.scene is not None
            and self.window is not None,
            "Application is not initialized."
        )

        width, height = (
            self.window.framebuffer_size
        )

        # -------------------------------------------------
        # Minimized Window
        # -------------------------------------------------

        if (
            width <= 0
            or height <= 0
        ):
            return

        # -------------------------------------------------
        # Scene, into the editor's viewport
        # -------------------------------------------------
        #
        # With the editor visible the 3D view is the docked
        # layout's central area (last frame's rectangle; the
        # UI is built after rendering); otherwise the whole
        # window. Clear first so nothing stale shows around
        # it.

        Framebuffer.bind_default(width, height)

        RenderCommand.set_clear_color((0.06, 0.065, 0.075, 1.0))
        RenderCommand.clear(depth=False)

        view_x, view_y, view_width, view_height = self._viewport_pixels(
            width,
            height
        )

        self.render_system.render(
            self.scene,
            view_width,
            view_height,
            output_origin=(view_x, view_y),
            extra_items=self.planet_system.draw_items,
            selected=(
                self.editor.selected
                if self.editor.visible
                else None
            )
        )

        # -------------------------------------------------
        # Debug UI (on top of the final image)
        # -------------------------------------------------

        profiler = self.profiler

        with profiler.scope("UI"):

            with profiler.scope("Build"):

                self.imgui_layer.begin_frame()

                self.editor.draw(
                    looking=self._looking
                )

                self.planet_panel.draw(
                    self.timer.delta_time,
                    self.timer.fps
                )

                self.tectonics_panel.draw()

                self.climate_panel.draw()

                self.debug_panel.draw(
                    DebugContext(
                        resources=self.resources,
                        settings=self.render_system.settings,
                        stats=self.renderer.stats,
                        timer=self.timer,
                        reload_shaders=lambda: self.resources.reload_changed_shaders(
                            force=True
                        ),
                        profiler=profiler,
                        cprofile_capture=self.cprofile_capture,
                        window=self.window,
                        jobs=self.jobs,
                        planet_stats=self.planet_system.stats
                    )
                )

            with profiler.scope("Draw", gpu=True):

                self.imgui_layer.end_frame()

    def _viewport_pixels(
        self,
        width: int,
        height: int
    ) -> tuple[int, int, int, int]:
        """
        The editor viewport in framebuffer pixels
        (x, y from the bottom-left, width, height).
        """

        rect = self.editor.viewport_rect

        io = imgui.get_io()

        scale_x = io.display_framebuffer_scale.x or 1.0
        scale_y = io.display_framebuffer_scale.y or 1.0

        x = int(round(rect.x * scale_x))
        top = int(round(rect.y * scale_y))

        view_width = max(1, min(width - x, int(round(rect.width * scale_x))))
        view_height = max(1, min(height - top, int(round(rect.height * scale_y))))

        y = max(0, height - top - view_height)

        return x, y, view_width, view_height

    # =====================================================
    # Screenshot
    # =====================================================

    def save_screenshot(
        self,
        path
    ):
        """
        Save the window's current back buffer (scene + UI)
        as an image. Call after render(), before
        swap_buffers().
        """

        engine_assert(
            self.window is not None,
            "Application has no Window."
        )

        from pathlib import Path

        import numpy as np

        from OpenGL.GL import (
            GL_BACK,
            GL_FRAMEBUFFER,
            GL_PACK_ALIGNMENT,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            glBindFramebuffer,
            glPixelStorei,
            glReadBuffer,
            glReadPixels
        )
        from PIL import Image

        width, height = self.window.framebuffer_size

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glReadBuffer(GL_BACK)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)

        pixels = glReadPixels(
            0,
            0,
            width,
            height,
            GL_RGB,
            GL_UNSIGNED_BYTE
        )

        # GL rows start at the bottom.

        image = np.frombuffer(
            pixels,
            dtype=np.uint8
        ).reshape(height, width, 3)[::-1]

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        Image.fromarray(image).save(path)

        Logger.info(
            "[Application] Saved screenshot '%s'.",
            path
        )

    # =====================================================
    # Events
    # =====================================================

    def on_event(
        self,
        event
    ):

        dispatcher = EventDispatcher(
            event
        )

        dispatcher.dispatch(
            WindowCloseEvent,
            self.on_window_close
        )

        dispatcher.dispatch(
            WindowResizeEvent,
            self.on_window_resize
        )

        dispatcher.dispatch(
            WindowLostFocusEvent,
            self.on_window_lost_focus
        )

        dispatcher.dispatch(
            MouseButtonPressedEvent,
            self.on_mouse_button_pressed
        )

        dispatcher.dispatch(
            MouseButtonReleasedEvent,
            self.on_mouse_button_released
        )

    def on_window_close(
        self,
        event
    ):

        Logger.info(
            "[Application] Window close requested."
        )

        # With unsaved changes, keep the window open and ask
        # first; the editor closes it after Save / Discard.

        if self.editor is not None and self.editor.dirty:

            self.window.set_should_close(
                False
            )

            self.editor.request_close(
                lambda: self.window.set_should_close(True)
            )

        return True

    def on_window_resize(
        self,
        event
    ):

        # RenderSystem sets the viewport and resizes its
        # framebuffers every frame from the framebuffer
        # size, so nothing to do here.

        return False

    def on_window_lost_focus(
        self,
        event
    ):

        self._set_looking(
            False
        )

        return False

    # -----------------------------------------------------
    # Mouse Look
    # -----------------------------------------------------
    #
    # Hold the right mouse button to capture the cursor and
    # look around. Releasing it frees the cursor for the
    # debug UI. Clicks that land on the UI never start
    # looking.

    def on_mouse_button_pressed(
        self,
        event
    ):

        if (
            event.button == MouseButton.RIGHT
            and not self.imgui_layer.wants_mouse
        ):

            self._set_looking(
                True
            )

            return True

        return False

    def on_mouse_button_released(
        self,
        event
    ):

        if event.button == MouseButton.RIGHT:

            self._set_looking(
                False
            )

            return True

        return False

    def _set_looking(
        self,
        looking: bool
    ):

        if looking == self._looking:
            return

        self._looking = looking

        self.window.set_cursor_captured(
            looking
        )

        self.imgui_layer.set_mouse_enabled(
            not looking
        )

    # =====================================================
    # Shutdown
    # =====================================================

    def shutdown(self):

        if self._shutdown:
            return

        Logger.info(
            "[Application] Shutting down."
        )

        # =================================================
        # UI
        # =================================================

        if self.imgui_layer is not None:

            self.imgui_layer.shutdown()

            self.imgui_layer = None

        self.debug_panel = None

        self.editor = None
        self.serializer = None

        # =================================================
        # Scene
        # =================================================

        if self.scene is not None:

            self.scene.shutdown()

            self.scene = None

        # =================================================
        # Systems
        # =================================================

        if self.climate_system is not None:

            self.climate_system.shutdown()

            self.climate_system = None

        if self.tectonics_system is not None:

            self.tectonics_system.shutdown()

            self.tectonics_system = None

        # Chunk meshes live in the renderer's geometry pool.
        if self.planet_system is not None:

            self.planet_system.shutdown()

            self.planet_system = None

        if self.render_system is not None:

            self.render_system.shutdown()

            self.render_system = None

        self.camera_controller_system = None
        self.transform_system = None
        self.rotator_system = None

        if self.jobs is not None:

            self.jobs.shutdown()

            self.jobs = None

        self.handles.clear()

        # =================================================
        # Resources
        # =================================================

        if self.resources is not None:

            self.resources.shutdown()

            self.resources = None

        # =================================================
        # Profiling
        # =================================================

        self.cprofile_capture.cancel()

        self.profiler.set_gpu_backend(
            None
        )

        if self.gpu_timer is not None:

            self.gpu_timer.delete()

            self.gpu_timer = None

        # =================================================
        # Renderer
        # =================================================

        if self.renderer is not None:

            self.renderer.shutdown()

            self.renderer = None

        # =================================================
        # Timer
        # =================================================

        self.timer = None

        # =================================================
        # Window
        # =================================================

        if self.window is not None:

            self.window.close()

            self.window = None

        # =================================================
        # Input
        # =================================================

        Input.clear()

        self._shutdown = True

        Logger.info(
            "[Application] Shutdown complete."
        )

    # =====================================================
    # State
    # =====================================================

    @property
    def initialized(self) -> bool:

        return self._initialized

    @property
    def running(self) -> bool:

        return self._running

    @property
    def is_shutdown(self) -> bool:

        return self._shutdown
