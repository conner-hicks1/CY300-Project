from pathlib import Path

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

from graphics.gpu_timer import GpuTimer
from graphics.material import Material
from graphics.mesh_factory import MeshFactory
from graphics.model_loader import load_gltf_material
from graphics.render_command import RenderCommand
from graphics.renderer import Renderer
from graphics.shader import Shader
from graphics.texture import Texture2D

from math3d.transform import Transform

from resources.resources import Resources

from systems.camera_controller_system import (
    CameraControllerSystem
)
from systems.render_system import RenderSystem
from systems.rotator_system import RotatorSystem
from systems.transform_system import TransformSystem

from scene.scene import Scene
from scene.scene_serializer import (
    SceneSerializer,
    model_material_key
)

from editor.scene_editor import SceneEditor

from ui.debug_panel import DebugContext, DebugPanel
from ui.imgui_layer import ImGuiLayer


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

        # -------------------------------------------------
        # UI
        # -------------------------------------------------

        self.imgui_layer: ImGuiLayer | None = None
        self.debug_panel: DebugPanel | None = None

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

        self.debug_panel = DebugPanel()

        # -------------------------------------------------
        # Content
        # -------------------------------------------------

        self._load_resources()

        self.serializer = SceneSerializer(
            self.resources,
            load_material=self._create_model_material
        )

        self.editor = SceneEditor(
            self.scene,
            self.resources,
            self.render_system,
            self.window,
            self.serializer,
            populate_demo_scene=self._populate_demo_scene,
            load_model=MeshFactory.load_model,
            load_model_material=self._load_model_material_handle
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

    DEFAULT_SCENE_PATH = Path("assets/scenes/demo.scene.json")

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

        # Model meshes are keyed by file path: scene files
        # reference them by key, and the serializer can load
        # any model path on demand.

        handles["torus"] = resources.meshes.load(
            "assets/models/torus.obj",
            lambda: MeshFactory.load_model(
                "assets/models/torus.obj"
            )
        )

        handles["pyramid"] = resources.meshes.load(
            "assets/models/pyramid.gltf",
            lambda: MeshFactory.load_model(
                "assets/models/pyramid.gltf"
            )
        )

        # The pyramid's own glTF material.
        pyramid_material_key = model_material_key(
            "assets/models/pyramid.gltf"
        )

        handles["pyramid_material"] = resources.materials.load(
            pyramid_material_key,
            lambda: self._create_model_material(pyramid_material_key)
        )

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

    def _populate_demo_scene(
        self,
        scene: Scene
    ):
        """
        Fill `scene` with the built-in demo content (used for
        File > New Demo Scene and when no scene file exists).
        """

        engine_assert(
            scene is self.scene,
            "The demo builder fills the application's scene."
        )

        handles = self.handles

        # -------------------------------------------------
        # Camera
        # -------------------------------------------------

        self._create_entity(
            "Camera",
            Transform(
                position=(0.0, 1.2, 4.5),
                rotation=(-12.0, 0.0, 0.0)
            ),
            CameraComponent(
                fov=50.0,
                near=0.1,
                far=100.0,
                primary=True
            ),
            CameraControllerComponent(
                movement_speed=3.0,
                mouse_sensitivity=0.1
            )
        )

        # -------------------------------------------------
        # Lights
        # -------------------------------------------------

        # Pitched down 50 degrees, yawed 30 degrees.

        self._create_entity(
            "Sun",
            Transform(
                rotation=(-50.0, 30.0, 0.0)
            ),
            DirectionalLightComponent(
                color=(1.0, 0.96, 0.9),
                intensity=5.0
            )
        )

        # A point light orbits the scene: it is a child of
        # a spinning pivot, so its world position comes
        # from the transform hierarchy.

        pivot = self._create_entity(
            "Orbit Pivot",
            Transform(
                position=(0.0, 0.6, 0.0)
            ),
            RotatorComponent(
                degrees_per_second=(0.0, 40.0, 0.0)
            )
        )

        self._create_entity(
            "Orbiting Point Light",
            Transform(
                position=(2.2, 0.0, 0.0)
            ),
            PointLightComponent(
                color=(1.0, 0.55, 0.25),
                intensity=15.0,
                range=6.0
            ),
            parent=pivot
        )

        # Blue cone pointing straight down onto the floor.
        # Pitch -90 turns forward (-Z) into -Y.

        self._create_entity(
            "Spot Light",
            Transform(
                position=(-2.2, 2.5, -1.5),
                rotation=(-90.0, 0.0, 0.0)
            ),
            SpotLightComponent(
                color=(0.3, 0.5, 1.0),
                intensity=30.0,
                range=8.0,
                inner_angle=18.0,
                outer_angle=28.0
            )
        )

        # -------------------------------------------------
        # Floor
        # -------------------------------------------------

        self._create_entity(
            "Floor",
            Transform(
                position=(0.0, -0.5, 0.0),
                scale=(12.0, 1.0, 12.0)
            ),
            MeshRendererComponent(
                mesh=handles["plane"],
                material=handles["floor_material"],
                casts_shadows=False
            )
        )

        # -------------------------------------------------
        # Objects
        # -------------------------------------------------

        cube = self._create_entity(
            "Cube",
            Transform(
                position=(0.0, 0.25, 0.0)
            ),
            MeshRendererComponent(
                mesh=handles["cube"],
                material=handles["cube_material"]
            ),
            RotatorComponent(
                degrees_per_second=(0.0, 30.0, 0.0)
            )
        )

        # Child of the cube: rides along as it spins.

        self._create_entity(
            "Moon",
            Transform(
                position=(0.0, 0.85, 0.0),
                scale=(0.35, 0.35, 0.35)
            ),
            MeshRendererComponent(
                mesh=handles["sphere"],
                material=handles["gold_material"]
            ),
            parent=cube
        )

        # Emissive: shows off bloom.

        self._create_entity(
            "Glow Orb",
            Transform(
                position=(-0.4, -0.3, 1.5),
                scale=(0.3, 0.3, 0.3)
            ),
            MeshRendererComponent(
                mesh=handles["sphere"],
                material=handles["glow_material"],
                casts_shadows=False
            )
        )

        self._create_entity(
            "Sphere",
            Transform(
                position=(1.6, 0.0, -0.8)
            ),
            MeshRendererComponent(
                mesh=handles["sphere"],
                material=handles["red_material"]
            )
        )

        self._create_entity(
            "Torus (OBJ)",
            Transform(
                position=(-1.6, 0.0, 0.6),
                rotation=(60.0, 0.0, 0.0)
            ),
            MeshRendererComponent(
                mesh=handles["torus"],
                material=handles["teal_material"]
            ),
            RotatorComponent(
                degrees_per_second=(0.0, 0.0, 45.0)
            )
        )

        self._create_entity(
            "Pyramid (glTF)",
            Transform(
                position=(1.4, -0.5, 1.4)
            ),
            MeshRendererComponent(
                mesh=handles["pyramid"],
                material=handles["pyramid_material"]
            )
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

        if not ui_wants_keyboard:

            if Input.is_key_pressed(Key.F1):

                visible = not self.debug_panel.visible

                self.debug_panel.visible = visible
                self.editor.visible = visible

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

            # Editor-style flying: WASD/QE only while the
            # right mouse button is held, so the same keys
            # can switch gizmo modes the rest of the time.

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
        # Scene
        # -------------------------------------------------

        self.render_system.render(
            self.scene,
            width,
            height,
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
                        jobs=self.jobs
                    )
                )

            with profiler.scope("Draw", gpu=True):

                self.imgui_layer.end_frame()

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
