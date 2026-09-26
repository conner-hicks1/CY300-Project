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

        self.camera_entity: Entity | None = None

    # =====================================================
    # Initialization
    # =====================================================

    def initialize(self):

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
        self._create_scene()

        self._initialized = True

        Logger.info(
            "[Application] Initialization complete."
        )

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

        handles["steve"] = resources.textures.load(
            "steve",
            lambda: Texture2D(
                "assets/textures/steve_gilland.jpg",
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

        handles["tiles_specular"] = resources.textures.load(
            "tiles_specular",
            lambda: Texture2D(
                "assets/textures/tiles_specular.png",
                srgb=False
            )
        )

        # -------------------------------------------------
        # Materials
        # -------------------------------------------------

        lit = handles["lit"]

        def cube_material():

            material = Material(lit)

            material.set_texture("uTexture", handles["steve"])
            material.set_float("uSpecularStrength", 0.4)
            material.set_float("uShininess", 32.0)

            return material

        def floor_material():

            material = Material(lit)

            material.set_texture("uTexture", handles["tiles_albedo"])
            material.set_texture("uNormalMap", handles["tiles_normal"])
            material.set_texture("uSpecularMap", handles["tiles_specular"])
            material.set_float("uSpecularStrength", 0.8)
            material.set_float("uShininess", 64.0)
            material.set_vec2("uUVScale", (3.0, 3.0))

            return material

        def glossy_material(color):

            def create():

                material = Material(lit)

                material.set_vec3("uBaseColor", color)
                material.set_float("uSpecularStrength", 1.0)
                material.set_float("uShininess", 128.0)

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

        handles["red_material"] = resources.materials.load(
            "glossy_red",
            glossy_material((0.8, 0.1, 0.08))
        )

        handles["gold_material"] = resources.materials.load(
            "glossy_gold",
            glossy_material((0.9, 0.65, 0.2))
        )

        handles["teal_material"] = resources.materials.load(
            "glossy_teal",
            glossy_material((0.1, 0.6, 0.55))
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

        handles["torus"] = resources.meshes.load(
            "torus",
            lambda: MeshFactory.load_model(
                "assets/models/torus.obj"
            )
        )

        handles["pyramid"] = resources.meshes.load(
            "pyramid",
            lambda: MeshFactory.load_model(
                "assets/models/pyramid.gltf"
            )
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

    def _create_scene(self):

        engine_assert(
            self.scene is not None,
            "Application has no Scene."
        )

        handles = self.handles

        # -------------------------------------------------
        # Camera
        # -------------------------------------------------

        self.camera_entity = self._create_entity(
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
                intensity=1.2,
                ambient=0.05
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
                intensity=5.0,
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
                intensity=10.0,
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
                material=handles["cube_material"]
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

        ui_wants_keyboard = (
            self.imgui_layer.wants_keyboard
            and not self._looking
        )

        if not ui_wants_keyboard:

            if Input.is_key_pressed(Key.ESCAPE):

                self.window.set_should_close(
                    True
                )

            if Input.is_key_pressed(Key.F1):

                self.debug_panel.visible = (
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

        self._fixed_accumulator += dt

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
        # Camera
        # -------------------------------------------------

        with profiler.scope("Camera"):

            self.camera_controller_system.update(
                self.scene,
                dt,
                look_enabled=self._looking,
                move_enabled=not ui_wants_keyboard
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
            height
        )

        # -------------------------------------------------
        # Debug UI (on top of the final image)
        # -------------------------------------------------

        profiler = self.profiler

        with profiler.scope("UI"):

            with profiler.scope("Build"):

                self.imgui_layer.begin_frame()

                self.debug_panel.draw(
                    DebugContext(
                        scene=self.scene,
                        resources=self.resources,
                        settings=self.render_system.settings,
                        stats=self.renderer.stats,
                        timer=self.timer,
                        reload_shaders=lambda: self.resources.reload_changed_shaders(
                            force=True
                        ),
                        profiler=profiler,
                        cprofile_capture=self.cprofile_capture,
                        window=self.window
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

        # =================================================
        # Scene
        # =================================================

        if self.scene is not None:

            self.scene.shutdown()

            self.scene = None

        self.camera_entity = None

        # =================================================
        # Systems
        # =================================================

        if self.render_system is not None:

            self.render_system.shutdown()

            self.render_system = None

        self.camera_controller_system = None
        self.transform_system = None
        self.rotator_system = None

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
