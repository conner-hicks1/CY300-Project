from core.assertions import engine_assert
from core.events import (
    EventDispatcher,
    WindowCloseEvent,
    WindowResizeEvent
)
from core.handle import Handle
from core.input import Input
from core.key_codes import Key
from core.logger import Logger
from core.timer import Timer
from core.window import Window

from ecs.components import (
    CameraComponent,
    CameraControllerComponent,
    DirectionalLightComponent,
    MeshRendererComponent,
    PointLightComponent,
    SpotLightComponent,
    TransformComponent
)
from ecs.entity import Entity

from graphics.material import Material
from graphics.render_command import RenderCommand
from graphics.renderer import Renderer
from graphics.shader import Shader
from graphics.texture import Texture2D
from graphics.mesh_factory import MeshFactory

from math3d.transform import Transform

from resources.resources import Resources

from systems.render_system import RenderSystem
from systems.camera_controller_system import (
    CameraControllerSystem
)

from scene.scene import Scene


class Application:

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

        # -------------------------------------------------
        # Graphics
        # -------------------------------------------------

        self.renderer: Renderer | None = None
        self.render_system: RenderSystem | None = None
        self.camera_controller_system: (
            CameraControllerSystem | None
        ) = None

        # -------------------------------------------------
        # Resources
        # -------------------------------------------------

        self.resources: Resources | None = None

        self.shader_handle: Handle | None = None
        self.texture_handle: Handle | None = None
        self.material_handle: Handle | None = None
        self.cube_handle: Handle | None = None
        self.plane_handle = None
        self.floor_entity = None

        # -------------------------------------------------
        # Scene
        # -------------------------------------------------

        self.scene: Scene | None = None

        self.cube_entity: Entity | None = None
        self.camera_entity: Entity | None = None
        self.light_entity: Entity | None = None
        self.point_light_entity: Entity | None = None
        self.spot_light_entity: Entity | None = None

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
            800,
            600,
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

        self.camera_controller_system = (
            CameraControllerSystem()
        )

        self.render_system = RenderSystem(
            self.renderer,
            self.resources
        )

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

        # =====================================================
        # Shader
        # =====================================================

        self.shader_handle = (
            self.resources.shaders.load(
                "basic",
                lambda: Shader(
                    "assets/shaders/vertex_shader.glsl",
                    "assets/shaders/fragment_shader.glsl"
                )
            )
        )

        # =====================================================
        # Texture
        # =====================================================

        self.texture_handle = (
            self.resources.textures.load(
                "steve",
                lambda: Texture2D(
                    "assets/textures/steve_gilland.jpg"
                )
            )
        )

        # =====================================================
        # Material
        # =====================================================

        def create_material():

            material = Material(
                self.shader_handle
            )

            material.set_texture(
                "uTexture",
                self.texture_handle
            )

            material.set_float(
                "uSpecularStrength",
                0.5
            )

            material.set_float(
                "uShininess",
                32.0
            )

            return material

        self.material_handle = (
            self.resources.materials.load(
                "checker_material",
                create_material
            )
        )

        # =====================================================
        # Meshes
        # =====================================================

        self.cube_handle = (
            self.resources.meshes.load(
                "cube",
                MeshFactory.create_cube
            )
        )

        self.plane_handle = (
            self.resources.meshes.load(
                "plane",
                MeshFactory.create_plane
            )
        )
    # =====================================================
    # Scene Creation
    # =====================================================
    def _create_scene(self):

        engine_assert(
            self.scene is not None,
            "Application has no Scene."
        )

        engine_assert(
            self.cube_handle is not None,
            "Cube mesh has not been loaded."
        )

        engine_assert(
            self.plane_handle is not None,
            "Plane mesh has not been loaded."
        )

        engine_assert(
            self.material_handle is not None,
            "Material has not been loaded."
        )

        # =====================================================
        # Camera
        # =====================================================

        self.camera_entity = (
            self.scene.create_entity()
        )

        self.scene.add_component(
            self.camera_entity,
            TransformComponent(
                transform=Transform(
                    position=(0.0, 0.0, 3.0)
                )
            )
        )

        self.scene.add_component(
            self.camera_entity,
            CameraComponent(
                fov=45.0,
                near=0.1,
                far=100.0,
                primary=True
            )
        )

        self.scene.add_component(
            self.camera_entity,
            CameraControllerComponent(
                movement_speed=3.0,
                mouse_sensitivity=0.1
            )
        )

        # =====================================================
        # Directional Light
        # =====================================================
        #
        # Pitched down 45 degrees and yawed 30 degrees so
        # the light hits the cube at an angle.

        self.light_entity = (
            self.scene.create_entity()
        )

        self.scene.add_component(
            self.light_entity,
            TransformComponent(
                transform=Transform(
                    rotation=(-45.0, 30.0, 0.0)
                )
            )
        )

        self.scene.add_component(
            self.light_entity,
            DirectionalLightComponent(
                color=(1.0, 1.0, 1.0),
                intensity=0.4,
                ambient=0.1
            )
        )

        # =====================================================
        # Point Light
        # =====================================================
        #
        # Warm light hovering beside the cube.

        self.point_light_entity = (
            self.scene.create_entity()
        )

        self.scene.add_component(
            self.point_light_entity,
            TransformComponent(
                transform=Transform(
                    position=(1.5, 0.5, 1.0)
                )
            )
        )

        self.scene.add_component(
            self.point_light_entity,
            PointLightComponent(
                color=(1.0, 0.6, 0.3),
                intensity=4.0,
                range=6.0
            )
        )

        # =====================================================
        # Spot Light
        # =====================================================
        #
        # Blue cone pointing straight down onto the floor.
        # Pitch -90 turns forward (-Z) into -Y.

        self.spot_light_entity = (
            self.scene.create_entity()
        )

        self.scene.add_component(
            self.spot_light_entity,
            TransformComponent(
                transform=Transform(
                    position=(-2.0, 2.0, -1.0),
                    rotation=(-90.0, 0.0, 0.0)
                )
            )
        )

        self.scene.add_component(
            self.spot_light_entity,
            SpotLightComponent(
                color=(0.3, 0.5, 1.0),
                intensity=8.0,
                range=8.0,
                inner_angle=15.0,
                outer_angle=25.0
            )
        )

        # =====================================================
        # Cube
        # =====================================================

        self.cube_entity = (
            self.scene.create_entity()
        )

        self.scene.add_component(
            self.cube_entity,
            TransformComponent(
                transform=Transform(
                    position=(0.0, 0.0, 0.0)
                )
            )
        )

        self.scene.add_component(
            self.cube_entity,
            MeshRendererComponent(
                mesh=self.cube_handle,
                material=self.material_handle
            )
        )

        # =====================================================
        # Floor
        # =====================================================

        self.floor_entity = (
            self.scene.create_entity()
        )

        self.scene.add_component(
            self.floor_entity,
            TransformComponent(
                transform=Transform(
                    position=(0.0, -1.0, 0.0),
                    scale=(10.0, 1.0, 10.0)
                )
            )
        )

        self.scene.add_component(
            self.floor_entity,
            MeshRendererComponent(
                mesh=self.plane_handle,
                material=self.material_handle
            )
        )
    # =====================================================
    # Main Loop
    # =====================================================

    def run(self):

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

        try:

            while not self.window.should_close():

                Input.begin_frame()

                self.window.poll_events()

                self.timer.update()

                self.update()

                self.render()

                self.window.swap_buffers()

        finally:

            self._running = False

            Logger.info(
                "[Application] Leaving main loop."
            )

    # =====================================================
    # Update
    # =====================================================

    def update(self):

        engine_assert(
            self.timer is not None,
            "Application has no Timer."
        )

        engine_assert(
            self.window is not None,
            "Application has no Window."
        )

        engine_assert(
            self.scene is not None,
            "Application has no Scene."
        )

        engine_assert(
            self.camera_controller_system is not None,
            "Application has no CameraControllerSystem."
        )

        dt = self.timer.delta_time

        # -------------------------------------------------
        # Camera
        # -------------------------------------------------

        self.camera_controller_system.update(
            self.scene,
            dt
        )

        # -------------------------------------------------
        # Temporary Cube Rotation
        # -------------------------------------------------

        engine_assert(
            self.cube_entity is not None,
            "Application has no cube entity."
        )

        transform_component = (
            self.scene.get_component(
                self.cube_entity,
                TransformComponent
            )
        )

        transform_component.transform.rotation[1] += (
            30.0 * dt
        )

        # -------------------------------------------------
        # Application Input
        # -------------------------------------------------

        if Input.is_key_pressed(
            Key.ESCAPE
        ):

            self.window.set_should_close(
                True
            )
            
    # =====================================================
    # Render
    # =====================================================

    def render(self):

        engine_assert(
            self.render_system is not None,
            "Application has no RenderSystem."
        )

        engine_assert(
            self.scene is not None,
            "Application has no Scene."
        )

        engine_assert(
            self.window is not None,
            "Application has no Window."
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
        # Scene Rendering
        # -------------------------------------------------

        self.render_system.render(
            self.scene,
            width / height
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

        if (
            event.width <= 0
            or event.height <= 0
        ):
            return False

        RenderCommand.set_viewport(
            0,
            0,
            event.width,
            event.height
        )

        return False
    
    # =====================================================
    # Shutdown
    # =====================================================
    def shutdown(self):

        if self._shutdown:
            return

        Logger.info(
            "[Application] Shutting down."
        )

        # =====================================================
        # Systems
        # =====================================================

        self.camera_controller_system = None
        self.render_system = None

        # =====================================================
        # Scene
        # =====================================================

        self.shader_handle: Handle | None = None
        self.texture_handle: Handle | None = None
        self.material_handle: Handle | None = None
        self.cube_handle: Handle | None = None
        self.plane_handle: Handle | None = None

        self.scene: Scene | None = None

        self.camera_entity: Entity | None = None
        self.cube_entity: Entity | None = None
        self.floor_entity: Entity | None = None
        self.light_entity: Entity | None = None
        self.point_light_entity: Entity | None = None
        self.spot_light_entity: Entity | None = None

        # =====================================================
        # Resources
        # =====================================================

        if self.resources is not None:

            self.resources.shutdown()

            self.resources = None

        # =====================================================
        # Renderer
        # =====================================================

        if self.renderer is not None:

            self.renderer.shutdown()

            self.renderer = None

        # =====================================================
        # Timer
        # =====================================================

        self.timer = None

        # =====================================================
        # Window
        # =====================================================

        if self.window is not None:

            self.window.close()

            self.window = None

        # =====================================================
        # Input
        # =====================================================

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