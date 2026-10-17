# PyOpenGL checks glGetError after every call by default,
# a round trip per call that costs milliseconds a frame. GL
# errors still reach the log through the driver's debug
# output (graphics/gl_debug.py). Must be set before
# OpenGL.GL is first imported.
import OpenGL

OpenGL.ERROR_CHECKING = False

import math
import os
import time

from datetime import datetime, timezone

import numpy as np

from imgui_bundle import imgui

from core.assertions import engine_assert
from core.cprofile_capture import CProfileCapture
from core.disk_cache import DiskCache, disk_cache, set_disk_cache
from core.events import (
    EventDispatcher,
    MouseButtonPressedEvent,
    MouseButtonReleasedEvent,
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
    BodyComponent,
    CameraComponent,
    ClockComponent,
    OrbitComponent,
    StarComponent,
    CameraControllerComponent,
    DirectionalLightComponent,
    HierarchyComponent,
    NameComponent,
    TransformComponent
)
from ecs.entity import Entity

from game.chase_camera import ChaseCamera
from game.flight import FlightContext, frame_body, go_to_orbit, go_to_surface, read_flight
from game.hud import FlightHud
from game.computer_panel import ComputerPanel
from game.maneuver_panel import ManeuverPanel
from game.programme_panel import ProgrammePanel
from game.vessels_panel import VesselsPanel
from game.science_panel import SciencePanel
from game.map_view import MapView
from game.part_editor import EditorActions, PartEditor
from game.pause_menu import Destination, MenuActions, PauseMenu
from game.quality import DEFAULT_QUALITY, apply_quality, current_quality
from game.time_warp import TimeWarp

from graphics.framebuffer import Framebuffer
from graphics.gpu_timer import GpuTimer
from graphics.material import Material
from graphics.render_command import RenderCommand
from graphics.renderer import Renderer
from graphics.shader import Shader

from math3d import quaternion
from math3d.transform import Transform

from planet import solar
from planet.bodies import (
    ProfileError,
    components_for,
    load_presets,
    preset_groups,
    star_color,
    sun_intensity,
    solar_system,
)
from planet.orbits import DAY, seconds_since_j2000
from planet.spawn import find_spawn
from planet.terrain import Terrain

from resources.resources import Resources

from controllers.comms import CommsController
from controllers.docking import DockingController
from controllers.editor import EditorController
from controllers.fleet import FleetController
from controllers.maneuvers import ManeuverController
from controllers.navigation import NavigationController
from controllers.programme import ProgrammeController
from controllers.science import ScienceController

from systems.camera_controller_system import (
    CameraControllerSystem
)
from systems.planet_system import PlanetSystem, planet_extent, terrain_settings_for
from systems.orbit_system import OrbitSystem, body_states
from systems.render_system import RenderSystem
from systems.transform_system import TransformSystem
from systems.comms_system import CommsSystem, Link
from systems.programme_system import ProgrammeSystem
from systems.science_system import ScienceSystem
from systems.vessel_system import LaunchSite, OrbitStart, VesselSystem

from programme import fleet
from programme.career import CAREER_PATH, Career, load_campaign

from science.record import PROGRESS_PATH, SANDBOX_PROGRESS_PATH, ScienceRecord
from science.sites import load_sites

from vessel.parts import load_design, load_parts

from scene.scene import Scene

from ui.debug_panel import DebugContext, DebugPanel
from ui.imgui_layer import ImGuiLayer
from ui.panels import PanelRegistry


class Application(
    EditorController,
    FleetController,
    DockingController,
    CommsController,
    ScienceController,
    ProgrammeController,
    NavigationController,
    ManeuverController
):

    # =====================================================
    # Timing
    # =====================================================
    #
    # Vessel physics steps in simulated time, the clock's
    # (VesselSystem: fixed 0.02 s steps, time warp
    # included); everything else once per rendered frame.

    # How often to check shader files for edits.
    SHADER_POLL_INTERVAL = 0.5

    # Main-thread time per frame for job completions.
    JOB_BUDGET_SECONDS = 0.002

    # =====================================================
    # Construction
    # =====================================================

    def __init__(self):

        # A sandbox (initialize): no career.
        self.sandbox = False

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
        self.camera_controller_system: (
            CameraControllerSystem | None
        ) = None

        # Background work (mesh generation, loading);
        # results are handed back in update().
        self.jobs: JobSystem | None = None

        self.planet_system: PlanetSystem | None = None
        self.orbit_system: OrbitSystem | None = None
        self.vessel_system: VesselSystem | None = None

        # -------------------------------------------------
        # UI
        # -------------------------------------------------

        self.imgui_layer: ImGuiLayer | None = None

        self.hud = FlightHud()
        self.pause_menu: PauseMenu | None = None

        # Developer tools (F3): stats, render settings,
        # profiler.
        self.dev_panels = PanelRegistry()
        self.debug_panel: DebugPanel | None = None
        self.dev_tools = False

        # -------------------------------------------------
        # Game
        # -------------------------------------------------

        self.time_warp = TimeWarp()

        # The vessel every journey starts with
        # (assets/vessels/), and where it started.
        self.vessel_design = None
        self.body_id: str | None = None

        # Following the vessel, or flying free (V); the map
        # (M) over either.
        self.chase_camera = ChaseCamera()
        self.free_camera = False
        self.map_view = MapView()

        # Moons' orbits for the map: key -> (hour, points).
        self._moon_orbits: dict = {}

        # Last frame's parachute state (for the notice).
        self._parachute = ""

        # The departure window to the map's target: (origin,
        # target, clock time found, transfer.Window).
        self._window = None

        # Course correction toward the target, round the Sun:
        # the arrival aimed at ((target, clock time) or None),
        # when it was last found (s, real time), and the burn
        # left (m/s, or None).
        self._arrival = None
        self._arrival_found = -1.0e9
        self.maneuver_dv: float | None = None
        self._correction: np.ndarray | None = None

        # The map's maneuver planning; warping to the node:
        # until this clock time (or None).
        self.maneuver_panel = ManeuverPanel()
        self._warp_until: float | None = None

        # The science site the vessel was last in (its id).
        self._site: str | None = None

        # Other vessels' orbits on the map: entity -> (minute,
        # points).
        self._other_orbits: dict = {}

        # Camera speed for the HUD (smoothed).
        self._last_camera_position: np.ndarray | None = None
        self._speed = 0.0

        self._quit_requested = False

        # -------------------------------------------------
        # Resources
        # -------------------------------------------------

        self.resources: Resources | None = None

        self.handles: dict[str, Handle] = {}

        # -------------------------------------------------
        # Scene
        # -------------------------------------------------

        self.scene: Scene | None = None

    # =====================================================
    # Initialization
    # =====================================================

    def initialize(
        self,
        body: str | None = None,
        dev: bool = False,
        sandbox: bool = False
    ):
        """
        body: where to start (a profile id in assets/bodies/;
            Earth by default; a career always starts at home).
        dev: start with the developer tools open (F3).
        sandbox: no funds, missions or calendar to keep: launch
            anything free, travel anywhere; its own saves.
        """

        self.sandbox = sandbox

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
            "CY300 Spaceflight"
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
            "Flight"
        )

        # -------------------------------------------------
        # Systems
        # -------------------------------------------------

        self.transform_system = TransformSystem()
        self.orbit_system = OrbitSystem()
        self.camera_controller_system = CameraControllerSystem()

        self.jobs = JobSystem()

        # Terrain chunks kept on disk between runs (core/disk_cache.py), oldest
        # pruned in the background.
        if not os.environ.get("ENGINE_NO_DISK_CACHE"):

            cache = DiskCache()

            set_disk_cache(cache)

            self.jobs.submit(cache.prune, name="disk cache prune")

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

        # The sun's size follows each body's distance from it.
        self.render_system.star_provider = self.orbit_system.star_distance
        self.render_system.star_locator = self.orbit_system.star_position
        self.render_system.scatter_provider = lambda: self.planet_system.scatter.frame

        self.planet_system = PlanetSystem(
            self.resources,
            self.jobs,
            self.handles["planet_material"]
        )

        # Body profiles (assets/bodies/*.json): every place
        # to travel to.
        self.body_presets = self._load_body_presets()

        self.pause_menu = PauseMenu({
            group: [Destination(profile.id, profile.name, profile.description) for profile in members]
            for group, members in preset_groups(self.body_presets).items()
        })

        self.debug_panel = DebugPanel(
            self.dev_panels
        )

        self.dev_tools = dev

        # Rocket parts and the starting vessel.
        self.vessel_system = VesselSystem(
            self.resources,
            self.handles["lit"]
        )

        self.parts = load_parts()

        self.vessel_design = load_design("pathfinder", self.parts)

        # Science: the sites, the player's record (saved), the
        # panel (K).
        science_path = SANDBOX_PROGRESS_PATH if sandbox else PROGRESS_PATH

        self.science_system = ScienceSystem(load_sites(), ScienceRecord.load(science_path), science_path)
        self.science_panel = SciencePanel()

        # The link home (systems/comms_system.py), this frame;
        # the flight computer's panel (C).
        self.comms_system = CommsSystem()
        self.link = Link(False)
        self.computer_panel = ComputerPanel()

        # The space programme: funds, missions (saved); its
        # panel (P).
        campaign = load_campaign()

        if sandbox:
            self.programme_system = ProgrammeSystem(campaign, Career(funds=0.0), None, sandbox=True)
        else:
            self.programme_system = ProgrammeSystem(campaign, Career.load(campaign, CAREER_PATH), CAREER_PATH)

        self.programme_panel = ProgrammePanel()

        # Vessels left in flight (programme/fleet.py): saved,
        # brought into every scene; the list (L).
        self.fleet_path = fleet.SANDBOX_FLEET_PATH if sandbox else fleet.FLEET_PATH
        self._fleet_records = fleet.load(self.fleet_path)
        self._fleet_restored = False
        self._fleet_saved = 0.0
        self.vessels_panel = VesselsPanel()

        # The vessel flown before the part editor opened (to
        # go back to on Cancel).
        self._flying_before_edit = None

        # Docking automatically with this vessel (B), or None;
        # the approach as last guided (for the HUD).
        self._docking = None
        self._docking_command = None

        # A probe's controls as the pilot has sent them (the
        # vessel gets them a light time later): (vessel id,
        # throttle holder, Controls, last sent snapshot).
        self._sent = None

        self.part_editor = PartEditor(self.parts)

        # The design to put back if the editor is cancelled,
        # and the height the view was last fitted to.
        self._design_before_edit = None
        self._edited_height = 0.0

        apply_quality(self.render_system.settings, DEFAULT_QUALITY)

        if body and body != self.START_BODY and not sandbox:
            Logger.warning("[Application] A career starts at home (%s); --sandbox to start at %s.", self.START_BODY, body)

        self.travel_to((body or self.START_BODY) if sandbox else self.START_BODY)

        # Resolve world matrices before the first frame, so
        # everything has valid data even if update() is
        # skipped.
        self.transform_system.update(
            self.scene
        )

        self._initialized = True

        Logger.info(
            "[Application] Initialization complete."
        )

    # =====================================================
    # Travel
    # =====================================================

    def travel_to(
        self,
        body_id: str,
        keep_flown: bool = True
    ):
        """
        Start over at a body: a fresh scene with it, its
        system, the sun and the camera (on its surface, or
        in space over a giant or a small body). The vessels in
        flight come along (the one flown now too, unless
        keep_flown is False: a revert); a new one stands on
        the pad.
        """

        body_id = body_id.partition("@")[0]

        engine_assert(
            body_id in self.body_presets,
            f"Unknown body '{body_id}' (assets/bodies/)."
        )

        if self._fleet_restored:
            self._fleet_records = fleet.snapshot(self.scene, self.vessel_system, self.science_system, self.orbit_system.time, keep_flown)

        self.scene.clear()

        self.scene.name = self.body_presets[body_id].name

        self._populate_body_scene(self.scene, body_id)

        self.time_warp.stop()

        self.body_id = body_id

        self.chase_camera = ChaseCamera()
        self.free_camera = False
        self.map_view = MapView()
        self._moon_orbits = {}
        self._parachute = ""
        self._window = None
        self._arrival = None
        self._warp_until = None
        self._site = None
        self._sent = None
        self._other_orbits = {}
        self._docking = None
        self.hud.notices.clear()

        self._last_camera_position = None
        self._speed = 0.0

        # The fleet, brought up to now (at the start, flying
        # the one flown when it was saved).
        for notice in fleet.restore(
            self._fleet_records,
            self.scene,
            self.vessel_system,
            self.orbit_system,
            self.parts,
            self.orbit_system.time,
            activate=not self._fleet_restored
        ):
            self.hud.notify(notice)

        self._fleet_restored = True

        Logger.info("[Application] Traveled to %s.", self.scene.name)


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

        lit = handles["lit"]

        # -------------------------------------------------
        # Materials
        # -------------------------------------------------

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

            # A broken profile only loses that body; the start
            # needs Earth though.
            Logger.error("[Application] Body profiles: %s", error)

            presets = {}

        Logger.info(
            "[Application] %d body profile(s): %s.",
            len(presets),
            ", ".join(profile.name for profile in presets.values())
        )

        engine_assert(
            self.START_BODY in presets,
            f"assets/bodies/{self.START_BODY}.json is missing or invalid."
        )

        return presets

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

    # Where the game starts unless told otherwise.
    START_BODY = "earth"

    # Bodies smaller than this (km) start with the camera in
    # space, the whole body in view.
    SMALL_BODY_KM = 400.0

    # The date new planets start on (then moved on to the
    # morning where the camera is).
    START_DATE = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)

    def _add_system_members(
        self,
        body_id: str
    ):
        """
        Add every other body of the solar system (OrbitSystem
        places them; far ones stay dormant: PlanetSystem).
        """

        for member_id in solar_system(self.body_presets, body_id):

            member = self.body_presets[member_id]

            parts = components_for(member)

            self._create_entity(
                member.name,
                Transform(),
                *parts.all()
            )

    def _time_of_day(
        self,
        entity: Entity,
        direction: np.ndarray,
        start: float,
        hour: float
    ) -> float:
        """
        The first time from `start` at which it is `hour`
        (local solar time) at a direction on a body.
        """

        scene = self.scene

        bodies = list(scene.registry.view_with(TransformComponent, OrbitComponent))

        body = scene.try_get_component(entity, BodyComponent)

        solar_day = abs(body.solar_day_hours) * 3_600.0 if body is not None and body.solar_day_hours else DAY

        # Venus and Uranus turn backwards about their north
        # poles (planet/solar.py).
        retrograde = body is not None and body.rotation_hours < 0.0

        moment = start

        # The sun moves on in the sky while the body turns:
        # the first step goes forward to that hour, then
        # small corrections.
        for step in range(4):

            position, frame = body_states(scene, bodies, moment)[entity]

            sun = frame.T @ (-position / np.linalg.norm(position))

            now, _, _ = solar.solar_time(direction, sun, retrograde)

            error = (hour - now) % 24.0 if step == 0 else (hour - now + 12.0) % 24.0 - 12.0

            moment += error / 24.0 * solar_day

        return moment

    def _populate_body_scene(
        self,
        scene: Scene,
        body_id: str
    ):
        """
        Fill `scene` with a planet built from a body profile
        (assets/bodies/<body_id>.json), a sun, and a camera.

        Solid bodies: the planet is turned so a scenic spot
        (planet/spawn.py) is on top, and placed so that spot
        sits at the world origin (+Y up there); the camera
        stands above it.
        Gas giants: the camera starts in orbit.

        Bodies are as measured, wherever their maps are
        downloaded (tools/fetch_maps.py).
        """

        engine_assert(
            scene is self.scene,
            "The planet builder fills the application's scene."
        )

        profile = self.body_presets[body_id]

        parts = components_for(profile)

        planet = parts.planet

        terrain_settings = terrain_settings_for(planet)

        terrain = Terrain(terrain_settings)

        # -------------------------------------------------
        # Where to stand
        # -------------------------------------------------

        up = np.array([0.0, 1.0, 0.0])

        # Giants have no ground to stand on, and small bodies
        # are best seen whole: the camera starts in space.
        from_space = not profile.has_solid_surface or profile.radius_km < self.SMALL_BODY_KM

        if not from_space:

            spawn = find_spawn(terrain)

            spawn_direction = spawn.direction
            view_direction = spawn.view_direction

            ground_height = (
                max(spawn.elevation, 0.0)
                if terrain_settings.has_liquid
                else spawn.elevation
            )

            # Relief stands on the body's shape.
            ground_height += float(terrain.base_height(spawn_direction[None, :])[0])

        elif profile.has_solid_surface:

            # Over the equator, side-on to an irregular body's
            # long axis (z).
            spawn_direction = np.array([np.cos(np.radians(20.0)), np.sin(np.radians(20.0)), 0.0])
            view_direction = np.array([0.0, 0.0, 1.0])
            ground_height = 0.0

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

        if not from_space:

            camera_position = (0.0, 500.0, 0.0)

            camera_orientation = quaternion.multiply(
                quaternion.look_rotation(view, up),
                quaternion.from_euler((-6.0, 0.0, 0.0))
            )

        else:

            # Back far enough for the whole planet to fit.
            distance = planet_extent(planet) / math.sin(math.radians(fov) * 0.5) * 1.25

            camera_position = (0.0, distance - ground, 0.0)

            # Giants: north up the screen; small bodies: their
            # spin axis up, so a long body lies across it.
            camera_orientation = quaternion.look_rotation(
                np.array([0.0, -1.0, 0.0]),
                np.array([0.0, 0.0, -1.0])
                if not profile.has_solid_surface
                else quaternion.rotate_vector(orientation, np.array([0.0, 1.0, 0.0]))
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
        # Sun and clock (OrbitSystem moves everything)
        # -------------------------------------------------

        # The programme's calendar goes on (a revert does not
        # turn it back): from the latest it reached.
        clock = ClockComponent(time=max(seconds_since_j2000(self.START_DATE), self.programme_system.career.clock), rate=1.0)

        self._create_entity(
            "Sun",
            Transform(),
            # The star's color from its temperature; intensity
            # compressed from the real range (see
            # planet/bodies.py sun_intensity).
            DirectionalLightComponent(
                color=star_color(profile.star_temperature_k),
                intensity=sun_intensity(profile.sunlight)
            ),
            StarComponent(
                luminosity=profile.star_luminosity,
                temperature=profile.star_temperature_k,
                radius=profile.star_radius * 6.957e8
            ),
            clock
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

        # Its moons, or its planet and sibling moons, in the
        # sky.
        self._add_system_members(body_id)

        # Their survey masks (mapped ground: science/survey.py);
        # the body launched from is home, mapped already.
        self.science_system.register_bodies(scene, body_id)

        # Mid-morning where the camera stands, for a new game
        # (or a sandbox); a career's calendar goes on as it is
        # (a revert there must not skip ahead: a solar day on
        # Venus is 117 Earth days).
        if self.sandbox or self.programme_system.career.clock <= seconds_since_j2000(self.START_DATE):
            clock.time = self._time_of_day(planet_entity, spawn_direction, clock.time, 10.5)

        self.orbit_system.anchor_to(scene, planet_entity)

        # -------------------------------------------------
        # The vessel: on the ground where the camera looks,
        # or in a low orbit over a giant or a small body
        # -------------------------------------------------

        if from_space:

            start = OrbitStart(
                direction=spawn_direction,
                radius=max(planet.radius * 1.25, planet_extent(planet) * 1.3)
            )

        else:

            start = LaunchSite(
                direction=spawn_direction,
                radius=ground
            )

        self.vessel_system.spawn(
            scene,
            self.vessel_design,
            planet_entity,
            start,
            self.orbit_system
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

            while not self.window.should_close() and not self._quit_requested:

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
        """The HUD (F2); the developer tools have their own switch (F3)."""

        self.hud.visible = visible

    def update(self):

        engine_assert(
            self.timer is not None
            and self.window is not None
            and self.scene is not None,
            "Application is not initialized."
        )

        dt = self.timer.delta_time

        # -------------------------------------------------
        # Game Keys
        # -------------------------------------------------

        self._handle_keys()

        building = self.part_editor.open

        paused = self.pause_menu.open or building

        self._update_maneuver()

        self._update_comms(paused)

        self._pilot(dt, paused)

        self._update_docking(paused)

        # The clock runs at the warp's rate (no faster than
        # the vessel allows); the menu stops it.
        self.time_warp.paused = paused

        if self.time_warp.limit(
            min(
                self.vessel_system.max_warp(self.scene, self.orbit_system, self.planet_system),
                self._computer_warp(),
                # (Docking: in real time.)
                1.0 if self._docking is not None else math.inf
            )
        ):
            Logger.info("[Application] Time warp held to %s.", self.time_warp.label())

        clock = FlightContext.find(self.scene).clock()

        self.time_warp.apply(clock)

        profiler = self.profiler

        # -------------------------------------------------
        # Shader Hot Reload (developer tools)
        # -------------------------------------------------

        self._shader_poll_timer += dt

        if self.dev_tools and self._shader_poll_timer >= self.SHADER_POLL_INTERVAL:

            self._shader_poll_timer = 0.0

            with profiler.scope("Shader poll"):

                self.resources.reload_changed_shaders()

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
        # Orbits: the clock, the bodies and the sun
        # -------------------------------------------------

        with profiler.scope("Orbits"):

            before = clock.time if clock is not None else 0.0

            self.orbit_system.update(
                self.scene,
                dt,
                self.render_system.camera_position(self.scene)
            )

            simulated = (clock.time - before) if clock is not None else 0.0

        # -------------------------------------------------
        # The vessel, over the same simulated time
        # -------------------------------------------------

        with profiler.scope("Vessel"):

            events = self.vessel_system.update(
                self.scene,
                simulated,
                self.orbit_system,
                self.planet_system
            )

            if "destroyed" in events or "landed" in events:
                self.time_warp.stop()

            self._notify(events)

        with profiler.scope("Science"):

            self._update_science(events, simulated)

            self.programme_system.update(self, events, simulated)

            for event in events:

                kind, _, name = event.partition(":")

                if kind == "lost":
                    self.hud.notify(f"{name} came down and was lost")
                elif kind == "landed" and name:
                    self.hud.notify(f"{name} landed under its parachute")

            now = time.monotonic()

            if now - self._fleet_saved > 30.0:

                self._fleet_saved = now

                self.save_fleet()

            for news in self.programme_system.news:
                self.hud.notify(news)

            self.programme_system.news.clear()

            self._take_recovered()

        self.hud.tick(dt)

        with profiler.scope("Camera"):

            if building:

                self._update_editor()

                self._chase(paused=self.pause_menu.open)

            elif self.map_view.open and self.vessel_system.find(self.scene) is not None:

                self._map_camera(paused)

            elif self.free_camera or self.vessel_system.find(self.scene) is None:

                # Planet-mode cameras fly relative to the
                # ground under them.
                self.planet_system.update_camera_ground(
                    self.scene
                )

                # Free flight while the right mouse button is
                # held; nothing moves while paused.

                flying = self._looking and not paused

                self.camera_controller_system.update(
                    self.scene,
                    dt,
                    look_enabled=flying,
                    move_enabled=flying
                )

            else:

                self._chase(paused)

            self._track_speed(dt)

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

        with profiler.scope("Planet"):

            # Rocks, trees and grass (Render settings).
            self.planet_system.scatter.enabled = self.render_system.settings.scatter

            self.planet_system.update(
                self.scene,
                self.render_system.camera_position(self.scene)
            )

    # -----------------------------------------------------
    # Communications and the Flight Computer
    # -----------------------------------------------------


    def _chase(
        self,
        paused: bool
    ):
        """The camera round the vessel (right mouse: swing round it; scroll: closer / farther)."""

        found = self.vessel_system.find(self.scene)

        context = FlightContext.find(self.scene)

        if found is None or context.camera is None:
            return

        entity, component = found

        if self._looking and not paused:

            scroll = 0.0 if self.imgui_layer.wants_mouse else Input.scroll_delta()[1]

            self.chase_camera.orbit(Input.mouse_delta(), scroll)

        elif not self.imgui_layer.wants_mouse and not paused:

            self.chase_camera.orbit((0.0, 0.0), Input.scroll_delta()[1])

        up = self.vessel_system.world_up(self.scene, self.orbit_system)

        if up is None:
            return

        transform = self.scene.get_component(entity, TransformComponent).transform

        axis_world = quaternion.rotate_vector(transform.orientation, np.array([0.0, 1.0, 0.0]))

        target = np.asarray(transform.position, dtype=np.float64) + axis_world * component.vessel.center_of_mass

        body = self.scene.try_get_component(component.body, TransformComponent) if component.body is not None else None

        if body is not None:

            spin_axis = np.asarray(body.world_matrix, dtype=np.float64)[:3, 1]

        else:

            # Around the Sun: the orbit's pole.
            pole = np.cross(component.state.position, component.state.velocity)

            spin_axis = self.orbit_system.inertial_to_world @ (pole / max(float(np.linalg.norm(pole)), 1e-9))

        # Close to the ground: keep the camera out of it.
        height = float(np.dot(target - np.asarray(transform.position, dtype=np.float64), up))

        floor = 2.0 - height if component.state.landed else None

        self.chase_camera.place(
            context.camera_transform(),
            target,
            up,
            spin_axis,
            floor
        )

    def _notify(
        self,
        events: list[str]
    ):
        """Notices for what happened to the vessel."""

        readout = self.vessel_system.readout(self.scene, self.orbit_system, self.planet_system)

        if readout is None:
            return

        if "soi" in events:
            self.hud.notify(f"Now in {readout.body}'s sphere of influence")

        if "node done" in events:

            found = self.vessel_system.find(self.scene)

            if found is not None:
                found[1].vessel.throttle = 0.0

            self.hud.notify("Maneuver done: throttle cut")

        if "landed" in events:
            self.hud.notify(f"Landed on {readout.body}")

        if readout.parachute == "open" and self._parachute != "open":
            self.hud.notify("Parachute opening")

        self._parachute = readout.parachute


    # Colors for the navigation panel.
    GOOD = (0.5, 1.0, 0.6, 1.0)
    NOTE = (0.75, 0.75, 0.8, 1.0)


    # -----------------------------------------------------
    # The Part Editor
    # -----------------------------------------------------


    def revert(self):
        """Back to the launch: the same body, from the start (the other vessels in flight stay where they are)."""

        if self.body_id is not None:
            self.travel_to(self.body_id, keep_flown=False)

    # -----------------------------------------------------
    # Keys
    # -----------------------------------------------------

    def _handle_keys(self):

        # Keys typed into a developer window are its own.
        if self.imgui_layer.wants_keyboard and not self._looking:
            return

        menu = self.pause_menu

        if self.part_editor.open:

            # Esc leaves the editor as Cancel does.
            if Input.is_key_pressed(Key.ESCAPE):

                self.part_editor.open = False

                self._cancel_editing()

            return

        if Input.is_key_pressed(Key.ESCAPE):

            if menu.open:
                menu.back()
            else:
                menu.show()
                self._set_looking(False)

        if Input.is_key_pressed(Key.F2):
            self.hud.visible = not self.hud.visible

        if Input.is_key_pressed(Key.F3):
            self.dev_tools = not self.dev_tools

        if self.dev_tools and Input.is_key_pressed(Key.F5):

            self.resources.reload_changed_shaders(
                force=True
            )

        if menu.open:
            return

        if Input.is_key_pressed(Key.PERIOD):
            self.time_warp.faster()

        if Input.is_key_pressed(Key.COMMA):
            self.time_warp.slower()
            self._warp_until = None

        if Input.is_key_pressed(Key.SLASH):
            self.time_warp.stop()
            self._warp_until = None

        if Input.is_key_pressed(Key.V) and self.vessel_system.find(self.scene) is not None:
            self.free_camera = not self.free_camera

        if Input.is_key_pressed(Key.M) and self.vessel_system.find(self.scene) is not None:
            self.map_view.toggle()

        if Input.is_key_pressed(Key.K) and self.vessel_system.find(self.scene) is not None:
            self.science_panel.toggle()

        if Input.is_key_pressed(Key.R) and self.vessel_system.find(self.scene) is not None:
            self._run_science()

        if Input.is_key_pressed(Key.C) and self.vessel_system.find(self.scene) is not None:
            self.computer_panel.toggle()

        if Input.is_key_pressed(Key.P):
            self.programme_panel.toggle()

        if Input.is_key_pressed(Key.L):
            self.vessels_panel.toggle()

        if Input.is_key_pressed(Key.B) and self.vessel_system.find(self.scene) is not None:
            self._toggle_docking()

        if Input.is_key_pressed(Key.LEFT_BRACKET):
            self._cycle_vessel(-1)

        if Input.is_key_pressed(Key.RIGHT_BRACKET):
            self._cycle_vessel(1)

    def _track_speed(
        self,
        delta_time: float
    ):
        """The camera's speed (smoothed; teleports settle within a second)."""

        context = FlightContext.find(self.scene)

        if context.camera is None or delta_time <= 0.0:
            return

        position = np.asarray(context.camera_transform().position, dtype=np.float64)

        if self._last_camera_position is not None:

            speed = float(np.linalg.norm(position - self._last_camera_position)) / delta_time

            self._speed += (speed - self._speed) * min(1.0, delta_time * 6.0)

        self._last_camera_position = position

    # -----------------------------------------------------
    # Menu Actions
    # -----------------------------------------------------

    def _menu_actions(
        self,
        context: FlightContext
    ) -> MenuActions:

        def travel(body_id):
            self.pause_menu.close()
            self.travel_to(body_id)

        def camera_move(move):

            def run():

                if context.body is not None and context.camera is not None:

                    # (Moving the camera means flying it.)
                    self.free_camera = True

                    move()

                    self.pause_menu.close()

                    # (A jump, not a speed.)
                    self._last_camera_position = None

            return run

        def to_surface():

            spawn = self.planet_system.spawn_point(context.body)

            if spawn is not None:
                go_to_surface(context, spawn, self.planet_system.ground)

        def revert():
            self.pause_menu.close()
            self.revert()

        def toggle_camera():
            self.free_camera = not self.free_camera
            self.pause_menu.close()

        def build():
            self.pause_menu.close()
            self.start_editing()

        return MenuActions(
            resume=self.pause_menu.close,
            revert=revert,
            build=build,
            toggle_camera=toggle_camera,
            travel=travel,
            to_surface=camera_move(to_surface),
            to_orbit=camera_move(lambda: go_to_orbit(context, 400_000.0)),
            whole_body=camera_move(
                lambda: frame_body(context, self.planet_system.extent(context.body) or context.planet.radius)
            ),
            set_quality=lambda name: apply_quality(self.render_system.settings, name),
            quit=self.request_quit
        )

    def request_quit(self):

        self._quit_requested = True

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
        # Scene, over the whole window
        # -------------------------------------------------

        Framebuffer.bind_default(width, height)

        self.render_system.render(
            self.scene,
            width,
            height,
            extra_items=self.planet_system.draw_items
        )

        # -------------------------------------------------
        # HUD, menu and developer tools (over the image)
        # -------------------------------------------------

        profiler = self.profiler

        with profiler.scope("UI"):

            with profiler.scope("Build"):

                self.imgui_layer.begin_frame()

                self._draw_ui()

            with profiler.scope("Draw", gpu=True):

                self.imgui_layer.end_frame()

    def _draw_ui(self):

        if self.part_editor.open:

            self.part_editor.funds = self.programme_system.career.funds

            self.part_editor.draw(
                EditorActions(
                    launch=self._launch_design,
                    cancel=self._cancel_editing
                )
            )

            return

        found = self.vessel_system.find(self.scene)

        # Where the player is: the vessel, else the camera.
        focus = (
            np.asarray(self.scene.get_component(found[0], TransformComponent).transform.position, dtype=np.float64)
            if found is not None
            else None
        )

        context = FlightContext.find(self.scene, focus)

        planets = self.planet_system.stats

        vessel = self.vessel_system.readout(self.scene, self.orbit_system, self.planet_system)

        # (The vessel's own ground lookup, reused.)
        accuracy = self.vessel_system.ground_accuracy if found is not None else 0.0

        def ground(body, direction):
            return self.planet_system.ground(body, direction, accuracy)

        self.hud.draw(
            read_flight(context, ground, focus),
            vessel.surface_speed if vessel is not None and not self.free_camera else self._speed,
            self.time_warp,
            loading=bool(planets.chunks_building or planets.chunks_queued),
            fps=self.timer.fps if self.dev_tools else None,
            vessel=vessel,
            maneuver=self._node_readout(found[1]) if found is not None else None,
            braking=self.vessel_system.braking(self.scene, self.orbit_system, self.planet_system),
            link=self._link_line(found[1]) if found is not None else None,
            docking=self._docking_line(),
            navball=self.vessel_system.navball_frame(self.scene, self.orbit_system, self.planet_system)
        )

        self._draw_map()

        self._draw_science()

        self._draw_computer()

        self._draw_programme()

        self._draw_vessels()

        self.pause_menu.draw(
            self._menu_actions(context),
            context.name() if context.body is not None else "",
            current_quality(self.render_system.settings),
            free_camera=self.free_camera,
            vessel=found is not None,
            travel=self.sandbox
        )

        if not self.dev_tools:
            return

        if imgui.begin_main_menu_bar():

            if imgui.begin_menu("Developer"):

                self.dev_panels.draw_menu_items()

                imgui.separator()

                if imgui.menu_item("Reload shaders", "F5", False)[0]:
                    self.resources.reload_changed_shaders(force=True)

                if imgui.menu_item("Hide developer tools", "F3", False)[0]:
                    self.dev_tools = False

                imgui.end_menu()

            imgui.end_main_menu_bar()

        profiler = self.profiler

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
    # menu and developer tools. Clicks that land on them
    # never start looking.

    def on_mouse_button_pressed(
        self,
        event
    ):

        if (
            event.button == MouseButton.RIGHT
            and not self.imgui_layer.wants_mouse
            and not self.pause_menu.open
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

        science = getattr(self, "science_system", None)

        if science is not None:
            science.save()

        programme = getattr(self, "programme_system", None)

        if programme is not None:
            programme.save()

        if getattr(self, "_fleet_restored", False) and self.scene is not None:
            self.save_fleet()

        # =================================================
        # UI
        # =================================================

        if self.imgui_layer is not None:

            self.imgui_layer.shutdown()

            self.imgui_layer = None

        self.debug_panel = None
        self.pause_menu = None

        # =================================================
        # Scene
        # =================================================

        if self.scene is not None:

            self.scene.shutdown()

            self.scene = None

        # =================================================
        # Systems
        # =================================================

        # Chunk meshes live in the renderer's geometry pool.
        if self.planet_system is not None:

            self.planet_system.shutdown()

            self.planet_system = None

        if self.render_system is not None:

            self.render_system.shutdown()

            self.render_system = None

        self.camera_controller_system = None
        self.transform_system = None
        self.vessel_system = None

        if self.jobs is not None:

            self.jobs.shutdown()

            self.jobs = None

        cache = disk_cache()

        if cache is not None:

            Logger.info("[Cache] %d hit(s), %d miss(es) this run.", cache.hits, cache.misses)

            set_disk_cache(None)

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
