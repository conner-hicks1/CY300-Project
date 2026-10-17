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
from pathlib import Path
from types import SimpleNamespace

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
    PlanetComponent,
    TransformComponent,
    VesselComponent
)
from ecs.entity import Entity

from game.chase_camera import ChaseCamera
from game.flight import FlightContext, frame_body, go_to_orbit, go_to_surface, read_flight
from game.hud import FlightHud
from game.computer_context import GameFlightContext, hold_sink_rate
from game.computer_panel import ComputerInfo, ComputerPanel
from game.maneuver_panel import ManeuverPanel, NodeInfo
from game.programme_panel import ProgrammeInfo, ProgrammePanel, deadline_text, reward_text
from game.vessels_panel import VesselRow, VesselsInfo, VesselsPanel
from game.science_panel import ScienceInfo, SciencePanel
from game.map_view import MapView, ScreenProjection
from game.part_editor import EditorActions, LaunchBody, PartEditor
from game.pause_menu import Destination, MenuActions, PauseMenu
from game.quality import DEFAULT_QUALITY, apply_quality, current_quality
from game.time_warp import TimeWarp
from game.vessel_controls import read_pilot, release

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
from planet.orbits import DAY, date_of, seconds_since_j2000
from planet.spawn import find_spawn
from planet.terrain import Terrain

from resources.resources import Resources

from systems.camera_controller_system import (
    CameraControllerSystem
)
from systems.planet_system import PlanetSystem, planet_extent, terrain_settings_for
from systems.orbit_system import OrbitSystem, body_states
from systems.render_system import RenderSystem
from systems.transform_system import TransformSystem
from systems.comms_system import CommsSystem, Link
from systems.comms_system import crew as crew_of
from systems.programme_system import ProgrammeSystem, move_propellant
from systems.science_system import ScienceSystem
from systems.vessel_system import LaunchSite, OrbitStart, Suggestion, VesselSystem

from programme import fleet
from programme.career import CAREER_PATH, RECOVERY, Career, load_campaign, today_of

from science.record import PROGRESS_PATH, SANDBOX_PROGRESS_PATH, ScienceRecord
from science.survey import direction_of as survey_direction
from science.sites import load_sites

from vessel import kepler
from vessel import docking, maneuver
from vessel.physics import Controls
from vessel.sequence import FlightComputer, save_program
from vessel.maneuver import ManeuverNode
from vessel.transfer import closest_approach, correction, find_window, plan_transfer
from game import format as readable
from vessel.design import DesignDraft
from vessel.parts import load_design, load_parts

from scene.scene import Scene

from ui.debug_panel import DebugContext, DebugPanel
from ui.imgui_layer import ImGuiLayer
from ui.panels import PanelRegistry


class Application:

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

    # A recovered vessel stays this long (s, real) before it
    # is taken away.
    RECOVERY_PAUSE = 3.0

    def _take_recovered(self):
        """Recovered vessels go (their flight is over); the flown one's place: a fresh rocket on the pad."""

        programme = self.programme_system

        now = time.monotonic()

        waiting = []

        for entity, when in programme.recovered:

            component = self.scene.try_get_component(entity, VesselComponent)

            # Gone, or off again (it pays again to fly).
            if component is None or not component.state.landed or component.paid:
                continue

            if now - when < self.RECOVERY_PAUSE:

                waiting.append((entity, when))

                continue

            found = self.vessel_system.find(self.scene)

            if found is not None and found[0] == entity:

                # (Rebuild takes the unpaid one off the pad.)
                self.vessel_system.rebuild(self.scene, self.vessel_design, self.orbit_system)

                self.chase_camera = ChaseCamera()

                self.hud.notify(f"{self.vessel_design.name} on the pad")

            else:

                self.vessel_system.remove(self.scene, entity)

        programme.recovered = waiting

    def save_fleet(self):

        if not self._fleet_restored:
            return

        try:
            fleet.save(
                fleet.snapshot(self.scene, self.vessel_system, self.science_system, self.orbit_system.time),
                self.fleet_path
            )
        except OSError as error:
            Logger.error("[Application] Could not save the fleet: %s", error)

    def switch_vessel(
        self,
        entity
    ) -> bool:
        """Fly another vessel (the camera follows it)."""

        why = self.vessel_system.switch(self.scene, entity, self.orbit_system, self.planet_system)

        if why:

            self.hud.notify(why)

            return False

        found = self.vessel_system.find(self.scene)

        # (Not a target of itself.)
        if found is not None and self.map_view.target == ("vessel", found[0]):
            self.map_view.target = None

        self.chase_camera = ChaseCamera()
        self._sent = None
        self._site = None
        self._window = None
        self._arrival = None
        self._warp_until = None

        if found is not None:
            self.hud.notify(f"Flying {found[1].vessel.name}")

        return True

    # Auto-docking starts within this of the target (m).
    DOCKING_RANGE = 2_000.0

    def _toggle_docking(self):
        """B: dock with the targeted vessel automatically (again: stop)."""

        if self._docking is not None:

            self._docking = None

            found = self.vessel_system.find(self.scene)

            if found is not None:
                found[1].controls.translate = np.zeros(3)

            self.hud.notify("Auto-docking off")

            return

        target = self.map_view.target

        found = self.vessel_system.find(self.scene)

        if not (isinstance(target, tuple) and target and target[0] == "vessel") or found is None:

            self.hud.notify("Target a vessel to dock with first (Vessels: L)")

            return

        pair = self.vessel_system.docking_pair(self.scene, found[1], target[1])

        if pair is None:

            self.hud.notify("No free docking ports to join (one at an end of each stack)")

            return

        if found[1].vessel.rcs_thrust <= 0.0:

            self.hud.notify("No translation thrusters (RCS) aboard")

            return

        distance = float(np.linalg.norm(pair[1][0] - pair[3][0]))

        if distance > self.DOCKING_RANGE:

            self.hud.notify(f"Too far to dock: {readable.distance(distance)} (within {readable.distance(self.DOCKING_RANGE)})")

            return

        self._docking = target[1]

        self.hud.notify("Auto-docking: the nose turns, the thrusters bring it in")

    def _update_docking(
        self,
        paused: bool
    ):
        """Each frame: the approach to the targeted vessel's port (the HUD's line; auto-docking flies it); docked when they touch."""

        self._docking_command = None

        found = self.vessel_system.find(self.scene)

        target = self.map_view.target

        if found is None or not (isinstance(target, tuple) and target and target[0] == "vessel"):

            self._docking = None

            return

        entity, component = found

        pair = self.vessel_system.docking_pair(self.scene, component, target[1])

        if pair is None:

            self._docking = None

            return

        own, own_point, theirs, their_point, other = pair

        if float(np.linalg.norm(own_point[0] - their_point[0])) > 5.0 * self.DOCKING_RANGE:
            return

        command = docking.docking_command(
            own_point,
            component.state.velocity,
            component.state.orientation,
            own.end,
            their_point,
            other.state.velocity,
            component.vessel.rcs_thrust / component.vessel.mass
        )

        self._docking_command = command

        if docking.can_capture(own_point, component.state.velocity, their_point, other.state.velocity):

            name = other.vessel.name

            if not self.vessel_system.dock(self.scene, entity, own.end, target[1], theirs.end, self.orbit_system, own.copy, theirs.copy):

                self._docking = None

                self.hud.notify(docking.can_hang(component.vessel if theirs.end == "side" else other.vessel) or "Could not dock")

                return

            component.controls.translate = np.zeros(3)

            self._docking = None
            self.map_view.target = None

            self.hud.notify(f"Docked with {name}")

            return

        if self._docking != target[1] or paused:
            return

        controls = component.controls

        controls.sas = True
        controls.sas_mode = "maneuver"
        controls.maneuver = command.up

        component.vessel.throttle = 0.0

        # (Translating only once pointing near enough.)
        controls.translate = command.translate if command.misalignment < 10.0 else np.zeros(3)

    def undock(self):
        """Split the flown vessel at its last docking joint."""

        found = self.vessel_system.find(self.scene)

        if found is None:
            return

        other = self.vessel_system.undock(self.scene, found[0], self.orbit_system)

        if other is None:

            self.hud.notify("Nothing docked to undock")

            return

        name = self.scene.get_component(other, VesselComponent).vessel.name

        self.hud.notify(f"Undocked: {name} drifts away")

    def _cycle_vessel(
        self,
        step: int
    ):

        vessels = [entity for entity, component in self.vessel_system.vessels(self.scene) if not component.state.destroyed]

        if len(vessels) < 2:
            return

        found = self.vessel_system.find(self.scene)

        index = vessels.index(found[0]) if found is not None and found[0] in vessels else 0

        self.switch_vessel(vessels[(index + step) % len(vessels)])

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
    # The Vessel
    # -----------------------------------------------------

    def _pilot(
        self,
        dt: float,
        paused: bool
    ):
        """The pilot's keys (not while paused, typing, or flying the free camera)."""

        found = self.vessel_system.find(self.scene)

        if found is None:
            return

        _, component = found

        busy = (
            paused
            or (self.imgui_layer.wants_keyboard and not self._looking)
            or (self.free_camera and self._looking)
            or component.state.destroyed
        )

        if self.link.crewed:

            if busy:

                release(component.controls)

                return

            if read_pilot(component.vessel, component.controls, dt):
                self.stage_vessel(component)

            return

        # A probe: the keys go home's way to it, a light time
        # late; without a link, nowhere.
        sent = self._sent

        if sent is None or sent[0] != id(component.vessel):
            sent = self._sent = (id(component.vessel), SimpleNamespace(throttle=component.vessel.throttle), Controls(), None)

        _, throttle, controls, last = sent

        def snapshot():
            return (controls.pitch, controls.yaw, controls.roll, controls.sas, controls.sas_mode, throttle.throttle, tuple(controls.translate))

        # Nothing on the way: what is sent is what it has (and
        # what its computer set: not to be sent back over it).
        if last is None or not any(kind == "controls" for _, kind, _ in component.uplink):

            throttle.throttle = component.vessel.throttle

            for name in ("pitch", "yaw", "roll", "sas", "sas_mode"):
                setattr(controls, name, getattr(component.controls, name))

            controls.translate = np.array(component.controls.translate, dtype=np.float64)

            last = snapshot()

        if busy:
            release(controls)
            staged = False
        else:
            staged = read_pilot(throttle, controls, dt)

        now_sent = snapshot()

        if not self.link.connected:

            if now_sent != last or staged:
                self.hud.notify(f"No link ({self.link.reason}): the probe cannot hear you")

            # Forget what could not be sent.
            self._sent = (sent[0], throttle, controls, None)

            return

        if now_sent != last:
            self._uplink(component, "controls", now_sent)

        if staged:
            self._uplink(component, "stage", None)

        self._sent = (sent[0], throttle, controls, now_sent)

    def stage_vessel(
        self,
        component
    ):
        """Fire the next stage (the pilot, the computer), with a notice of what it did."""

        # The first: the launch is paid for (or not, short of
        # funds: then nothing fires).
        if not self.programme_system.pay_launch(component):

            for news in self.programme_system.news:
                self.hud.notify(news)

            self.programme_system.news.clear()

            return

        vessel = component.vessel

        parts = len(vessel.parts)

        stage = vessel.stage()

        if stage is None:
            return

        Logger.info("[Application] Stage %d.", stage)

        what = []

        if len(vessel.parts) < parts:
            what.append("separated")

        if any(state.ignited and state.stage == stage for state in vessel.parts):
            what.append("engines lit")

        if any(state.armed and state.stage == stage for state in vessel.parts):
            what.append("parachute armed")

        self.hud.notify(f"Stage {stage}: {', '.join(what) or 'fired'}")

    # -----------------------------------------------------
    # Communications and the Flight Computer
    # -----------------------------------------------------

    def _uplink(
        self,
        component,
        kind: str,
        payload
    ):
        """Send a command; it arrives a light time from now (now, crewed)."""

        delay = 0.0 if self.link.crewed else self.link.delay

        component.uplink.append((self.orbit_system.time + delay, kind, payload))

    def _update_comms(
        self,
        paused: bool
    ):
        """Each frame: the link; commands arriving; the computer's turn."""

        found = self.vessel_system.find(self.scene)

        if found is None:

            self.link = Link(False)

            return

        _, component = found

        self.link = self.comms_system.link(self.scene, self.vessel_system, self.orbit_system, self.planet_system)

        now = self.orbit_system.time

        arrived = [item for item in component.uplink if item[0] <= now]

        if arrived:

            component.uplink = [item for item in component.uplink if item[0] > now]

            for _, kind, payload in sorted(arrived, key=lambda item: item[0]):
                self._command(component, kind, payload)

        computer = component.computer

        if computer is not None and not computer.done and not component.state.destroyed and not paused:

            line = computer.current

            computer.tick(GameFlightContext(self, component))

            for when, text in computer.log[-3:]:
                if when == now and text:
                    self.hud.notify(f"Computer: {text}")

            if computer.done:
                component.hover = None

            elif line is not computer.current and computer.current is not None:
                Logger.debug("[Computer] Line %d.", computer.current.line)

        hold_sink_rate(self, component)

    def _command(
        self,
        component,
        kind: str,
        payload
    ):
        """A command arriving at the vessel."""

        if kind == "controls":

            pitch, yaw, roll, sas, sas_mode, throttle, translate = payload

            controls = component.controls

            controls.pitch, controls.yaw, controls.roll = pitch, yaw, roll
            controls.translate = np.array(translate, dtype=np.float64)
            controls.sas, controls.sas_mode = sas, sas_mode

            component.vessel.throttle = throttle

        elif kind == "stage":

            self.stage_vessel(component)

        elif kind == "science":

            self.run_science_aboard(component)

        elif kind == "program":

            component.computer = FlightComputer(payload)
            component.hover = None

            self.hud.notify("Computer: program received" if not self.link.crewed else "Computer: running")

        elif kind == "stop":

            if component.computer is not None:

                component.computer.stop(GameFlightContext(self, component))

                component.hover = None

    def _computer_warp(self) -> float:
        """Fastest the clock may run not to pass the computer's next deadline."""

        found = self.vessel_system.find(self.scene)

        if found is None:
            return math.inf

        computer = found[1].computer

        if computer is None or computer.done:
            return math.inf

        gap = computer.time_left(GameFlightContext(self, found[1]))

        if gap is None:
            return math.inf

        return max(1.0, gap / (4.0 / 30.0))

    def run_science_aboard(
        self,
        component
    ):
        """Run the experiments (the pilot's R arriving, or the computer)."""

        place = self._science_place()

        if place is None:
            return

        for notice in self.science_system.run(component, place, self._has_air()):
            self.hud.notify(notice)

    def transmit_all(
        self,
        component
    ):
        """Send every result aboard home (half value), if there is a link."""

        if not component.science:
            return

        if not self.link.connected:

            self.hud.notify(f"No link to transmit ({self.link.reason})")

            return

        gained = 0.0

        while component.science:
            gained += self.science_system.transmit(component, 0)

        self.hud.notify(f"Transmitted: +{gained:,.1f} science")

    def _draw_vessels(self):

        if not self.vessels_panel.open:
            return

        vessels = self.vessel_system

        found = vessels.find(self.scene)

        info = VesselsInfo()

        here = vessels.position_of(found[1], self.orbit_system) if found is not None else None

        for entity, component in vessels.vessels(self.scene):

            if component.state.destroyed:
                continue

            where = vessels.position_of(component, self.orbit_system)

            active = found is not None and entity == found[0]

            info.rows.append(VesselRow(
                entity=entity,
                name=component.vessel.name,
                body=vessels.body_name(self.scene, component.body),
                situation=vessels.brief_situation(self.scene, component, self.orbit_system),
                distance=None if active or here is None or where is None else float(np.linalg.norm(where - here)),
                active=active,
                targeted=self.map_view.target == ("vessel", entity),
                crewed=crew_of(component.vessel) > 0,
                propellant=component.vessel.propellant,
                capacity=component.vessel.propellant_capacity,
                docked=docking.modules(component.vessel)
            ))

        if found is not None:

            info.propellant = found[1].vessel.propellant
            info.capacity = found[1].vessel.propellant_capacity

            info.nearby = vessels.alongside(self.scene, found[1], self.orbit_system)

        for action in self.vessels_panel.draw(info):

            if action[0] == "fly":

                self.switch_vessel(action[1])

            elif action[0] == "target":

                self.map_view.target = ("vessel", action[1]) if action[1] is not None else None

            elif action[0] == "end":

                if found is not None and action[1] == found[0]:

                    self.hud.notify("Fly another vessel first")

                    continue

                self.vessel_system.remove(self.scene, action[1])

                self.hud.notify("Mission ended")

            elif action[0] == "undock":

                self.undock()

            elif action[0] == "transfer" and found is not None:

                other = self.scene.try_get_component(action[1], VesselComponent)

                if other is None:
                    continue

                amount = action[2]

                giver, taker = (other, found[1]) if amount > 0 else (found[1], other)

                moved = move_propellant(giver.vessel, taker.vessel, abs(amount))

                self.hud.notify(f"Moved {moved / 1000.0:,.1f} t of propellant to {taker.vessel.name}")

    @property
    def current_site(self) -> str | None:
        """The science site the vessel is in (its id), or None."""

        return self._site

    def _draw_programme(self):

        if not self.programme_panel.open:
            return

        programme = self.programme_system

        career = programme.career

        today = today_of(date_of(self.orbit_system.time))

        info = ProgrammeInfo(
            funds=career.funds,
            budget=career.budget(programme.campaign),
            reputation=career.reputation,
            science=self.science_system.record.points,
            today=today.strftime("%d %B %Y")
        )

        found = self.vessel_system.find(self.scene)

        if found is not None:

            vessel = found[1].vessel

            info.vessel = (vessel.name, vessel.cost, vessel.cost * RECOVERY, found[1].paid)

            if any(state.part.isru is not None for state in vessel.parts):

                if programme.making > 0.0:
                    source = programme.resources[0].name if programme.resources else ""
                elif programme.resources:
                    source = f"{programme.resources[0].name} here: land to make propellant"
                else:
                    source = "nothing to make it from here"

                info.plant = (programme.making, source, career.made)

        for mission in programme.campaign.missions:

            state = career.state(mission, today)

            info.missions.append((
                mission.title,
                mission.about,
                state,
                deadline_text(mission.deadline, today) if state != "waiting" else f"opens {mission.opens.isoformat()}",
                reward_text(mission.reward)
            ))

        now = self.orbit_system.time

        for window in programme.windows(self):

            info.windows.append((
                window.target,
                date_of(window.departure).strftime("%d %b %Y"),
                (window.departure - now) / 86_400.0,
                window.flight / 86_400.0,
                window.excess
            ))

        self.programme_panel.draw(info)

    def _docking_line(self) -> tuple[str, str] | None:
        """For the HUD: the approach to the target's port, or None."""

        command = self._docking_command

        if command is None:
            return None

        text = (
            f"Dock: {readable.distance(command.distance)}, closing {command.closing:+.2f} m/s, "
            f"{command.misalignment:.0f} deg off, {command.lateral:.1f} m off axis"
        )

        if self._docking is not None:
            return text + "  (auto, B: stop)", "good"

        return text + "  (B: auto-dock)", "warn"

    def _link_line(
        self,
        component
    ) -> tuple[str, str] | None:
        """For the HUD: (link text, "good" / "warn" / "bad"), and the computer's line."""

        link = self.link

        if link.connected:
            text = f"Link {link.station}" + (f", {readable.countdown(link.delay)} delay" if link.delay >= 0.5 else "")
            tone = "warn" if link.delay > 5.0 and not link.crewed else "good"
        else:
            text = f"No link: {link.reason}"
            tone = "warn" if link.crewed else "bad"

        computer = component.computer

        if computer is not None and not computer.done and computer.current is not None:
            text += f"  |  Computer line {computer.current.line}"

        return text, tone

    def _draw_computer(self):

        found = self.vessel_system.find(self.scene)

        if not self.computer_panel.open or found is None:
            return

        _, component = found

        link = self.link

        now = self.orbit_system.time

        info = ComputerInfo(
            connected=link.connected,
            link=link.station if link.connected else link.reason,
            delay=link.delay,
            crewed=link.crewed
        )

        names = {"program": "the program", "stop": "Stop", "controls": "controls", "stage": "Stage", "science": "Run experiments"}

        info.in_flight = [(names.get(kind, kind), when - now) for when, kind, _ in component.uplink if when > now and kind != "controls"]

        computer = component.computer

        if computer is not None:

            info.running = True
            info.done = computer.done
            info.log = computer.log

            current = computer.current

            if current is not None:
                info.line = current.text
                info.line_number = current.line

            info.wait = computer.deadline - now if computer.deadline is not None else None

        for action in self.computer_panel.draw(info):

            if action[0] == "save":

                try:
                    path = save_program(action[1], action[2])
                except OSError as error:
                    self.hud.notify(f"Could not save: {error}")
                else:
                    self.hud.notify(f"Saved {path}")

            elif action[0] in ("send", "stop"):

                if not link.crewed and not link.connected:

                    self.hud.notify(f"No link ({link.reason}): cannot send")

                    continue

                self._uplink(component, "program" if action[0] == "send" else "stop", action[1] if action[0] == "send" else None)

                if not link.crewed and link.delay > 1.0:
                    self.hud.notify(f"Sent: arrives in {readable.countdown(link.delay)}")

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

    def _map_camera(
        self,
        paused: bool
    ):
        """The map's camera, far round the body the vessel flies about."""

        context = FlightContext.find(self.scene)

        found = self.vessel_system.find(self.scene)

        if found is None or context.camera is None:
            return

        _, component = found

        scroll = 0.0 if self.imgui_layer.wants_mouse or paused else Input.scroll_delta()[1]

        self.map_view.orbit(Input.mouse_delta() if self._looking and not paused else (0.0, 0.0), scroll)

        center, axis, radius = self._map_body(component.body, component)

        prediction = self.vessel_system.prediction(self.scene, self.orbit_system)

        fit = radius * 2.0

        if prediction is not None and prediction.patches:

            reach = float(np.linalg.norm(prediction.patches[0].points, axis=1).max())

            # (Round a planet, no farther than its sphere of
            # influence; round the Sun, the whole path.)
            bodies = self.vessel_system.soi_bodies(self.scene)

            limit = bodies[component.body].soi if component.body in bodies else math.inf

            fit = max(fit, min(reach, limit if math.isfinite(limit) else reach))

        self.map_view.place_camera(context.camera_transform(), center, axis, fit, radius * 1.3)

    def _map_body(
        self,
        key,
        component
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """(world center, spin axis, radius) of a body in patched conics (None: the Sun)."""

        bodies = self.vessel_system.soi_bodies(self.scene)

        body = bodies.get(key)

        radius = body.radius if body is not None else 1.0

        if key is None:

            center = self.orbit_system.star_position()

            pole = np.cross(component.state.position, component.state.velocity)

            axis = self.orbit_system.inertial_to_world @ (pole / max(float(np.linalg.norm(pole)), 1e-9))

            return (center if center is not None else np.zeros(3)), axis, radius

        transform = self.scene.get_component(key, TransformComponent)

        matrix = np.asarray(transform.world_matrix, dtype=np.float64)

        return matrix[:3, 3].copy(), matrix[:3, 1] / np.linalg.norm(matrix[:3, 1]), radius

    def _draw_map(self):
        """The map's orbit lines over the view."""

        found = self.vessel_system.find(self.scene)

        context = FlightContext.find(self.scene)

        if not self.map_view.open or found is None or context.camera is None:
            return

        entity, component = found

        camera = self.scene.get_component(context.camera, CameraComponent)

        io = imgui.get_io()

        projection = ScreenProjection.of(context.camera_transform(), camera.fov, io.display_size.x, io.display_size.y)

        bodies = self.vessel_system.soi_bodies(self.scene)

        reference = component.body

        keys = [reference] + [
            key for key, body in bodies.items()
            if key is not None and body.parent == reference and key != reference
        ]

        body_world = {}
        body_radius = {}
        names = {}

        for key in keys:

            center, _, radius = self._map_body(key, component)

            body_world[key] = center
            body_radius[key] = radius
            names[key] = bodies[key].name if key in bodies else ""

        # The moons' orbits about the reference body (redrawn
        # every simulated hour).
        hour = int(self.orbit_system.time // 3600.0)

        children = [key for key in keys[1:] if key in self.orbit_system.states]

        relative = self.vessel_system.relative_states(self.scene, self.orbit_system, reference, children)

        moon_orbits = {}

        for key in children:

            cached = self._moon_orbits.get(key)

            if cached is None or cached[0] != hour:

                position, velocity = relative[key]

                shape = kepler.orbit_shape(position, velocity, bodies[reference].mu)

                span = shape.period if shape.period is not None else 30.0 * 86_400.0

                points, _ = kepler.propagate_many(position, velocity, bodies[reference].mu, np.linspace(0.0, span, 129))

                cached = (hour, points)

                self._moon_orbits[key] = cached

            moon_orbits[key] = cached[1]

        prediction = self.vessel_system.prediction(self.scene, self.orbit_system)

        planned = self.vessel_system.planned(self.scene, self.orbit_system)

        node_world = None

        if component.node is not None:

            where = self.vessel_system.node_state(self.scene, self.orbit_system, component.node.time)

            if where is not None and reference in body_world:
                node_world = body_world[reference] + self.orbit_system.inertial_to_world @ where[0]

        # The other vessels round the body (orbits redrawn every
        # simulated minute).
        others = []

        if reference in body_world:

            minute = int(self.orbit_system.time // 60.0)

            for other_entity, other in self.vessel_system.vessels(self.scene):

                if other_entity == entity or other.body != reference or other.state.destroyed:
                    continue

                point = body_world[reference] + self.orbit_system.inertial_to_world @ other.state.position

                orbit = None

                if not other.state.landed and reference in bodies:

                    cached = self._other_orbits.get(other_entity)

                    if cached is None or cached[0] != minute:

                        shape = kepler.orbit_shape(other.state.position, other.state.velocity, bodies[reference].mu)

                        span = shape.period if shape.period is not None and (shape.apoapsis or 0.0) < bodies[reference].soi else 86_400.0

                        points, _ = kepler.propagate_many(other.state.position, other.state.velocity, bodies[reference].mu, np.linspace(0.0, span, 129))

                        cached = (minute, points[np.isfinite(points).all(axis=1)])

                        self._other_orbits[other_entity] = cached

                    orbit = body_world[reference] + cached[1] @ self.orbit_system.inertial_to_world.T

                others.append((point, other.vessel.name, orbit, self.map_view.target == ("vessel", other_entity)))

        # The science sites on the body, where they are now.
        sites = []

        body_id = self.science_system.body_id(self.scene, reference)

        if reference is not None and reference in self.orbit_system.states and reference in body_world:

            _, frame = self.orbit_system.states[reference]

            for site in self.science_system.catalog.of(body_id):

                local = survey_direction(site.latitude, site.longitude)

                sites.append((
                    body_world[reference] + self.orbit_system.inertial_to_world @ (frame @ local) * body_radius[reference],
                    site.name
                ))

        self.map_view.draw(
            projection,
            self.orbit_system.inertial_to_world,
            body_world,
            body_radius,
            names,
            prediction,
            np.asarray(self.scene.get_component(entity, TransformComponent).transform.position, dtype=np.float64),
            moon_orbits,
            body_radius.get(reference, 0.0),
            planned=planned,
            node_world=node_world,
            node_label=f"Node {self.maneuver_dv:,.0f} m/s" if self.maneuver_dv is not None else "",
            sites=sites,
            others=others
        )

        targets = sorted(
            (
                (key, body.name)
                for key, body in bodies.items()
                if key is not None and key != reference and key in self.orbit_system.states
            ),
            # Planets first, then moons (by name).
            key=lambda item: (bodies[item[0]].parent is not None, item[1])
        )

        # Other vessels too (a rendezvous).
        targets += [
            (("vessel", other_entity), f"{other.vessel.name} (vessel)")
            for other_entity, other in self.vessel_system.vessels(self.scene)
            if other_entity != entity and not other.state.destroyed
        ]

        # (Planning a burn, the encounter it makes shows.)
        lines, burn, suggested = self._navigation(component, bodies, planned or prediction)

        self.map_view.draw_navigation(targets, lines, burn, projection)

        self._plan_node(
            component,
            planned,
            suggested + self.vessel_system.suggested_burns(self.scene, self.orbit_system, self.planet_system)
        )

    # Colors for the navigation panel.
    GOOD = (0.5, 1.0, 0.6, 1.0)
    NOTE = (0.75, 0.75, 0.8, 1.0)

    def _update_maneuver(self):
        """
        Each frame: the maneuver SAS mode points along what is
        left of the node's burn (the HUD counts it down); and
        round the Sun with a planet targeted, the course
        correction that reaches it (vessel/transfer.py), for
        the navigation panel to suggest as a node.
        """

        self.maneuver_dv = None
        self._correction = None

        found = self.vessel_system.find(self.scene)

        if found is None:
            return

        _, component = found

        remaining = self.vessel_system.node_remaining(component)

        component.controls.maneuver = remaining

        if remaining is not None:
            self.maneuver_dv = float(np.linalg.norm(remaining))

        # Warping to the node: as fast as allowed (the node
        # holds it back as the burn comes), until there.
        if self._warp_until is not None:

            if component.node is None or self.orbit_system.time >= self._warp_until:
                self._warp_until = None
            else:
                self.time_warp.index = len(self.time_warp.rates) - 1

        self._correction = self._course_correction(component)

    def _rendezvous(
        self,
        component,
        bodies: dict,
        prediction,
        target_entity
    ) -> list:
        """Navigation lines toward another vessel: how close the paths come, how fast past each other."""

        other = self.scene.try_get_component(target_entity, VesselComponent)

        if other is None or other.state.destroyed:

            self.map_view.target = None

            return [("That vessel is gone.", self.NOTE)]

        name = other.vessel.name

        if other.body != component.body:
            return [(f"{name} is round {self.vessel_system.body_name(self.scene, other.body)}: get there first.", self.NOTE)]

        distance = float(np.linalg.norm(other.state.position - component.state.position))
        speed = float(np.linalg.norm(other.state.velocity - component.state.velocity))

        lines = [(f"{name}: {readable.distance(distance)} away, {speed:,.1f} m/s apart", None)]

        near = self.vessel_system.alongside(self.scene, component, self.orbit_system)

        if near is not None and near[0] == target_entity:

            lines.append(("Alongside: move propellant in Vessels (L)", self.GOOD))

            return lines

        reference = bodies.get(component.body)

        if prediction is None or not prediction.patches or reference is None or other.state.landed or component.state.landed:

            lines.append(("Both in flight round the same body for an approach.", self.NOTE))

            return lines

        patch = prediction.patches[0]

        where, moving = kepler.propagate_many(other.state.position, other.state.velocity, reference.mu, patch.times)

        gap, when = closest_approach(patch.points, patch.times, where)

        index = int(np.argmin(np.abs(patch.times - when)))

        _, own = kepler.propagate(component.state.position, component.state.velocity, reference.mu, float(when))

        relative = float(np.linalg.norm(moving[index] - own)) if np.isfinite(moving[index]).all() else float("nan")

        lines.append((f"Closest approach: {readable.distance(gap)} in {readable.countdown(when)}", self.GOOD if gap < 5_000.0 else None))
        lines.append((f"Passing at {relative:,.0f} m/s: brake to it there (a node at the time)", self.NOTE))

        return lines

    def _course_correction(
        self,
        component
    ) -> np.ndarray | None:
        """Round the Sun, the map's target a planet: the velocity change now that meets it, or None."""

        target = self.map_view.target

        bodies = self.vessel_system.soi_bodies(self.scene)

        if (
            component.body is not None
            or target not in bodies
            or bodies[target].parent is not None
            or target not in self.orbit_system.states
            or component.state.destroyed
        ):
            return None

        now = self.orbit_system.time

        clock_now = self.timer.time_ns / 1e9 if self.timer is not None else 0.0

        # When it gets closest, from the predicted path (found
        # again every few seconds; the aim is that moment).
        if self._arrival is None or self._arrival[0] != target or clock_now - self._arrival_found > 3.0:

            prediction = self.vessel_system.prediction(self.scene, self.orbit_system)

            if prediction is None or not prediction.patches:
                return None

            # Already meeting it: trim inside its sphere instead.
            if any(patch.body == target for patch in prediction.patches[1:]):

                self._arrival = (target, None)

            else:

                patch = prediction.patches[0]

                relative = self.vessel_system.relative_states(self.scene, self.orbit_system, None, [target])[target]

                where, _ = kepler.propagate_many(*relative, bodies[None].mu, patch.times)

                _, when = closest_approach(patch.points, patch.times, where)

                self._arrival = (target, now + when if when > 86_400.0 else None)

            self._arrival_found = clock_now

        arrival = self._arrival[1]

        if arrival is None or arrival - now < 86_400.0:
            return None

        velocities = self.orbit_system.velocities(self.scene)

        # The cheapest of a few arrival times round the closest
        # approach (near half a turn away the transfer's plane
        # is ill-defined, and one exact time can cost a lot).
        candidates = []

        for factor in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15):

            try:
                candidates.append(correction(
                    bodies[None].mu,
                    component.state.position,
                    component.state.velocity,
                    self.orbit_system.states[target][0],
                    velocities[target],
                    (arrival - now) * factor
                ))
            except (ValueError, ZeroDivisionError, OverflowError):
                continue

        if not candidates:
            return None

        return min(candidates, key=lambda vector: float(np.linalg.norm(vector)))

    # -----------------------------------------------------
    # Science
    # -----------------------------------------------------

    def _science_place(self):

        return self.science_system.place(self.scene, self.vessel_system, self.orbit_system, self.planet_system)

    def _has_air(self) -> bool:
        """Air outside the vessel now."""

        found = self.vessel_system.find(self.scene)

        if found is None or self.vessel_system.reference_state(self.orbit_system, found[1].body) is None:
            return False

        environment = self.vessel_system.environment(self.scene, found[1].body, self.orbit_system, self.planet_system)

        state = found[1].state

        radius = float(np.linalg.norm(state.position))

        return environment.air(radius, state.position / radius)[0] > 0.0

    def _run_science(self):
        """R: run the experiments aboard (a probe: when the command gets there)."""

        found = self.vessel_system.find(self.scene)

        if found is None:
            return

        if self.link.crewed:

            self.run_science_aboard(found[1])

        elif self.link.connected:

            self._uplink(found[1], "science", None)

            if self.link.delay > 1.0:
                self.hud.notify(f"Run experiments: arrives in {readable.countdown(self.link.delay)}")

        else:

            self.hud.notify(f"No link ({self.link.reason}): the probe cannot hear you")

    def _update_science(
        self,
        events: list[str],
        simulated: float
    ):
        """Each frame: the camera maps; landing home recovers what is aboard; sites passed are named."""

        found = self.vessel_system.find(self.scene)

        if found is None:
            return

        _, component = found

        for notice in self.science_system.map_ground(self.scene, self.vessel_system, self.orbit_system, simulated):
            self.hud.notify(notice)

        if "landed" in events and component.body == self.vessel_system.launch_body and component.science:

            gained = self.science_system.recover(component)

            self.hud.notify(f"Science recovered: +{gained:,.1f}")

        # Arriving at a named site (landed or low over it).
        place = self._science_place() if events or self.timer.frame_count % 15 == 0 else None

        if place is not None:

            site = place.site.id if place.site is not None else None

            # (Not at high warp: an orbit passes a site every
            # few frames.)
            if site != self._site and site is not None and self.time_warp.rate <= 10.0:

                verb = "Landed at" if place.situation == "landed" else "Over"

                self.hud.notify(f"{verb} {place.site.name}")

            self._site = site

        self.science_system.upkeep(self.planet_system)

    def _draw_science(self):

        found = self.vessel_system.find(self.scene)

        if not self.science_panel.open or found is None:
            return

        _, component = found

        science = self.science_system

        place = self._science_place()

        info = ScienceInfo(points=science.record.points)

        launch = self.vessel_system.launch_body

        info.home = self.vessel_system.body_name(self.scene, launch) if launch is not None else "home"

        if place is not None:

            info.here = place.description

            if place.site is not None:
                info.about = place.site.about

            if place.nearest is not None and place.site is None:
                info.nearest = (place.nearest[0].name, place.nearest[1])

            info.experiments = science.status(component, place, self._has_air())

            mask = science.record.surveys.get(place.body_id)

            planet = self.scene.try_get_component(place.body, PlanetComponent) if place.body is not None else None

            if mask is not None and planet is not None and planet.palette != "bands":

                info.mapped = mask.coverage()
                info.body = place.body_name
                info.mapping = science.mapping_status(self.scene, self.vessel_system, self.orbit_system)

        info.results = [
            (result.title, science.record.worth(result), science.record.worth(result, transmitted=True))
            for result in component.science
        ]

        for action in self.science_panel.draw(info):

            if action[0] == "run":

                self._run_science()

            elif action[0] == "transmit":

                if not self.link.connected:

                    self.hud.notify(f"No link to transmit ({self.link.reason})")

                    continue

                gained = science.transmit(component, action[1])

                self.hud.notify(f"Transmitted: +{gained:,.1f} science")

    # -----------------------------------------------------
    # Maneuver Nodes
    # -----------------------------------------------------

    def _node_readout(
        self,
        component
    ) -> tuple[float, float, float] | None:
        """For the HUD: (m/s left, s until the burn should start, burn s), or None."""

        if component.node is None or self.maneuver_dv is None:
            return None

        burn = self.vessel_system.node_burn_time(component)

        start = component.node.time - 0.5 * (burn if math.isfinite(burn) else 0.0)

        return self.maneuver_dv, start - self.orbit_system.time, burn

    def _plan_node(
        self,
        component,
        planned,
        suggestions: list
    ):
        """The maneuver panel: show the node (or how to add one), carry out what was clicked."""

        vessels = self.vessel_system

        state = component.state

        now = self.orbit_system.time

        info = NodeInfo(can_plan=not state.landed and not state.destroyed)

        bodies = vessels.soi_bodies(self.scene)

        mu = bodies[component.body].mu if component.body in bodies else None

        if mu is not None and info.can_plan:

            info.apsides = tuple(
                apsis
                for apsis in ("apoapsis", "periapsis")
                if maneuver.time_to_apsis(state.position, state.velocity, mu, apsis) is not None
            )

        node = component.node

        info.suggestions = [(item.label, item.why) for item in suggestions]

        if node is not None:

            remaining = vessels.node_remaining(component)

            info.exists = True
            info.time_to = node.time - now
            info.components = {"prograde": node.prograde, "normal": node.normal, "radial": node.radial}
            info.remaining = float(np.linalg.norm(remaining))
            info.burn_time = vessels.node_burn_time(component)
            info.delta_v = component.vessel.delta_v()
            info.after = self._after_node(component, planned, bodies)

            if planned is None:
                info.note = "Past the path's next sphere: move it sooner"

        for action in self.maneuver_panel.draw(info):
            self._node_action(component, action, suggestions, mu)

    def _node_action(
        self,
        component,
        action: tuple,
        suggestions: list,
        mu: float | None
    ):

        vessels = self.vessel_system

        state = component.state

        now = self.orbit_system.time

        kind = action[0]

        def apsis_time(which: str) -> float | None:

            if mu is None:
                return None

            wait = maneuver.time_to_apsis(state.position, state.velocity, mu, which)

            return None if wait is None else now + max(wait, 30.0)

        if kind == "new":

            time = now + 300.0 if action[1] == "soon" else apsis_time(action[1])

            if time is not None:
                vessels.set_node(self.scene, self.orbit_system, ManeuverNode(time))

        elif kind == "suggest":

            item = suggestions[action[1]]

            where = vessels.node_state(self.scene, self.orbit_system, item.time)

            if where is not None:

                prograde, normal, radial = maneuver.from_inertial(*where, item.vector)

                vessels.set_node(self.scene, self.orbit_system, ManeuverNode(item.time, prograde, normal, radial))

        elif component.node is not None:

            node = component.node

            if kind == "delete":

                vessels.set_node(self.scene, self.orbit_system, None)

                self._warp_until = None

            elif kind == "warp":

                self._warp_until = node.time - 0.5 * min(vessels.node_burn_time(component), 1.0e7) - vessels.NODE_LEAD

            else:

                moved = ManeuverNode(node.time, node.prograde, node.normal, node.radial)

                if kind == "time":
                    moved.time = max(node.time + action[1], now + 1.0)
                elif kind == "time_to":
                    moved.time = apsis_time(action[1]) or node.time
                elif kind == "set":
                    setattr(moved, action[1], float(action[2]))

                if not vessels.set_node(self.scene, self.orbit_system, moved):
                    self.hud.notify("No orbit there to burn on")

    def _after_node(
        self,
        component,
        planned,
        bodies: dict
    ) -> list:
        """What the orbit becomes after the node, for the panel."""

        if planned is None or not planned.patches:
            return []

        first = planned.patches[0]

        body = bodies.get(first.body)

        if body is None:
            return []

        lines = []

        sea = body.radius

        if first.end == "orbit":

            distances = np.linalg.norm(first.points, axis=1)

            lines.append((f"After: Ap {readable.distance(float(distances.max()) - sea)}  Pe {readable.distance(first.periapsis - sea)}", None))

        elif first.end == "impact":

            lines.append((f"After: comes down on {body.name}", (1.0, 0.6, 0.4, 1.0)))

        elif first.end == "escape":

            lines.append((f"After: escapes {body.name}", (1.0, 0.6, 1.0, 1.0)))

        if first.end == "encounter" and len(planned.patches) > 1:

            inner = planned.patches[1]

            goal = bodies.get(inner.body)

            if goal is not None:
                lines.append((f"After: meets {goal.name}, Pe {readable.distance(inner.periapsis - goal.radius)}", self.GOOD))

        return lines

    def _navigation(
        self,
        component,
        bodies: dict,
        prediction
    ) -> tuple[list, np.ndarray | None, list]:
        """
        (lines, burn point in the world or None, Suggestions)
        for the map's navigation panel: how to get from here
        to the target, and the burns to plan for it.
        """

        target = self.map_view.target

        if isinstance(target, tuple) and target and target[0] == "vessel":
            return self._rendezvous(component, bodies, prediction, target[1]), None, []

        if target is None or target not in bodies:
            return [("Choose a target: how to get there shows here.", self.NOTE)], None, []

        reference = component.body

        state = component.state

        here = bodies.get(reference)
        goal = bodies[target]

        days = lambda seconds: seconds / 86_400.0

        # Encountering it already?
        if prediction is not None:

            for patch in prediction.patches[1:]:

                if patch.body == target:

                    return [
                        (f"Encounter with {goal.name} in {days(patch.start):,.1f} days", self.GOOD),
                        (f"Closest: {readable.distance(patch.periapsis - goal.radius)} above it", None),
                        ("Tune it with small burns; the map follows.", self.NOTE),
                        ("Inside its sphere, plan the capture at Pe.", self.NOTE),
                    ], None, []

        # There already.
        if target == reference:

            if state.landed:
                return [(f"Landed on {goal.name}.", self.GOOD)], None, []

            return [
                (f"In {goal.name}'s sphere of influence.", self.GOOD),
                ("Capture at the periapsis to stay (Maneuver),", self.NOTE),
                ("or come down through the air (heat shield first).", self.NOTE),
            ], None, []

        # From a planet's orbit to another planet: the window.
        if (
            here is not None
            and reference is not None
            and here.parent is None
            and goal.parent is None
            and not state.landed
        ):

            velocities = self.orbit_system.velocities(self.scene)

            now = self.orbit_system.time

            cached = self._window

            # The cheapest departure (~0.1 s to find): found
            # again for a new target, once it has passed, or
            # every couple of months (the orbits drift from
            # their conics).
            if (
                cached is None
                or cached[:2] != (reference, target)
                or now > cached[2] + cached[3].departure + 2.0 * 86_400.0
                or now - cached[2] > 60.0 * 86_400.0
            ):

                window = find_window(
                    bodies[None].mu,
                    (self.orbit_system.states[reference][0], velocities[reference]),
                    (self.orbit_system.states[target][0], velocities[target])
                )

                cached = self._window = (reference, target, now, window)

            plan = plan_transfer(
                bodies[None].mu,
                self.orbit_system.states[reference][0],
                velocities[reference],
                self.orbit_system.states[target][0],
                velocities[target],
                here.mu,
                state.position,
                state.velocity,
                window=cached[3],
                window_wait_time=cached[2] + cached[3].departure - now
            )

            wait = days(plan.wait)

            lines = [
                (f"To {goal.name}: {days(plan.transfer_time):,.0f} days on the way", None),
                (f"{goal.name} leads by {plan.phase_now:+.0f} deg (window at {plan.phase_wanted:+.0f})", None),
            ]

            # (Within a day either side.)
            if abs(wait) < 1.0:
                lines.append(("Window: now", self.GOOD))
            else:
                lines.append((f"Window in {wait:,.0f} days (time warp: . )", None))

            minutes, seconds = divmod(int(plan.time_to_burn_point), 60)

            lines += [
                (f"At the window: ~{plan.departure_burn:,.0f} m/s prograde", None),
                (f"Burning now at the marker: {plan.burn_now:,.0f} m/s", self.GOOD if plan.burn_now < plan.departure_burn * 1.1 else None),
                (f"(over it in {minutes}:{seconds:02d})", self.NOTE),
                (f"Arrive {plan.arrival_excess:,.0f} m/s fast: burn or brake in its air", self.NOTE),
            ]

            center, _, _ = self._map_body(reference, component)

            burn = center + self.orbit_system.inertial_to_world @ (
                plan.burn_direction * float(np.linalg.norm(state.position))
            )

            suggested = []

            # The window open: a node over the marker.
            if abs(wait) < 1.0:

                when = now + plan.time_to_burn_point

                where = self.vessel_system.node_state(self.scene, self.orbit_system, when)

                if where is not None:
                    suggested.append(Suggestion(
                        f"Burn for {goal.name}",
                        f"{plan.burn_now:,.0f} m/s prograde over the marker",
                        when,
                        maneuver.to_inertial(*where, plan.burn_now, 0.0, 0.0)
                    ))

            return lines, burn, suggested

        # Both round the same body: how close the path comes.
        if goal.parent == reference and prediction is not None and prediction.patches:

            patch = prediction.patches[0]

            relative = self.vessel_system.relative_states(self.scene, self.orbit_system, reference, [target])[target]

            where, _ = kepler.propagate_many(*relative, bodies[reference].mu if reference in bodies else here.mu, patch.times)

            gap, when = closest_approach(patch.points, patch.times, where)

            lines = [(f"Closest approach: {readable.distance(gap)} in {days(when):,.1f} days", None)]

            if gap < goal.soi:
                lines.append(("Inside its sphere of influence: an encounter", self.GOOD))
            else:
                lines.append((f"(its sphere of influence: {readable.distance(goal.soi)})", self.NOTE))

            suggested = []

            if self._correction is not None:

                size = float(np.linalg.norm(self._correction))

                lines.append((f"Course correction: {size:,.0f} m/s (Maneuver: suggested)", self.GOOD))

                suggested.append(Suggestion(
                    "Course correction",
                    f"{size:,.0f} m/s to meet {goal.name}",
                    self.orbit_system.time + 10.0,
                    self._correction
                ))

            return lines, None, suggested

        parent = bodies.get(goal.parent)

        return [(f"Reach {parent.name if parent is not None else 'orbit'} first.", self.NOTE)], None, []

    # -----------------------------------------------------
    # The Part Editor
    # -----------------------------------------------------

    def start_editing(self):
        """Build on the launch pad: the current design, there, to edit."""

        launch = self.vessel_system.launch_body

        if launch is None:
            return

        body = self.scene.try_get_component(launch, BodyComponent)

        self._design_before_edit = self.vessel_design
        self._edited_height = 0.0

        found = self.vessel_system.find(self.scene)

        self._flying_before_edit = found[0] if found is not None and found[1].paid else None

        self.free_camera = False
        self.map_view.open = False
        self.time_warp.stop()

        self.part_editor.start(
            DesignDraft.from_design(self.vessel_design),
            LaunchBody(
                self.vessel_system.body_name(self.scene, launch),
                body.surface_pressure_bar if body is not None else 0.0,
                body.surface_gravity if body is not None else 9.80665
            )
        )

    def _update_editor(self):
        """The rocket on the pad follows the draft; the selected part lights up."""

        editor = self.part_editor

        if not editor.changed:
            return

        editor.changed = False

        draft = editor.draft

        self.vessel_system.rebuild(
            self.scene,
            draft.preview() if draft.entries else None,
            self.orbit_system
        )

        self.vessel_system.highlight(self.scene, editor.selected)

        found = self.vessel_system.find(self.scene)

        height = found[1].vessel.height if found is not None else 0.0

        # The whole stack in view, refitted when it grows or
        # shrinks (a zoom of your own stays otherwise).
        if abs(height - self._edited_height) > 0.5:

            self.chase_camera.distance = max(height * 1.8, 15.0)

            self._edited_height = height

    def _launch_design(
        self,
        design
    ):

        self.vessel_design = design

        self.vessel_system.rebuild(self.scene, design, self.orbit_system)

        self.hud.notify(f"{design.name} on the pad: Z for full throttle, Space to launch")

    def _cancel_editing(self):

        if self._design_before_edit is not None:

            self.vessel_design = self._design_before_edit

            self.vessel_system.rebuild(self.scene, self.vessel_design, self.orbit_system)

        # Back to the vessel in flight it was opened from.
        if self._flying_before_edit is not None:

            self.vessel_system.switch(self.scene, self._flying_before_edit, self.orbit_system, self.planet_system)

            self._flying_before_edit = None

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
