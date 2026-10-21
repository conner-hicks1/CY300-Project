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
    TransformComponent,
    VesselComponent
)
from ecs.entity import Entity

from game.chase_camera import ChaseCamera
from game.flight import FlightContext, frame_body, go_to_orbit, go_to_surface, read_flight
from game import style
from game.hud import FlightHud
from game.navball import NavballFrame
from game.computer_panel import ComputerPanel
from game.maneuver_panel import ManeuverPanel
from game.programme_panel import ProgrammePanel
from game.research_panel import ResearchPanel
from game.vessels_panel import VesselsPanel
from game.science_panel import SciencePanel
from game.map_view import MapView
from game.part_editor import EditorActions, PartEditor
from game.pause_menu import Destination, MenuActions, PauseMenu
from game.title_screen import TitleActions, TitleScreen
from game.guide import HINTS_PATH, FlightSchool, Hints
from game.bindings import CONTROLS_PATH, ControlSettings, keys
from game.encyclopaedia import Encyclopaedia, entries
from game.settings_menu import SettingsContext, SettingsMenu
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
from controllers.landing import LandingController
from controllers.sound import SoundController
from controllers.guide import GuideController
from controllers.lessons import LessonsController
from controllers.navigation import NavigationController
from controllers.power import PowerController
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
from systems.power_system import PowerSystem

from audio.sound import SoundEngine
from game.hud import notice_tone
from systems.crew_system import CrewSystem
from systems.programme_system import ProgrammeSystem
from systems.science_system import ScienceSystem
from systems.vessel_system import LaunchSite, OrbitStart, VesselSystem

from programme import fleet
from programme.career import Career, load_campaign
from programme import saves as games

from science.record import ScienceRecord
from science.sites import load_sites

from vessel.parts import load_design, load_parts

from scene.scene import Scene

from ui.debug_panel import DebugContext, DebugPanel
from ui.imgui_layer import ImGuiLayer
from ui.panels import PanelRegistry


class Application(
    LessonsController,
    GuideController,
    SoundController,
    LandingController,
    PowerController,
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

    # Headless: each frame this long (s).
    HEADLESS_STEP = 1.0 / 60.0

    # =====================================================
    # Construction
    # =====================================================

    def __init__(self):

        # A sandbox (initialize): no career; headless: stepped
        # from outside (step()), a hidden window.
        self.sandbox = False
        self.headless = False

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

        # The player's keys, mouse, interface size
        # (game/bindings.py).
        self.controls: ControlSettings | None = None

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
        sandbox: bool = False,
        headless: bool = False,
        saves: Path | None = None,
        slot: str | None = None,
        title: bool = False
    ):
        """
        body: where to start (a profile id in assets/bodies/;
            Earth by default; a career always starts at home).
        dev: start with the developer tools open (F3).
        sandbox: no funds, missions or calendar to keep: launch
            anything free, travel anywhere.
        headless: a hidden window and a fixed 1/60 s frame, for
            stepping the game with step() (tests: see
            tests/integration/).
        saves: a folder for the career, science and fleet
            saves (tests); else the game is a saved game in
            data/saves/ (programme/saves.py): the one named
            slot (made if there is none), or with title the
            last played behind the title screen, or the one
            called Career (Sandbox) otherwise. Headless
            without saves: nothing kept.
        """

        self.headless = headless

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
            "CY300 Spaceflight",
            visible=not headless
        )

        self.window.set_event_callback(
            self.on_event
        )

        # -------------------------------------------------
        # Core
        # -------------------------------------------------

        self.timer = Timer(fixed_step=self.HEADLESS_STEP if headless else None)

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

        # Science: the sites; the panel (K). (The record: with
        # the saves, below.)
        self._sites = load_sites()
        self.science_panel = SciencePanel()

        # The link home (systems/comms_system.py), this frame;
        # the flight computer's panel (C).
        self.comms_system = CommsSystem()
        self.link = Link(False)
        self.computer_panel = ComputerPanel()

        # Every vessel's power (systems/power_system.py).
        self.power_system = PowerSystem()

        # Every crew's life (systems/crew_system.py).
        self.crew_system = CrewSystem()

        # The space programme: its campaign; its panel (P).
        self._campaign = load_campaign()

        self.programme_panel = ProgrammePanel()

        # The tech tree (programme/research.py).
        self.research_panel = ResearchPanel()

        # Vessels left in flight (programme/fleet.py): saved,
        # brought into every scene; the list (L).
        self._fleet_restored = False
        self._fleet_saved = 0.0
        self.vessels_panel = VesselsPanel()

        # The game played (programme/saves.py): its saves,
        # read; the title screen.
        self.title = TitleScreen()

        self.slot = None

        if saves is not None:
            folder = Path(saves)
        elif headless:
            folder = None
        else:

            games.migrate()

            if slot is not None:
                self.slot = games.find(slot) or games.create(slot, sandbox)
            elif title:
                self.slot = next(iter(games.slots()), None)
            else:
                name = "Sandbox" if sandbox else "Career"
                self.slot = games.find(name) or games.create(name, sandbox)

            folder = self.slot.folder if self.slot is not None else None

            # (Behind the title with no game yet: a sandbox,
            # nothing kept.)
            sandbox = self.slot.sandbox if self.slot is not None else True

        self._open_saves(folder, sandbox)

        self._autosaved = time.monotonic()

        # The last risky moment's copy (s, monotonic); the
        # flight as last looked at for them.
        self._risk_saved = 0.0
        self._risk = None

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

        # Sound (audio/sound.py): none headless (tests run
        # quiet).
        self.sound = SoundEngine(enabled=not headless)

        self.hud.on_notice = lambda text: self.sound.notice(notice_tone(text))

        # Flight school and the hints (game/guide.py); the
        # hints seen kept across games (not headless).
        self.school = FlightSchool()
        self.hints = Hints(None if headless else HINTS_PATH)
        self._guided = 0.0

        # Docked since a lesson began; the docking lesson's
        # station (controllers/lessons.py).
        self._docked = False
        self._lesson_station = None

        # The player's keys, mouse and interface size
        # (game/bindings.py; defaults headless), the settings
        # (game/settings_menu.py), the encyclopaedia (J).
        self.controls = ControlSettings.load(None if headless else CONTROLS_PATH)
        self.controls.apply()
        self._apply_ui_scale()
        self.hud.rebound = self.controls.changed()

        self.settings_menu = SettingsMenu()

        self.encyclopaedia = Encyclopaedia(lambda: entries(self.parts, self.body_presets, self.programme_system.tree))

        # Headless, nothing is drawn: no terrain meshes, no
        # rocks or trees to build.
        if headless:
            self.planet_system.stream = False
            self.render_system.settings.scatter = False

        if body and body != self.START_BODY and not self.sandbox:
            Logger.warning("[Application] A career starts at home (%s); --sandbox to start at %s.", self.START_BODY, body)

        self.travel_to(self._start_body(body))

        if title and not headless:
            self.title.show()

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
    # Saved Games
    # =====================================================

    # Every so often (s, real, playing) a copy of the game is
    # kept in its autosave/ (programme/saves.py); and before
    # a risky moment (a launch, a new sphere of influence,
    # coming down to land), no more often than RISK_EVERY.
    AUTOSAVE_EVERY = 300.0
    RISK_EVERY = 60.0

    # Coming down to land: below this (m) over the ground,
    # falling faster than LANDING_SPEED (m/s).
    LANDING_HEIGHT = 5_000.0
    LANDING_SPEED = 5.0

    def _open_saves(
        self,
        folder: Path | None,
        sandbox: bool
    ):
        """The game's record, career and fleet, read from a folder of saves (None: a game nothing is kept of)."""

        self.saves = Path(folder) if folder is not None else None
        self.sandbox = sandbox

        def path(name):
            return self.saves / name if self.saves is not None else None

        science_path = path("science.json")

        self.science_system = ScienceSystem(
            self._sites,
            ScienceRecord.load(science_path) if science_path is not None else ScienceRecord(),
            science_path
        )

        campaign = self._campaign

        career_path = path("career.json")

        if sandbox:

            # (No funds or missions; its calendar kept all the
            # same: it goes on from where it was saved.)
            career = Career.load(campaign, career_path) if career_path is not None else Career(funds=0.0)

            career.funds = 0.0

            self.programme_system = ProgrammeSystem(campaign, career, career_path, sandbox=True)

        elif career_path is not None:
            self.programme_system = ProgrammeSystem(campaign, Career.load(campaign, career_path), career_path)
        else:
            self.programme_system = ProgrammeSystem(campaign, Career(funds=campaign.start_funds), None)

        self.fleet_path = path("fleet.json")

        self._fleet_records = fleet.load(self.fleet_path) if self.fleet_path is not None else []

    def _start_body(
        self,
        wanted: str | None = None
    ) -> str:
        """Where a game starts: a career at home; a sandbox where asked, else where the flown vessel was."""

        if not self.sandbox:
            return self.START_BODY

        if wanted is None:
            wanted = next((record.get("body") for record in self._fleet_records if record.get("active")), None)

        return wanted if wanted in self.body_presets else self.START_BODY

    def _summary(self) -> dict:
        """The game in brief, for the title screen."""

        return {
            "clock": self.orbit_system.time,
            "funds": None if self.sandbox else self.programme_system.career.funds,
            "vessels": sum(1 for _, component in self.vessel_system.vessels(self.scene) if component.paid and not component.state.destroyed),
            "science": self.science_system.record.points,
            "body": self.body_id,
        }

    def save_game(self):
        """Everything written now (the game saves itself as it goes too)."""

        self.science_system.save()
        self.programme_system.save(force=True)
        self.save_fleet()

        if self.slot is None:
            return

        self.slot.played = time.time()
        self.slot.summary = self._summary()

        try:
            games.write(self.slot)
        except OSError as error:
            Logger.error("[Application] Could not save the game's summary: %s", error)

    def open_game(
        self,
        slot: games.Slot | None,
        kind: str | None = None,
        folder: Path | None = None,
        sandbox: bool | None = None
    ):
        """
        Play a saved game (the one now saved first): as it
        was left, or its quicksave / autosave (kind). slot
        None: a folder of saves (tests; None: nothing kept),
        a sandbox or not (as now, by default).
        """

        self.save_game()

        if slot is not None:
            folder, sandbox = slot.folder, slot.sandbox
        elif sandbox is None:
            sandbox = self.sandbox

        self.school.stop()

        if kind is not None and folder is not None:
            games.restore(folder, kind)

        self.slot = slot

        self.pause_menu.close()
        self.part_editor.open = False
        self._set_looking(False)

        self._open_saves(folder, sandbox)

        self.vessel_design = load_design("pathfinder", self.parts)
        self._flying_before_edit = None
        self._design_before_edit = None

        # (A new start: nothing of the last game's vessels
        # carried over.)
        self._fleet_restored = False

        self._fleet_saved = self._autosaved = time.monotonic()
        self._risk = None

        self.travel_to(self._start_body())

        if slot is not None:
            Logger.info("[Application] Playing '%s'%s.", slot.name, f" ({kind})" if kind else "")

    def quicksave(self) -> bool:

        if self.saves is None:

            self.hud.notify("Nothing is kept of this game: start or load one from the title screen")

            return False

        if self.part_editor.open:
            return False

        try:
            self.save_game()
            games.keep(self.saves, games.QUICKSAVE)
        except OSError as error:

            Logger.error("[Application] Quicksave failed: %s", error)

            self.hud.notify("Could not quicksave")

            return False

        self.hud.notify(keys("Quicksaved ({F9} to go back to it)"))

        return True

    def quickload(self) -> bool:

        if self.saves is None or games.kept(self.saves, games.QUICKSAVE) is None:

            self.hud.notify(keys("No quicksave yet ({F5})"))

            return False

        self.open_game(self.slot, games.QUICKSAVE, self.saves)

        self.hud.notify("Quickloaded")

        return True

    def _autosave(
        self,
        why: str | None = None
    ) -> bool:
        """Every AUTOSAVE_EVERY s of play, or now for a risky moment (why): a copy kept."""

        now = time.monotonic()

        if self.saves is None:
            return False

        if why is None and now - self._autosaved < self.AUTOSAVE_EVERY:
            return False

        if why is not None and now - self._risk_saved < self.RISK_EVERY:
            return False

        self._autosaved = now

        if why is not None:
            self._risk_saved = now

        try:
            self.save_game()
            games.keep(self.saves, games.AUTOSAVE)
        except OSError as error:

            Logger.error("[Application] Autosave failed: %s", error)

            return False

        if why is not None:

            Logger.info("[Application] Autosaved: %s.", why)

            self.hud.notify(f"Autosaved: {why}")

        return True

    def _autosave_at_risk(self):
        """A copy kept as something risky begins: a launch, a new sphere of influence, coming down to land."""

        found = self.vessel_system.find(self.scene)

        if found is None or self.saves is None:

            self._risk = None

            return

        entity, component = found

        state = component.state

        staged = component.vessel.next_stage > 0

        landing = (
            not state.landed
            and self.vessel_system.height < self.LANDING_HEIGHT
            and float(state.velocity @ state.position) / max(float(np.linalg.norm(state.position)), 1.0) < -self.LANDING_SPEED
        )

        now = (entity, staged, component.body, landing)

        before, self._risk = self._risk, now

        # (Another vessel, or the first look: nothing begun.)
        if before is None or before[0] != entity or state.destroyed:
            return

        if staged and not before[1]:
            self._autosave("the launch")
        elif component.body != before[2]:
            self._autosave(f"into {self.vessel_system.body_name(self.scene, component.body)}'s sphere of influence")
        elif landing and not before[3]:
            self._autosave("coming down to land")

    # =====================================================
    # Settings
    # =====================================================

    def _apply_ui_scale(self):
        """The interface's size (the style's from game/style.py, scaled)."""

        if self.imgui_layer is None:
            return

        style.apply()

        # (No bigger than the window holds: the HUD's panels
        # side by side, the menus whole.)
        scale = min(self.controls.ui_scale, self._ui_fits())

        imgui_style = imgui.get_style()

        if abs(scale - 1.0) > 1e-3:
            imgui_style.scale_all_sizes(scale)

        imgui_style.font_scale_main = scale

    # px of window height the interface needs at 1x.
    UI_HEIGHT = 540.0

    def _ui_fits(self) -> float:
        """The biggest interface size the window holds."""

        _, height = self.window.framebuffer_size

        return max(1.0, height / self.UI_HEIGHT)

    def _controls_changed(self):
        """Keys, mouse or size changed: into the input and the style, kept."""

        self.controls.apply()

        self.controls.save(None if self.headless else CONTROLS_PATH)

        self._apply_ui_scale()

        self.hud.rebound = self.controls.changed()

    def _settings_context(self) -> SettingsContext:

        return SettingsContext(
            quality=current_quality(self.render_system.settings),
            set_quality=lambda name: apply_quality(self.render_system.settings, name),
            sound=self.sound,
            hints=self.hints,
            controls=self.controls,
            controls_changed=self._controls_changed,
            ui_fits=self._ui_fits()
        )

    def _draw_overlays(self):

        self.encyclopaedia.draw()

        self.settings_menu.draw(self._settings_context())

    def leave_flight_school(self):

        self.school.stop()

        self.show_title()

    def show_title(self):
        """Back to the title screen (the game saved, behind it)."""

        self.save_game()

        self.pause_menu.close()
        self._set_looking(False)

        self.title.show()

    def _title_actions(self) -> TitleActions:

        def play(slot, kind):
            self.title.close()
            self.open_game(slot, kind)

        def new(name, sandbox):

            slot = games.create(name, sandbox, self.title.directory)

            self.title.close()

            self.open_game(slot)

            self.hud.notify(f"A new {'sandbox' if sandbox else 'career'}: {name}")

        return TitleActions(
            resume=self.title.close,
            play=play,
            new=new,
            quit=self.request_quit,
            school=self.start_lesson,
            settings=self.settings_menu.show,
            encyclopaedia=self.encyclopaedia.show
        )

    # =====================================================
    # Travel
    # =====================================================

    def travel_to(
        self,
        body_id: str,
        keep_flown: bool = True,
        morning: bool = False
    ):
        """
        Start over at a body: a fresh scene with it, its
        system, the sun and the camera (on its surface, or
        in space over a giant or a small body). The vessels in
        flight come along (the one flown now too, unless
        keep_flown is False: a revert); a new one stands on
        the pad. morning: on to mid-morning where it starts (a
        sandbox's trip somewhere: arrive in daylight).
        """

        self._morning = morning

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
        # (or a sandbox's trip somewhere); otherwise the
        # calendar goes on as it is: a game loaded (or
        # quickloaded) at the moment it was saved, a revert not
        # skipping ahead (a solar day on Venus is 117 Earth
        # days).
        if getattr(self, "_morning", False) or self.programme_system.career.clock <= seconds_since_j2000(self.START_DATE):
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

    def step(
        self,
        render: bool = False,
        keys: tuple = ()
    ):
        """
        One frame of the main loop, run from outside it (a
        headless run): input, the clock, update; render only if
        asked (the game's state never depends on drawing).
        keys: pressed this frame, as if typed.
        """

        engine_assert(self._initialized, "Application must be initialized before step().")

        Input.begin_frame()

        self.window.poll_events()

        for key in keys:
            Input.simulate_press(key)

        self.timer.update()

        self.update()

        if render:

            self.render()

            self.window.swap_buffers()

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

        paused = self.pause_menu.open or building or self.title.open

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

            if "destroyed" in events or "landed" in events or "toppled" in events:
                self.time_warp.stop()

            self._notify(events)

            self._update_power(simulated)

            self._update_crew(simulated)

        with profiler.scope("Science"):

            self._update_science(events, simulated)

            self.programme_system.update(self, events, simulated)

            for event in events:

                kind, _, name = event.partition(":")

                if kind == "bump":

                    other, _, speed = name.rpartition(":")

                    if float(speed) >= 1.0:
                        self.hud.notify(f"Bumped {other} at {float(speed):,.1f} m/s")

                elif kind == "reentered":

                    if not name.startswith("Fragment"):
                        self.hud.notify(f"{name} came down through the air and burned up")

                elif kind == "lost":
                    self.hud.notify(f"{name} came down and was lost")
                elif kind == "landed" and name:
                    self.hud.notify(f"{name} landed under its parachute")

            now = time.monotonic()

            if now - self._fleet_saved > 30.0:

                self._fleet_saved = now

                self.save_game()

            if not paused:

                self._autosave()

                self._autosave_at_risk()

            for news in self.programme_system.news:
                self.hud.notify(news)

            self.programme_system.news.clear()

            self._take_recovered()

        self.hud.tick(dt)

        self._sound_events(events)

        self._update_sound(paused)

        self._update_guide(paused)

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

            if readout.slope is not None and not readout.toppled:
                self.hud.notify(f"Standing on a {readout.slope:,.0f} deg slope")

        if "toppled" in events:
            self.hud.notify(f"{readout.name} fell over: {readout.fell}. It cannot lift off again")

        if readout.parachute == "open" and self._parachute != "open":
            self.hud.notify("Parachute opening")

            self.sound.play("chute", 0.9)

        self._parachute = readout.parachute


    # Colors for the navigation panel.
    GOOD = style.tone("good")
    NOTE = style.tone("muted")


    # -----------------------------------------------------
    # The Part Editor
    # -----------------------------------------------------


    def revert(self):
        """Back to the launch: the same body, from the start (the other vessels in flight stay where they are)."""

        # The flight called off: its crew back in the corps.
        found = self.vessel_system.find(self.scene)

        if found is not None and found[1].paid:
            self.programme_system.stand_down(found[1])

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

        # The settings: Esc closes them (or stops choosing a
        # key); no game keys meanwhile.
        if self.settings_menu.open:

            if Input.is_key_pressed(Key.ESCAPE):
                self.settings_menu.back()

            return

        # The encyclopaedia: Esc (or J) closes it.
        if self.encyclopaedia.open and (Input.is_key_pressed(Key.ESCAPE) or Input.is_key_pressed(Key.J)):

            self.encyclopaedia.open = False

            return

        # The title screen: Esc back a page; no game keys.
        if self.title.open:

            if Input.is_key_pressed(Key.ESCAPE):
                self.title.back()

            return

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

        if Input.is_key_pressed(Key.F1):
            self.hud.show_controls = not self.hud.show_controls

        if Input.is_key_pressed(Key.F2):
            self.hud.visible = not self.hud.visible

        if Input.is_key_pressed(Key.F3):
            self.dev_tools = not self.dev_tools

        if self.dev_tools and Input.is_key_pressed(Key.F6):

            self.resources.reload_changed_shaders(
                force=True
            )

        if menu.open:
            return

        if Input.is_key_pressed(Key.F5):
            self.quicksave()

        if Input.is_key_pressed(Key.F9):
            self.quickload()

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

        if Input.is_key_pressed(Key.J):
            self.encyclopaedia.toggle()

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

        if Input.is_key_pressed(Key.U) and self.vessel_system.find(self.scene) is not None:
            self._toggle_panels()

        if Input.is_key_pressed(Key.G) and self.vessel_system.find(self.scene) is not None:
            self._toggle_legs()

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
            self.travel_to(body_id, morning=True)

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
            quit=self.request_quit,
            title=self.show_title,
            settings=self.settings_menu.show,
            encyclopaedia=self.encyclopaedia.show
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

                # Over everything (the title screen too): the
                # settings, the encyclopaedia.
                self._draw_overlays()

                # The interface's click: a button, a box, a tab.
                if imgui.is_mouse_clicked(imgui.MouseButton_.left) and imgui.is_any_item_hovered():
                    self.sound.play("click", 0.7, alert=True)

            with profiler.scope("Draw", gpu=True):

                self.imgui_layer.end_frame()

    def _draw_ui(self):

        # The title screen over the game behind it, and
        # nothing else.
        if self.title.open:

            self.title.draw(self._title_actions(), self.slot)

            return

        # What is not researched yet, and the flight data, for
        # the editor.
        self.part_editor.locked = self.programme_system.locked_parts()

        failures = self.programme_system.failures

        self.part_editor.reliability = failures.record if failures.enabled else None

        if self.part_editor.open:

            self.part_editor.funds = self.programme_system.career.funds

            self.part_editor.draw(
                EditorActions(
                    launch=self._launch_design,
                    cancel=self._cancel_editing
                )
            )

            self._draw_research()

            return

        found = self.vessel_system.find(self.scene)

        # Where the player is: the vessel, else the camera.
        focus = (
            np.asarray(self.scene.get_component(found[0], TransformComponent).transform.position, dtype=np.float64)
            if found is not None
            else None
        )

        # (The body it orbits, not just the nearest ground.)
        context = FlightContext.find(self.scene, focus, found[1].body if found is not None else None)

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
            navball=self._navball() if found is not None else None,
            power=found[1].power if found is not None else None,
            life=found[1].life if found is not None else None
        )

        self._draw_map()

        self._draw_science()

        self._draw_computer()

        self._draw_programme()

        self._draw_research()

        self._draw_vessels()

        self.school.draw(self.leave_flight_school, self.lesson_again)

        if not self.pause_menu.open and not self.school.active:
            self.hints.draw()
        else:
            self.hints.height = 0.0

        # (Next frame's notices under them.)
        above = max(self.school.height, self.hints.height)

        self.hud.notices_below = above + style.GAP if above > 0.0 else 0.0

        self.pause_menu.draw(
            self._menu_actions(context),
            context.name() if context.body is not None else "",
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

                if imgui.menu_item("Reload shaders", "F6", False)[0]:
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

    def _navball(self):
        """What the attitude ball shows: the attitude, the orbit's directions, the burn left, the way to the target."""

        vessels = self.vessel_system

        frame = vessels.navball_frame(self.scene, self.orbit_system, self.planet_system)

        found = vessels.find(self.scene)

        if frame is None or found is None:
            return None

        _, component = found

        state = component.state

        orientation, up, north, motion = frame

        target = None

        chosen = self.map_view.target

        here = vessels.position_of(component, self.orbit_system)

        if isinstance(chosen, tuple) and chosen and chosen[0] == "vessel":

            other = self.scene.try_get_component(chosen[1], VesselComponent)

            there = vessels.position_of(other, self.orbit_system) if other is not None else None

            if there is not None and here is not None:
                target = there - here

        elif chosen is not None and chosen in self.orbit_system.states and here is not None:

            target = np.asarray(self.orbit_system.states[chosen][0], dtype=np.float64) - here

        return NavballFrame(
            orientation=orientation,
            up=up,
            north=north,
            motion=motion,
            position=None if state.landed else state.position,
            velocity=None if state.landed else state.velocity,
            maneuver=vessels.node_remaining(component),
            target=target
        )

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
        # size; the interface's size follows what the window
        # holds now.

        if self.controls is not None:
            self._apply_ui_scale()

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
            and not self.title.open
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

        sound = getattr(self, "sound", None)

        if sound is not None:
            sound.close()

        if getattr(self, "programme_system", None) is not None and self.scene is not None:

            try:
                self.save_game()
            except Exception:
                Logger.exception("[Application] Could not save the game.")

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
