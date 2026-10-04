import math
import random

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from imgui_bundle import imgui

from ecs.components import (
    AtmosphereComponent,
    BodyComponent,
    CameraComponent,
    ClimateComponent,
    ClockComponent,
    OrbitComponent,
    CameraControllerComponent,
    DirectionalLightComponent,
    NameComponent,
    PlanetComponent,
    TransformComponent
)
from ecs.entity import Entity

from editor.inspector_panel import body_facts

from math3d import quaternion

from planet import solar
from planet.air import AirColumn
from planet.orbits import date_of, seconds_since_j2000

from systems.planet_system import TERRAIN_VIEWS, PlanetSystem

from ui.panels import DockSlot

if TYPE_CHECKING:
    from editor.scene_editor import SceneEditor


# =========================================================
# Terrain Presets
# =========================================================
#
# Starting points for the terrain sliders; every field not
# listed keeps its value.

TERRAIN_PRESETS: dict[str, dict[str, float]] = {
    "Earth-like": dict(
        continent_frequency=1.2, continent_height=2_500.0, land_bias=0.05,
        mountain_frequency=150.0, mountain_height=5_000.0, detail_height=300.0
    ),
    "Mountainous": dict(
        continent_frequency=1.0, continent_height=3_000.0, land_bias=0.15,
        mountain_frequency=110.0, mountain_height=7_500.0, detail_height=450.0
    ),
    "Archipelago": dict(
        continent_frequency=3.5, continent_height=1_500.0, land_bias=-0.12,
        mountain_frequency=220.0, mountain_height=2_500.0, detail_height=250.0
    ),
    "Supercontinent": dict(
        continent_frequency=0.6, continent_height=2_500.0, land_bias=0.05,
        mountain_frequency=150.0, mountain_height=5_500.0, detail_height=300.0
    ),
    "Ocean world": dict(
        continent_frequency=2.0, continent_height=2_500.0, land_bias=-0.3,
        mountain_frequency=180.0, mountain_height=3_000.0, detail_height=200.0
    ),
    "Flat plains": dict(
        continent_frequency=1.2, continent_height=1_200.0, land_bias=0.2,
        mountain_frequency=150.0, mountain_height=600.0, detail_height=120.0
    ),
}

EARTH_RAYLEIGH = AtmosphereComponent().rayleigh_scattering
EARTH_MIE_SCATTERING = AtmosphereComponent().mie_scattering
EARTH_MIE_ABSORPTION = AtmosphereComponent().mie_absorption

HUD_FLAGS = (
    imgui.WindowFlags_.no_decoration.value
    | imgui.WindowFlags_.no_inputs.value
    | imgui.WindowFlags_.no_docking.value
    | imgui.WindowFlags_.no_saved_settings.value
    | imgui.WindowFlags_.no_focus_on_appearing.value
    | imgui.WindowFlags_.no_nav.value
    | imgui.WindowFlags_.always_auto_resize.value
)


class PlanetPanel:

    # =====================================================
    # Planet Panel + Viewport HUD
    # =====================================================
    #
    # The planet-specific controls in one place, in the
    # terms you think in rather than raw component fields:
    #
    #   Camera   jump to the surface / orbit / whole planet,
    #            fly speed
    #   Sun      local time of day and season at the camera
    #            (positions the Sun entity), optional day
    #            cycle animation
    #   Terrain  presets, seed, sizes (rebuilds once you
    #            stop dragging)
    #   Air      atmosphere on/off, density, haze
    #   View     data views (elevation, slope...) and level
    #            of detail
    #
    # It edits the selected planet, or the first one in the
    # scene. The raw fields stay available in the Inspector.
    #
    # The HUD shows altitude, speed, position and local time
    # over the viewport.

    def __init__(
        self,
        editor: "SceneEditor",
        planet_system: PlanetSystem,
        apply_preset: Callable[[Entity, str], None] | None = None,
        descriptions: dict[str, str] | None = None,
        maps: dict[str, tuple[str, str]] | None = None
    ):
        """
        apply_preset(planet entity, profile id): turn the
            planet into another body (Application).
        descriptions: profile id -> one-paragraph description.
        """

        self._editor = editor
        self._planets = planet_system
        self._apply_preset = apply_preset
        self._descriptions = descriptions or {}

        # profile id -> (elevation, color) real map datasets.
        self._maps = maps or {}

        self._preset_choice = 0

        editor.panels.register(
            "Planet",
            DockSlot.LEFT_BOTTOM,
            description="Camera, sun, terrain, atmosphere and views for the planet."
        )

        self.animate_day = False

        # Simulated hours per real second while animating.
        self.day_speed = 1.0

        self._preset = 0

        # Camera speed estimate for the HUD.
        self._last_camera_position: np.ndarray | None = None
        self._speed = 0.0

    # =====================================================
    # Per Frame
    # =====================================================

    def draw(
        self,
        delta_time: float,
        fps: float
    ):

        editor = self._editor

        context = self._context()

        if (
            self.animate_day
            and context is not None
            and context.sun is not None
            and context.clock() is None
        ):
            self._advance_day(context, delta_time)

        self._track_speed(context, delta_time)

        if not editor.visible:
            return

        if editor.panels.is_visible("Planet"):

            if editor.panels.begin("Planet"):
                self._draw_contents(context)

            editor.panels.end()

        if editor.show_hud:
            self._draw_hud(context, fps)

    # =====================================================
    # Scene Lookup
    # =====================================================

    def _context(
        self
    ) -> "_PlanetContext | None":

        scene = self._editor.scene
        registry = scene.registry

        planet_entity = None

        selected = self._editor.selected

        if selected is not None and scene.try_get_component(selected, PlanetComponent) is not None:

            planet_entity = selected

        else:

            for entity, _, _ in registry.view_with(TransformComponent, PlanetComponent):

                planet_entity = entity

                break

        camera = next(
            (
                entity
                for entity, component in registry.view_with(CameraComponent)
                if component.primary
            ),
            None
        )

        sun = next(
            (entity for entity, _ in registry.view_with(DirectionalLightComponent)),
            None
        )

        if planet_entity is None:
            return _PlanetContext(scene, None, camera, sun)

        return _PlanetContext(scene, planet_entity, camera, sun)

    # =====================================================
    # Panel
    # =====================================================

    def _draw_contents(
        self,
        context: "_PlanetContext"
    ):

        if context.planet is None:

            imgui.text_wrapped("There is no planet in this scene.")

            if imgui.button("Create Earth-sized planet"):
                self._create_planet()

            imgui.text_disabled("Or File > New Demo Scene.")

            return

        name = context.name()

        imgui.text(name)

        # Leave room for labels on the right of each widget.
        imgui.push_item_width(-125.0)

        try:
            self._draw_planet(context)
        finally:
            imgui.pop_item_width()

    def _draw_planet(
        self,
        context: "_PlanetContext"
    ):

        editor = self._editor

        component = context.planet_component

        if self._planets.rebuild_pending(context.planet):

            imgui.same_line()
            imgui.text_colored((0.95, 0.8, 0.35, 1.0), "(rebuilding...)")

        elif self._planets.stats.chunks_building or self._planets.stats.chunks_queued:

            imgui.same_line()
            imgui.text_colored((0.95, 0.8, 0.35, 1.0), "(loading detail...)")

        imgui.text_disabled(
            f"Radius {component.radius / 1000.0:,.0f} km  |  seed {component.seed}"
        )

        if imgui.collapsing_header("Body", imgui.TreeNodeFlags_.default_open.value):
            self._draw_body_section(context)

        if imgui.collapsing_header("Camera", imgui.TreeNodeFlags_.default_open.value):
            self._draw_camera_section(context)

        if context.sun is not None and imgui.collapsing_header("Sun & time", imgui.TreeNodeFlags_.default_open.value):
            self._draw_sun_section(context)

        if context.has_system() and imgui.collapsing_header("System", imgui.TreeNodeFlags_.default_open.value):
            self._draw_system_section(context)

        if imgui.collapsing_header("Terrain", imgui.TreeNodeFlags_.default_open.value):
            self._draw_terrain_section(context)

        if imgui.collapsing_header("Atmosphere"):
            self._draw_atmosphere_section(context)

        if imgui.collapsing_header("View & detail", imgui.TreeNodeFlags_.default_open.value):
            self._draw_view_section(context)

        if editor.playing:
            imgui.text_disabled("Playing: changes are undone on Stop.")

    # -----------------------------------------------------
    # Body
    # -----------------------------------------------------

    def _draw_body_section(
        self,
        context: "_PlanetContext"
    ):

        body = context.scene.try_get_component(context.planet, BodyComponent)

        if body is not None:

            description = self._descriptions.get(body.profile)

            if description:
                imgui.text_wrapped(description)

            if imgui.begin_table("planet body facts", 2, imgui.TableFlags_.row_bg.value):

                for label, value in body_facts(body)[:8]:

                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.text_disabled(label)
                    imgui.table_next_column()
                    imgui.text(value)

                imgui.end_table()

            self._draw_real_maps(context, body)

        else:

            imgui.text_disabled("No body profile: a custom planet.")

        presets = [
            (profile_id, name)
            for members in self._editor.body_presets.values()
            for profile_id, name in members
        ]

        if not presets or self._apply_preset is None:
            return

        names = [name for _, name in presets]

        self._preset_choice = min(self._preset_choice, len(names) - 1)

        _, self._preset_choice = imgui.combo("##body preset", self._preset_choice, names)

        imgui.same_line()

        if imgui.button("Become"):

            profile_id, name = presets[self._preset_choice]

            self._apply_preset(context.planet, profile_id)

        imgui.set_item_tooltip(
            "Turn this planet into the chosen body: size, surface,\n"
            "atmosphere, climate and geology (undo restores it)."
        )

    def _draw_real_maps(
        self,
        context: "_PlanetContext",
        body: BodyComponent
    ):
        """Measured elevation and colors instead of generated ones."""

        from planet.maps import DATASETS

        elevation, color = self._maps.get(body.profile, ("", ""))

        datasets = [DATASETS[d] for d in (elevation, color) if d]

        if not datasets:
            return

        component = context.planet_component

        available = all(d.available for d in datasets)

        using = bool(component.elevation_map or component.color_map)

        imgui.begin_disabled(not available and not using)

        changed, using = imgui.checkbox("Real maps", using)

        imgui.end_disabled()

        if changed:

            component.elevation_map = elevation if using else ""
            component.color_map = color if using else ""

            self._editor.record("Real maps" if using else "Generated terrain")

        tooltip = "\n".join(f"{d.description}\n  {d.credit}" for d in datasets)

        if not available:
            tooltip += f"\n\nNot downloaded: python tools/fetch_maps.py {body.profile}"

        imgui.set_item_tooltip(tooltip)

    # -----------------------------------------------------
    # Camera
    # -----------------------------------------------------

    def _draw_camera_section(
        self,
        context: "_PlanetContext"
    ):

        if context.camera is None:

            imgui.text_disabled("No primary camera.")

            return

        width = (imgui.get_content_region_avail().x - 2 * imgui.get_style().item_spacing.x) / 3.0

        if imgui.button("Surface", (width, 0)):
            self._go_to_surface(context)

        imgui.set_item_tooltip("A valley with mountains in view.")

        imgui.same_line()

        if imgui.button("Low orbit", (width, 0)):
            self._go_to_orbit(context, 400_000.0)

        imgui.set_item_tooltip("400 km up, looking at the horizon.")

        imgui.same_line()

        if imgui.button("Whole planet", (width, 0)):
            self._editor.frame_planet(
                context.camera,
                context.planet_center(),
                self._planets.extent(context.planet) or context.planet_component.radius
            )

        imgui.set_item_tooltip("Back off until the planet fits the view (F).")

        controller = context.scene.try_get_component(context.camera, CameraControllerComponent)

        if controller is not None:

            changed, value = imgui.slider_float(
                "Fly speed",
                controller.altitude_speed,
                0.1,
                5.0,
                "%.1f x altitude/s",
                imgui.SliderFlags_.logarithmic.value
            )

            if changed:
                controller.altitude_speed = value
                controller.planet_mode = True

            self._record_on_release("Fly speed")

            imgui.set_item_tooltip(
                "Speed grows with height above the ground:\n"
                "at 1x you cover your altitude every second."
            )

        imgui.text_disabled("Hold right mouse button: look and fly.")

    def _go_to_surface(
        self,
        context: "_PlanetContext"
    ):

        spawn = self._planets.spawn_point(context.planet)

        if spawn is None:
            return

        rotation = context.planet_rotation()
        center = context.planet_center()

        up = rotation @ spawn.direction
        view = rotation @ spawn.view_direction

        base, surface = self._planets.ground(context.planet, spawn.direction) or (0.0, max(spawn.elevation, 0.0))

        ground = context.planet_component.radius + base + surface

        transform = context.camera_transform()

        transform.position = center + up * (ground + 150.0)

        transform.orientation = quaternion.multiply(
            quaternion.look_rotation(view, up),
            quaternion.from_euler((-4.0, 0.0, 0.0))
        )

        self._editor.record("Camera to surface")

    def _go_to_orbit(
        self,
        context: "_PlanetContext",
        altitude: float
    ):

        center = context.planet_center()
        transform = context.camera_transform()

        offset = np.asarray(transform.position, dtype=np.float64) - center

        up = offset / max(float(np.linalg.norm(offset)), 1e-9)

        heading = transform.forward - np.dot(transform.forward, up) * up

        if np.linalg.norm(heading) < 1e-6:
            heading = transform.up - np.dot(transform.up, up) * up

        heading /= np.linalg.norm(heading)

        radius = context.planet_component.radius

        transform.position = center + up * (radius + altitude)

        # Pitched down so the curved horizon sits in view.
        dip = math.degrees(math.acos(radius / (radius + altitude)))

        transform.orientation = quaternion.multiply(
            quaternion.look_rotation(heading, up),
            quaternion.from_euler((-(dip + 8.0), 0.0, 0.0))
        )

        self._editor.record("Camera to orbit")

    # -----------------------------------------------------
    # Sun & Time
    # -----------------------------------------------------

    def _draw_sun_section(
        self,
        context: "_PlanetContext"
    ):

        current = context.solar_time()

        if current is None:

            imgui.text_disabled("Needs a camera to define \"local\" time.")

            return

        clock = context.clock()

        if clock is not None:

            self._draw_clock_section(context, clock, current)

            return

        hour, declination, elevation = current

        # The sun's latitude swings between +-tilt over the
        # year (the climate's axial tilt, Earth's without).
        climate = context.scene.try_get_component(context.planet, ClimateComponent)

        tilt = max(
            0.1,
            climate.axial_tilt if climate is not None else solar.MAX_DECLINATION
        )

        hours = int(hour)
        minutes = int((hour - hours) * 60.0)

        changed, new_hour = imgui.slider_float(
            "Time of day",
            hour,
            0.0,
            24.0,
            f"{hours:02d}:{minutes:02d}"
        )

        if changed:
            context.set_solar_time(new_hour, declination)

        self._record_on_release("Time of day")

        imgui.set_item_tooltip("Local solar time where the camera is (12:00 = sun highest).")

        changed, new_declination = imgui.slider_float(
            "Season",
            math.degrees(declination),
            -tilt,
            tilt,
            "sun %+.1f deg"
        )

        if changed:
            context.set_solar_time(hour, math.radians(new_declination))

        self._record_on_release("Season")

        imgui.set_item_tooltip(
            "Sun's latitude (declination): +23.4 = northern summer,\n"
            "0 = equinox, -23.4 = northern winter."
        )

        if elevation > 0.0:
            imgui.text_disabled(f"Sun {math.degrees(elevation):.1f} deg above the horizon")
        else:
            imgui.text_disabled(f"Night: sun {-math.degrees(elevation):.1f} deg below the horizon")

        _, self.animate_day = imgui.checkbox("Animate day", self.animate_day)

        if self.animate_day:

            imgui.same_line()

            imgui.set_next_item_width(-1.0)

            _, self.day_speed = imgui.slider_float(
                "##day_speed",
                self.day_speed,
                0.1,
                24.0,
                "%.1f h/s",
                imgui.SliderFlags_.logarithmic.value
            )

    # Simulated time per real second: (label, seconds).
    CLOCK_RATES = (
        ("Stopped", 0.0),
        ("Real time", 1.0),
        ("1 minute / s", 60.0),
        ("10 minutes / s", 600.0),
        ("1 hour / s", 3_600.0),
        ("6 hours / s", 21_600.0),
        ("1 day / s", 86_400.0),
        ("1 week / s", 604_800.0),
        ("1 month / s", 2_629_800.0),
    )

    def _draw_clock_section(
        self,
        context: "_PlanetContext",
        clock: ClockComponent,
        current
    ):
        """
        The simulation clock: the date, how fast time runs,
        and local time at the camera (moving the clock, so
        the whole system follows: moons, seasons, eclipses).
        """

        hour, declination, elevation = current

        imgui.text(date_of(clock.time).strftime("%Y-%m-%d  %H:%M UTC"))

        names = [name for name, _ in self.CLOCK_RATES]

        rates = [rate for _, rate in self.CLOCK_RATES]

        index = min(range(len(rates)), key=lambda i: abs(rates[i] - clock.rate))

        changed, index = imgui.combo("Time runs", index, names)

        if changed:

            clock.rate = rates[index]

            self._editor.record("Clock rate")

        body = context.scene.try_get_component(context.planet, BodyComponent)

        solar_day = (
            abs(body.solar_day_hours) * 3_600.0
            if body is not None and body.solar_day_hours
            else 86_400.0
        )

        hours = int(hour)
        minutes = int((hour - hours) * 60.0)

        changed, new_hour = imgui.slider_float(
            "Time of day",
            hour,
            0.0,
            24.0,
            f"{hours:02d}:{minutes:02d}"
        )

        if changed:

            # The nearest way round to the new hour.
            difference = (new_hour - hour + 12.0) % 24.0 - 12.0

            clock.time += difference / 24.0 * solar_day

        self._record_on_release("Time of day")

        imgui.set_item_tooltip(
            "Local solar time where the camera is (12:00 = sun highest).\n"
            "Moves the clock: the moons and the season follow."
        )

        width = (imgui.get_content_region_avail().x - 3 * imgui.get_style().item_spacing.x) / 4.0

        for label, days in (("-30 d", -30.0), ("-1 d", -1.0), ("+1 d", 1.0), ("+30 d", 30.0)):

            if imgui.button(label, (width, 0)):

                # Whole local days (the same time of day) where
                # days are short; Earth days otherwise.
                clock.time += days * (solar_day if solar_day <= 2.0 * 86_400.0 else 86_400.0)

                self._editor.record("Clock date")

            imgui.same_line()

        imgui.new_line()

        if imgui.button("Now"):

            from datetime import datetime, timezone

            clock.time = seconds_since_j2000(datetime.now(timezone.utc))

            self._editor.record("Clock to now")

        imgui.set_item_tooltip("The real date and time.")

        imgui.same_line()

        imgui.text_disabled(f"Season: sun at {math.degrees(declination):+.1f} deg latitude")

        if elevation > 0.0:
            imgui.text_disabled(f"Sun {math.degrees(elevation):.1f} deg above the horizon")
        else:
            imgui.text_disabled(f"Night: sun {-math.degrees(elevation):.1f} deg below the horizon")

    # -----------------------------------------------------
    # System
    # -----------------------------------------------------

    def _draw_system_section(
        self,
        context: "_PlanetContext"
    ):
        """The bodies of the planetary system, and a way to each."""

        scene = context.scene

        camera = context.camera

        position = (
            np.asarray(context.camera_transform().position, dtype=np.float64)
            if camera is not None
            else None
        )

        for entity, transform, orbit in list(scene.registry.view_with(TransformComponent, OrbitComponent)):

            planet = scene.try_get_component(entity, PlanetComponent)

            if planet is None:
                continue

            name = scene.try_get_component(entity, NameComponent)

            label = name.name if name is not None else "Body"

            center = np.asarray(transform.world_matrix, dtype=np.float64)[:3, 3]

            imgui.push_id(int(entity.index))

            if imgui.small_button("Go") and camera is not None:

                extent = self._planets.extent(entity) or planet.radius

                self._editor.frame_planet(camera, center, extent)

                self._editor.select(entity)

            imgui.set_item_tooltip("Fly there: the whole body in view.")

            imgui.same_line()

            if position is not None:

                distance = float(np.linalg.norm(center - position)) - planet.radius

                imgui.text(f"{label}")

                imgui.same_line()

                imgui.text_disabled(_distance(max(distance, 0.0)) + " away" if distance > 1.0 else "here")

            else:

                imgui.text(label)

            imgui.pop_id()

    def _advance_day(
        self,
        context: "_PlanetContext",
        delta_time: float
    ):

        current = context.solar_time()

        if current is None:
            return

        hour, declination, _ = current

        context.set_solar_time((hour + self.day_speed * delta_time) % 24.0, declination)

    # -----------------------------------------------------
    # Terrain
    # -----------------------------------------------------

    def _draw_terrain_section(
        self,
        context: "_PlanetContext"
    ):

        component = context.planet_component

        names = list(TERRAIN_PRESETS)

        imgui.set_next_item_width(imgui.get_content_region_avail().x * 0.6)

        changed, self._preset = imgui.combo("##preset", self._preset, names)

        imgui.same_line()

        if imgui.button("Apply preset"):

            for field, value in TERRAIN_PRESETS[names[self._preset]].items():
                setattr(component, field, value)

            self._editor.record(f"Terrain preset {names[self._preset]}")

        changed, seed = imgui.input_int("Seed", component.seed)

        if changed:
            component.seed = seed
            self._editor.record("Planet seed")

        imgui.same_line()

        if imgui.button("Random"):
            component.seed = random.randint(1, 999_999)
            self._editor.record("Random planet seed")

        self._slider(component, "radius", "Radius", 50_000.0, 10_000_000.0, "%.0f m", logarithmic=True,
                     tooltip="Earth: 6,371 km.")

        self._slider(component, "land_bias", "Land amount", -0.5, 0.5, "%.2f",
                     tooltip="Higher = more land, lower = more ocean.")

        self._slider(component, "continent_frequency", "Continent count", 0.3, 6.0, "%.1f",
                     tooltip="Low = a few large continents, high = many small ones.")

        self._slider(component, "continent_height", "Continent relief", 0.0, 8_000.0, "%.0f m",
                     tooltip="Highlands vs ocean depth.")

        self._slider(component, "mountain_height", "Mountain height", 0.0, 15_000.0, "%.0f m")

        self._slider(component, "mountain_frequency", "Mountain spacing", 20.0, 600.0, "%.0f", logarithmic=True,
                     tooltip="Higher = narrower ranges with more peaks.")

        self._slider(component, "detail_height", "Hills", 0.0, 1_500.0, "%.0f m")

        imgui.text_disabled("The planet rebuilds when you release a slider.")

    # -----------------------------------------------------
    # Atmosphere
    # -----------------------------------------------------

    def _draw_atmosphere_section(
        self,
        context: "_PlanetContext"
    ):

        scene = context.scene

        atmosphere = scene.try_get_component(context.planet, AtmosphereComponent)

        enabled = atmosphere is not None

        changed, enabled = imgui.checkbox("Atmosphere", enabled)

        if changed:

            if enabled:
                scene.add_component(context.planet, AtmosphereComponent())
            else:
                scene.remove_component(context.planet, AtmosphereComponent)

            self._editor.record("Toggle atmosphere")

            return

        if atmosphere is None:
            return

        density = atmosphere.rayleigh_scattering[0] / EARTH_RAYLEIGH[0]

        changed, density = imgui.slider_float("Air density", density, 0.0, 5.0, "%.2f x Earth")

        if changed:
            atmosphere.rayleigh_scattering = tuple(v * density for v in EARTH_RAYLEIGH)

        self._record_on_release("Air density")

        haze = atmosphere.mie_scattering / EARTH_MIE_SCATTERING

        changed, haze = imgui.slider_float("Haze", haze, 0.0, 20.0, "%.1f x Earth",
                                           imgui.SliderFlags_.logarithmic.value)

        if changed:
            atmosphere.mie_scattering = EARTH_MIE_SCATTERING * haze
            atmosphere.mie_absorption = EARTH_MIE_ABSORPTION * haze

        self._record_on_release("Haze")

        self._slider(atmosphere, "height", "Thickness", 10_000.0, 500_000.0, "%.0f m", logarithmic=True)

        imgui.text_disabled("All coefficients: Inspector > Atmosphere.")

    # -----------------------------------------------------
    # View & Detail
    # -----------------------------------------------------

    def _draw_view_section(
        self,
        context: "_PlanetContext"
    ):

        names = [name for name, _ in TERRAIN_VIEWS]

        changed, mode = imgui.combo("Show", self._planets.view_mode, names)

        if changed:
            self._planets.view_mode = mode

        imgui.set_item_tooltip(TERRAIN_VIEWS[self._planets.view_mode][1])

        component = context.planet_component

        self._slider(component, "split_factor", "Detail distance", 0.75, 3.0, "%.2f",
                     tooltip="How far out full detail reaches. Higher = sharper, slower.")

        changed, depth = imgui.slider_int("Max detail level", component.max_depth, 6, 20)

        if changed:
            component.max_depth = depth

        self._record_on_release("Max detail level")

        spacing = (math.pi * 0.5 * component.radius) / (1 << component.max_depth) / max(component.resolution - 1, 1)

        imgui.text_disabled(f"Finest vertex spacing: {_distance(spacing)}")

        stats = self._planets.stats

        imgui.text_disabled(f"{stats.chunks_drawn} chunks drawn, {stats.chunks_loaded} in memory")

    # -----------------------------------------------------
    # Widgets
    # -----------------------------------------------------

    def _slider(
        self,
        component,
        field: str,
        label: str,
        low: float,
        high: float,
        fmt: str,
        logarithmic: bool = False,
        tooltip: str = ""
    ):

        flags = imgui.SliderFlags_.logarithmic.value if logarithmic else 0

        changed, value = imgui.slider_float(label, float(getattr(component, field)), low, high, fmt, flags)

        if changed:
            setattr(component, field, value)

        self._record_on_release(label)

        if tooltip:
            imgui.set_item_tooltip(tooltip)

    def _record_on_release(
        self,
        label: str
    ):

        if imgui.is_item_deactivated_after_edit():
            self._editor.record(label)

    def _create_planet(self):

        editor = self._editor

        planet = PlanetComponent()

        entity = editor.create_entity("Planet", [planet, AtmosphereComponent()])

        # Surface just below the world origin, where the
        # camera usually is.
        transform = editor.scene.get_component(entity, TransformComponent).transform

        transform.position = (0.0, -planet.radius - 2_000.0, 0.0)

    # =====================================================
    # HUD
    # =====================================================

    def _track_speed(
        self,
        context: "_PlanetContext | None",
        delta_time: float
    ):

        if context is None or context.camera is None or delta_time <= 0.0:
            return

        position = np.asarray(context.camera_transform().position, dtype=np.float64)

        if self._last_camera_position is not None:

            speed = float(np.linalg.norm(position - self._last_camera_position)) / delta_time

            # Smoothed; teleports settle within a second.
            self._speed += (speed - self._speed) * min(1.0, delta_time * 6.0)

        self._last_camera_position = position

    def _air(
        self,
        context
    ) -> AirColumn | None:
        """The planet's air column (planet/air.py), None without air."""

        scene = self._editor.scene

        atmosphere = scene.try_get_component(context.planet, AtmosphereComponent)
        body = scene.try_get_component(context.planet, BodyComponent)

        if atmosphere is None or body is None or body.surface_pressure_bar <= 0.0:
            return None

        giant = context.planet_component.palette == "bands"

        # Radiative skin temperature: equilibrium / 2^(1/4).
        equilibrium = (
            278.6
            * max(body.star_luminosity, 0.0) ** 0.25
            * max(1.0 - body.bond_albedo, 0.0) ** 0.25
            / math.sqrt(max(body.orbit_distance_au, 1e-6))
        )

        return AirColumn.from_scale_height(
            surface_pressure=body.surface_pressure_bar,
            surface_temperature=body.mean_temperature + 273.15,
            scale_height_km=float(atmosphere.rayleigh_scale_height) / 1000.0,
            gravity=body.surface_gravity,
            adiabatic_exponent=float(atmosphere.adiabatic_exponent),
            lapse_rate=body.lapse_rate,
            giant=giant,
            skin_temperature=equilibrium / 2.0 ** 0.25
        )

    def _draw_hud(
        self,
        context: "_PlanetContext | None",
        fps: float
    ):

        rect = self._editor.viewport_rect

        # Bottom-left: the toolbar is top-center, status
        # messages bottom-right.
        imgui.set_next_window_pos(
            (rect.x + 10.0, rect.y + rect.height - 10.0),
            imgui.Cond_.always,
            (0.0, 1.0)
        )
        imgui.set_next_window_bg_alpha(0.55)

        imgui.begin("##hud", None, HUD_FLAGS)

        imgui.text(f"{fps:5.1f} FPS  ({1000.0 / fps if fps > 0 else 0.0:.1f} ms)")

        if context is not None and context.planet is not None and context.camera is not None:

            center = context.planet_center()
            position = np.asarray(context.camera_transform().position, dtype=np.float64)

            offset = position - center
            distance = float(np.linalg.norm(offset))

            radius = context.planet_component.radius

            local = context.planet_rotation().T @ (offset / max(distance, 1e-9))

            ground = self._planets.ground(context.planet, local)

            # Sea level (the datum on dry worlds): the body's
            # shape.
            base, surface = ground if ground is not None else (0.0, 0.0)

            above_sea = distance - radius - base

            if ground is not None and context.planet_component.palette == "bands":

                # Giants: no ground, cloud tops (and below them,
                # air ever thicker and hotter).
                above_tops = above_sea

                if above_tops >= 0.0:

                    imgui.text(f"Altitude  {_distance(above_tops)} above the cloud tops")

                else:

                    imgui.text(f"Depth     {_distance(-above_tops)} below the cloud tops")

            elif ground is not None:

                above_ground = above_sea - surface

                imgui.text(f"Altitude  {_distance(above_ground)} above ground")

                if abs(above_ground - above_sea) > 1.0:
                    imgui.text_disabled(f"          {_distance(above_sea)} above sea level")

            else:

                imgui.text(f"Altitude  {_distance(above_sea)}")

            # The air around the camera.
            air = self._air(context)

            if air is not None:

                altitude_km = above_sea / 1000.0

                pressure = air.pressure(altitude_km)

                if pressure > 1e-6:
                    imgui.text(f"Air       {_pressure(pressure)}, {air.temperature(altitude_km) - 273.15:,.0f} C")

            imgui.text(f"Speed     {_distance(self._speed)}/s")

            latitude = math.degrees(math.asin(float(np.clip(local[1], -1.0, 1.0))))
            longitude = math.degrees(math.atan2(local[0], local[2]))

            imgui.text(
                f"Position  {abs(latitude):.2f} {'N' if latitude >= 0 else 'S'}  "
                f"{abs(longitude):.2f} {'E' if longitude >= 0 else 'W'}"
            )

            current = context.solar_time()

            clock = context.clock()

            if clock is not None:
                imgui.text(f"Date      {date_of(clock.time).strftime('%Y-%m-%d %H:%M UTC')}")

            if current is not None:

                hour, _, elevation = current

                imgui.text(
                    f"Time      {int(hour):02d}:{int((hour % 1.0) * 60):02d}  "
                    f"(sun {math.degrees(elevation):+.0f} deg)"
                )

        imgui.text_disabled("Hold right mouse to fly  |  Help > Controls")

        imgui.end()


# =========================================================
# Helpers
# =========================================================

class _PlanetContext:

    # The planet being edited, the camera and the sun, with
    # the geometry the panel needs.

    def __init__(
        self,
        scene,
        planet: Entity | None,
        camera: Entity | None,
        sun: Entity | None
    ):

        self.scene = scene
        self.planet = planet
        self.camera = camera
        self.sun = sun

    @property
    def planet_component(
        self
    ) -> PlanetComponent:

        return self.scene.get_component(self.planet, PlanetComponent)

    def name(
        self
    ) -> str:

        component = self.scene.try_get_component(self.planet, NameComponent)

        return component.name if component is not None else "Planet"

    def _planet_world(
        self
    ) -> np.ndarray:

        matrix = np.asarray(
            self.scene.get_component(self.planet, TransformComponent).world_matrix,
            dtype=np.float64
        )

        basis = matrix[:3, :3]

        rigid = np.eye(4)
        rigid[:3, :3] = basis / np.linalg.norm(basis, axis=0, keepdims=True)
        rigid[:3, 3] = matrix[:3, 3]

        return rigid

    def planet_center(
        self
    ) -> np.ndarray:

        return self._planet_world()[:3, 3]

    def planet_rotation(
        self
    ) -> np.ndarray:

        return self._planet_world()[:3, :3]

    def clock(
        self
    ) -> ClockComponent | None:

        return next((c for _, c in self.scene.registry.view_with(ClockComponent)), None)

    def has_system(
        self
    ) -> bool:
        """Several bodies on orbits."""

        return sum(1 for _ in self.scene.registry.view_with(OrbitComponent)) > 1

    def camera_transform(self):

        return self.scene.get_component(self.camera, TransformComponent).transform

    def _sun_transform(self):

        return self.scene.get_component(self.sun, TransformComponent).transform

    # -----------------------------------------------------
    # Solar time
    # -----------------------------------------------------
    #
    # In planet space (y = the planet's axis), the sun's
    # direction has a latitude (declination = season) and a
    # longitude. Local solar time at the camera is the
    # sun's longitude relative to the camera's: 12:00 when
    # they match (sun due south/north, highest).

    def solar_time(
        self
    ) -> tuple[float, float, float] | None:
        """(hour 0-24, declination rad, sun elevation rad) at the camera."""

        if self.planet is None or self.camera is None or self.sun is None:
            return None

        point = self._camera_direction()

        if point is None:
            return None

        rotation = self.planet_rotation()

        sun_world = -np.asarray(self._sun_transform().forward, dtype=np.float64)

        return solar.solar_time(point, rotation.T @ sun_world)

    def set_solar_time(
        self,
        hour: float,
        declination: float
    ):

        point = self._camera_direction()

        if point is None:
            return

        sun_world = self.planet_rotation() @ solar.sun_direction(point, hour, declination)

        self._sun_transform().orientation = sun_orientation(sun_world)

    def _camera_direction(
        self
    ) -> np.ndarray | None:
        """Planet-space unit direction from the center to the camera."""

        offset = np.asarray(self.camera_transform().position, dtype=np.float64) - self.planet_center()

        length = float(np.linalg.norm(offset))

        if length <= 0.0:
            return None

        return self.planet_rotation().T @ (offset / length)


def sun_orientation(
    sun_world: np.ndarray
) -> np.ndarray:
    """Orientation of a directional light shining from `sun_world`."""

    travel = -np.asarray(sun_world, dtype=np.float64)

    hint = np.array([0.0, 1.0, 0.0]) if abs(travel[1]) < 0.99 else np.array([0.0, 0.0, -1.0])

    return quaternion.look_rotation(travel, hint)


def _pressure(
    bar: float
) -> str:
    """Human-readable pressure."""

    if bar >= 10.0:
        return f"{bar:,.0f} bar"

    if bar >= 0.01:
        return f"{bar:.2f} bar"

    return f"{bar * 1e5:.3g} Pa"


def _distance(
    meters: float
) -> str:
    """Human-readable distance."""

    magnitude = abs(meters)

    if magnitude < 1_000.0:
        return f"{meters:,.0f} m"

    if magnitude < 100_000.0:
        return f"{meters / 1000.0:,.1f} km"

    return f"{meters / 1000.0:,.0f} km"
