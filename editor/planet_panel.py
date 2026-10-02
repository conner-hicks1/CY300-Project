import math
import random

from typing import TYPE_CHECKING

import numpy as np

from imgui_bundle import imgui

from ecs.components import (
    AtmosphereComponent,
    CameraComponent,
    ClimateComponent,
    CameraControllerComponent,
    DirectionalLightComponent,
    NameComponent,
    PlanetComponent,
    TransformComponent
)
from ecs.entity import Entity

from math3d import quaternion

from planet import solar

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
        planet_system: PlanetSystem
    ):

        self._editor = editor
        self._planets = planet_system

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

        if self.animate_day and context is not None and context.sun is not None:
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

        if imgui.collapsing_header("Camera", imgui.TreeNodeFlags_.default_open.value):
            self._draw_camera_section(context)

        if context.sun is not None and imgui.collapsing_header("Sun & time", imgui.TreeNodeFlags_.default_open.value):
            self._draw_sun_section(context)

        if imgui.collapsing_header("Terrain", imgui.TreeNodeFlags_.default_open.value):
            self._draw_terrain_section(context)

        if imgui.collapsing_header("Atmosphere"):
            self._draw_atmosphere_section(context)

        if imgui.collapsing_header("View & detail", imgui.TreeNodeFlags_.default_open.value):
            self._draw_view_section(context)

        if editor.playing:
            imgui.text_disabled("Playing: changes are undone on Stop.")

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
                context.planet_component.radius
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

        ground = context.planet_component.radius + max(spawn.elevation, 0.0)

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

            ground = self._planets.ground_elevation(context.planet, local)

            above_sea = distance - radius

            if ground is not None:

                above_ground = above_sea - max(ground, 0.0)

                imgui.text(f"Altitude  {_distance(above_ground)} above ground")

                if abs(above_ground - above_sea) > 1.0:
                    imgui.text_disabled(f"          {_distance(above_sea)} above sea level")

            else:

                imgui.text(f"Altitude  {_distance(above_sea)}")

            imgui.text(f"Speed     {_distance(self._speed)}/s")

            latitude = math.degrees(math.asin(float(np.clip(local[1], -1.0, 1.0))))
            longitude = math.degrees(math.atan2(local[0], local[2]))

            imgui.text(
                f"Position  {abs(latitude):.2f} {'N' if latitude >= 0 else 'S'}  "
                f"{abs(longitude):.2f} {'E' if longitude >= 0 else 'W'}"
            )

            current = context.solar_time()

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
