from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from imgui_bundle import imgui

from core.cprofile_capture import CProfileCapture
from core.jobs import JobSystem
from ui.panels import DockSlot, PanelRegistry
from core.profiler import Profiler
from core.timer import Timer
from core.window import Window

from graphics.lighting import MAX_CASCADES
from graphics.render_settings import (
    RenderSettings,
    Tonemapper
)
from graphics.renderer import (
    Renderer,
    RenderStats
)

from resources.resources import Resources
from systems.planet_system import PlanetStats


# =========================================================
# Debug Context
# =========================================================
#
# Everything the panel can inspect or edit, gathered by
# the Application each frame.

@dataclass(slots=True)
class DebugContext:

    resources: Resources
    settings: RenderSettings
    stats: RenderStats
    timer: Timer

    # Forces a rebuild of every shader; returns how many
    # were reloaded.
    reload_shaders: Callable[[], int]

    profiler: Profiler
    cprofile_capture: CProfileCapture
    window: Window
    jobs: JobSystem
    planet_stats: PlanetStats


class DebugPanel:

    # =====================================================
    # Construction
    # =====================================================

    def __init__(
        self,
        panels: PanelRegistry
    ):

        self.visible = True

        self._panels = panels

        panels.register(
            "Stats",
            DockSlot.BOTTOM,
            description="Frame rate, draw calls, streaming."
        )

        panels.register(
            "Render",
            DockSlot.BOTTOM,
            description="Exposure, bloom, sky, shadows, materials."
        )

        panels.register(
            "Profiler",
            DockSlot.BOTTOM,
            description="CPU / GPU time per frame stage."
        )

        self._last_reload_message = ""

        self._capture_frames = 120

    # =====================================================
    # Draw
    # =====================================================

    def draw(
        self,
        context: DebugContext
    ):

        if not self.visible:
            return

        if self._panels.is_visible("Stats"):
            self._draw_stats_window(context)

        if self._panels.is_visible("Render"):
            self._draw_engine_window(context)

        if self._panels.is_visible("Profiler"):
            self._draw_profiler_window(context)

    # =====================================================
    # Profiler Window
    # =====================================================

    # Refresh-rate reference lines for the frame graph.
    _BUDGET_60_HZ_MS = 1000.0 / 60.0

    def _draw_profiler_window(
        self,
        context: DebugContext
    ):

        self._panels.begin("Profiler")


        profiler = context.profiler

        # -------------------------------------------------
        # Controls
        # -------------------------------------------------

        _, profiler.enabled = imgui.checkbox(
            "Enabled",
            profiler.enabled
        )

        imgui.same_line()

        _, profiler.paused = imgui.checkbox(
            "Pause",
            profiler.paused
        )

        imgui.same_line()

        changed, vsync = imgui.checkbox(
            "VSync",
            context.window.vsync
        )

        if changed:
            context.window.set_vsync(vsync)

        if imgui.is_item_hovered():

            imgui.set_tooltip(
                "With VSync on, 'Swap' includes waiting for the display.\n"
                "Turn it off to see the real cost of a frame."
            )

        imgui.same_line()

        if imgui.button("Reset"):
            profiler.reset()

        # -------------------------------------------------
        # Frame Graph
        # -------------------------------------------------

        frame_times = profiler.frame_times_ms

        average = profiler.frame_average_ms

        imgui.text(
            f"Frame: {average:6.2f} ms avg "
            f"({1000.0 / average if average > 0 else 0.0:5.1f} FPS)   "
            f"max {profiler.frame_max_ms:6.2f} ms   "
            f"GPU timing: {'on' if profiler.has_gpu_timing else 'unavailable'}"
        )

        if frame_times:

            imgui.plot_lines(
                "##frame_times",
                np.asarray(frame_times, dtype=np.float32),
                overlay_text=f"60 Hz budget = {self._BUDGET_60_HZ_MS:.1f} ms",
                scale_min=0.0,
                scale_max=max(
                    2.0 * self._BUDGET_60_HZ_MS,
                    profiler.frame_max_ms * 1.1
                ),
                graph_size=(-1, 60)
            )

        # -------------------------------------------------
        # Scope Table
        # -------------------------------------------------

        self._draw_scope_table(
            profiler,
            average
        )

        # -------------------------------------------------
        # cProfile Capture
        # -------------------------------------------------

        if imgui.collapsing_header(
            "Python profile (cProfile)"
        ):

            self._draw_cprofile_section(
                context.cprofile_capture
            )

        imgui.end()

    @staticmethod
    def _draw_scope_table(
        profiler: Profiler,
        frame_average_ms: float
    ):

        flags = (
            imgui.TableFlags_.borders.value
            | imgui.TableFlags_.row_bg.value
            | imgui.TableFlags_.resizable.value
        )

        if not imgui.begin_table("scopes", 6, flags):
            return

        imgui.table_setup_column("Scope", imgui.TableColumnFlags_.width_stretch)

        for label in ("CPU avg", "CPU max", "GPU avg", "% frame", "Calls"):

            imgui.table_setup_column(
                label,
                imgui.TableColumnFlags_.width_fixed,
                64.0
            )

        imgui.table_headers_row()

        for stat in profiler.stats():

            imgui.table_next_row()

            imgui.table_next_column()

            imgui.text(
                "  " * stat.depth + stat.name
            )

            if stat.name == "Swap" and imgui.is_item_hovered():

                imgui.set_tooltip(
                    "Buffer swap. Includes the VSync wait when VSync is on."
                )

            imgui.table_next_column()
            imgui.text(f"{stat.cpu_average_ms:7.3f}")

            imgui.table_next_column()
            imgui.text(f"{stat.cpu_max_ms:7.3f}")

            imgui.table_next_column()

            if stat.gpu_average_ms is None:
                imgui.text_disabled("      -")
            else:
                imgui.text(f"{stat.gpu_average_ms:7.3f}")

            imgui.table_next_column()

            share = (
                stat.cpu_average_ms / frame_average_ms
                if frame_average_ms > 0
                else 0.0
            )

            imgui.progress_bar(
                min(share, 1.0),
                (-1, 0),
                f"{share * 100.0:4.1f}%"
            )

            imgui.table_next_column()
            imgui.text(f"{stat.calls:5.1f}")

        imgui.end_table()

    def _draw_cprofile_section(
        self,
        capture: CProfileCapture
    ):

        imgui.text_disabled(
            "Records every Python call for N frames. Adds heavy\n"
            "overhead: compare proportions, not absolute times."
        )

        _, self._capture_frames = imgui.slider_int(
            "Frames",
            self._capture_frames,
            10,
            600
        )

        if capture.active:

            done, total = capture.progress

            imgui.progress_bar(
                done / total if total else 0.0,
                (-1, 0),
                f"Capturing {done}/{total}"
            )

        elif imgui.button("Capture"):

            capture.request(
                self._capture_frames
            )

        result = capture.last_result

        if result is None:
            return

        imgui.text(
            f"Last capture: {result.frames} frames"
        )

        imgui.text_disabled(
            f"{result.profile_path}\n{result.summary_path}"
        )

        flags = (
            imgui.TableFlags_.borders.value
            | imgui.TableFlags_.row_bg.value
            | imgui.TableFlags_.resizable.value
        )

        if not imgui.begin_table("cprofile", 3, flags):
            return

        imgui.table_setup_column("Function (by own time)", imgui.TableColumnFlags_.width_stretch)
        imgui.table_setup_column("Own ms/frame", imgui.TableColumnFlags_.width_fixed, 90.0)
        imgui.table_setup_column("Cum ms/frame", imgui.TableColumnFlags_.width_fixed, 90.0)

        imgui.table_headers_row()

        for label, cumulative_ms, own_ms in result.top_functions:

            imgui.table_next_row()

            imgui.table_next_column()
            imgui.text(label)

            imgui.table_next_column()
            imgui.text(f"{own_ms:8.3f}")

            imgui.table_next_column()
            imgui.text(f"{cumulative_ms:8.3f}")

        imgui.end_table()

    # =====================================================
    # Stats and Render Windows
    # =====================================================

    def _draw_stats_window(
        self,
        context: DebugContext
    ):

        self._panels.begin("Stats")

        timer = context.timer
        stats = context.stats

        fps = timer.fps

        imgui.text(
            f"FPS: {fps:6.1f}   "
            f"({1000.0 / fps if fps > 0 else 0.0:5.2f} ms)"
        )

        imgui.text(
            f"Draw calls: {stats.draw_calls}   "
            f"Triangles: {stats.triangles:,}"
        )

        imgui.text(
            f"Objects drawn: {stats.objects_drawn} / {stats.objects_total} "
            f"(after culling)"
        )

        imgui.text(
            f"Background jobs: {context.jobs.pending} pending "
            f"({context.jobs.workers} workers)"
        )

        planet = context.planet_stats

        if planet.planets:

            imgui.separator_text("Planet streaming")

            imgui.text(
                f"Chunks: {planet.chunks_drawn} drawn, "
                f"{planet.chunks_loaded} in memory"
            )

            if planet.chunks_building or planet.chunks_queued:

                imgui.text_colored(
                    (0.95, 0.8, 0.35, 1.0),
                    f"Building {planet.chunks_building}, "
                    f"{planet.chunks_queued} waiting"
                )

            else:

                imgui.text_disabled("Up to date")

        imgui.end()

    def _draw_engine_window(
        self,
        context: DebugContext
    ):

        self._panels.begin("Render")

        settings = context.settings

        # -------------------------------------------------
        # Output
        # -------------------------------------------------

        if imgui.collapsing_header(
            "Output",
            imgui.TreeNodeFlags_.default_open
        ):

            _slider(settings, "exposure", "Exposure", 0.05, 8.0, logarithmic=True)

            _checkbox(settings, "auto_exposure", "Auto exposure")

            imgui.set_item_tooltip(
                "Eye adaptation: brightens dim scenes (Titan, dusk) and "
                "darkens glaring ones, over about half a second."
            )

            if settings.auto_exposure:
                _slider(settings, "exposure_adaptation", "Adaptation", 0.0, 1.0)

            names = [
                tonemapper.name.title()
                for tonemapper in Tonemapper
            ]

            changed, index = imgui.combo(
                "Tone mapper",
                int(settings.tonemapper),
                names
            )

            if changed:
                settings.tonemapper = Tonemapper(index)

            _slider(settings, "gamma", "Gamma", 1.0, 3.0)

            _checkbox(settings, "fxaa_enabled", "FXAA anti-aliasing")

        # -------------------------------------------------
        # Bloom
        # -------------------------------------------------

        if imgui.collapsing_header("Bloom"):

            _checkbox(settings, "bloom_enabled", "Enabled##bloom")
            _slider(settings, "bloom_intensity", "Intensity##bloom", 0.0, 0.3, "%.3f")
            _slider(settings, "bloom_radius", "Radius##bloom", 0.001, 0.02, "%.4f")

        # -------------------------------------------------
        # Sky & IBL
        # -------------------------------------------------

        if imgui.collapsing_header("Sky & IBL"):

            _checkbox(settings, "show_sky", "Show sky")

            if not settings.show_sky:
                _color(settings, "clear_color", "Background")

            _color(settings, "sky_zenith_color", "Zenith")
            _color(settings, "sky_horizon_color", "Horizon")
            _color(settings, "ground_color", "Ground")
            _slider(settings, "sky_intensity", "Sky intensity", 0.0, 4.0)
            _slider(settings, "sun_size", "Sun size (deg)", 0.1, 5.0)
            _slider(settings, "ibl_intensity", "Ambient (IBL)", 0.0, 3.0)

            imgui.text_disabled(
                "The sun follows the directional light.\n"
                "Lighting re-bakes when the sky changes."
            )

            imgui.separator_text("Atmosphere")

            _checkbox(settings, "atmosphere_enabled", "Enabled##atmosphere")
            _checkbox(settings, "show_stars", "Stars and Milky Way")
            _checkbox(settings, "aerial_perspective", "Haze over terrain")

            changed, samples = imgui.slider_int(
                "Samples##atmosphere",
                settings.atmosphere_samples,
                4,
                64
            )

            if changed:
                settings.atmosphere_samples = samples

            imgui.text_disabled(
                "Planets with an Atmosphere component\n"
                "replace the sky settings above."
            )

        # -------------------------------------------------
        # Shadows
        # -------------------------------------------------

        if imgui.collapsing_header("Shadows"):

            _checkbox(settings, "shadows_enabled", "Enabled##shadows")

            _slider(settings, "shadow_distance", "Distance", 2.0, 150.0, "%.1f", logarithmic=True)

            changed, count = imgui.slider_int(
                "Cascades",
                int(settings.cascade_count),
                1,
                MAX_CASCADES
            )

            if changed:
                settings.cascade_count = count

            _slider(settings, "cascade_split_lambda", "Split lambda", 0.0, 1.0)

            _size_combo(settings, "shadow_map_size", "Cascade size")
            _size_combo(settings, "spot_shadow_map_size", "Spot map size")

            _slider(settings, "shadow_normal_offset", "Normal offset", 0.0, 5.0)
            _slider(settings, "shadow_depth_bias", "Depth bias", 0.0, 0.005, "%.5f")

            _checkbox(settings, "visualize_cascades", "Visualize cascades")

        # -------------------------------------------------
        # Debug
        # -------------------------------------------------

        if imgui.collapsing_header(
            "Debug",
            imgui.TreeNodeFlags_.default_open
        ):

            _, settings.show_light_gizmos = imgui.checkbox(
                "Light gizmos",
                settings.show_light_gizmos
            )

            _, settings.gizmo_scale = imgui.slider_float(
                "Gizmo scale",
                settings.gizmo_scale,
                0.02,
                0.5,
                "%.2f"
            )

            if imgui.button("Reload shaders (F5)"):

                count = context.reload_shaders()

                self._last_reload_message = (
                    f"Reloaded {count} shader(s)."
                )

            if self._last_reload_message:

                imgui.same_line()

                imgui.text_disabled(
                    self._last_reload_message
                )

        self._draw_materials_section(
            context
        )

        imgui.end()

    # =====================================================
    # Materials Section
    # =====================================================

    def _draw_materials_section(
        self,
        context: DebugContext
    ):

        # A section of the Render panel rather than its own
        # window, so it never overlaps the other panels.

        if not imgui.collapsing_header("Materials"):
            return

        defaults = Renderer.DEFAULT_MATERIAL_VALUES

        for key, material in context.resources.materials.items():

            imgui.push_id(
                key
            )

            if imgui.tree_node(
                key
            ):

                changed, color = imgui.color_edit3(
                    "Base color",
                    list(material.get_value("uBaseColor", defaults["uBaseColor"]))
                )

                if changed:
                    material.set_vec3("uBaseColor", color)

                changed, value = imgui.slider_float(
                    "Metallic",
                    material.get_value("uMetallic", defaults["uMetallic"]),
                    0.0,
                    1.0,
                    "%.2f"
                )

                if changed:
                    material.set_float("uMetallic", value)

                changed, value = imgui.slider_float(
                    "Roughness",
                    material.get_value("uRoughness", defaults["uRoughness"]),
                    0.0,
                    1.0,
                    "%.2f"
                )

                if changed:
                    material.set_float("uRoughness", value)

                changed, color = imgui.color_edit3(
                    "Emissive",
                    list(material.get_value("uEmissive", defaults["uEmissive"])),
                    imgui.ColorEditFlags_.hdr.value | imgui.ColorEditFlags_.float.value
                )

                if changed:
                    material.set_vec3("uEmissive", color)

                changed, value = imgui.slider_float(
                    "Occlusion",
                    material.get_value("uOcclusionStrength", defaults["uOcclusionStrength"]),
                    0.0,
                    1.0,
                    "%.2f"
                )

                if changed:
                    material.set_float("uOcclusionStrength", value)

                changed, value = imgui.slider_float(
                    "Normal strength",
                    material.get_value("uNormalStrength", defaults["uNormalStrength"]),
                    0.0,
                    3.0,
                    "%.2f"
                )

                if changed:
                    material.set_float("uNormalStrength", value)

                uv_scale = material.get_value(
                    "uUVScale",
                    defaults["uUVScale"]
                )

                changed, value = imgui.slider_float(
                    "UV scale",
                    uv_scale[0],
                    0.1,
                    20.0,
                    "%.2f"
                )

                if changed:
                    material.set_vec2("uUVScale", (value, value))

                texture_names = ", ".join(
                    name
                    for name, _ in material.textures
                )

                imgui.text_disabled(
                    f"Textures: {texture_names or 'defaults'}"
                )

                imgui.tree_pop()

            imgui.pop_id()


# =========================================================
# Settings Widgets
# =========================================================

def _slider(
    settings,
    field: str,
    label: str,
    minimum: float,
    maximum: float,
    fmt: str = "%.2f",
    logarithmic: bool = False
):

    changed, value = imgui.slider_float(
        label,
        float(getattr(settings, field)),
        minimum,
        maximum,
        fmt,
        imgui.SliderFlags_.logarithmic.value if logarithmic else 0
    )

    if changed:
        setattr(settings, field, value)


def _checkbox(
    settings,
    field: str,
    label: str
):

    changed, value = imgui.checkbox(
        label,
        bool(getattr(settings, field))
    )

    if changed:
        setattr(settings, field, value)


def _color(
    settings,
    field: str,
    label: str
):

    changed, value = imgui.color_edit3(
        label,
        list(getattr(settings, field)),
        imgui.ColorEditFlags_.hdr.value | imgui.ColorEditFlags_.float.value
    )

    if changed:
        setattr(settings, field, tuple(value))


_MAP_SIZES = [512, 1024, 2048, 4096]


def _size_combo(
    settings,
    field: str,
    label: str
):

    current = int(getattr(settings, field))

    index = (
        _MAP_SIZES.index(current)
        if current in _MAP_SIZES
        else 2
    )

    changed, index = imgui.combo(
        label,
        index,
        [str(size) for size in _MAP_SIZES]
    )

    if changed:
        setattr(settings, field, _MAP_SIZES[index])
