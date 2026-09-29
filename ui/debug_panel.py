from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from imgui_bundle import imgui

from core.cprofile_capture import CProfileCapture
from core.profiler import Profiler
from core.timer import Timer
from core.window import Window

from graphics.render_settings import (
    RenderSettings,
    Tonemapper
)
from graphics.renderer import (
    Renderer,
    RenderStats
)

from resources.resources import Resources


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


class DebugPanel:

    # =====================================================
    # Construction
    # =====================================================

    def __init__(self):

        self.visible = True

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

        self._draw_engine_window(
            context
        )

        self._draw_profiler_window(
            context
        )

    # =====================================================
    # Profiler Window
    # =====================================================

    # Refresh-rate reference lines for the frame graph.
    _BUDGET_60_HZ_MS = 1000.0 / 60.0

    def _draw_profiler_window(
        self,
        context: DebugContext
    ):

        display = imgui.get_io().display_size

        imgui.set_next_window_pos(
            (350, max(10.0, display.y - 310)),
            imgui.Cond_.first_use_ever
        )

        imgui.set_next_window_size(
            (max(460.0, display.x - 700), 300),
            imgui.Cond_.first_use_ever
        )

        # Starts collapsed so it does not cover the middle
        # of the viewport; the layout file remembers it.
        imgui.set_next_window_collapsed(
            True,
            imgui.Cond_.first_use_ever
        )

        imgui.begin("Profiler")

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
    # Engine Window
    # =====================================================

    def _draw_engine_window(
        self,
        context: DebugContext
    ):

        imgui.set_next_window_pos(
            (10, imgui.get_frame_height() + 4.0),
            imgui.Cond_.first_use_ever
        )

        imgui.set_next_window_size(
            (330, 0),
            imgui.Cond_.first_use_ever
        )

        imgui.begin("Engine")

        timer = context.timer
        stats = context.stats
        settings = context.settings

        fps = timer.fps

        imgui.text(
            f"FPS: {fps:6.1f}   "
            f"({1000.0 / fps if fps > 0 else 0.0:5.2f} ms)"
        )

        imgui.text(
            f"Draw calls: {stats.draw_calls}   "
            f"Triangles: {stats.triangles}"
        )

        imgui.text_disabled(
            "Hold RMB + WASD/QE: fly  |  Click: select\n"
            "Q/W/E/R: select/move/rotate/scale  |  F: focus\n"
            "Ctrl+Z/Y: undo/redo  |  Ctrl+S: save  |  Ctrl+P: play\n"
            "F1: UI  |  F5: reload shaders"
        )

        # -------------------------------------------------
        # Output
        # -------------------------------------------------

        if imgui.collapsing_header(
            "Output",
            imgui.TreeNodeFlags_.default_open
        ):

            _, settings.exposure = imgui.slider_float(
                "Exposure",
                settings.exposure,
                0.05,
                8.0,
                "%.2f"
            )

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

            _, settings.gamma = imgui.slider_float(
                "Gamma",
                settings.gamma,
                1.0,
                3.0,
                "%.2f"
            )

            changed, color = imgui.color_edit3(
                "Background",
                list(settings.clear_color)
            )

            if changed:
                settings.clear_color = tuple(color)

        # -------------------------------------------------
        # Shadows
        # -------------------------------------------------

        if imgui.collapsing_header(
            "Shadows",
            imgui.TreeNodeFlags_.default_open
        ):

            _, settings.shadows_enabled = imgui.checkbox(
                "Enabled",
                settings.shadows_enabled
            )

            sizes = [512, 1024, 2048, 4096]

            current = (
                sizes.index(settings.shadow_map_size)
                if settings.shadow_map_size in sizes
                else 2
            )

            changed, index = imgui.combo(
                "Map size",
                current,
                [str(size) for size in sizes]
            )

            if changed:
                settings.shadow_map_size = sizes[index]

            _, settings.shadow_extent = imgui.slider_float(
                "Extent",
                settings.shadow_extent,
                1.0,
                50.0,
                "%.1f"
            )

            changed, center = imgui.drag_float3(
                "Center",
                list(settings.shadow_center),
                0.05
            )

            if changed:
                settings.shadow_center = tuple(center)

            _, settings.shadow_bias_min = imgui.slider_float(
                "Bias min",
                settings.shadow_bias_min,
                0.0,
                0.01,
                "%.5f"
            )

            _, settings.shadow_bias_max = imgui.slider_float(
                "Bias max",
                settings.shadow_bias_max,
                0.0,
                0.05,
                "%.4f"
            )

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

        # A section of the Engine window rather than its own
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
                    "Specular",
                    material.get_value("uSpecularStrength", defaults["uSpecularStrength"]),
                    0.0,
                    2.0,
                    "%.2f"
                )

                if changed:
                    material.set_float("uSpecularStrength", value)

                changed, value = imgui.slider_float(
                    "Shininess",
                    material.get_value("uShininess", defaults["uShininess"]),
                    1.0,
                    512.0,
                    "%.0f",
                    imgui.SliderFlags_.logarithmic
                )

                if changed:
                    material.set_float("uShininess", value)

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

