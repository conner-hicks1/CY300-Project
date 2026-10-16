from imgui_bundle import imgui

from game import format
from game.flight import FlightReadout
from game.time_warp import TimeWarp

from planet.orbits import date_of


PANEL_FLAGS = (
    imgui.WindowFlags_.no_decoration.value
    | imgui.WindowFlags_.no_inputs.value
    | imgui.WindowFlags_.no_docking.value
    | imgui.WindowFlags_.no_saved_settings.value
    | imgui.WindowFlags_.no_focus_on_appearing.value
    | imgui.WindowFlags_.no_nav.value
    | imgui.WindowFlags_.always_auto_resize.value
)

WARP_COLOR = (1.0, 0.82, 0.3, 1.0)
LOADING_COLOR = (0.95, 0.8, 0.35, 1.0)


# =========================================================
# Flight HUD
# =========================================================
#
# Over the 3D view, nothing to click:
#
#   top center    the clock: date and time warp
#   bottom left   where you are: body, altitude, speed, the
#                 air outside, position, local time
#   bottom right  key hints
#
# F2 hides it (screenshots).

class FlightHud:

    def __init__(self):

        self.visible = True

    def draw(
        self,
        readout: FlightReadout | None,
        speed: float,
        warp: TimeWarp,
        loading: bool = False,
        fps: float | None = None
    ):

        if not self.visible:
            return

        viewport = imgui.get_main_viewport()

        x, y = viewport.work_pos.x, viewport.work_pos.y
        width, height = viewport.work_size.x, viewport.work_size.y

        self._draw_clock(readout, warp, (x + width * 0.5, y + 10.0))

        self._draw_flight(readout, speed, loading, fps, (x + 10.0, y + height - 10.0))

        self._draw_hints((x + width - 10.0, y + height - 10.0))

    # -----------------------------------------------------
    # Clock
    # -----------------------------------------------------

    @staticmethod
    def _draw_clock(
        readout: FlightReadout | None,
        warp: TimeWarp,
        position
    ):

        imgui.set_next_window_pos(position, imgui.Cond_.always, (0.5, 0.0))
        imgui.set_next_window_bg_alpha(0.55)

        imgui.begin("##hud clock", None, PANEL_FLAGS)

        if readout is not None and readout.time is not None:
            imgui.text(date_of(readout.time).strftime("%Y-%m-%d  %H:%M:%S UTC"))
            imgui.same_line()

        if warp.paused or warp.warping:
            imgui.text_colored(WARP_COLOR, warp.label())
        else:
            imgui.text_disabled(warp.label())

        imgui.end()

    # -----------------------------------------------------
    # Flight
    # -----------------------------------------------------

    @staticmethod
    def _draw_flight(
        readout: FlightReadout | None,
        speed: float,
        loading: bool,
        fps: float | None,
        position
    ):

        imgui.set_next_window_pos(position, imgui.Cond_.always, (0.0, 1.0))
        imgui.set_next_window_bg_alpha(0.55)

        imgui.begin("##hud flight", None, PANEL_FLAGS)

        if readout is None:

            imgui.text_disabled("No body nearby.")

        else:

            imgui.text(readout.body)

            if loading:
                imgui.same_line()
                imgui.text_colored(LOADING_COLOR, "(loading terrain...)")

            if readout.above_ground is None:

                # Giants: no ground; cloud tops (below them,
                # air ever thicker and hotter).
                if readout.altitude >= 0.0:
                    imgui.text(f"Altitude  {format.distance(readout.altitude)} above the cloud tops")
                else:
                    imgui.text(f"Depth     {format.distance(-readout.altitude)} below the cloud tops")

            else:

                imgui.text(f"Altitude  {format.distance(readout.above_ground)} above ground")

                if abs(readout.above_ground - readout.altitude) > 1.0:
                    imgui.text_disabled(f"          {format.distance(readout.altitude)} above sea level")

            imgui.text(f"Speed     {format.distance(speed)}/s")

            if readout.pressure is not None:
                imgui.text(f"Air       {format.pressure(readout.pressure)}, {readout.temperature:,.0f} C")
            else:
                imgui.text_disabled("Air       vacuum")

            latitude, longitude = readout.latitude, readout.longitude

            imgui.text(
                f"Position  {abs(latitude):.2f} {'N' if latitude >= 0 else 'S'}  "
                f"{abs(longitude):.2f} {'E' if longitude >= 0 else 'W'}"
            )

            if readout.local_hour is not None:

                hour = readout.local_hour

                imgui.text(
                    f"Local     {int(hour):02d}:{int((hour % 1.0) * 60):02d}  "
                    f"(sun {readout.sun_elevation:+.0f} deg)"
                )

        if fps is not None:
            imgui.text_disabled(f"{fps:5.1f} FPS  ({1000.0 / fps if fps > 0 else 0.0:.1f} ms)")

        imgui.end()

    # -----------------------------------------------------
    # Hints
    # -----------------------------------------------------

    @staticmethod
    def _draw_hints(
        position
    ):

        imgui.set_next_window_pos(position, imgui.Cond_.always, (1.0, 1.0))
        imgui.set_next_window_bg_alpha(0.55)

        imgui.begin("##hud hints", None, PANEL_FLAGS)

        imgui.text_disabled("Esc  menu   |   , .  time warp   |   hold right mouse  look and fly")

        imgui.end()
