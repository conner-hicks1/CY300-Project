import math

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
GOOD_COLOR = (0.5, 1.0, 0.6, 1.0)
WARN_COLOR = (1.0, 0.82, 0.3, 1.0)
BAD_COLOR = (1.0, 0.45, 0.35, 1.0)

SITUATION_COLORS = {
    "Landed": (0.6, 0.85, 0.6, 1.0),
    "Flying": (0.55, 0.75, 1.0, 1.0),
    "Sub-orbital": (1.0, 0.82, 0.3, 1.0),
    "Orbiting": (0.5, 1.0, 0.7, 1.0),
    "Escaping": (1.0, 0.6, 1.0, 1.0),
    "Flyby": (1.0, 0.6, 1.0, 1.0),
    "Destroyed": (1.0, 0.45, 0.35, 1.0),
}
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
#   bottom center the vessel: situation, throttle, stage,
#                 propellant, thrust to weight, speeds, the
#                 orbit's apoapsis and periapsis; beside it
#                 the attitude ball (game/navball.py)
#   top right     key hints
#   below clock   notices (a stage separating, a new sphere
#                 of influence...), a few seconds each
#
# F2 hides it (screenshots).

NOTICE_SECONDS = 4.0


class FlightHud:

    def __init__(self):

        self.visible = True

        # (text, seconds left), newest last.
        self.notices: list[list] = []

    def notify(
        self,
        text: str
    ):

        self.notices.append([text, NOTICE_SECONDS])

        del self.notices[:-3]

    def tick(
        self,
        delta_time: float
    ):

        for notice in self.notices:
            notice[1] -= delta_time

        self.notices = [notice for notice in self.notices if notice[1] > 0.0]

    def draw(
        self,
        readout: FlightReadout | None,
        speed: float,
        warp: TimeWarp,
        loading: bool = False,
        fps: float | None = None,
        vessel=None,
        navball=None,
        maneuver: tuple[float, float, float] | None = None,
        braking=None,
        link: tuple[str, str] | None = None
    ):
        """
        vessel: the vessel's readout (systems/vessel_system.py
        VesselReadout), if flying one. navball: (orientation,
        up, north, motion) for the attitude ball.
        """

        if not self.visible:
            return

        viewport = imgui.get_main_viewport()

        x, y = viewport.work_pos.x, viewport.work_pos.y
        width, height = viewport.work_size.x, viewport.work_size.y

        self._draw_clock(readout, warp, (x + width * 0.5, y + 10.0))

        self._draw_flight(readout, speed, loading, fps, (x + 10.0, y + height - 10.0))

        self._draw_hints((x + width - 10.0, y + 10.0), vessel is not None)

        self._draw_notices((x + width * 0.5, y + 48.0))

        if vessel is not None:

            self._draw_vessel(vessel, (x + width * 0.5, y + height - 10.0), maneuver, braking, link)

            if navball is not None and not vessel.destroyed:

                from game.navball import attitude, draw_navball

                radius = 62.0

                center = (x + width * 0.5 + 290.0 + radius, y + height - 22.0 - radius)

                draw_navball(center, radius, *navball)

                heading, pitch = attitude(*navball[:3])

                draw = imgui.get_foreground_draw_list()

                draw.add_text(
                    (center[0] - 46.0, center[1] + radius + 4.0),
                    imgui.get_color_u32((0.9, 0.9, 0.9, 1.0)),
                    f"HDG {heading:03.0f}  PIT {pitch:+.0f}"
                )

            if vessel.destroyed:
                self._draw_banner(
                    f"{vessel.name} burned up coming in: its {vessel.burned_up.lower()} overheated."
                    if vessel.burned_up
                    else f"{vessel.name} hit the ground at {vessel.impact_speed:,.0f} m/s and was destroyed.",
                    "Esc > Revert to launch",
                    (x + width * 0.5, y + height * 0.3)
                )

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

        if not warp.paused and warp.capped:

            imgui.same_line()

            imgui.text_disabled("(max)")

            imgui.set_item_tooltip(
                "Faster time warp waits until the vessel is coasting above the air:\n"
                "under thrust, in the air or close to the ground physics runs at most 5x."
            )

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
        position,
        piloting: bool
    ):

        imgui.set_next_window_pos(position, imgui.Cond_.always, (1.0, 0.0))
        imgui.set_next_window_bg_alpha(0.55)

        imgui.begin("##hud hints", None, PANEL_FLAGS)

        if piloting:
            imgui.text_disabled("Z / X  throttle   |   Space  stage   |   W A S D Q E  steer")
            imgui.text_disabled("T  SAS   |   F  SAS mode   |   M  map   |   V  free camera")
            imgui.text_disabled("R  run experiments   |   K  science   |   C  flight computer   |   P  programme")
            imgui.text_disabled("Esc  menu   |   , .  time warp   |   right mouse  look around")
        else:
            imgui.text_disabled("Esc  menu   |   , .  time warp   |   hold right mouse  look and fly")

        imgui.end()

    # -----------------------------------------------------
    # Vessel
    # -----------------------------------------------------

    @staticmethod
    def _draw_vessel(
        vessel,
        position,
        maneuver: tuple[float, float, float] | None = None,
        braking=None,
        link: tuple[str, str] | None = None
    ):
        """
        maneuver: the node's (m/s left, s until its burn
        starts, burn s); braking: vessel/maneuver.py Braking;
        link: (text, "good" / "warn" / "bad").
        """

        imgui.set_next_window_pos(position, imgui.Cond_.always, (0.5, 1.0))
        imgui.set_next_window_bg_alpha(0.55)

        imgui.begin("##hud vessel", None, PANEL_FLAGS)

        imgui.text(vessel.name)
        imgui.same_line()
        imgui.text_colored(SITUATION_COLORS.get(vessel.situation, (0.8, 0.8, 0.8, 1.0)), vessel.situation)
        imgui.same_line()
        imgui.text_disabled(f"at {vessel.body}")

        if link is not None:

            text, tone = link

            imgui.text_colored({"good": (0.6, 0.85, 0.6, 1.0), "warn": WARN_COLOR, "bad": BAD_COLOR}[tone], text)

        if imgui.begin_table("vessel", 2):

            imgui.table_setup_column("left", imgui.TableColumnFlags_.width_fixed.value, 210.0)
            imgui.table_setup_column("right", imgui.TableColumnFlags_.width_fixed.value, 230.0)

            imgui.table_next_row()
            imgui.table_next_column()

            imgui.text("Throttle")
            imgui.same_line(80.0)
            imgui.progress_bar(vessel.throttle, (120.0, 0.0), f"{vessel.throttle * 100.0:.0f}%")

            share = vessel.propellant / vessel.propellant_capacity if vessel.propellant_capacity > 0.0 else 0.0

            imgui.text("Fuel")
            imgui.same_line(80.0)
            imgui.progress_bar(share, (120.0, 0.0), f"{vessel.propellant / 1000.0:,.1f} t")

            if vessel.stages_left:
                imgui.text(f"Stage     {vessel.next_stage}  ({vessel.stages_left} left: Space)")
            else:
                imgui.text_disabled("Stage     none left")

            imgui.text(f"TWR       {vessel.twr:.2f}")
            imgui.text(f"Delta-v   {vessel.delta_v:,.0f} m/s")
            imgui.text(f"Mass      {vessel.mass / 1000.0:,.1f} t")

            imgui.table_next_column()

            imgui.text(f"Surface   {format.distance(vessel.surface_speed)}/s")
            imgui.text(f"Orbital   {format.distance(vessel.orbital_speed)}/s")
            imgui.text(f"Vertical  {vessel.vertical_speed:+,.0f} m/s")

            if vessel.periapsis is None:

                imgui.text_disabled("Apoapsis  -")
                imgui.text_disabled("Periapsis -")

            else:

                imgui.text(
                    f"Apoapsis  {format.distance(vessel.apoapsis)}"
                    if vessel.apoapsis is not None
                    else "Apoapsis  escaping"
                )

                if vessel.periapsis < -1.0e5:
                    imgui.text_disabled("Periapsis below ground")
                else:
                    imgui.text(f"Periapsis {format.distance(vessel.periapsis)}")

            if vessel.sas:
                imgui.text(f"SAS {vessel.sas_mode} (T, F)")
            else:
                imgui.text_colored(WARP_COLOR, "SAS off (T)")

            if maneuver is not None:

                left, start_in, burn = maneuver

                imgui.text_colored(GOOD_COLOR, f"Maneuver {left:,.1f} m/s")

                if start_in > 0.0:
                    imgui.text_colored(WARN_COLOR if start_in < 30.0 else GOOD_COLOR, f"  burn in {format.countdown(start_in)}")
                else:
                    imgui.text_colored(WARN_COLOR, f"  burn now ({format.countdown(burn)})")

            if braking is not None and math.isfinite(braking.start_in):

                urgent = braking.start_in < 10.0

                imgui.text_colored(
                    BAD_COLOR if urgent else WARN_COLOR,
                    f"Brake in {format.countdown(braking.start_in)}" if braking.start_in > 0.5 else "BRAKE NOW"
                )

                imgui.text_disabled(f"  at {format.distance(braking.height)}; ground in {format.countdown(braking.impact_in)}")

            elif braking is not None:

                imgui.text_colored(BAD_COLOR, "Engines cannot stop the fall")

            heat = vessel.heat

            if heat is not None and heat[0] > 0.35 * heat[1]:

                share = heat[0] / heat[1]

                imgui.text_colored(
                    BAD_COLOR if share > 0.85 else WARN_COLOR if share > 0.6 else (0.85, 0.85, 0.85, 1.0),
                    f"Heat {heat[0]:,.0f} / {heat[1]:,.0f} K"
                )

                imgui.text_disabled(f"  {heat[2]}")

            if vessel.parachute == "armed":
                imgui.text_colored(WARP_COLOR, "Parachute armed")
            elif vessel.parachute == "open":
                imgui.text_colored((0.6, 0.85, 0.6, 1.0), "Parachute open")

            imgui.end_table()

        imgui.end()

    def _draw_notices(
        self,
        position
    ):

        if not self.notices:
            return

        imgui.set_next_window_pos(position, imgui.Cond_.always, (0.5, 0.0))
        imgui.set_next_window_bg_alpha(0.45)

        imgui.begin("##hud notices", None, PANEL_FLAGS)

        for text, left in self.notices:
            imgui.text_colored((1.0, 0.92, 0.6, min(1.0, left)), text)

        imgui.end()

    @staticmethod
    def _draw_banner(
        message: str,
        hint: str,
        position
    ):

        imgui.set_next_window_pos(position, imgui.Cond_.always, (0.5, 0.5))
        imgui.set_next_window_bg_alpha(0.75)

        imgui.begin("##hud banner", None, PANEL_FLAGS)

        imgui.text_colored((1.0, 0.45, 0.35, 1.0), message)
        imgui.text_disabled(hint)

        imgui.end()
