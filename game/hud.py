import math

from imgui_bundle import imgui

from game import format, style

from planet import weather
from game.flight import FlightReadout
from game.time_warp import TimeWarp

from planet.orbits import date_of


# =========================================================
# Flight HUD
# =========================================================
#
# Over the 3D view, nothing to click (game/style.py: one
# look for all of it):
#
#   top center    the clock: date, time, time warp
#   under it      notices, a few seconds each, colored by
#                 what they say
#   top right     "F1 controls"; F1: every key
#   bottom left   where you are: body, altitude, speed, the
#                 air outside, position, local time
#   bottom center the vessel: its name and situation, then
#                 propulsion (throttle, fuel, stage, thrust
#                 to weight, delta-v, mass), motion (speeds),
#                 orbit (apoapsis, periapsis), its state as
#                 chips (SAS, link, parachute), and what needs
#                 attention (a burn coming, the braking point,
#                 heat, a docking approach); beside it the
#                 attitude ball (game/navball.py)
#
# F2 hides it (screenshots).

NOTICE_SECONDS = 4.0

# Words that color a notice.
NOTICE_TONES = (
    (("lost", "destroyed", "burned up", "missed", "out of power", "torn off", "fell over", "failure"), "bad"),
    (("no link", "not enough", "cannot", "too far", "could not", "nothing", "no power"), "warn"),
    (("mission done", "recovered", "docked", "landed", "science", "transmitted", "budget", "mapped", "done", "power again"), "good"),
)

CONTROLS = (
    ("Z / X", "throttle full / cut"),
    ("Shift / Ctrl", "throttle up / down"),
    ("Space", "next stage"),
    ("W A S D Q E", "steer"),
    ("T / F", "SAS on / its mode"),
    ("arrows, H / N", "translate (RCS)"),
    ("M", "map"),
    ("R / K", "experiments / science"),
    ("C", "flight computer"),
    ("P", "programme"),
    ("L, [ ]", "vessels"),
    ("B", "auto-dock"),
    ("U", "solar wings"),
    ("G", "landing legs"),
    (", . /", "time warp"),
    ("V", "free camera"),
    ("Esc", "menu"),
    ("F1 / F2", "these keys / the HUD"),
)


# The landing legs, as a chip: (text, tone).
LEG_CHIPS = {
    "up": ("Legs up", "faint"),
    "lowering": ("Legs lowering", "accent"),
    "raising": ("Legs raising", "accent"),
    "down": ("Legs down", "good"),
}

# The solar wings, as a chip: (text, tone).
PANEL_CHIPS = {
    "folded": ("Wings folded", "faint"),
    "unfolding": ("Wings unfolding", "accent"),
    "folding": ("Wings folding", "accent"),
    "out": ("Wings out", "good"),
    "torn off": ("Wings torn off", "bad"),
}


def notice_tone(
    text: str
) -> str:

    lowered = text.lower()

    for words, tone in NOTICE_TONES:
        if any(word in lowered for word in words):
            return tone

    return "text"


class FlightHud:

    def __init__(self):

        self.visible = True

        # Every key listed (F1).
        self.show_controls = False

        # (text, seconds left), newest last.
        self.notices: list[list] = []

    def notify(
        self,
        text: str
    ):

        self.notices.append([text, NOTICE_SECONDS])

        del self.notices[:-4]

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
        link: tuple[str, str] | None = None,
        docking: tuple[str, str] | None = None,
        power=None
    ):
        """
        vessel: the vessel's readout (systems/vessel_system.py
        VesselReadout), if flying one. navball: a
        game/navball.py NavballFrame for the attitude ball.
        power: its vessel/power.py PowerStatus.
        """

        if not self.visible:
            return

        viewport = imgui.get_main_viewport()

        x, y = viewport.work_pos.x, viewport.work_pos.y
        width, height = viewport.work_size.x, viewport.work_size.y

        margin = style.MARGIN

        self._draw_clock(readout, warp, (x + width * 0.5, y + margin))

        self._draw_notices((x + width * 0.5, y + margin + 44.0))

        self._draw_flight(readout, speed, loading, fps, (x + margin, y + height - margin))

        try:
            self._draw_ship(vessel, navball, maneuver, braking, link, docking, power, x, y, width, height)
        finally:
            # The left side, between the navigation and the
            # location panels: clear of the rest.
            self._draw_hints((x + width - margin, y + margin), (x + margin, y + height * 0.45))

    def _draw_ship(
        self,
        vessel,
        navball,
        maneuver,
        braking,
        link,
        docking,
        power,
        x: float,
        y: float,
        width: float,
        height: float
    ):

        margin = style.MARGIN

        if vessel is None:
            return

        right, bottom = self._draw_vessel(vessel, (x + width * 0.5, y + height - margin), maneuver, braking, link, docking, power)

        if navball is not None and not vessel.destroyed:

            from game.navball import attitude, draw_navball

            radius = 64.0

            # Beside the vessel panel, its readout under it.
            center = (right + style.GAP + 14.0 + radius, bottom - 26.0 - radius)

            draw_navball(center, radius, navball)

            heading, pitch = attitude(navball.orientation, navball.up, navball.north)

            draw = imgui.get_foreground_draw_list()

            text = f"HDG {heading:03.0f}   PIT {pitch:+03.0f}"

            where = (center[0] - 52.0, center[1] + radius + 5.0)

            color = imgui.get_color_u32(style.tone("muted"))

            mono = style.FONTS.get("mono")

            # On a dark backing: it reads over bright ground.
            with style.font("mono"):
                measured = imgui.calc_text_size(text)

            size = (measured.x, measured.y)

            draw.add_rect_filled(
                (where[0] - 6.0, where[1] - 1.0),
                (where[0] + size[0] + 6.0, where[1] + size[1] + 1.0),
                imgui.get_color_u32(style.PANEL),
                4.0
            )

            if mono is not None:
                draw.add_text(mono[0], mono[1], where, color, text)
            else:
                draw.add_text(where, color, text)

        if vessel.destroyed:
            self._draw_banner(
                f"{vessel.name} burned up coming in: its {vessel.burned_up.lower()} overheated."
                if vessel.burned_up
                else (
                    f"{vessel.name} {vessel.fell}."
                    if vessel.fell
                    else f"{vessel.name} hit the ground at {vessel.impact_speed:,.0f} m/s and was destroyed."
                ),
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

        with style.panel("##hud clock", position, (0.5, 0.0)):

            if readout is not None and readout.time is not None:

                moment = date_of(readout.time)

                with style.font("bold"):
                    imgui.text(moment.strftime("%d %b %Y"))

                imgui.same_line()

                with style.font("mono"):
                    imgui.text_colored(style.tone("muted"), moment.strftime("%H:%M:%S UTC"))

                imgui.same_line()

            if warp.paused:
                style.chip("Paused", "warn")
            elif warp.warping:
                style.chip(warp.label(), "warn")
            else:
                style.chip("Real time", "faint")

            if not warp.paused and warp.capped:

                imgui.same_line()

                imgui.text_colored(style.tone("faint"), "max")

                imgui.set_item_tooltip(
                    "Faster time warp waits until the vessel is coasting above the air:\n"
                    "under thrust, in the air or close to the ground physics runs at most 5x."
                )

    # -----------------------------------------------------
    # Where you are
    # -----------------------------------------------------

    @staticmethod
    def _draw_flight(
        readout: FlightReadout | None,
        speed: float,
        loading: bool,
        fps: float | None,
        position
    ):

        with style.panel("##hud flight", position, (0.0, 1.0)):

            if readout is None:

                style.say("No body nearby.", "faint")

            else:

                with style.font("bold"):
                    imgui.text(readout.body)

                if loading:
                    style.chip("loading terrain", "warn", same_line=True)

                if readout.above_ground is None:

                    # Giants: no ground; cloud tops (below them,
                    # air ever thicker and hotter).
                    if readout.altitude >= 0.0:
                        style.row("Altitude", f"{format.distance(readout.altitude)} over clouds")
                    else:
                        style.row("Depth", f"{format.distance(-readout.altitude)} in clouds", "warn")

                else:

                    style.row("Altitude", format.distance(readout.above_ground))

                    if abs(readout.above_ground - readout.altitude) > 1.0:
                        style.row("Sea level", format.distance(readout.altitude), "muted")

                style.row("Speed", f"{format.distance(speed)}/s")

                if readout.pressure is not None:
                    style.row("Air", f"{format.pressure(readout.pressure)}, {readout.temperature:,.0f} C")
                else:
                    style.row("Air", "vacuum", "faint")

                latitude, longitude = readout.latitude, readout.longitude

                style.row(
                    "Position",
                    f"{abs(latitude):.2f} {'N' if latitude >= 0 else 'S'}  {abs(longitude):.2f} {'E' if longitude >= 0 else 'W'}"
                )

                if readout.local_hour is not None:

                    hour = readout.local_hour

                    style.row("Local", f"{int(hour):02d}:{int((hour % 1.0) * 60):02d}  sun {readout.sun_elevation:+.0f} deg")

            if fps is not None:
                style.row("Frame", f"{fps:5.1f} FPS  {1000.0 / fps if fps > 0 else 0.0:.1f} ms", "faint")

    # -----------------------------------------------------
    # Keys
    # -----------------------------------------------------

    def _draw_hints(
        self,
        corner,
        middle
    ):

        if not self.show_controls:

            with style.panel("##hud hints", corner, (1.0, 0.0), alpha=0.6):
                imgui.text_colored(style.tone("faint"), "F1  controls")

            return

        # Every key, two columns (F1 again to close).
        with style.panel("##hud hints", middle, (0.0, 0.5), alpha=0.97):

            style.heading("Controls  (F1 to close)")

            half = (len(CONTROLS) + 1) // 2

            if imgui.begin_table("##hud keys", 4, imgui.TableFlags_.sizing_fixed_fit.value):

                for index in range(half):

                    imgui.table_next_row()

                    for keys, what in CONTROLS[index::half]:

                        imgui.table_next_column()

                        with style.font("mono"):
                            imgui.text_colored(style.tone("accent"), keys)

                        imgui.table_next_column()

                        imgui.text_colored(style.tone("muted"), f"{what}      ")

                imgui.end_table()

    # -----------------------------------------------------
    # Vessel
    # -----------------------------------------------------

    @staticmethod
    def _draw_vessel(
        vessel,
        position,
        maneuver: tuple[float, float, float] | None = None,
        braking=None,
        link: tuple[str, str] | None = None,
        docking: tuple[str, str] | None = None,
        power=None
    ):
        """
        maneuver: the node's (m/s left, s until its burn
        starts, burn s); braking: vessel/maneuver.py Braking;
        link, docking: (text, "good" / "warn" / "bad");
        power: vessel/power.py PowerStatus.
        Returns the panel's right edge and bottom (px).
        """

        with style.panel("##hud vessel", position, (0.5, 1.0)):

            where = imgui.get_window_pos()
            size = imgui.get_window_size()

            with style.font("large"):
                imgui.text(vessel.name)

            imgui.same_line()

            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 3.0)

            style.chip(vessel.situation, style.SITUATIONS.get(vessel.situation, "muted"))

            imgui.same_line()

            imgui.text_colored(style.tone("muted"), f"at {vessel.body}")

            flags = imgui.TableFlags_.sizing_fixed_fit.value

            if imgui.begin_table("vessel", 3, flags):

                imgui.table_setup_column("propulsion", imgui.TableColumnFlags_.width_fixed.value, 214.0)
                imgui.table_setup_column("motion", imgui.TableColumnFlags_.width_fixed.value, 164.0)
                imgui.table_setup_column("orbit", imgui.TableColumnFlags_.width_fixed.value, 164.0)

                imgui.table_next_row()

                # Propulsion.
                imgui.table_next_column()

                style.heading("Propulsion")

                style.meter("Throttle", vessel.throttle, f"{vessel.throttle * 100.0:.0f}%", "accent", 124.0, 74.0)

                share = vessel.propellant / vessel.propellant_capacity if vessel.propellant_capacity > 0.0 else 0.0

                style.meter(
                    "Fuel",
                    share,
                    f"{vessel.propellant / 1000.0:,.1f} t {vessel.propellant_kind}".rstrip(),
                    "warn" if share < 0.15 else "money",
                    124.0,
                    74.0
                )

                if vessel.stages_left:
                    style.row("Stage", f"{vessel.next_stage}  ({vessel.stages_left} left)", label_width=74.0)
                else:
                    style.row("Stage", "none left", "faint", label_width=74.0)

                weak = 0.0 < vessel.twr < 1.0 and vessel.situation in ("Landed", "Flying")

                style.row("TWR", f"{vessel.twr:.2f}", "warn" if weak else None, label_width=74.0)
                style.row("Delta-v", f"{vessel.delta_v:,.0f} m/s", label_width=74.0)
                style.row("Mass", f"{vessel.mass / 1000.0:,.1f} t", label_width=74.0)

                # Cryogenic propellant boiling away.
                if power is not None and power.boiloff >= 0.5:
                    style.row("Boil-off", f"{power.boiloff:,.0f} kg/day", "warn", label_width=74.0)

                # Motion.
                imgui.table_next_column()

                style.heading("Motion")

                falling = vessel.vertical_speed < -50.0 and vessel.situation == "Flying"

                style.row("Surface", f"{format.distance(vessel.surface_speed)}/s", label_width=66.0)
                style.row("Orbital", f"{format.distance(vessel.orbital_speed)}/s", label_width=66.0)
                style.row("Vertical", f"{vessel.vertical_speed:+,.0f} m/s", "warn" if falling else None, label_width=66.0)

                if power is not None:
                    FlightHud._draw_power(power)

                # Orbit.
                imgui.table_next_column()

                style.heading("Orbit")

                if vessel.periapsis is None:

                    style.row("Apoapsis", "-", "faint", label_width=78.0)
                    style.row("Periapsis", "-", "faint", label_width=78.0)

                else:

                    if vessel.apoapsis is not None:
                        style.row("Apoapsis", format.distance(vessel.apoapsis), label_width=78.0)
                    else:
                        style.row("Apoapsis", "escaping", "science", label_width=78.0)

                    if vessel.periapsis < -1.0e5:
                        style.row("Periapsis", "underground", "warn", label_width=78.0)
                    else:
                        style.row("Periapsis", format.distance(vessel.periapsis), "warn" if vessel.periapsis < 0.0 else None, label_width=78.0)

                imgui.end_table()

            # State, as chips.
            style.chip(f"SAS {vessel.sas_mode}" if vessel.sas else "SAS off", "accent" if vessel.sas else "warn")

            if link is not None:

                text, tone = link

                style.chip(text, tone, same_line=True)

            if vessel.parachute == "armed":
                style.chip("Parachute armed", "warn", same_line=True)
            elif vessel.parachute == "open":
                style.chip("Parachute open", "good", same_line=True)

            legs = LEG_CHIPS.get(vessel.legs)

            if legs is not None:
                style.chip(legs[0], legs[1], same_line=True)

            if vessel.toppled:
                style.chip("On its side", "bad", same_line=True)

            if vessel.failed:
                style.chip(f"Failed: {vessel.failed[0].lower()}" + (f" +{len(vessel.failed) - 1}" if len(vessel.failed) > 1 else ""), "bad", same_line=True)

            sky = vessel.weather

            if sky is not None and sky.storm:
                style.chip("Global dust storm" if sky.storm.startswith("Global") else "Dust storm", "bad" if sky.tau >= 5.0 else "warn", same_line=True)

            if power is not None:

                if not power.powered:
                    style.chip("No power", "bad", same_line=True)

                wings = PANEL_CHIPS.get(power.panels)

                if wings is not None:
                    style.chip(wings[0], wings[1], same_line=True)

            # What needs attention.
            if maneuver is not None:

                left, start_in, burn = maneuver

                if start_in > 0.0:
                    style.say(f"Burn {left:,.1f} m/s in {format.countdown(start_in)}", "warn" if start_in < 30.0 else "accent")
                else:
                    style.say(f"Burn now: {left:,.1f} m/s left ({format.countdown(burn)})", "warn")

            if braking is not None and math.isfinite(braking.start_in):

                urgent = braking.start_in < 10.0

                style.say(
                    f"Brake in {format.countdown(braking.start_in)} at {format.distance(braking.height)}" if braking.start_in > 0.5 else "BRAKE NOW",
                    "bad" if urgent else "warn"
                )

                imgui.same_line()

                imgui.text_colored(style.tone("faint"), f"ground in {format.countdown(braking.impact_in)}")

            elif braking is not None:

                style.say("Engines cannot stop the fall", "bad")

            # Mars's weather, low or landed there.
            if sky is not None:

                style.say(
                    f"Mars Ls {sky.longitude:,.0f} ({sky.season.split(' / ')[0]}): dust {sky.tau:.1f}, "
                    f"sunlight {sky.light:.0%}, wind {sky.wind:,.0f} m/s from {weather.wind_from(sky)}",
                    "bad" if sky.tau >= 5.0 else ("warn" if sky.tau >= 2.0 else "muted")
                )

            # The ground below: will it stand there?
            if vessel.slope is not None and not vessel.destroyed and not vessel.toppled:

                steep = vessel.slope > vessel.standing

                if vessel.situation == "Landed":
                    style.say(f"Standing on a {vessel.slope:,.0f} deg slope (holds {vessel.standing:,.0f})", "muted")
                else:

                    style.say(
                        f"Ground below: {vessel.slope:,.0f} deg slope; stands on {vessel.standing:,.0f}"
                        + ("  (too steep)" if steep else ""),
                        "warn" if steep else "muted"
                    )

                    if vessel.legs in ("up", "raising"):

                        imgui.same_line()

                        imgui.text_colored(style.tone("warn"), "  legs up: G")

            heat = vessel.heat

            if heat is not None and heat[0] > 0.35 * heat[1]:

                share = heat[0] / heat[1]

                style.meter("Heat", share, f"{heat[0]:,.0f} / {heat[1]:,.0f} K  {heat[2]}", "bad" if share > 0.85 else "warn", 300.0, 74.0)

            if docking is not None:

                text, tone = docking

                style.say(text, tone)

        return where.x + size.x, where.y + size.y

    @staticmethod
    def _draw_power(
        power
    ):
        """Under Motion: the batteries, what comes in or goes out, the Sun."""

        imgui.spacing()

        style.heading("Power")

        if power.capacity > 0.0:

            share = power.share

            style.meter(
                "Charge",
                share,
                format.energy(power.charge),
                "bad" if share < 0.1 else ("warn" if share < 0.3 else "good"),
                88.0,
                66.0
            )

        net = power.net

        # (Full and making more than it draws: steady.)
        full = power.capacity > 0.0 and power.share > 0.999 and net > 0.0

        style.row(
            "Net",
            f"{'+' if net >= 0.0 else '-'}{format.watts(abs(net))}" + ("  full" if full else ""),
            "good" if net >= 0.0 else ("bad" if power.share < 0.1 else "warn"),
            label_width=66.0
        )

        # The Sun's strength here (1 AU: 100%), and how much of
        # the time it shines (in time warp, round the orbit).
        if power.sunlight <= 0.001:
            sun = "in shadow"
        elif power.sunlight >= 0.999:
            sun = f"{power.flux:.0%}"
        else:
            sun = f"{power.flux:.0%}, {power.sunlight:.0%} lit"

        style.row("Sun", sun, "faint" if power.sunlight <= 0.001 else "muted", label_width=66.0)

    # -----------------------------------------------------
    # Notices
    # -----------------------------------------------------

    def _draw_notices(
        self,
        position
    ):

        if not self.notices:
            return

        with style.panel("##hud notices", position, (0.5, 0.0), alpha=0.7):

            for text, left in self.notices:

                color = style.tone(notice_tone(text))

                imgui.text_colored((color[0], color[1], color[2], min(1.0, left)), text)

    @staticmethod
    def _draw_banner(
        message: str,
        hint: str,
        position
    ):

        with style.panel("##hud banner", position, (0.5, 0.5), alpha=0.88):

            with style.font("large"):
                imgui.text_colored(style.tone("bad"), message)

            imgui.text_colored(style.tone("muted"), hint)
