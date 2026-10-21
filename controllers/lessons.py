import numpy as np

from programme import fleet

from vessel import placement, power
from vessel.parts import load_design

from game.lessons import lesson as lesson_of


# =========================================================
# The Application: Flight School's Lessons
# =========================================================
#
# A lesson (game/lessons.py) starts in a sandbox nothing is
# kept of, its vessels put where it begins:
#
#   pad       Pathfinder on the launch pad
#   orbit     Pathfinder's upper stage, 250 km round the Earth
#   docking   the Station Core 300 km up, a ferry 120 m off
#             its port, drifting with it
#   landing   Mariner's lander 10 km over the Moon's near
#             side, falling
#   power     Mariner's lander 5,000 km round the Earth, its
#             wings folded, its battery at 15%
#   comms     Mariner's lander 500 km round Mars, wings out,
#             on an orbit round the axis pointing home (never
#             behind Mars from the Earth)
#
# A part of Application (application.py), apart for its
# size: its methods run on the application (self is it).

class LessonsController:

    # (LESSON_: the application's own names are not these.)

    LESSON_ORBIT_ALTITUDE = 250_000.0
    LESSON_STATION_ALTITUDE = 300_000.0
    LESSON_DOCKING_DISTANCE = 120.0
    LESSON_LANDING_HEIGHT = 10_000.0
    LESSON_LANDING_SPEED = 40.0
    # (High enough to see a ground station: a probe obeys
    # only with a link.)
    LESSON_POWER_ALTITUDE = 5_000_000.0
    LESSON_POWER_CHARGE = 0.15
    LESSON_MARS_ALTITUDE = 500_000.0

    def start_lesson(
        self,
        lesson_id: str
    ):
        """Flight school: a lesson, from where it starts."""

        lesson = lesson_of(lesson_id)

        self.title.close()

        # (No hints over a lesson: it says it all.)
        self.hints.dismiss()

        self.open_game(None, folder=None, sandbox=True)

        self._docked = False

        getattr(self, f"_set_up_{lesson.setup}")()

        self.school.start(lesson)

    def start_flight_school(self):
        """The first lesson: to orbit from the pad."""

        self.start_lesson("first_orbit")

    def lesson_again(self):

        if self.school.lesson is not None:
            self.start_lesson(self.school.lesson.id)

    # -----------------------------------------------------
    # Where they start
    # -----------------------------------------------------

    def _set_up_pad(self):

        pass

    def _lesson_vessel(
        self,
        design: str,
        stages: int
    ):
        """A design on the pad, fired through its first stages (paid: a sandbox's are free); the one flown."""

        self._launch_design(load_design(design, self.parts))

        _, component = self.vessel_system.find(self.scene)

        self.programme_system.pay_launch(component)

        for _ in range(stages):
            self.stage_vessel(component)

        component.vessel.throttle = 0.0

        return component

    def _body(
        self,
        profile_id: str
    ):

        return fleet.body_profiles(self.scene)[profile_id]

    def _earthward(
        self,
        body
    ) -> np.ndarray:
        """The inertial direction from a body to the Earth (a probe's link home)."""

        states = self.orbit_system.states

        way = states[self._body("earth")][0] - states[body][0]

        return way / float(np.linalg.norm(way))

    def _in_orbit(
        self,
        component,
        profile_id: str,
        altitude: float,
        facing_home: bool = False
    ):
        """A circular orbit; facing_home: round the axis pointing at the Earth (never behind the body from home)."""

        body = self._body(profile_id)

        reference = self.vessel_system.soi_bodies(self.scene)[body]

        component.body = body

        if facing_home:

            pole = self._earthward(body)

            direction = np.cross(pole, [0.0, 1.0, 0.0])

            if float(np.linalg.norm(direction)) < 1e-6:
                direction = np.cross(pole, [1.0, 0.0, 0.0])

            placement.circular_orbit(component.state, reference.mu, reference.radius + altitude, direction, pole)

        else:
            placement.circular_orbit(component.state, reference.mu, reference.radius + altitude)

        self.vessel_system.placed(component.state)

    def _set_up_orbit(self):

        component = self._lesson_vessel("pathfinder", 2)

        self._in_orbit(component, "earth", self.LESSON_ORBIT_ALTITUDE)

        component.controls.sas = True
        component.controls.sas_mode = "hold"

        self.hud.notices.clear()

    def _set_up_docking(self):

        station = self._lesson_vessel("station_core", 4)

        self._in_orbit(station, "earth", self.LESSON_STATION_ALTITUDE)

        station.controls.sas = True
        station.controls.sas_mode = "hold"

        station_entity = self.vessel_system.active

        ferry = self._lesson_vessel("ferry", 2)

        # Off the station's port, a little aside, drifting
        # with it.
        _, (point, face) = next(iter(self.vessel_system.port_points(station)))

        aside = np.cross(face, [0.0, 1.0, 0.0])

        if float(np.linalg.norm(aside)) < 1e-6:
            aside = np.cross(face, [1.0, 0.0, 0.0])

        aside = aside / np.linalg.norm(aside)

        ferry.body = station.body
        ferry.state.position = point + face * self.LESSON_DOCKING_DISTANCE + aside * 8.0
        ferry.state.velocity = station.state.velocity.copy()
        ferry.state.landed = False
        ferry.state.angular_velocity = np.zeros(3)

        ferry.controls.sas = True
        ferry.controls.sas_mode = "hold"

        self.vessel_system.placed(ferry.state)

        # (Not the target yet: the lesson's first step.)
        self.map_view.target = None

        self._lesson_station = station_entity

        self.hud.notices.clear()

    def _set_up_landing(self):

        lander = self._lesson_vessel("mariner", 6)

        moon = self._body("moon")

        reference = self.vessel_system.soi_bodies(self.scene)[moon]

        environment = self.vessel_system.environment(self.scene, moon, self.orbit_system, self.planet_system)

        lander.body = moon

        # (Where the Earth is up - a probe obeys only with a
        # link - and the Sun too: halfway between them.)
        sunward = -self.orbit_system.states[moon][0]

        over = self._earthward(moon) + sunward / float(np.linalg.norm(sunward))

        placement.coming_down(lander.state, reference.radius + self.LESSON_LANDING_HEIGHT, environment.spin, direction=over, falling=self.LESSON_LANDING_SPEED)

        lander.controls.sas = False

        self.vessel_system.placed(lander.state)

        self.hud.notices.clear()

    def _set_up_power(self):

        probe = self._lesson_vessel("mariner", 6)

        self._in_orbit(probe, "earth", self.LESSON_POWER_ALTITUDE)

        _fold(probe.vessel, out=False)

        power.set_charge(probe.vessel, probe.vessel.charge_capacity * self.LESSON_POWER_CHARGE)

        probe.controls.sas = True
        probe.controls.sas_mode = "hold"

        self.hud.notices.clear()

    def _set_up_comms(self):

        probe = self._lesson_vessel("mariner", 6)

        self._in_orbit(probe, "mars", self.LESSON_MARS_ALTITUDE, facing_home=True)

        _fold(probe.vessel, out=True)

        probe.controls.sas = False

        self.hud.notices.clear()


def _fold(
    vessel,
    out: bool
):
    """Every solar wing out (or folded), as if U had been pressed long ago."""

    for state in vessel.parts:

        if state.part.solar is None:
            continue

        state.extending = out
        state.extended = 1.0 if out else 0.0
