import time

from dataclasses import dataclass

from core.logger import Logger

from planet.orbits import date_of

from programme.career import Campaign, Career, Flight, today_of
from programme.resources import production, resources_at

from vessel import soi
from vessel.docking import modules
from vessel.transfer import find_window

from systems.comms_system import crew


@dataclass(frozen=True, slots=True)
class LaunchWindow:
    """The next cheap departure to a planet, from home's orbit."""

    target: str                     # its name
    departure: float                # clock time (s)
    flight: float                   # s on the way
    excess: float                   # m/s beyond escaping home


class ProgrammeSystem:

    # =====================================================
    # The Space Programme
    # =====================================================
    #
    # The career (programme/career.py) in the game:
    #
    #   * the budget comes in month by month on the clock;
    #   * a vessel is paid for when it first stages (no funds,
    #     no staging), and landing back on
    #     the body it launched from recovers 80% of what came
    #     back (then it is paid for again if it flies again);
    #   * missions are checked against what the vessel does
    #     (its body, situation, crew, the site it is in, the
    #     bodies it has landed on this flight), the science
    #     banked, the ground mapped, the samples recovered;
    #   * a propellant plant landed on ice or in a usable air
    #     fills the tanks, day by day (time warp included);
    #   * the next launch windows to the planets, from home's
    #     orbit (vessel/transfer.py find_window).

    SAVE_INTERVAL = 30.0

    # Real seconds between mission checks, and between
    # launch window searches.
    CHECK_INTERVAL = 0.5
    WINDOW_INTERVAL = 5.0

    WINDOW_TARGETS = ("venus", "mars", "jupiter")

    def __init__(
        self,
        campaign: Campaign,
        career: Career,
        path=None,
        sandbox: bool = False
    ):

        self.campaign = campaign
        self.career = career
        self.path = path

        # A sandbox: launches free, no budget, no missions.
        self.sandbox = sandbox

        # Vessels recovered (landed home): (entity, real time),
        # for the game to take away shortly.
        self.recovered: list = []

        self._last_check = 0.0
        self._last_save = time.monotonic()
        self._unsaved = False

        # Notices for the HUD.
        self.news: list[str] = []

        # (real time searched, [LaunchWindow]).
        self._windows: tuple[float, list] | None = None

        # Tanks full (noted once).
        self._full = False

        # kg a day the plant makes here now (for the panel).
        self.making = 0.0
        self.resources = []

    # =====================================================
    # Flights
    # =====================================================

    def pay_launch(
        self,
        component
    ) -> bool:
        """Before the first stage: pay for the vessel. False when the funds are short."""

        if component.paid:
            return True

        if self.sandbox:

            component.paid = True

            return True

        cost = component.vessel.cost

        if not self.career.can_afford(cost):

            self.news.append(f"Not enough funds: {component.vessel.name} costs ${cost:,.1f}M, ${self.career.funds:,.1f}M left")

            return False

        self.career.launch(cost)

        component.paid = True

        self.news.append(f"Launch: -${cost:,.1f}M")

        self._changed()

        return True

    # =====================================================
    # Each Frame
    # =====================================================

    def update(
        self,
        app,
        events: list[str],
        simulated: float
    ):

        found = app.vessel_system.find(app.scene)

        career = self.career

        career.clock = max(career.clock, app.orbit_system.time)

        today = today_of(date_of(app.orbit_system.time))

        income = 0.0 if self.sandbox else career.pay_budget(self.campaign, today)

        if income > 0.0:
            self.news.append(f"Budget: +${income:,.1f}M")
            self._changed(save_now=False)

        flight = None

        if found is not None:

            _, component = found

            flight = self._flight(app, component)

            if "destroyed" in events and crew(component.vessel) > 0:
                career.crew_lost()
                self._changed()

            if flight is not None and flight.situation == "Landed" and flight.body != flight.home:
                component.landed_on.add(flight.body)

            self._isru(app, component, flight, simulated)

        now = time.monotonic()

        landed_home = flight is not None and "landed" in events and flight.body == flight.home

        if not self.sandbox and (landed_home or now - self._last_check > self.CHECK_INTERVAL):

            self._last_check = now

            self._check(app, today, flight)

        if landed_home and found is not None and found[1].paid:

            if not self.sandbox:

                refund = career.recover(found[1].vessel.cost)

                self.news.append(f"Recovered: +${refund:,.1f}M")

            found[1].paid = False
            found[1].landed_on = set()

            # (Taken away in a moment: its flight is over.)
            self.recovered.append((found[0], now))

            self._changed()

        self.news.extend(career.news)
        career.news.clear()

        if self._unsaved and now - self._last_save > self.SAVE_INTERVAL:
            self.save()

    def _flight(
        self,
        app,
        component
    ) -> Flight | None:

        readout = app.vessel_system.readout(app.scene, app.orbit_system, app.planet_system)

        if readout is None:
            return None

        science = app.science_system

        home = app.vessel_system.launch_body

        return Flight(
            body=science.body_id(app.scene, component.body),
            situation=readout.situation,
            crewed=crew(component.vessel) > 0,
            site=app.current_site or "",
            home=science.body_id(app.scene, home) if home is not None else "",
            landed_on=frozenset(component.landed_on),
            docked=modules(component.vessel)
        )

    def _check(
        self,
        app,
        today,
        flight: Flight | None
    ):

        record = app.science_system.record

        mapped = {body: mask.coverage() for body, mask in record.surveys.items() if mask.cells.any()}

        sampled = {
            key.split("@", 1)[1].split(":", 1)[0]
            for key, share in record.subjects.items()
            if key.startswith("samples@") and share > 0.0
        }

        done = self.career.check(self.campaign, today, flight, record.points, mapped, sampled)

        for mission in done:

            if mission.reward.science:
                record.points += mission.reward.science
                app.science_system._changed()

            Logger.info("[Programme] Mission done: %s.", mission.title)

        if done or self.career.news:
            self._changed()

    # =====================================================
    # Making Propellant
    # =====================================================

    def _isru(
        self,
        app,
        component,
        flight: Flight | None,
        simulated: float
    ):

        self.making = 0.0
        self.resources = []

        vessel = component.vessel

        plants = [state for state in vessel.parts if state.part.isru is not None]

        if not plants or flight is None or flight.body == flight.home:
            return

        place = app._science_place()

        if place is None:
            return

        self.resources = resources_at(flight.body, place.latitude, flight.site, app._has_air())

        if flight.situation != "Landed":
            return

        rate = sum(
            production(self.resources, state.part.isru.rate_ice, state.part.isru.rate_air) * state.count
            for state in plants
        )

        self.making = rate

        if rate <= 0.0 or simulated <= 0.0:
            return

        made = fill_tanks(vessel, rate * simulated / 86_400.0)

        self.career.made += made

        if made > 0.0:

            self._full = False

            self._changed(save_now=False)

        elif not self._full:

            self._full = True

            self.news.append("Propellant plant: tanks full")

    # =====================================================
    # Launch Windows
    # =====================================================

    def windows(
        self,
        app
    ) -> list[LaunchWindow]:
        """The next windows from home's orbit to the planets (searched every few seconds)."""

        now = time.monotonic()

        cached = self._windows

        if cached is not None and now - cached[0] < self.WINDOW_INTERVAL and all(
            window.departure > app.orbit_system.time - 86_400.0 for window in cached[1]
        ):
            return cached[1]

        windows = []

        vessels = app.vessel_system

        home = vessels.launch_body

        bodies = vessels.soi_bodies(app.scene)

        if home is not None and home in bodies and bodies[home].parent is None and home in app.orbit_system.states:

            velocities = app.orbit_system.velocities(app.scene)

            home_id = app.science_system.body_id(app.scene, home)

            for key, body in bodies.items():

                body_id = app.science_system.body_id(app.scene, key)

                if key is None or body.parent is not None or body_id not in self.WINDOW_TARGETS or body_id == home_id:
                    continue

                if key not in app.orbit_system.states or key not in velocities:
                    continue

                try:
                    window = find_window(
                        soi.SUN_MU,
                        (app.orbit_system.states[home][0], velocities[home]),
                        (app.orbit_system.states[key][0], velocities[key])
                    )
                except (ValueError, ZeroDivisionError, OverflowError):
                    continue

                windows.append(LaunchWindow(body.name, app.orbit_system.time + window.departure, window.transfer_time, window.excess))

        windows.sort(key=lambda window: window.departure)

        self._windows = (now, windows)

        return windows

    # =====================================================
    # Saving
    # =====================================================

    def _changed(
        self,
        save_now: bool = True
    ):

        self._unsaved = True

        if save_now:
            self.save()

    def save(self):

        if self.path is None or not self._unsaved:
            return

        try:
            self.career.save(self.path)
        except OSError as error:
            Logger.error("[Programme] Could not save the career: %s", error)
            return

        self._unsaved = False
        self._last_save = time.monotonic()


def drain_tanks(
    vessel,
    amount: float
) -> float:
    """Take up to `amount` kg of propellant from the tanks, evenly by what each holds; returns what came out."""

    return vessel.draw_propellant(amount)


def move_propellant(
    giver,
    taker,
    amount: float
) -> float:
    """From one vessel's tanks to another's (a depot): what fits, of what there is, up to `amount` kg."""

    room = taker.propellant_capacity - taker.propellant

    taken = drain_tanks(giver, min(amount, room))

    put = fill_tanks(taker, taken)

    # (Anything that did not fit goes back.)
    if taken - put > 1e-6:
        fill_tanks(giver, taken - put)

    return put


def fill_tanks(
    vessel,
    amount: float
) -> float:
    """Put up to `amount` kg of propellant into the tanks with room, evenly by room; returns what went in."""

    tanks = [state for state in vessel.parts if state.part.propellant > state.propellant + 1e-9]

    room = sum((state.part.propellant - state.propellant) * state.count for state in tanks)

    if room <= 0.0 or amount <= 0.0:
        return 0.0

    share = min(amount / room, 1.0)

    for state in tanks:
        state.propellant += (state.part.propellant - state.propellant) * share

    return room * share
