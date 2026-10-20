import math
import time

from dataclasses import dataclass

from core.logger import Logger

from planet.orbits import date_of

import numpy as np

from programme import events, research
from vessel.reliability import Failures
from programme.career import EVENT_KINDS, Campaign, Career, Flight, today_of
from programme.resources import products, resources_at

from vessel import propellants, soi
from vessel.docking import modules
from vessel.transfer import find_window

from types import SimpleNamespace

from programme.crew import HIRE_COST
from systems.comms_system import crew, crew_names
from vessel import life


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
        sandbox: bool = False,
        tree: research.TechTree | None = None
    ):

        self.campaign = campaign
        self.career = career
        self.path = path

        # The tech tree (programme/research.py).
        self.tree = tree if tree is not None else research.load_tree()

        # Parts failing (vessel/reliability.py): learned from the
        # career's flight data; none in a sandbox.
        self.failures = Failures(career.reliability, enabled=not sandbox)

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

        # kg a day the plant makes here now, and of what (for
        # the panel); "" when it has no tanks for what it could
        # make.
        self.making = 0.0
        self.making_kind = ""
        self.resources = []

    # =====================================================
    # Flights
    # =====================================================

    def until_affordable(
        self,
        cost: float
    ) -> str:
        """When the monthly budget will have brought enough, in words."""

        career = self.career

        monthly = career.budget(self.campaign)

        if monthly <= 0.0:
            return "Missions done bring funds."

        months = max(math.ceil((cost - career.funds) / monthly), 1)

        today = today_of(date_of(career.clock))

        index = today.year * 12 + (today.month - 1) + months

        when = today.replace(year=index // 12, month=index % 12 + 1, day=1)

        return f"The budget brings ${monthly:,.1f}M on the 1st of each month: enough on {when:%d %b %Y} (time warp with .), or build something cheaper."

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

        waiting = self.unresearched(component.vessel)

        if waiting:

            self.news.append(f"Not researched yet: {waiting[0]}" + (f" and {len(waiting) - 1} more" if len(waiting) > 1 else ""))

            return False

        cost = component.vessel.cost

        # Its seats: filled from the corps (programme/crew.py).
        seats = sum(part.part.crew * part.count for part in component.vessel.parts if part.crew_names is None)

        chosen = self.career.corps.choose(seats) if seats > 0 else []

        if seats > 0 and not chosen:

            self.news.append("No astronauts ready to fly: hire one (Programme panel)")

            return False

        if not self.career.can_afford(cost):

            self.news.append(f"Not enough funds: {component.vessel.name} costs ${cost:,.1f}M, ${self.career.funds:,.1f}M left. {self.until_affordable(cost)}")

            return False

        self.career.launch(cost)

        component.paid = True

        if chosen:

            names = [astronaut.name for astronaut in chosen]

            # Into the seats, part by part.
            for part in component.vessel.parts:

                if part.part.crew > 0 and part.crew_names is None:

                    take = part.part.crew * part.count

                    part.crew_names = tuple(names[:take])

                    names = names[take:]

            self.career.corps.launch([astronaut.name for astronaut in chosen], component.vessel.name, self.career.clock)

            self.news.append("Crew: " + ", ".join(f"{astronaut.name} ({astronaut.role})" for astronaut in chosen))

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

        if found is not None and simulated > 0.0 and not found[1].state.destroyed:
            self._burn_failures(found[1], simulated)

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

                self.crew_lost(crew_names(component.vessel), f"with {component.vessel.name}")

                for part in component.vessel.parts:
                    if part.part.crew > 0:
                        part.crew_names = ()

            # Crews lost with vessels not flown (on rails).
            for event in events:

                kind, _, names = event.partition(":")

                if kind == "crew gone":
                    self.crew_lost([name for name in names.split(";") if name], "with their vessel")

            if flight is not None and flight.situation == "Landed" and flight.body != flight.home:
                component.landed_on.add(flight.body)

            if not self.sandbox:

                for mission in career.events(self.campaign, today, flight):

                    if mission.reward.science:
                        app.science_system.record.points += mission.reward.science
                        app.science_system._changed()

                    Logger.info("[Programme] Event mission done: %s.", mission.title)

                    self._changed()

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

            # The crew home.
            names = crew_names(found[1].vessel)

            if names and not self.sandbox:

                for name in career.corps.home(names, app.orbit_system.time, life.CAREER_LIMIT):
                    self.news.append(f"{name} retires: past the career's radiation limit")

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

        eclipsed, apophis = self._events_seen(app, component)

        return Flight(
            body=science.body_id(app.scene, component.body),
            situation=readout.situation,
            crewed=crew(component.vessel) > 0,
            site=app.current_site or "",
            home=science.body_id(app.scene, home) if home is not None else "",
            landed_on=frozenset(component.landed_on),
            docked=modules(component.vessel),
            eclipsed=eclipsed,
            apophis=apophis
        )

    def _events_seen(
        self,
        app,
        component,
        always: bool = False
    ) -> tuple[dict, float | None]:
        """
        Real events from the vessel: how much of the Sun each
        body hides; how far Apophis is. Looked at only on an
        event mission's days (it is not free), or `always`.
        """

        today = today_of(date_of(app.orbit_system.time))

        wanted = {
            mission.kind
            for mission in self.campaign.missions
            if mission.kind in EVENT_KINDS
            and self.career.state(mission, today) == "open"
            and (mission.deadline is None or today <= mission.deadline)
        }

        if always:
            wanted = set(EVENT_KINDS)

        if not wanted:
            return {}, None

        vessels = app.vessel_system

        where = vessels.position_of(component, app.orbit_system)

        if where is None:
            return {}, None

        science = app.science_system

        sun_radius = vessels._star_radius(app.scene)

        eclipsed = {}

        earth = None

        for key, body in vessels.soi_bodies(app.scene).items():

            if key is None:
                continue

            reference = vessels.reference_state(app.orbit_system, key)

            if reference is None:
                continue

            body_id = science.body_id(app.scene, key)

            center = np.asarray(reference[0], dtype=np.float64)

            if body_id == "earth":
                earth = center

            covered = events.sun_coverage(where, np.zeros(3), sun_radius, center, body.radius)

            if covered > 0.0:
                eclipsed[body_id] = covered

        apophis = None

        now = app.orbit_system.time

        if "apophis" in wanted and earth is not None and events.apophis_near(now):
            apophis = float(np.linalg.norm(earth + events.apophis_position(now) - where))

        return eclipsed, apophis

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
        self.making_kind = ""
        self.resources = []

        vessel = component.vessel

        # (Wanted: drawing its power, systems/power_system.py.)
        vessel.isru_wanted = False

        plants = [state for state in vessel.parts if state.part.isru is not None]

        if not plants or flight is None or flight.body == flight.home:
            return

        place = app._science_place()

        if place is None:
            return

        self.resources = resources_at(flight.body, place.latitude, flight.site, app._has_air())

        if flight.situation != "Landed":
            return

        # What it could make here (kg a day of each), and of
        # that what it has tanks with room for: the fastest.
        could: dict[str, float] = {}

        for state in plants:
            for kind, rate in products(self.resources, state.part.isru.rate_ice, state.part.isru.rate_air).items():
                could[kind] = could.get(kind, 0.0) + rate * state.count

        room = {
            kind: rate
            for kind, rate in could.items()
            if (aboard := propellants.held(vessel, kind))[1] - aboard[0] > 1e-3
        }

        if not room:

            # (Something to make, nowhere to put it: the panel
            # says which tanks it needs.)
            self.making_kind = "" if not could else "needs:" + ",".join(sorted(could))

            return

        kind = max(room, key=room.get)

        rate = room[kind]

        # Running while there is room, as fast as its power
        # lets it.
        vessel.isru_wanted = True

        rate *= vessel.isru_share

        self.making = rate
        self.making_kind = kind

        if rate <= 0.0 or simulated <= 0.0:
            return

        made = fill_tanks(vessel, rate * simulated / 86_400.0, kind)

        self.career.made += made

        if made > 0.0:

            self._full = False

            self._changed(save_now=False)

        elif not self._full:

            self._full = True

            self.news.append(f"Propellant plant: {propellants.of(kind).short} tanks full")

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

    # =====================================================
    # Crews
    # =====================================================

    def crew_lost(
        self,
        names: list[str],
        why: str
    ):
        """Astronauts lost: noted, mourned (reputation)."""

        if self.sandbox:
            return

        lost = self.career.corps.lose(names)

        if not lost and not names:

            # (An unassigned crew: a sandbox's, a pad's.)
            self.career.crew_lost()

        if lost:

            self.career.crew_lost()

            self.news.append(f"Lost {why}: {', '.join(lost)}")

        self._changed()

    def crew_losses(
        self,
        losses
    ):
        """systems/crew_system.py's losses (out of supplies, of power, the radiation)."""

        for loss in losses:

            self.news.append(f"The crew of {loss.vessel} {loss.why}")

            self.crew_lost(list(loss.names), f"aboard {loss.vessel}")

    def add_doses(
        self,
        doses: dict
    ):
        """Each astronaut's radiation this frame."""

        if self.sandbox:
            return

        corps = self.career.corps

        for name, dose in doses.items():

            astronaut = corps.named(name)

            if astronaut is not None:
                astronaut.dose += dose

        if doses:
            self._changed(save_now=False)

    def stand_down(
        self,
        component
    ):
        """A flight called off (a revert): its crew ready again."""

        names = crew_names(component.vessel)

        if names:

            self.career.corps.stand_down(names)

            self._changed()

    def hire(
        self,
        role: str
    ) -> str:
        """A new astronaut ($HIRE_COST M): "" hired, else why not."""

        if self.sandbox:
            return "A sandbox needs no astronauts"

        if not self.career.can_afford(HIRE_COST):
            return f"Hiring costs ${HIRE_COST:,.0f}M"

        self.career.funds -= HIRE_COST

        astronaut = self.career.corps.hire(role)

        self.news.append(f"Hired {astronaut.name}, {astronaut.role}: -${HIRE_COST:,.0f}M")

        self._changed()

        return ""

    @staticmethod
    def has_role(
        vessel,
        corps,
        role: str
    ) -> bool:

        return any((corps.named(name) or SimpleNamespace(role="")).role == role for name in crew_names(vessel))

    # =====================================================
    # Reliability
    # =====================================================

    def _burn_failures(
        self,
        component,
        simulated: float
    ):
        """The burning engines' seconds: flight data, and the odd failure."""

        vessel = component.vessel

        if vessel.throttle <= 0.0:
            return

        engines = vessel.burning_engines()

        if not engines:
            return

        scale = 0.5 if self.has_role(vessel, self.career.corps, "engineer") else 1.0

        for _, state in engines:

            if not self.failures.burns(state.part.id, simulated, state.count, scale):

                state.failed = True
                state.ignited = False

                self.news.append(f"Engine failure: the {state.part.name.lower()} shut down")

                Logger.info("[Programme] %s: the %s failed.", vessel.name, state.part.name)

        self._changed(save_now=False)

    def staged(
        self,
        component,
        stage: int
    ) -> list[str]:
        """
        A stage just fired: its engines lighting, its
        parachutes arming, each may fail (marked failed).
        Notices for what did.
        """

        notices = []

        scale = 0.5 if self.has_role(component.vessel, self.career.corps, "engineer") else 1.0

        for state in component.vessel.parts:

            if state.stage != stage:
                continue

            if state.part.engine is not None and state.ignited and not self.failures.lights(state.part.id, state.count, scale):

                state.ignited = False
                state.failed = True

                notices.append(f"Engine failure: the {state.part.name.lower()} did not light")

            if state.part.parachute is not None and state.armed and not self.failures.opens(state.part.id, state.count, scale):

                state.armed = False
                state.failed = True

                notices.append(f"Parachute failure: the {state.part.name.lower()} will not open")

        if notices:
            Logger.info("[Programme] Stage %d: %s.", stage, "; ".join(notices))

        self._changed(save_now=False)

        return notices

    # =====================================================
    # Research
    # =====================================================

    def science_to_spend(
        self,
        record
    ) -> float:
        """Science earned (science/record.py) less what research took."""

        return max(record.points - self.career.research_spent, 0.0)

    def locked_parts(
        self
    ) -> dict:
        """Part id -> the tech node it waits on (none in a sandbox)."""

        if self.sandbox:
            return {}

        return research.locked_parts(self.tree, self.career.researched)

    def unresearched(
        self,
        vessel
    ) -> list[str]:
        """The vessel's parts not researched yet: "Part (Node)" each."""

        locked = self.locked_parts()

        found = []

        for state in vessel.parts:

            node = locked.get(state.part.id)

            if node is not None:

                text = f"{state.part.name} ({node.name})"

                if text not in found:
                    found.append(text)

        return found

    def research(
        self,
        node_id: str,
        record
    ) -> str:
        """Research a node with the science to spend: "" done (with news), else why not."""

        if self.sandbox:
            return "A sandbox knows everything"

        why = research.research(self.tree, node_id, self.career.researched, self.science_to_spend(record))

        if why:
            return why

        node = self.tree.node(node_id)

        self.career.research_spent += node.cost

        Logger.info("[Programme] Researched %s (%.0f science).", node.name, node.cost)

        self.news.append(f"Researched {node.name}: {len(node.parts)} new parts")

        self._changed()

        return ""

    def _changed(
        self,
        save_now: bool = True
    ):

        self._unsaved = True

        if save_now:
            self.save()

    def save(
        self,
        force: bool = False
    ):
        """Written if anything changed (force: anyway; the clock moves on)."""

        if self.path is None or not (self._unsaved or force):
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
    amount: float,
    kind: str | None = None
) -> float:
    """Take up to `amount` kg of propellant (of a kind: None, any) from the tanks, evenly by what each holds; returns what came out."""

    return vessel.draw_propellant(amount, kind)


def move_propellant(
    giver,
    taker,
    amount: float
) -> float:
    """
    From one vessel's tanks to another's (a depot): up to
    `amount` kg in all, each propellant into tanks of its own
    kind, what fits of what there is.
    """

    moved = 0.0

    for kind in propellants.kinds_aboard(giver):

        if amount - moved <= 1e-6:
            break

        have, _ = propellants.held(giver, kind)
        aboard, capacity = propellants.held(taker, kind)

        wanted = min(amount - moved, have, capacity - aboard)

        if wanted <= 1e-6:
            continue

        taken = drain_tanks(giver, wanted, kind)

        put = fill_tanks(taker, taken, kind)

        # (Anything that did not fit goes back.)
        if taken - put > 1e-6:
            fill_tanks(giver, taken - put, kind)

        moved += put

    return moved


def fill_tanks(
    vessel,
    amount: float,
    kind: str | None = None
) -> float:
    """Put up to `amount` kg of propellant (into tanks of a kind: None, any) into the tanks with room, evenly by room; returns what went in."""

    tanks = [
        state
        for state in vessel.parts
        if state.part.propellant > state.propellant + 1e-9 and (kind is None or state.part.propellant_type == kind)
    ]

    room = sum((state.part.propellant - state.propellant) * state.count for state in tanks)

    if room <= 0.0 or amount <= 0.0:
        return 0.0

    share = min(amount / room, 1.0)

    for state in tanks:
        state.propellant += (state.part.propellant - state.propellant) * share

    return room * share
