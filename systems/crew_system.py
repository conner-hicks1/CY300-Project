from dataclasses import dataclass

import numpy as np

from scene.scene import Scene

from systems.comms_system import crew, crew_names
from vessel import life


# =========================================================
# Crews, Each Frame
# =========================================================
#
# Every crewed vessel over the frame's simulated time
# (vessel/life.py): its supplies used (a habitat's life
# support recycling a share, with power), how long it has
# gone without supplies or without power, the radiation it
# is taking (where it is, the vessel's shielding, a solar
# storm) and the dose it has had this flight. A crew that
# runs out (STARVING days without supplies, DARK days
# without power) or takes ACUTE_LIMIT in one flight is lost.
#
# Returns what happened for the programme (programme/
# crew.py keeps each astronaut's dose and fate).

SUN = None


@dataclass(frozen=True, slots=True)
class LifeStatus:
    """A crewed vessel's life, for the HUD."""

    crew: int
    supplies: float                 # kg
    days: float                     # of supplies left
    recycling: float                # the share recycled now
    rate: float                     # mSv a day
    storm: float                    # of it from a solar storm
    dose: float                     # mSv this flight
    starving: float                 # s without supplies
    unpowered: float                # s without power


@dataclass(frozen=True, slots=True)
class Loss:

    vessel: str
    names: tuple
    why: str


class CrewSystem:

    # A solar storm worth a notice (mSv a day here).
    STORM_NOTICE = 5.0

    def __init__(self):

        # Doses this frame: astronaut name -> mSv.
        self.doses: dict[str, float] = {}

        # Vessels told of the storm now blowing (entity ids).
        self._warned: set = set()

    def update(
        self,
        scene: Scene,
        vessel_system,
        orbit_system,
        dt: float
    ) -> tuple[list[Loss], list[str]]:
        """(crews lost, notices)."""

        self.doses = {}

        losses: list[Loss] = []
        notices: list[str] = []

        now = orbit_system.time

        for entity, component in vessel_system.vessels(scene):

            vessel = component.vessel
            state = component.state

            people = crew(vessel)

            if state.destroyed or people <= 0:

                component.life = None

                continue

            reference = vessel_system.reference_state(orbit_system, component.body)

            if reference is None:
                continue

            physics = vessel_system._body_physics(scene, component.body, orbit_system)

            # Supplies: used, a share recycled while powered.
            stores = [part for part in vessel.parts if part.part.supplies > 0.0]

            held = sum(part.supplies * part.count for part in stores)

            recycling = max((part.part.recycling for part in vessel.parts), default=0.0) if vessel.powered else 0.0

            if dt > 0.0:

                left = life.use(held, people, recycling, dt)

                capacity = sum(part.part.supplies * part.count for part in stores)

                share = left / capacity if capacity > 0.0 else 0.0

                for part in stores:
                    part.supplies = part.part.supplies * share

                held = left

                component.starving = component.starving + dt if held <= 1e-6 else 0.0
                component.unpowered = component.unpowered + dt if not vessel.powered else 0.0

            # Radiation.
            radius = float(np.linalg.norm(state.position))

            pressure = 0.0

            if physics.column is not None:
                pressure = max(physics.column.pressure((radius - physics.radius) / 1000.0), 0.0)

            where = np.asarray(reference[0], dtype=np.float64) + state.position

            shielding = min((part.part.shielding for part in vessel.parts if part.part.crew > 0 or part.part.shielding < 1.0), default=1.0)

            place = life.Place(
                body=vessel_system.body_name(scene, component.body),
                radius=radius,
                body_radius=physics.radius,
                landed=state.landed,
                pressure=pressure,
                sun_distance=float(np.linalg.norm(where)),
                shielding=shielding
            )

            rate, storm = life.dose_rate(place, now)

            added = rate * max(dt, 0.0) / life.DAY

            component.dose += added

            names = crew_names(vessel)

            for name in names:
                self.doses[name] = self.doses.get(name, 0.0) + added

            if storm >= self.STORM_NOTICE and entity not in self._warned:

                self._warned.add(entity)

                notices.append(f"Solar storm: {vessel.name}'s crew taking {rate:,.0f} mSv a day")

            elif storm < 0.5 * self.STORM_NOTICE:

                self._warned.discard(entity)

            component.life = LifeStatus(
                crew=people,
                supplies=held,
                days=life.days_left(held, people, recycling),
                recycling=recycling,
                rate=rate,
                storm=storm,
                dose=component.dose,
                starving=component.starving,
                unpowered=component.unpowered
            )

            why = ""

            if component.starving > life.STARVING * life.DAY:
                why = "ran out of supplies"
            elif component.unpowered > life.DARK * life.DAY:
                why = "went without power too long"
            elif component.dose > life.ACUTE_LIMIT:
                why = f"took {component.dose:,.0f} mSv of radiation"

            if why:

                losses.append(Loss(vessel.name, tuple(names), why))

                # Nobody aboard now.
                for part in vessel.parts:
                    if part.part.crew > 0:
                        part.crew_names = ()

                component.life = None

        return losses, notices
