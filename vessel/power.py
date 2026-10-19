import math

from dataclasses import dataclass

import numpy as np

from vessel.vessel import PartState, Vessel


# =========================================================
# Power
# =========================================================
#
# Electric charge, in real units: W for power, Wh stored.
#
# Made by
#
#   * solar wings: their rating at 1 AU, by the inverse
#     square of the distance to the Sun (Mars gets 43%,
#     Jupiter 4%), nothing in a body's shadow, less through
#     thick air (none to speak of under Venus's clouds). A
#     wing turns about its long axis to face the Sun, so it
#     gets the sun's share across that axis: the vessel
#     turned with the Sun along the wings, nothing. Wings
#     unfold (U) and fold over a few seconds; out in fast,
#     thick air they tear off;
#   * generators: an RTG's or a reactor's steady watts,
#     day and night, anywhere.
#
# Stored in batteries (command parts carry some).
#
# Drawn by avionics and life support (all the time),
# instruments and transmitters (all the time), reaction
# wheels as they work, a propellant plant while it runs
# (last: only what is spare, and the batteries down to
# half), an experiment's run and a result sent home (their
# Wh at once).
#
# With nothing left (empty, and making less than it
# draws): no reaction wheels (SAS cannot hold), no link
# home, no experiments, the flight computer stops; a crew
# can still throttle and stage. The charge coming back
# brings it all back.

# m: the Earth's mean distance from the Sun.
AU = 1.495978707e11

# Seconds the wings take to unfold (or fold).
PANEL_TIME = 6.0

# Pa of dynamic pressure (half rho v^2) that tears unfolded
# wings off: well short of a launch's (~30 kPa).
PANEL_MAX_Q = 1_500.0

# Light lost through the air, per bar of it overhead (the
# Earth's clear sky at noon lets ~88% through).
AIR_EXTINCTION = 0.13

# Wh to send one result home.
TRANSMIT_ENERGY = 20.0

# A propellant plant leaves this share of the batteries.
ISRU_RESERVE = 0.5


@dataclass(slots=True)
class PowerStatus:
    """A vessel's power now, for the HUD and the editor."""

    charge: float = 0.0             # Wh
    capacity: float = 0.0           # Wh
    generation: float = 0.0         # W made (on average over the step)
    draw: float = 0.0               # W drawn (the plant's included)
    sunlight: float = 0.0           # 0..1 of the step in sunlight
    flux: float = 0.0               # the Sun's strength here (1: at 1 AU)
    powered: bool = True
    panels: str = ""                # "", "folded", "unfolding", "out", "folding", "torn off"
    isru: float = 0.0               # W the plant got

    # The Sun's direction in the vessel's frame (None: in
    # shadow): its wings turn to it.
    sun_local: object = None

    # kg a day of propellant boiling off now
    # (vessel/propellants.py).
    boiloff: float = 0.0

    @property
    def net(
        self
    ) -> float:

        return self.generation - self.draw

    @property
    def share(
        self
    ) -> float:
        """0..1 of the batteries full (1 without any)."""

        return self.charge / self.capacity if self.capacity > 0.0 else (1.0 if self.powered else 0.0)


# ---------------------------------------------------------
# Making it
# ---------------------------------------------------------

def solar_parts(
    vessel: Vessel
) -> list[PartState]:

    return [state for state in vessel.parts if state.part.solar is not None]


def wing_axes(
    state: PartState
) -> list[np.ndarray]:
    """Each copy's wing axis in the vessel's frame (a stack part's across it; a radial copy's straight out)."""

    base = math.radians(state.angle)

    if not state.radial:
        return [np.array([math.cos(base), 0.0, math.sin(base)])]

    return [
        np.array([math.cos(base + 2.0 * math.pi * k / state.count), 0.0, math.sin(base + 2.0 * math.pi * k / state.count)])
        for k in range(state.count)
    ]


def facing(
    axis: np.ndarray,
    sun: np.ndarray
) -> float:
    """0..1: how square on to the Sun a wing turning about `axis` gets (both unit, the same frame)."""

    along = float(axis @ sun)

    return math.sqrt(max(1.0 - along * along, 0.0))


def solar_power(
    vessel: Vessel,
    sun_local: np.ndarray | None,
    light: float
) -> float:
    """
    W the wings make: sun_local the Sun's direction in the
    vessel's frame (None: no sun), light its strength there
    (1 at 1 AU, in the open; shadow and air taken off).
    """

    if sun_local is None or light <= 0.0:
        return 0.0

    total = 0.0

    # (Dust on them: Mars.)
    light = light * (1.0 - vessel.panel_dust)

    for state in solar_parts(vessel):

        if state.broken or state.extended <= 0.0:
            continue

        rating = state.part.solar.power * state.extended

        for axis in wing_axes(state):
            total += rating * facing(axis, sun_local) * light

    return total


def generator_power(
    vessel: Vessel
) -> float:

    return sum(state.part.generator * state.count for state in vessel.parts)


def strength(
    distance: float
) -> float:
    """The Sun's light at `distance` m from it (1 at 1 AU)."""

    return (AU / max(distance, 1.0)) ** 2


def through_air(
    pressure_bar: float,
    sun_height: float
) -> float:
    """0..1 of the light that gets down through `pressure_bar` of air with the Sun `sun_height` (its sine) up."""

    if pressure_bar <= 0.0:
        return 1.0

    return math.exp(-AIR_EXTINCTION * pressure_bar / max(sun_height, 0.05))


def shadowed(
    where: np.ndarray,
    sun: np.ndarray,
    bodies
) -> bool:
    """Whether a body (center, radius) stands between `where` and the Sun at `sun`."""

    segment = sun - where

    length2 = float(segment @ segment)

    if length2 <= 0.0:
        return False

    for center, radius in bodies:

        t = float((center - where) @ segment) / length2

        # (Behind the vessel, or beyond the Sun: not between.)
        if t <= 0.0 or t >= 1.0:
            continue

        closest = where + segment * t

        if float(np.linalg.norm(center - closest)) < radius:
            return True

    return False


# ---------------------------------------------------------
# Drawing it
# ---------------------------------------------------------

def base_draw(
    vessel: Vessel
) -> float:
    """W drawn all the time: avionics, life support, instruments, transmitters; the wheels as they work."""

    steady = sum(state.part.power_draw * state.count for state in vessel.parts)

    wheels = sum(state.part.wheel_power * state.count for state in vessel.parts) * min(max(vessel.wheel_use, 0.0), 1.0)

    return steady + wheels


def isru_draw(
    vessel: Vessel
) -> float:
    """W the propellant plants want, running."""

    if not vessel.isru_wanted:
        return 0.0

    return sum(state.part.isru.power * state.count for state in vessel.parts if state.part.isru is not None)


def set_charge(
    vessel: Vessel,
    total: float
):
    """Spread `total` Wh over the batteries, each the same share full."""

    capacity = vessel.charge_capacity

    if capacity <= 0.0:
        return

    share = min(max(total / capacity, 0.0), 1.0)

    for state in vessel.parts:
        state.charge = state.part.battery * share


def take(
    vessel: Vessel,
    energy: float
) -> bool:
    """Spend `energy` Wh at once (an experiment, a transmission): False, and nothing spent, if there is not that much."""

    if energy <= 0.0:
        return True

    charge = vessel.charge

    if not vessel.powered or charge < energy:
        return False

    set_charge(vessel, charge - energy)

    return True


def settle(
    vessel: Vessel,
    generation: float,
    dt: float
) -> PowerStatus:
    """
    `dt` s of the vessel's power: `generation` W made (on
    average), its draw taken, the batteries charged or run
    down, the plant given what is spare. Sets
    vessel.powered and vessel.isru_share.
    """

    capacity = vessel.charge_capacity
    charge = vessel.charge

    essential = base_draw(vessel)

    plant = isru_draw(vessel)

    surplus = generation - essential

    share = 0.0

    if plant > 0.0:

        # What is spare, and the batteries down to the
        # reserve over the step.
        spare = max(surplus, 0.0)

        if dt > 0.0:
            spare += max(charge - ISRU_RESERVE * capacity, 0.0) * 3600.0 / dt

        share = min(max(spare / plant, 0.0), 1.0)

    vessel.isru_share = share if plant > 0.0 else 1.0

    net = surplus - share * plant

    after = charge + net * dt / 3600.0

    if capacity > 0.0:

        powered = after > 1e-9 or net >= 0.0

        after = min(max(after, 0.0), capacity)

        set_charge(vessel, after)

    else:

        powered = net >= 0.0
        after = 0.0

    # (Nothing to run on: off, and drawing nothing.)
    vessel.powered = powered

    return PowerStatus(
        charge=after,
        capacity=capacity,
        generation=generation,
        draw=essential + share * plant,
        powered=powered,
        panels=panel_state(vessel),
        isru=share * plant
    )


# ---------------------------------------------------------
# The wings
# ---------------------------------------------------------

def toggle_panels(
    vessel: Vessel
) -> bool | None:
    """Unfold the wings (all folded or folding) or fold them; the new direction, None without any."""

    wings = [state for state in solar_parts(vessel) if not state.broken]

    if not wings:
        return None

    out = not any(state.extending for state in wings)

    set_panels(vessel, out)

    return out


def set_panels(
    vessel: Vessel,
    out: bool
):

    for state in solar_parts(vessel):
        if not state.broken:
            state.extending = out


def move_panels(
    vessel: Vessel,
    dt: float
):
    """The wings unfolding or folding over PANEL_TIME."""

    step = dt / PANEL_TIME

    for state in solar_parts(vessel):

        if state.broken:
            continue

        if state.extending:
            state.extended = min(state.extended + step, 1.0)
        else:
            state.extended = max(state.extended - step, 0.0)


def tear_panels(
    vessel: Vessel,
    dynamic_pressure: float
) -> list[str]:
    """Unfolded wings in air too fast and thick: torn off. Their parts' names."""

    if dynamic_pressure <= PANEL_MAX_Q:
        return []

    torn = []

    for state in solar_parts(vessel):

        if not state.broken and state.extended > 0.05:

            state.broken = True
            state.extending = False
            state.extended = 0.0

            torn.append(state.part.name)

    return torn


def panel_state(
    vessel: Vessel
) -> str:

    wings = solar_parts(vessel)

    if not wings:
        return ""

    working = [state for state in wings if not state.broken]

    if not working:
        return "torn off"

    if all(state.extending and state.extended >= 1.0 for state in working):
        return "out"

    if any(state.extending for state in working):
        return "unfolding"

    if any(state.extended > 0.0 for state in working):
        return "folding"

    return "folded"


# ---------------------------------------------------------
# For the editor
# ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PowerBudget:
    """A design's power, as the editor shows it."""

    capacity: float                 # Wh
    solar: float                    # W the wings make at the distance, square on
    generators: float               # W
    draw: float                     # W steady (wheels idle)
    wheels: float                   # W more turning hard

    @property
    def dark_hours(
        self
    ) -> float | None:
        """Hours the batteries last in the dark (None: forever, the generators keep up)."""

        deficit = self.draw - self.generators

        if deficit <= 0.0:
            return None

        return self.capacity / deficit


def budget(
    vessel: Vessel,
    distance: float = AU
) -> PowerBudget:

    solar = sum(
        state.part.solar.power * state.count * strength(distance)
        for state in solar_parts(vessel)
        if not state.broken
    )

    return PowerBudget(
        capacity=vessel.charge_capacity,
        solar=solar,
        generators=generator_power(vessel),
        draw=sum(state.part.power_draw * state.count for state in vessel.parts),
        wheels=sum(state.part.wheel_power * state.count for state in vessel.parts)
    )
