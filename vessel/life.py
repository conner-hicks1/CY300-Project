import math
import random

from dataclasses import dataclass


# =========================================================
# Life Support and Radiation
# =========================================================
#
# A crew lives on what it carries: SUPPLY_RATE kg a person a
# day of oxygen, water and food (supplies, in the command
# parts and supply modules). A habitat's life support
# recycles a share of it (its air and water) while it has
# power. With nothing left the crew lasts STARVING days;
# with no power (no air scrubbed, no heat) DARK days.
#
# Radiation, in mSv a day (measured or modeled):
#
#   the Earth's surface, thick air    ~0 (shielded)
#   low Earth orbit (under ~1,000 km) 0.5 (the ISS)
#   the Van Allen belts                tens to ~100 at their
#                                      hearts (1.7 and ~4.5
#                                      Earth radii): cross fast
#   inside the magnetosphere           1.0
#   deep space (the cosmic rays)       1.8 (MSL's cruise)
#   the Moon's surface                 1.4 (Chang'e 4)
#   Mars's surface                     0.7 (Curiosity)
#   Jupiter's belts                    thousands round Europa,
#                                      tens of thousands at
#                                      Io; Callisto little
#
# Solar particle events: now and then the Sun throws out a
# storm of protons (about one month in eight, a day and a
# half, hundreds of mSv a day at 1 AU unsheltered); the
# magnetosphere turns it from low orbit, air stops it, a
# habitat's water walls halve it, a planet under a lander
# blocks half the sky. Drawn from the calendar's month: the
# same in every game.
#
# A crew's dose adds up: past CAREER_LIMIT an astronaut
# retires on coming home; past ACUTE_LIMIT in one flight the
# crew is lost.

SUPPLY_RATE = 5.0                   # kg a person a day
STARVING = 3.0                      # days without supplies
DARK = 1.0                          # days without power

CAREER_LIMIT = 1000.0               # mSv
ACUTE_LIMIT = 3000.0                # mSv in a flight

DAY = 86_400.0

AU = 1.495978707e11

DEEP_SPACE = 1.8
LOW_ORBIT = 0.5
MAGNETOSPHERE = 1.0

EARTH_RADIUS = 6_371_000.0
JUPITER_RADIUS = 69_911_000.0


@dataclass(frozen=True, slots=True)
class Place:
    """Where a crew is, for its dose."""

    body: str                       # "Earth", "Moon", "Mars", "Jupiter"... (names)
    radius: float                   # m from the body's center
    body_radius: float              # m
    landed: bool
    pressure: float                 # bar outside
    sun_distance: float             # m
    shielding: float = 1.0          # the vessel's (1: none)


def background_rate(
    place: Place
) -> float:
    """mSv a day from the cosmic rays and the belts here (no solar storm)."""

    # Thick air above: next to nothing.
    if place.pressure > 0.1:
        return 0.01

    body = place.body

    r = place.radius / max(place.body_radius, 1.0)

    if body == "Earth":

        belts = 100.0 * math.exp(-((r - 1.7) / 0.25) ** 2) + 40.0 * math.exp(-((r - 4.5) / 1.0) ** 2)

        if place.radius - place.body_radius < 1_000_000.0:
            base = LOW_ORBIT
        elif r < 10.0:
            base = MAGNETOSPHERE
        else:
            base = DEEP_SPACE

        return base + belts

    if body == "Jupiter" or body in ("Io", "Europa", "Ganymede", "Callisto"):
        # (The moons' own radius is in place.radius only for
        # Jupiter: the moons' doses are their orbits'.)
        if body == "Jupiter":
            from_jupiter = r
        else:
            from_jupiter = {"Io": 5.9, "Europa": 9.4, "Ganymede": 15.0, "Callisto": 26.3}[body]

        belts = 5400.0 * math.exp(-(from_jupiter - 9.4) / 2.0)

        return DEEP_SPACE + belts * (0.5 if place.landed else 1.0)

    if place.landed:

        if body == "Mars":
            return 0.7

        return 1.4

    if body == "Mars" and r < 1.5:
        return DEEP_SPACE * 0.75

    return DEEP_SPACE


# ---------------------------------------------------------
# Solar storms
# ---------------------------------------------------------

STORM_MONTH = 30.0 * DAY
STORM_CHANCE = 0.12
STORM_LENGTH = 1.5 * DAY


@dataclass(frozen=True, slots=True)
class SolarStorm:

    start: float                    # s since J2000
    peak: float                     # mSv a day at 1 AU, unsheltered

    def rate(
        self,
        seconds: float
    ) -> float:
        """At 1 AU unsheltered: up in an hour, dying away over the day and a half."""

        age = seconds - self.start

        if age < 0.0 or age > STORM_LENGTH:
            return 0.0

        rise = min(age / 3600.0, 1.0)

        return self.peak * rise * math.exp(-age / (0.5 * DAY))


def storm_of(
    month: int
) -> SolarStorm | None:
    """The solar storm starting in a 30-day month (counted from J2000), or None."""

    rng = random.Random(month * 104_729 + 3)

    if rng.random() >= STORM_CHANCE:
        return None

    return SolarStorm(
        start=month * STORM_MONTH + rng.uniform(0.0, STORM_MONTH - 2.0 * DAY),
        peak=rng.uniform(100.0, 600.0)
    )


def storm_rate(
    seconds: float
) -> float:
    """A solar storm's mSv a day at 1 AU unsheltered now (0: none)."""

    month = math.floor(seconds / STORM_MONTH)

    total = 0.0

    for which in (month - 1, month):

        storm = storm_of(which)

        if storm is not None:
            total += storm.rate(seconds)

    return total


def storm_exposure(
    place: Place
) -> float:
    """Of a solar storm, the share reaching the crew here (before the vessel's shielding)."""

    if place.pressure > 0.1:
        return 0.0

    share = (AU / max(place.sun_distance, 1.0)) ** 2

    if place.body == "Earth":

        r = place.radius / max(place.body_radius, 1.0)

        # The magnetosphere turns it from low orbit; out
        # past ~6 Earth radii, little.
        share *= min(max((r - 1.3) / 4.7, 0.02), 1.0)

    if place.landed:
        share *= 0.5

    return share


def dose_rate(
    place: Place,
    seconds: float
) -> tuple[float, float]:
    """(mSv a day here now, of it from a solar storm), the vessel's shielding taken."""

    storm = storm_rate(seconds) * storm_exposure(place)

    return (background_rate(place) + storm) * place.shielding, storm * place.shielding


# ---------------------------------------------------------
# Supplies
# ---------------------------------------------------------

def use(
    held: float,
    crew: int,
    recycling: float,
    dt: float
) -> float:
    """kg of supplies left after `dt` s for a crew (a share recycled)."""

    return max(held - crew * SUPPLY_RATE * (1.0 - min(max(recycling, 0.0), 0.95)) * dt / DAY, 0.0)


def days_left(
    held: float,
    crew: int,
    recycling: float
) -> float:

    if crew <= 0:
        return math.inf

    return held / (crew * SUPPLY_RATE * (1.0 - min(max(recycling, 0.0), 0.95)))
