import math
import random

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache


# =========================================================
# Mars's Weather
# =========================================================
#
# The season, from the real calendar: Mars's solar longitude
# Ls (0 its northern spring equinox, 90 northern summer, 180
# autumn, 270 winter; Allison & McEwen 2000), and the Mars
# year (MY 1 began on 11 April 1955; MY 39 begins in October
# 2026).
#
#   pressure  CO2 freezes onto the winter pole and thaws: the
#             whole air is about 12% thicker at Ls ~255 (south
#             summer) than at Ls ~75
#   dust      the sky's optical depth (tau): ~0.4 clear, more
#             in the dusty season (southern spring and summer,
#             Ls 180-330). Each Mars year has a few regional
#             storms in that season (tau 1.5-4.5 over a
#             thousand-odd km, for a few weeks), and some years
#             a global one (tau 5-10 everywhere for months: the
#             2018 storm that ended Opportunity reached 10.8).
#             Storms are drawn from the Mars year (the same
#             every time it is played): a forecast can be made
#   wind      a few m/s, more round a storm (tens of m/s),
#             blowing round it
#
# Sunlight through it: what reaches the ground is mostly
# scattered, not lost: about exp(-0.4 tau / sin(elevation))
# of the clear light (tau 1: two thirds, tau 10: a couple of
# percent). Dust settles on a lander's panels; a storm's
# wind can blow them clean.

MARS_RADIUS = 3_389_500.0

DAY = 86_400.0

# A Mars solar day (s).
SOL = 88_775.244

_J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# MY 1 began on this day (Ls 0).
_MY1 = datetime(1955, 4, 11, 0, 0, 0, tzinfo=timezone.utc)

CLEAR_TAU = 0.4

DUST_SEASON = (180.0, 330.0)

# Of the dust's optical depth, the share that dims the
# light reaching the ground (the rest scatters down).
LIGHT_LOSS = 0.4


def _longitude_unwrapped(
    days: float
) -> float:
    """Ls in degrees, not wrapped (days since J2000)."""

    mean = math.radians(19.3871 + 0.52402073 * days)

    fictitious = 270.3871 + 0.524038496 * days

    center = (
        (10.691 + 3.0e-7 * days) * math.sin(mean)
        + 0.623 * math.sin(2.0 * mean)
        + 0.050 * math.sin(3.0 * mean)
        + 0.005 * math.sin(4.0 * mean)
        + 0.0005 * math.sin(5.0 * mean)
    )

    return fictitious + center


def solar_longitude(
    seconds: float
) -> float:
    """Mars's Ls (deg, 0-360) at a time (s since J2000)."""

    return _longitude_unwrapped(seconds / DAY) % 360.0


def _first_year_turn() -> int:
    """How many whole turns of Ls (unwrapped) came before MY 1."""

    start = (_MY1 - _J2000).total_seconds() / DAY

    return math.floor(_longitude_unwrapped(start + 1.0) / 360.0)


def mars_year(
    seconds: float
) -> int:

    return math.floor(_longitude_unwrapped(seconds / DAY) / 360.0) - _first_year_turn() + 1


def season(
    longitude: float
) -> str:

    names = ("northern spring", "northern summer", "northern autumn", "northern winter")
    south = ("southern autumn", "southern winter", "southern spring", "southern summer")

    index = int(longitude // 90.0) % 4

    return f"{names[index]} / {south[index]}"


def dusty_season(
    longitude: float
) -> bool:

    return DUST_SEASON[0] <= longitude <= DUST_SEASON[1]


def pressure_factor(
    longitude: float
) -> float:
    """The whole air's mass against its mean (CO2 frozen out at the winter pole)."""

    return 1.0 + 0.12 * math.cos(math.radians(longitude - 255.0))


# ---------------------------------------------------------
# Storms
# ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Storm:

    name: str
    start: float                    # Ls
    length: float                   # Ls it lasts
    latitude: float
    longitude: float
    radius: float                   # m (math.inf: global)
    tau: float                      # at its height, at its center
    wind: float                     # m/s at its height

    @property
    def global_storm(
        self
    ) -> bool:

        return math.isinf(self.radius)

    def strength(
        self,
        age: float
    ) -> float:
        """0..1 over its life (age: Ls since it began): building fast, a while at its height, dying away."""

        if age < 0.0 or age > self.length:
            return 0.0

        grow = 0.15 * self.length

        if age < grow:
            return age / grow

        fade = 0.45 * self.length

        if age > self.length - fade:
            return (self.length - age) / fade

        return 1.0

    def reach(
        self,
        latitude: float,
        longitude: float
    ) -> float:
        """0..1: how much of it is here (1 at its center, everywhere for a global one)."""

        if self.global_storm:
            return 1.0

        distance = great_circle(latitude, longitude, self.latitude, self.longitude) * MARS_RADIUS

        return math.exp(-(distance / self.radius) ** 2)


def great_circle(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float
) -> float:
    """Angle (rad) between two places."""

    a, b = math.radians(latitude_a), math.radians(latitude_b)

    cosine = math.sin(a) * math.sin(b) + math.cos(a) * math.cos(b) * math.cos(math.radians(longitude_a - longitude_b))

    return math.acos(min(max(cosine, -1.0), 1.0))


@lru_cache(maxsize=16)
def storms(
    year: int
) -> tuple[Storm, ...]:
    """A Mars year's dust storms (the same each time: drawn from the year)."""

    rng = random.Random(year * 7919 + 17)

    found = []

    for number in range(rng.randint(3, 6)):

        found.append(Storm(
            name=f"Regional storm {year}-{number + 1}",
            start=rng.uniform(DUST_SEASON[0], DUST_SEASON[1] - 10.0),
            length=rng.uniform(6.0, 20.0),
            latitude=rng.uniform(-65.0, 25.0),
            longitude=rng.uniform(-180.0, 180.0),
            radius=rng.uniform(600_000.0, 2_200_000.0),
            tau=rng.uniform(1.5, 4.5),
            wind=rng.uniform(15.0, 30.0)
        ))

    # About one year in three: a global storm.
    if rng.random() < 0.33:

        found.append(Storm(
            name=f"Global dust storm, MY {year}",
            start=rng.uniform(185.0, 260.0),
            length=rng.uniform(60.0, 90.0),
            latitude=rng.uniform(-40.0, 0.0),
            longitude=rng.uniform(-180.0, 180.0),
            radius=math.inf,
            tau=rng.uniform(5.0, 10.0),
            wind=rng.uniform(20.0, 35.0)
        ))

    return tuple(found)


def active(
    seconds: float
) -> list[tuple[Storm, float]]:
    """The storms blowing now, with their strength (0..1)."""

    # Ls counted on from MY 1 (not wrapped): a storm's age.
    turned = _longitude_unwrapped(seconds / DAY) - 360.0 * _first_year_turn()

    year = mars_year(seconds)

    found = []

    # (Last year's late storms may run into this one.)
    for which in (year, year - 1):

        for storm in storms(which):

            strength = storm.strength(turned - (360.0 * (which - 1) + storm.start))

            if strength > 0.0:
                found.append((storm, strength))

    return found


@dataclass(frozen=True, slots=True)
class Weather:
    """Mars's weather at a place and time."""

    longitude: float                # Ls
    season: str
    tau: float                      # dust optical depth
    wind: float                     # m/s
    wind_east: float                # m/s toward the east
    wind_north: float               # m/s toward the north
    pressure: float                 # the air's factor (pressure_factor)
    storm: str = ""                 # the strongest storm here, if any

    @property
    def light(
        self
    ) -> float:
        """Of the clear-sky light at noon, the share reaching the ground."""

        return sunlight(self.tau, 1.0)


def sunlight(
    tau: float,
    sun_height: float
) -> float:
    """0..1 of the light reaching the ground through dust of optical depth `tau` with the Sun `sun_height` (its sine) up."""

    return math.exp(-LIGHT_LOSS * tau / max(sun_height, 0.1))


def weather_at(
    seconds: float,
    latitude: float,
    longitude: float
) -> Weather:

    ls = solar_longitude(seconds)

    # The dusty season's background haze.
    bump = math.exp(-(((ls - 250.0 + 180.0) % 360.0 - 180.0) / 50.0) ** 2)

    tau = CLEAR_TAU + 0.35 * bump

    # A steady westerly breeze.
    east, north = 5.0, 0.0

    strongest = ("", 0.0)

    for storm, strength in active(seconds):

        here = storm.reach(latitude, longitude) * strength

        if here <= 1e-3:
            continue

        tau += storm.tau * here

        if storm.tau * here > strongest[1]:
            strongest = (storm.name, storm.tau * here)

        speed = storm.wind * here

        if storm.global_storm:

            east += speed

        else:

            # Round the storm (counterclockwise seen from
            # above in the north, as a low turns).
            bearing = _bearing(latitude, longitude, storm.latitude, storm.longitude)

            turn = bearing + (math.pi / 2.0 if storm.latitude >= 0.0 else -math.pi / 2.0)

            east += speed * math.sin(turn)
            north += speed * math.cos(turn)

    return Weather(
        longitude=ls,
        season=season(ls),
        tau=tau,
        wind=math.hypot(east, north),
        wind_east=east,
        wind_north=north,
        pressure=pressure_factor(ls),
        storm=strongest[0] if strongest[1] >= 1.0 else ""
    )


def _bearing(
    latitude: float,
    longitude: float,
    to_latitude: float,
    to_longitude: float
) -> float:
    """Initial bearing (rad, from north, east positive) toward a place."""

    a, b = math.radians(latitude), math.radians(to_latitude)

    delta = math.radians(to_longitude - longitude)

    return math.atan2(math.sin(delta) * math.cos(b), math.cos(a) * math.sin(b) - math.sin(a) * math.cos(b) * math.cos(delta))


def wind_from(
    weather: Weather
) -> str:
    """The compass point the wind blows from."""

    if weather.wind < 0.5:
        return "calm"

    toward = math.degrees(math.atan2(weather.wind_east, weather.wind_north)) % 360.0

    points = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

    return points[int(((toward + 180.0) % 360.0 + 22.5) // 45.0) % 8]


# ---------------------------------------------------------
# Dust on the panels
# ---------------------------------------------------------

# Share of a panel covered a sol, settling (more in a dusty
# sky), and cleaned a sol per m/s of wind over CLEANING_WIND
# (gusts in fairly clear air: in a thick storm the dust in it
# settles as fast as it is blown off; after, the panels can
# come up clean, as Spirit's and Opportunity's did).
SETTLING = 0.003
CLEANING_WIND = 12.0
CLEANING = 0.02
CLEAN_TAU = 4.0


def settle_dust(
    dust: float,
    weather: Weather,
    seconds: float
) -> float:
    """Dust on a lander's panels (0..1 covered) after `seconds` more on the ground."""

    sols = seconds / SOL

    dust += SETTLING * (0.5 + 0.5 * weather.tau) * sols

    if weather.wind > CLEANING_WIND:
        dust -= CLEANING * (weather.wind - CLEANING_WIND) * max(1.0 - weather.tau / CLEAN_TAU, 0.0) * sols

    return min(max(dust, 0.0), 0.95)
