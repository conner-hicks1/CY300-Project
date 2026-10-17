import math

from dataclasses import dataclass

import numpy as np

from vessel import kepler


# =========================================================
# Transfers Between Planets
# =========================================================
#
# What a pilot needs to fly from one planet to another
# without a planner: the Hohmann transfer (the half ellipse
# touching both orbits, the cheapest way between circles),
# when it opens (the target must lead by the right angle at
# departure, so it arrives where the vessel does), and the
# burn from a parking orbit: how hard, and where in the
# orbit (the escape hyperbola's asymptote must leave along
# the planet's own motion, or against it going inward).
#
# Orbits are treated as circular and coplanar for the
# window; the burn's real result shows on the map.


@dataclass(frozen=True, slots=True)
class Hohmann:

    transfer_time: float            # s, half the ellipse
    departure_excess: float         # m/s over the origin's orbital speed (< 0: inward)
    arrival_excess: float           # m/s relative to the target on arrival (magnitude)
    phase: float                    # rad the target should lead the origin by at departure


def hohmann(
    mu: float,
    r1: float,
    r2: float
) -> Hohmann:
    """Between circular orbits of radius r1 and r2 about a body of G M `mu`."""

    a = 0.5 * (r1 + r2)

    transfer = math.pi * math.sqrt(a ** 3 / mu)

    departure = math.sqrt(mu / r1) * (math.sqrt(2.0 * r2 / (r1 + r2)) - 1.0)
    arrival = math.sqrt(mu / r2) * (1.0 - math.sqrt(2.0 * r1 / (r1 + r2)))

    # The target moves on while the vessel flies half round.
    phase = math.pi - math.sqrt(mu / r2 ** 3) * transfer

    return Hohmann(transfer, departure, abs(arrival), phase)


def phase_angle(
    origin_position,
    origin_velocity,
    target_position
) -> float:
    """rad the target leads the origin round the star by (-pi..pi), in the origin's direction of motion."""

    r_o = np.asarray(origin_position, dtype=np.float64)
    r_t = np.asarray(target_position, dtype=np.float64)

    normal = np.cross(r_o, np.asarray(origin_velocity, dtype=np.float64))
    normal /= np.linalg.norm(normal)

    return math.atan2(float(np.dot(normal, np.cross(r_o, r_t))), float(np.dot(r_o, r_t)))


def window_wait(
    phase_now: float,
    phase_wanted: float,
    origin_rate: float,
    target_rate: float
) -> float:
    """s until the phase is right (rates: rad/s round the star)."""

    rate = target_rate - origin_rate

    if abs(rate) < 1e-15:
        return math.inf

    synodic = 2.0 * math.pi / abs(rate)

    return ((phase_wanted - phase_now) / rate) % synodic


def escape_burn(
    mu: float,
    parking_radius: float,
    excess: float
) -> float:
    """m/s to add in a circular parking orbit to leave with `excess` m/s to spare."""

    return math.sqrt(excess * excess + 2.0 * mu / parking_radius) - math.sqrt(mu / parking_radius)


def ejection_point(
    parking_normal,
    planet_velocity,
    mu: float,
    parking_radius: float,
    excess: float
) -> np.ndarray:
    """
    Unit direction (from the planet) of the point in the
    parking orbit to burn at, so the escape leaves along the
    planet's motion (excess > 0) or against it (< 0): the
    hyperbola turns by acos(-1 / e) from its periapsis to its
    asymptote, so the burn is that far before it.
    """

    normal = np.asarray(parking_normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)

    direction = np.asarray(planet_velocity, dtype=np.float64) * (1.0 if excess >= 0.0 else -1.0)

    # Into the parking orbit's plane.
    direction = direction - np.dot(direction, normal) * normal
    direction /= np.linalg.norm(direction)

    e = 1.0 + parking_radius * excess * excess / mu

    turn = math.acos(-1.0 / e)

    # Rotated back (against the orbit's motion) by the turn.
    return direction * math.cos(turn) - np.cross(normal, direction) * math.sin(turn)


@dataclass(frozen=True, slots=True)
class TransferPlan:

    phase_now: float                # deg the target leads by
    phase_wanted: float             # deg it should
    wait: float                     # s until the window
    synodic: float                  # s between windows
    transfer_time: float            # s
    departure_burn: float           # m/s prograde from the parking orbit (at the window, Hohmann)
    burn_now: float                 # m/s to leave now and arrive in transfer_time (Lambert)
    arrival_excess: float           # m/s
    burn_direction: np.ndarray      # unit, from the origin (its frame)
    time_to_burn_point: float       # s until the vessel is over it (in its orbit now)


@dataclass(frozen=True, slots=True)
class Window:
    """The cheapest departure found: when (s from the search's start), how long the flight, at what excess speed."""

    departure: float
    transfer_time: float
    excess: float


def find_window(
    star_mu: float,
    origin_state: tuple,
    target_state: tuple,
    horizon: float | None = None,
    steps: int = 120
) -> Window:
    """
    The cheapest departure from the origin's orbit to the
    target's within `horizon` s (default: one synodic
    period): Lambert's problem solved for each departure
    date and a few flight times round Hohmann's, refined
    round the best. The planets move on their osculating
    conics about the star (real orbits: eccentric,
    inclined), so the window is where it really is.
    """

    origin_position, origin_velocity = (np.asarray(v, dtype=np.float64) for v in origin_state)
    target_position, target_velocity = (np.asarray(v, dtype=np.float64) for v in target_state)

    r1 = float(np.linalg.norm(origin_position))
    r2 = float(np.linalg.norm(target_position))

    base = hohmann(star_mu, r1, r2).transfer_time

    if horizon is None:

        rate = abs(math.sqrt(star_mu / r1 ** 3) - math.sqrt(star_mu / r2 ** 3))

        horizon = 2.0 * math.pi / rate if rate > 0.0 else 365.25 * 86_400.0

    normal = np.cross(origin_position, origin_velocity)

    def excess_at(departures: np.ndarray, flight: float) -> np.ndarray:

        origins, speeds = kepler.propagate_many(origin_position, origin_velocity, star_mu, departures)
        targets, _ = kepler.propagate_many(target_position, target_velocity, star_mu, departures + flight)

        found = np.full(len(departures), np.inf)

        for k in range(len(departures)):

            try:
                velocity, _ = lambert(origins[k], targets[k], flight, star_mu, normal)
            except (ValueError, ZeroDivisionError, OverflowError):
                continue

            found[k] = float(np.linalg.norm(velocity - speeds[k]))

        return found

    flights = [base * factor for factor in (0.8, 0.9, 1.0, 1.1, 1.2)]

    departures = np.linspace(0.0, horizon, steps)

    best = (math.inf, 0.0, base)

    for flight in flights:

        excess = excess_at(departures, flight)

        k = int(np.argmin(excess))

        if excess[k] < best[0]:
            best = (float(excess[k]), float(departures[k]), flight)

    # Refine round the best: a finer grid of dates and times.
    step = horizon / (steps - 1)

    for _ in range(2):

        _, departure, flight = best

        dates = np.clip(np.linspace(departure - step, departure + step, 21), 0.0, horizon)

        for factor in np.linspace(0.92, 1.08, 9):

            excess = excess_at(dates, flight * factor)

            k = int(np.argmin(excess))

            if excess[k] < best[0]:
                best = (float(excess[k]), float(dates[k]), flight * factor)

        step /= 10.0

    return Window(departure=best[1], transfer_time=best[2], excess=best[0])


def plan_transfer(
    star_mu: float,
    origin_position,
    origin_velocity,
    target_position,
    target_velocity,
    origin_mu: float,
    vessel_position,
    vessel_velocity,
    window: Window | None = None,
    window_wait_time: float | None = None
) -> TransferPlan:
    """
    The vessel in orbit round the origin planet (positions
    of the planets about the star, the vessel's about the
    origin; one non-rotating frame), to the target planet.
    window (found by find_window) and window_wait_time (s
    from now until it): given, the window's timing and
    flight time are used, and the burn now is aimed by
    Lambert at where the target will really be; else all is
    the circular Hohmann estimate.
    """

    origin_position = np.asarray(origin_position, dtype=np.float64)
    origin_velocity = np.asarray(origin_velocity, dtype=np.float64)

    r1 = float(np.linalg.norm(origin_position))
    r2 = float(np.linalg.norm(target_position))

    transfer = hohmann(star_mu, r1, r2)

    now = phase_angle(origin_position, origin_velocity, target_position)

    origin_rate = math.sqrt(star_mu / r1 ** 3)
    target_rate = math.sqrt(star_mu / r2 ** 3)

    synodic = 2.0 * math.pi / abs(target_rate - origin_rate)

    if window is not None and window_wait_time is not None:
        wait = window_wait_time
        flight = window.transfer_time
        window_burn_excess = window.excess
    else:
        wait = window_wait(now, transfer.phase, origin_rate, target_rate)
        flight = transfer.transfer_time
        window_burn_excess = abs(transfer.departure_excess)

    vessel_position = np.asarray(vessel_position, dtype=np.float64)
    vessel_velocity = np.asarray(vessel_velocity, dtype=np.float64)

    parking = float(np.linalg.norm(vessel_position))

    normal = np.cross(vessel_position, vessel_velocity)

    # Leaving now, arriving where the target will be after
    # the flight.
    arrival, _ = kepler.propagate(target_position, target_velocity, star_mu, flight)

    departure, _ = lambert(origin_position, arrival, flight, star_mu, np.cross(origin_position, origin_velocity))

    excess_vector = departure - origin_velocity

    excess = float(np.linalg.norm(excess_vector))

    point = ejection_point(normal, excess_vector, origin_mu, parking, excess)

    # How far round the vessel has to go to get there.
    unit_normal = normal / np.linalg.norm(normal)

    here = vessel_position / parking

    ahead = math.atan2(float(np.dot(unit_normal, np.cross(here, point))), float(np.dot(here, point))) % (2.0 * math.pi)

    rate = float(np.linalg.norm(vessel_velocity)) / parking

    return TransferPlan(
        phase_now=math.degrees(now),
        phase_wanted=math.degrees(transfer.phase),
        wait=wait,
        synodic=synodic,
        transfer_time=flight,
        departure_burn=escape_burn(origin_mu, parking, window_burn_excess),
        burn_now=escape_burn(origin_mu, parking, excess),
        arrival_excess=transfer.arrival_excess,
        burn_direction=point,
        time_to_burn_point=ahead / rate
    )


def lambert(
    r1,
    r2,
    time: float,
    mu: float,
    normal
) -> tuple[np.ndarray, np.ndarray]:
    """
    The conic from r1 to r2 in `time` s (less than one
    revolution), going round `normal` the short or long way
    as that requires: (velocity at r1, velocity at r2).
    Universal variables (Curtis, algorithm 5.2), solved by
    bisection (the time grows with z).
    """

    from vessel.kepler import _stumpff_c, _stumpff_s

    r1 = np.asarray(r1, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)

    n1 = float(np.linalg.norm(r1))
    n2 = float(np.linalg.norm(r2))

    cos_angle = float(np.clip(np.dot(r1, r2) / (n1 * n2), -1.0, 1.0))

    angle = math.acos(cos_angle)

    # Prograde round the normal.
    if float(np.dot(np.cross(r1, r2), np.asarray(normal, dtype=np.float64))) < 0.0:
        angle = 2.0 * math.pi - angle

    a = math.sin(angle) * math.sqrt(n1 * n2 / (1.0 - math.cos(angle)))

    def y_of(z):
        return n1 + n2 + a * (z * _stumpff_s(z) - 1.0) / math.sqrt(_stumpff_c(z))

    def time_of(z):

        y = y_of(z)

        return ((y / _stumpff_c(z)) ** 1.5 * _stumpff_s(z) + a * math.sqrt(y)) / math.sqrt(mu)

    high = 4.0 * math.pi ** 2 * 0.999

    # The low end: where y turns positive (y > 0 at z = 0;
    # fast transfers are hyperbolic, z below 0).
    low = -1.0e4

    if y_of(low) < 0.0:

        below, above = low, 0.0

        for _ in range(100):

            middle = 0.5 * (below + above)

            if y_of(middle) < 0.0:
                below = middle
            else:
                above = middle

        low = above

    for _ in range(64):

        middle = 0.5 * (low + high)

        if time_of(middle) < time:
            low = middle
        else:
            high = middle

    z = 0.5 * (low + high)

    y = y_of(z)

    f = 1.0 - y / n1
    g = a * math.sqrt(y / mu)
    g_dot = 1.0 - y / n2

    return (r2 - f * r1) / g, (g_dot * r2 - r1) / g


def closest_approach(
    points: np.ndarray,
    times: np.ndarray,
    target_positions: np.ndarray
) -> tuple[float, float]:
    """(least distance m, at time s) between a sampled path and a target sampled at the same times."""

    gaps = np.linalg.norm(np.asarray(points) - np.asarray(target_positions), axis=1)

    index = int(np.argmin(gaps))

    return float(gaps[index]), float(times[index])


def correction(
    star_mu: float,
    vessel_position,
    vessel_velocity,
    target_position,
    target_velocity,
    arrival: float
) -> np.ndarray:
    """
    The velocity change (m/s, vector) that brings the vessel
    to the target's center `arrival` s from now (a course
    correction: aim, then trim the pass inside its sphere of
    influence).
    """

    target, _ = kepler.propagate(target_position, target_velocity, star_mu, arrival)

    needed, _ = lambert(
        vessel_position,
        target,
        arrival,
        star_mu,
        np.cross(np.asarray(vessel_position, dtype=np.float64), np.asarray(vessel_velocity, dtype=np.float64))
    )

    return needed - np.asarray(vessel_velocity, dtype=np.float64)
