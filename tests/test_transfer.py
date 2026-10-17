import math

import numpy as np
import pytest

from vessel import kepler
from vessel.transfer import (
    closest_approach,
    ejection_point,
    escape_burn,
    hohmann,
    phase_angle,
    plan_transfer,
    window_wait
)


SUN_MU = 1.32712440018e20
EARTH_MU = 3.986004418e14
AU = 1.495978707e11

EARTH_ORBIT = 1.0 * AU
MARS_ORBIT = 1.524 * AU


def test_earth_to_mars_hohmann():

    transfer = hohmann(SUN_MU, EARTH_ORBIT, MARS_ORBIT)

    # The textbook numbers: ~259 days, ~2.94 km/s leaving
    # Earth's orbit, ~2.65 km/s arriving, Mars ~44 deg ahead.
    assert transfer.transfer_time / 86_400.0 == pytest.approx(259.0, abs=2.0)
    assert transfer.departure_excess == pytest.approx(2_945.0, abs=30.0)
    assert transfer.arrival_excess == pytest.approx(2_650.0, abs=30.0)
    assert math.degrees(transfer.phase) == pytest.approx(44.3, abs=0.5)

    # Inward (to Venus): a negative excess, the target behind.
    inward = hohmann(SUN_MU, EARTH_ORBIT, 0.723 * AU)

    assert inward.departure_excess < 0.0
    assert math.degrees(inward.phase) < 0.0


def test_escape_burn_from_low_orbit():

    # ~3.6 km/s from a 300 km orbit for Mars.
    burn = escape_burn(EARTH_MU, 6_678_000.0, 2_945.0)

    assert burn == pytest.approx(3_590.0, abs=40.0)


def test_phase_and_wait():

    # Mars 90 deg ahead, both going counterclockwise about +Z.
    phase = phase_angle([EARTH_ORBIT, 0, 0], [0, 29_780, 0], [0, MARS_ORBIT, 0])

    assert math.degrees(phase) == pytest.approx(90.0)

    earth_rate = math.sqrt(SUN_MU / EARTH_ORBIT ** 3)
    mars_rate = math.sqrt(SUN_MU / MARS_ORBIT ** 3)

    wait = window_wait(phase, math.radians(44.3), earth_rate, mars_rate)

    # Earth gains on Mars ~0.46 deg a day: ~100 days to
    # close 45.7 deg.
    assert wait / 86_400.0 == pytest.approx(45.7 / math.degrees(earth_rate - mars_rate) / 86_400.0, rel=1e-6)
    assert 90.0 < wait / 86_400.0 < 110.0

    # Just past the window: nearly a whole synodic period
    # (~780 days).
    late = window_wait(math.radians(44.0), math.radians(44.3), earth_rate, mars_rate)

    assert late / 86_400.0 == pytest.approx(780.0, abs=5.0)


def test_ejection_leaves_along_the_planets_motion():

    parking = 6_678_000.0

    planet_velocity = np.array([0.0, 29_780.0, 0.0])

    point = ejection_point([0, 0, 1], planet_velocity, EARTH_MU, parking, 2_945.0)

    burn = escape_burn(EARTH_MU, parking, 2_945.0)

    # Burn prograde at that point in a counterclockwise orbit.
    position = point * parking
    velocity = np.cross([0, 0, 1], point) * (kepler.circular_speed(EARTH_MU, parking) + burn)

    # Far out, the escape heads along +Y.
    far, outgoing = kepler.propagate(position, velocity, EARTH_MU, 30 * 86_400.0)

    direction = outgoing / np.linalg.norm(outgoing)

    assert direction @ np.array([0, 1, 0]) > 0.999
    assert np.linalg.norm(outgoing) == pytest.approx(2_945.0, rel=0.01)


def test_plan_from_low_earth_orbit():

    parking = 6_678_000.0

    plan = plan_transfer(
        SUN_MU,
        [EARTH_ORBIT, 0, 0], [0, 29_780, 0],
        [0, MARS_ORBIT, 0], [-24_130, 0, 0],
        EARTH_MU,
        [parking, 0, 0], [0, kepler.circular_speed(EARTH_MU, parking), 0]
    )

    assert plan.phase_now == pytest.approx(90.0)
    assert plan.phase_wanted == pytest.approx(44.3, abs=0.5)
    assert plan.departure_burn == pytest.approx(3_590.0, abs=40.0)

    period = 2.0 * math.pi * parking / kepler.circular_speed(EARTH_MU, parking)

    assert 0.0 <= plan.time_to_burn_point < period


def test_closest_approach():

    times = np.linspace(0, 10, 11)

    points = np.stack([times, np.zeros(11), np.zeros(11)], axis=1)
    target = np.stack([np.full(11, 5.0), np.full(11, 2.0), np.zeros(11)], axis=1)

    assert closest_approach(points, times, target) == pytest.approx((2.0, 5.0))


@pytest.mark.parametrize("angle_deg, days", [(90.0, 120.0), (150.0, 200.0), (179.0, 259.0), (230.0, 300.0)])
def test_lambert_arrives(angle_deg, days):

    from vessel.transfer import lambert

    r1 = np.array([EARTH_ORBIT, 0.0, 0.0])

    angle = math.radians(angle_deg)

    # Slightly out of plane, as Mars is.
    r2 = MARS_ORBIT * np.array([math.cos(angle), math.sin(angle), 0.03])

    v1, v2 = lambert(r1, r2, days * 86_400.0, SUN_MU, [0.0, 0.0, 1.0])

    there, speed = kepler.propagate(r1, v1, SUN_MU, days * 86_400.0)

    np.testing.assert_allclose(there, r2, rtol=1e-6, atol=1.0)
    np.testing.assert_allclose(speed, v2, rtol=1e-5, atol=1e-3)


def test_window_search_finds_the_hohmann_window():

    from vessel.transfer import find_window

    earth_speed = math.sqrt(SUN_MU / EARTH_ORBIT)
    mars_speed = math.sqrt(SUN_MU / MARS_ORBIT)

    # Mars 90 deg ahead on circular coplanar orbits: the
    # window comes when it leads by 44.3 deg, ~100 days on.
    earth = ([EARTH_ORBIT, 0, 0], [0, earth_speed, 0])
    mars = ([0, MARS_ORBIT, 0], [-mars_speed, 0, 0])

    import time

    started = time.perf_counter()

    window = find_window(SUN_MU, earth, mars)

    assert time.perf_counter() - started < 3.0

    rates = math.sqrt(SUN_MU / EARTH_ORBIT ** 3) - math.sqrt(SUN_MU / MARS_ORBIT ** 3)

    assert window.departure / 86_400.0 == pytest.approx(math.radians(90.0 - 44.3) / rates / 86_400.0, abs=4.0)
    assert window.transfer_time / 86_400.0 == pytest.approx(259.0, abs=12.0)
    assert window.excess == pytest.approx(2_945.0, abs=60.0)


def test_correction_hits_the_target():

    from vessel.transfer import correction

    mars_speed = math.sqrt(SUN_MU / MARS_ORBIT)

    target = (np.array([0.0, MARS_ORBIT, 0.0]), np.array([-mars_speed, 0.0, 0.0]))

    # A vessel on a slightly wrong path out from Earth's orbit.
    position = np.array([EARTH_ORBIT * 1.01, 0.0, 0.0])
    velocity = np.array([0.0, 32_600.0, 300.0])

    arrival = 200 * 86_400.0

    change = correction(SUN_MU, position, velocity, *target, arrival)

    there, _ = kepler.propagate(position, velocity + change, SUN_MU, arrival)

    goal, _ = kepler.propagate(*target, SUN_MU, arrival)

    assert np.linalg.norm(there - goal) < 1_000.0
