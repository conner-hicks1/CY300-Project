import math

import numpy as np
import pytest

from vessel import kepler, maneuver
from vessel.maneuver import ManeuverNode
from vessel.parts import load_design, load_parts
from vessel.vessel import G0, Vessel


EARTH_MU = 3.986004418e14
EARTH_RADIUS = 6_371_000.0

LOW = EARTH_RADIUS + 200_000.0


def circular(radius: float = LOW):

    return np.array([radius, 0.0, 0.0]), np.array([0.0, kepler.circular_speed(EARTH_MU, radius), 0.0])


def test_node_frame_and_back():

    position, velocity = circular()

    prograde, normal, radial = maneuver.node_frame(position, velocity)

    np.testing.assert_allclose(prograde, [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(normal, [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(radial, [1, 0, 0], atol=1e-12)

    vector = maneuver.to_inertial(position, velocity, 100.0, -20.0, 5.0)

    assert maneuver.from_inertial(position, velocity, vector) == pytest.approx((100.0, -20.0, 5.0))


def test_remaining_counts_only_the_thrust():

    node = ManeuverNode(0.0, 100.0)
    node.vector = np.array([0.0, 100.0, 0.0])
    node.baseline = np.array([5.0, 5.0, 5.0])

    np.testing.assert_allclose(node.remaining(np.array([5.0, 65.0, 5.0])), [0.0, 40.0, 0.0])

    assert node.size == pytest.approx(100.0)


def test_apsis_times():

    # An ellipse from its periapsis.
    position = np.array([LOW, 0.0, 0.0])
    velocity = np.array([0.0, kepler.circular_speed(EARTH_MU, LOW) * 1.1, 0.0])

    shape = kepler.orbit_shape(position, velocity, EARTH_MU)

    assert maneuver.time_to_apsis(position, velocity, EARTH_MU, "periapsis") == pytest.approx(0.0, abs=1e-6) or \
        maneuver.time_to_apsis(position, velocity, EARTH_MU, "periapsis") == pytest.approx(shape.period, rel=1e-9)

    half = maneuver.time_to_apsis(position, velocity, EARTH_MU, "apoapsis")

    assert half == pytest.approx(shape.period / 2.0, rel=1e-9)

    there, _ = kepler.propagate(position, velocity, EARTH_MU, half)

    assert float(np.linalg.norm(there)) == pytest.approx(shape.apoapsis, rel=1e-9)

    # A quarter of the way round: still ahead, both.
    quarter, speed = kepler.propagate(position, velocity, EARTH_MU, shape.period / 4.0)

    to_peri = maneuver.time_to_apsis(quarter, speed, EARTH_MU, "periapsis")

    assert to_peri == pytest.approx(shape.period * 0.75, rel=1e-6)

    # Coming in on a hyperbola: periapsis ahead, no apoapsis.
    start = np.array([5.0e8, 1.0e7, 0.0])
    inbound = np.array([-3_000.0, 0.0, 0.0])

    wait = maneuver.time_to_apsis(start, inbound, EARTH_MU, "periapsis")

    assert maneuver.time_to_apsis(start, inbound, EARTH_MU, "apoapsis") is None

    lowest, _ = kepler.propagate(start, inbound, EARTH_MU, wait)

    open_shape = kepler.orbit_shape(start, inbound, EARTH_MU)

    assert float(np.linalg.norm(lowest)) == pytest.approx(open_shape.periapsis, rel=1e-6)

    # Going out, it is behind.
    assert maneuver.time_to_apsis(start, -inbound, EARTH_MU, "periapsis") is None


def test_circularize_and_capture():

    # At the apoapsis of a transfer ellipse: circular after.
    position = np.array([LOW, 0.0, 0.0])
    velocity = np.array([0.0, kepler.circular_speed(EARTH_MU, LOW) * 1.15, 0.0])

    half = maneuver.time_to_apsis(position, velocity, EARTH_MU, "apoapsis")

    there, speed = kepler.propagate(position, velocity, EARTH_MU, half)

    change = maneuver.circularize(there, speed, EARTH_MU)

    shape = kepler.orbit_shape(there, speed + change, EARTH_MU)

    assert shape.eccentricity < 1e-9

    # Arriving fast at a periapsis: a long captured orbit.
    fast = np.array([0.0, math.sqrt(2.0 * EARTH_MU / LOW) + 800.0, 0.0])

    capture = maneuver.capture(position, fast, EARTH_MU, 400_000_000.0)

    captured = kepler.orbit_shape(position, fast + capture, EARTH_MU)

    assert captured.apoapsis == pytest.approx(400_000_000.0, rel=1e-6)

    # Cheaper than circular.
    assert np.linalg.norm(capture) < np.linalg.norm(maneuver.circularize(position, fast, EARTH_MU))


def test_lower_periapsis():

    position, velocity = circular()

    change = maneuver.lower_periapsis(position, velocity, EARTH_MU, EARTH_RADIUS + 45_000.0)

    shape = kepler.orbit_shape(position, velocity + change, EARTH_MU)

    assert shape.periapsis == pytest.approx(EARTH_RADIUS + 45_000.0, abs=1.0)

    # Retrograde, ~50 m/s from 200 km.
    assert float(np.dot(change, velocity)) < 0.0
    assert 30.0 < float(np.linalg.norm(change)) < 70.0

    # Already low enough: nothing.
    assert maneuver.lower_periapsis(position, velocity, EARTH_MU, LOW + 1_000.0) is None


def test_burn_time():

    vessel = Vessel.from_design(load_design("pathfinder", load_parts()))

    # Up to the upper stage, in orbit.
    vessel.stage()
    vessel.stage()

    vessel.dropped.clear()

    thrust, flow = vessel.thrust(0.0)

    vessel.throttle = 1.0

    thrust, flow = vessel.thrust(0.0)

    mass = vessel.mass

    exhaust = thrust / flow

    # The rocket equation for 500 m/s.
    expected = mass / flow * (1.0 - math.exp(-500.0 / exhaust))

    assert maneuver.burn_time(vessel, 500.0) == pytest.approx(expected, rel=1e-9)

    assert maneuver.burn_time(vessel, 0.0) == 0.0

    # More than it has.
    assert maneuver.burn_time(vessel, 50_000.0) == math.inf

    # Nothing about the vessel changed.
    assert vessel.mass == mass


def test_braking():

    # 2 km up, falling at 100 m/s on the Moon, 5 m/s^2 of thrust.
    found = maneuver.braking(2_000.0, -100.0, 100.0, 1.62, 5.0)

    # Stopping height: (v^2 + 2 g H) / (2 a).
    assert found.height == pytest.approx((100.0 ** 2 + 2.0 * 1.62 * 2_000.0) / 10.0)

    # Falling from there, it stops at the ground.
    fall_speed = math.sqrt(100.0 ** 2 + 2.0 * 1.62 * (2_000.0 - found.height))

    assert fall_speed ** 2 / (2.0 * (5.0 - 1.62)) == pytest.approx(found.height)

    assert 0.0 < found.start_in < found.impact_in

    # Climbing: nothing to do; too weak: never stops.
    assert maneuver.braking(2_000.0, 10.0, 50.0, 1.62, 5.0) is None
    assert maneuver.braking(2_000.0, -10.0, 10.0, 9.8, 5.0).start_in == math.inf
