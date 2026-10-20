import math

import numpy as np
import pytest

from math3d import quaternion
from vessel import power, rover
from vessel.layout import copy_orientations, copy_positions
from vessel.parts import load_design, load_parts
from vessel.physics import Controls, Environment, VesselState, step, upright
from vessel.vessel import Vessel


EARTH_RADIUS = 6_371_000.0
EARTH_MU = 3.986004418e14


@pytest.fixture(scope="module")
def parts():

    return load_parts()


@pytest.fixture
def buggy(parts):

    return Vessel.from_design(load_design("rover", parts))


def test_four_wheels_in_pairs_their_axles_across(buggy):

    wheels = rover.wheels(buggy)

    assert len(wheels) == 1 and wheels[0].count == 4

    positions = np.array(copy_positions(wheels[0]))

    np.testing.assert_allclose(sorted(np.abs(positions[:, 0])), [0.7] * 4)
    assert sorted(np.sign(positions[:, 2])) == [-1.0, -1.0, 1.0, 1.0]

    for turned, position in zip(copy_orientations(wheels[0]), positions):

        axle = quaternion.rotate_vector(turned, np.array([0.0, 1.0, 0.0]))

        np.testing.assert_allclose(axle, [math.copysign(1.0, position[0]), 0.0, 0.0], atol=1e-9)

    # Its body held up off the ground.
    assert rover.clearance(buggy) == pytest.approx(0.1)


def test_driving_speeds_up_coasts_and_turns(buggy):

    state = VesselState(np.array([EARTH_RADIUS, 0.0, 0.0]), np.zeros(3), upright(np.array([1.0, 0.0, 0.0])), np.zeros(3), landed=True)

    up = np.array([1.0, 0.0, 0.0])

    start = state.position.copy()

    for _ in range(200):
        rover.drive(buggy, state, 1.0, 0.0, up, 0.05)

    # 10 s at 0.6 m/s^2, held to its top speed.
    assert state.drive == pytest.approx(rover.top_speed(buggy))

    moved = state.position - start

    assert float(np.linalg.norm(moved)) > 15.0
    assert abs(float(moved @ up)) < 1e-9

    # Its front is its -Z: where it went.
    front = quaternion.rotate_vector(state.orientation, np.array([0.0, 0.0, -1.0]))

    assert float(moved @ front) > 0.0

    assert rover.wheel_power(buggy, state.drive) == pytest.approx(4 * 60.0)

    # Let go: it coasts down.
    for _ in range(200):
        rover.drive(buggy, state, 0.0, 0.0, up, 0.05)

    assert state.drive < 0.1

    # Stopped, A turns it on the spot.
    before = front.copy()

    for _ in range(20):
        rover.drive(buggy, state, 0.0, 1.0, up, 0.05)

    after = quaternion.rotate_vector(state.orientation, np.array([0.0, 0.0, -1.0]))

    assert math.degrees(math.acos(min(float(before @ after), 1.0))) == pytest.approx(math.degrees(rover.SPIN_TURN * 1.0), abs=1.0)

    # Without power: it goes nowhere.
    buggy.powered = False

    rover.drive(buggy, state, 1.0, 0.0, up, 1.0)

    assert state.drive == pytest.approx(0.0, abs=0.1)


def test_a_rover_drives_on_the_ground_in_the_physics(buggy):

    environment = Environment(mu=EARTH_MU, spin=np.zeros(3), air=lambda r, d: (1.0, 1.2), ground=lambda d: EARTH_RADIUS)

    up = np.array([1.0, 0.0, 0.0])

    state = VesselState(up * EARTH_RADIUS, np.zeros(3), upright(up), np.zeros(3))

    from vessel.physics import land

    land(state, environment, buggy)

    assert float(np.linalg.norm(state.position)) == pytest.approx(EARTH_RADIUS + 0.1)

    controls = Controls(pitch=-1.0, sas=False)

    for _ in range(250):
        step(buggy, state, environment, controls, 0.02)

    assert state.landed and not state.toppled
    assert float(np.linalg.norm(state.position)) == pytest.approx(EARTH_RADIUS + 0.1)
    assert state.drive > 2.0

    # Its wheels' draw is in its power.
    assert power.base_draw(buggy) > 200.0


def test_a_helicopter_flies_in_thin_air(parts):

    helicopter = Vessel("Helicopter", [])

    from vessel.vessel import PartState

    helicopter.parts = [PartState(parts["helicopter_rotor"], None), PartState(parts["helicopter_body"], None, charge=40.0)]

    helicopter._restack()

    mars_g = 3.71

    weight = helicopter.mass * mars_g

    # Mars's air (0.015 kg/m^3): a little more than its weight.
    assert 1.0 < rover.rotor_lift(helicopter, 0.015) / weight < 1.4

    # Vacuum: nothing.
    assert rover.rotor_lift(helicopter, 0.0) == 0.0

    helicopter.throttle = 1.0

    assert power.base_draw(helicopter) >= 350.0

    radius = 3_389_500.0

    environment = Environment(
        mu=mars_g * radius ** 2, spin=np.zeros(3),
        air=lambda r, d: (0.006, 0.015), ground=lambda d: radius
    )

    up = np.array([1.0, 0.0, 0.0])

    state = VesselState(up * radius, np.zeros(3), upright(up), np.zeros(3), landed=True)

    for _ in range(500):
        step(helicopter, state, environment, Controls(), 0.02)

    assert not state.landed and float(np.linalg.norm(state.position)) - radius > 1.0
