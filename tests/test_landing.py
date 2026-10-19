import math

import numpy as np
import pytest

from math3d import quaternion
from vessel import landing
from vessel.design import DesignDraft
from vessel.parts import load_design, load_parts
from vessel.physics import Controls, Environment, VesselState, step, upright
from vessel.records import state_from_json, state_to_json
from vessel.sequence import parse
from vessel.vessel import Vessel


MOON_RADIUS = 1_737_400.0
MOON_MU = 4.9048695e12
MOON_G = MOON_MU / MOON_RADIUS ** 2


@pytest.fixture(scope="module")
def parts():

    return load_parts()


@pytest.fixture
def lander(parts):
    """Mariner's lander alone (every stage fired)."""

    vessel = Vessel.from_design(load_design("mariner", parts))

    for _ in range(6):
        vessel.stage()

    return vessel


def lower(vessel, down=True):

    landing.set_legs(vessel, down)
    landing.move_legs(vessel, landing.LEG_TIME)


def test_legs_widen_the_stance(lander):

    assert landing.leg_state(lander) == "up"

    # On its engine's bell: a narrow stance.
    bell = lander.core[-1].part.radius_bottom

    assert lander.core[-1].part.kind == "engine"
    assert landing.feet(lander) == pytest.approx(bell)
    assert landing.standing_angle(lander) == pytest.approx(math.degrees(math.atan2(bell, lander.center_of_mass)))

    landing.set_legs(lander, True)
    landing.move_legs(lander, 0.5 * landing.LEG_TIME)

    # Half way: still on the bell.
    assert landing.leg_state(lander) == "lowering" and landing.feet(lander) == pytest.approx(bell)

    landing.move_legs(lander, landing.LEG_TIME)

    assert landing.leg_state(lander) == "down"

    # Out at the mount (the tank's radius and the leg's) plus
    # the span.
    assert landing.feet(lander) == pytest.approx(0.625 + 0.15 + 1.4)
    assert landing.standing_angle(lander) > 40.0
    assert landing.crash_speed(lander) == 12.0

    assert landing.toggle_legs(lander) is False
    assert landing.leg_state(lander) == "raising"


def _touch(vessel, slope=0.0, tilt=0.0, down=-2.0, sideways=0.0, g=MOON_G):

    up = np.array([0.0, 1.0, 0.0])

    normal = np.array([math.sin(math.radians(slope)), math.cos(math.radians(slope)), 0.0])

    axis = np.array([0.0, math.cos(math.radians(tilt)), math.sin(math.radians(tilt))])

    velocity = up * down + np.array([0.0, 0.0, sideways])

    return landing.touch_down(vessel, axis, normal, up, velocity, g)


def test_it_stands_on_gentle_ground(lander):

    result = _touch(lander)

    assert result.outcome == "stand"
    np.testing.assert_allclose(result.axis, [0.0, 1.0, 0.0])


def test_too_fast_breaks_it(lander):

    assert _touch(lander, down=-13.0).outcome == "crash"


def test_steep_ground_tips_it_over_without_legs(lander):

    assert _touch(lander, slope=20.0).outcome == "topple"

    lower(lander)

    assert _touch(lander, slope=20.0).outcome == "stand"
    assert _touch(lander, slope=50.0).outcome == "topple"


def test_coming_down_tilted_tips_it(lander):

    lower(lander)

    assert _touch(lander, tilt=10.0).outcome == "stand"

    result = _touch(lander, tilt=50.0)

    assert result.outcome == "topple" and "off square" in result.reason


def test_sliding_sideways_rolls_it_over(lander):

    lower(lander)

    # The Moon's weak pull: about 1.7 m/s rolls it.
    assert _touch(lander, sideways=1.0).outcome == "stand"
    assert _touch(lander, sideways=2.5).outcome == "topple"

    # The Earth's holds it down harder.
    assert _touch(lander, sideways=2.5, g=9.81).outcome == "stand"


def test_falling_over(lander, parts):

    lying = landing.fallen_axis(
        np.array([math.sin(0.3), math.cos(0.3), 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.zeros(3)
    )

    # Downhill (the slope falls away to -X).
    assert lying[0] > 0.9 or lying[0] < -0.9
    assert abs(float(lying @ np.array([math.sin(0.3), math.cos(0.3), 0.0]))) < 1e-9

    # A small lander survives falling over; a whole rocket on
    # the Earth does not.
    rocket = Vessel.from_design(load_design("pathfinder", parts))

    assert landing.fall_speed(lander, MOON_G) < landing.TOPPLE_CRASH
    assert landing.fall_speed(rocket, 9.81) > landing.TOPPLE_CRASH


def test_the_ground_normal_of_a_slope():

    # A plane tilted 10 deg about Z through (R, 0, 0): the
    # ground's radius along each direction.
    tilt = math.radians(10.0)

    normal = np.array([math.cos(tilt), math.sin(tilt), 0.0])

    point = np.array([MOON_RADIUS, 0.0, 0.0])

    def ground(direction):
        return float(normal @ point) / float(normal @ direction)

    found = landing.ground_normal(ground, np.array([1.0, 0.0, 0.0]), 3.0)

    np.testing.assert_allclose(found, normal, atol=1e-4)

    assert landing.slope_of(found, np.array([1.0, 0.0, 0.0])) == pytest.approx(10.0, abs=0.05)


def _moon(normal=None):

    return Environment(
        mu=MOON_MU,
        spin=np.zeros(3),
        air=lambda r, d: (0.0, 0.0),
        ground=lambda d: MOON_RADIUS,
        normal=(lambda d: normal) if normal is not None else None
    )


def _falling(height=0.5, speed=2.0):

    up = np.array([1.0, 0.0, 0.0])

    return VesselState(
        position=up * (MOON_RADIUS + height),
        velocity=-up * speed,
        orientation=upright(up),
        angular_velocity=np.zeros(3)
    )


def test_it_settles_square_to_the_slope(lander):

    lower(lander)

    tilt = math.radians(12.0)

    normal = np.array([math.cos(tilt), math.sin(tilt), 0.0])

    state = _falling()

    controls = Controls(sas=False)

    for _ in range(100):

        step(lander, state, _moon(normal), controls, 0.02)

        if state.landed:
            break

    assert state.landed and not state.toppled and not state.destroyed

    np.testing.assert_allclose(state.up, normal, atol=1e-6)


def test_on_its_side_it_cannot_lift_off(lander):

    tilt = math.radians(25.0)

    normal = np.array([math.cos(tilt), math.sin(tilt), 0.0])

    state = _falling()

    for _ in range(100):

        step(lander, state, _moon(normal), Controls(sas=False), 0.02)

        if state.landed:
            break

    assert state.toppled and state.landed and not state.destroyed and "slope" in state.fell

    # Lying along the ground, raised off it by its radius.
    assert abs(float(state.up @ normal)) < 1e-6
    assert float(np.linalg.norm(state.position)) == pytest.approx(MOON_RADIUS + state.rest_height)

    lander.throttle = 1.0
    lander.parts[-1].ignited = True

    for _ in range(50):
        step(lander, state, _moon(normal), Controls(sas=False), 0.02)

    assert state.landed

    # Kept in the saves.
    loaded = state_from_json(state_to_json(state))

    assert loaded.toppled and loaded.rest_height == pytest.approx(state.rest_height) and loaded.fell == state.fell


def test_a_tall_rocket_falling_over_in_strong_gravity_breaks_up(parts):

    rocket = Vessel.from_design(load_design("pathfinder", parts))

    up = np.array([1.0, 0.0, 0.0])

    state = VesselState(
        position=up * (MOON_RADIUS + 0.2),
        velocity=-up * 1.0,
        # Lying nearly flat as it comes down.
        orientation=quaternion.look_rotation(up, np.array([0.0, 1.0, 0.0])),
        angular_velocity=np.zeros(3)
    )

    # The Earth's pull (on a Moon-sized ball).
    environment = Environment(mu=9.81 * MOON_RADIUS ** 2, spin=np.zeros(3), air=lambda r, d: (0.0, 0.0), ground=lambda d: MOON_RADIUS)

    for _ in range(50):

        step(rocket, state, environment, Controls(sas=False), 0.02)

        if state.landed or state.destroyed:
            break

    assert state.destroyed and "fell over" in state.fell


def test_the_computer_lowers_legs():

    program = parse("legs down\ngear up\nwait until slope < 10")

    assert [statement.value for statement in program.statements[:2]] == [True, False]


def test_radial_groups_keep_their_angle(parts):

    design = load_design("mariner", parts)

    vessel = Vessel.from_design(design)

    legs = landing.legs(vessel)

    assert legs and all(state.angle == 45.0 for state in legs)

    draft = DesignDraft.from_design(design)

    again = draft.to_design(parts)

    assert any(group.angle == 45.0 for entry in again.stack for group in entry.radial)
