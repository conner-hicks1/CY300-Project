import math

import numpy as np
import pytest

from vessel import kepler, soi


EARTH_MU = 3.986004418e14
MOON_MU = 4.9048695e12
EARTH_RADIUS = 6_371_000.0
MOON_RADIUS = 1_737_400.0
MOON_DISTANCE = 384_400_000.0


def bodies():

    earth_soi = soi.soi_radius(1.495978707e11, 5.972e24, 1.989e30)
    moon_soi = soi.soi_radius(MOON_DISTANCE, 7.342e22, 5.972e24)

    return {
        None: soi.Body(None, "the Sun", soi.SUN_MU, 6.957e8, math.inf, None),
        "earth": soi.Body("earth", "Earth", EARTH_MU, EARTH_RADIUS, earth_soi, None),
        "moon": soi.Body("moon", "Moon", MOON_MU, MOON_RADIUS, moon_soi, "earth"),
    }


def test_sphere_sizes():

    table = bodies()

    # Textbook values: the Moon's ~66,100 km, Earth's ~924,600 km.
    assert table["moon"].soi == pytest.approx(66.1e6, rel=0.01)
    assert table["earth"].soi == pytest.approx(924.6e6, rel=0.01)


def test_transitions():

    table = bodies()

    moon_at = np.array([MOON_DISTANCE, 0.0, 0.0])

    # Low orbit: stays with Earth.
    assert soi.transition([7e6, 0, 0], table["earth"], table, {"moon": moon_at}) == (False, None)

    # Inside the Moon's sphere: the Moon.
    assert soi.transition(moon_at + [30e6, 0, 0], table["earth"], table, {"moon": moon_at}) == (True, "moon")

    # Out past the Moon's sphere, from the Moon: back to Earth.
    assert soi.transition([70e6, 0, 0], table["moon"], table, {}) == (True, "earth")

    # Out of Earth's: the Sun (key None).
    assert soi.transition([1e9, 0, 0], table["earth"], table, {"moon": moon_at}) == (True, None)


def test_rebase_keeps_the_absolute_state():

    position, velocity = soi.rebase(
        [1.0, 2.0, 3.0], [0.1, 0.2, 0.3],
        [100.0, 0.0, 0.0], [10.0, 0.0, 0.0],
        [0.0, 50.0, 0.0], [0.0, 5.0, 0.0]
    )

    np.testing.assert_allclose(position + [0.0, 50.0, 0.0], [101.0, 2.0, 3.0])
    np.testing.assert_allclose(velocity + [0.0, 5.0, 0.0], [10.1, 0.2, 0.3])


def test_closed_orbit_prediction():

    table = bodies()

    r = EARTH_RADIUS + 400_000.0

    prediction = soi.predict([r, 0, 0], [0, 0, kepler.circular_speed(EARTH_MU, r)], table["earth"], table, {})

    (patch,) = prediction.patches

    assert patch.end == "orbit"
    np.testing.assert_allclose(np.linalg.norm(patch.points, axis=1), r, rtol=1e-6)

    # It closes: the last point is the first.
    np.testing.assert_allclose(patch.points[-1], patch.points[0], atol=1.0)


def test_impact_prediction():

    table = bodies()

    r = EARTH_RADIUS + 100_000.0

    prediction = soi.predict([r, 0, 0], [0, 0, 6_000.0], table["earth"], table, {})

    assert prediction.patches[0].end == "impact"
    assert np.linalg.norm(prediction.patches[0].points[-1]) < EARTH_RADIUS


def test_lunar_transfer_finds_the_moon():
    """A Hohmann-like transfer from low orbit, timed to meet the Moon."""

    table = bodies()

    r = EARTH_RADIUS + 200_000.0

    # Transfer ellipse from r to the Moon's distance.
    a = 0.5 * (r + MOON_DISTANCE)

    speed = math.sqrt(EARTH_MU * (2.0 / r - 1.0 / a))

    transfer_time = math.pi * math.sqrt(a ** 3 / EARTH_MU)

    # The Moon on a circular orbit in the same plane, placed
    # so it arrives at the apogee (-x) with the vessel.
    moon_speed = kepler.circular_speed(EARTH_MU, MOON_DISTANCE)

    angle = math.pi - moon_speed / MOON_DISTANCE * transfer_time

    moon_position = MOON_DISTANCE * np.array([math.cos(angle), 0.0, -math.sin(angle)])
    moon_velocity = moon_speed * np.array([-math.sin(angle), 0.0, -math.cos(angle)])

    prediction = soi.predict(
        [r, 0.0, 0.0],
        [0.0, 0.0, -speed],
        table["earth"],
        table,
        {"moon": (moon_position, moon_velocity)},
        samples=2_000
    )

    first = prediction.patches[0]

    assert first.end == "encounter" and first.next_body == "moon"

    encounter = prediction.encounter

    assert encounter is not None and encounter.body == "moon"

    # Passes the Moon (or hits it) well inside its sphere.
    assert encounter.periapsis < 20e6


def test_vectorized_propagation_matches():

    r = EARTH_RADIUS + 300_000.0

    position = np.array([r, 0.0, 0.0])

    for speed in (7_000.0, 7_800.0, 11_500.0):

        velocity = np.array([0.0, 1_000.0, speed])

        times = np.linspace(0.0, 20_000.0, 17)

        many, many_v = kepler.propagate_many(position, velocity, EARTH_MU, times)

        for t, p, v in zip(times, many, many_v):

            one, one_v = kepler.propagate(position, velocity, EARTH_MU, t)

            np.testing.assert_allclose(p, one, rtol=1e-8, atol=1e-3)
            np.testing.assert_allclose(v, one_v, rtol=1e-8, atol=1e-6)


@pytest.mark.parametrize("speed", [8_000.0, 10_500.0, 11_500.0, 15_000.0])
def test_time_to_radius(speed):

    r = EARTH_RADIUS + 300_000.0

    position = np.array([r, 0.0, 0.0])
    velocity = np.array([-500.0, 0.0, speed])

    target = 50_000_000.0

    t = kepler.time_to_radius(position, velocity, EARTH_MU, target)

    shape = kepler.orbit_shape(position, velocity, EARTH_MU)

    if shape.apoapsis is not None and shape.apoapsis < target:
        assert t is None
        return

    there, _ = kepler.propagate(position, velocity, EARTH_MU, t)

    assert np.linalg.norm(there) == pytest.approx(target, rel=1e-6)


def test_a_small_moon_has_room_to_be_landed_on():

    # Phobos: Laplace's sphere (7 km) is within it (11 km).
    laplace = soi.soi_radius(9_376_000.0, 1.0659e16, 6.4171e23)

    assert laplace < 11_100.0

    assert soi.sphere(9_376_000.0, 1.0659e16, 6.4171e23, 11_100.0) == soi.MIN_SOI_RADII * 11_100.0

    # The Moon's own is far bigger than that: kept.
    assert soi.sphere(MOON_DISTANCE, 7.342e22, 5.972e24, 1_737_400.0) == soi.soi_radius(MOON_DISTANCE, 7.342e22, 5.972e24)


def test_a_flyby_is_followed_in_and_out_again():

    earth_mu = 3.986004418e14
    moon_mu = 4.9048695e12

    earth = soi.Body("earth", "Earth", earth_mu, 6.371e6, 9.25e8, None)
    moon = soi.Body("moon", "Moon", moon_mu, 1.7374e6, soi.soi_radius(MOON_DISTANCE, 7.342e22, 5.972e24), "earth")

    bodies = {"earth": earth, "moon": moon}

    moon_speed = math.sqrt(earth_mu / MOON_DISTANCE)

    moon_position = np.array([MOON_DISTANCE, 0.0, 0.0])
    moon_velocity = np.array([0.0, moon_speed, 0.0])

    # 150,000 km short of the Moon, passing it 20,000 km to
    # one side at 1 km/s relative: in through its sphere and
    # out again.
    position = moon_position + np.array([-150_000_000.0, 20_000_000.0, 0.0])
    velocity = moon_velocity + np.array([1_000.0, 0.0, 0.0])

    prediction = soi.predict(position, velocity, earth, bodies, {"moon": (moon_position, moon_velocity)}, horizon=20.0 * 86_400.0)

    assert [patch.body for patch in prediction.patches][:3] == ["earth", "moon", "earth"]

    into, round_moon, out = prediction.patches[:3]

    assert into.end == "encounter" and round_moon.end == "escape" and round_moon.next_body == "earth"

    # Out again where it left the Moon's sphere, then on round
    # the Earth (not caught by the Moon again at once).
    moon_then, _ = kepler.propagate(moon_position, moon_velocity, earth_mu, float(out.times[0]))

    np.testing.assert_allclose(out.points[0], round_moon.points[-1] + moon_then, atol=5_000.0)

    assert float(np.linalg.norm(out.points[0] - moon_then)) > 0.95 * moon.soi

    assert out.times[0] == pytest.approx(round_moon.times[-1], rel=1e-6)
