import numpy as np
import pytest

from vessel import debris, kepler
from vessel.parts import load_design, load_parts
from vessel.vessel import Vessel


MU = 3.986004418e14
R = 6_371_000.0


def test_the_upper_air_thins_with_height():

    heights = [150e3, 250e3, 400e3, 700e3, 1_200e3]

    densities = [debris.thermosphere(h) for h in heights]

    assert all(a > b for a, b in zip(densities, densities[1:]))
    assert debris.thermosphere(400e3) == pytest.approx(2.8e-12, rel=1e-6)


def test_low_orbits_come_down_soon_high_ones_never():

    b = 0.01

    days = [debris.lifetime(h, b) / 86_400.0 for h in (200e3, 400e3, 600e3, 800e3)]

    assert days[0] < 5.0
    assert 100.0 < days[1] < 2_000.0
    assert days[2] > 3_000.0 and days[3] > 30_000.0


def test_decay_lowers_the_orbit_and_keeps_its_shape():

    radius = R + 400e3

    position = np.array([radius, 0.0, 0.0])
    velocity = np.array([0.0, kepler.circular_speed(MU, radius) * 1.01, 0.0])

    before = kepler.orbit_shape(position, velocity, MU)

    after_p, after_v = debris.decay(position, velocity, MU, R, 0.01, 30 * 86_400.0)

    after = kepler.orbit_shape(after_p, after_v, MU)

    assert after.semi_major_axis < before.semi_major_axis
    assert after.eccentricity == pytest.approx(before.eccentricity, abs=1e-9)

    # Same direction, same plane.
    np.testing.assert_allclose(after_p / np.linalg.norm(after_p), position / radius)


def test_a_stage_tumbling_has_a_ballistic_coefficient():

    parts = load_parts()

    upper = Vessel.from_design(load_design("pathfinder", parts))

    assert 1e-4 < debris.ballistic(upper) < 0.05


def test_a_collision_throws_a_cloud():

    position = np.array([R + 500e3, 0.0, 0.0])
    velocity = np.array([0.0, 7_600.0, 0.0])

    cloud = debris.fragments(position, velocity, 1_000.0, 20_000.0, seed=7)

    assert 4 <= len(cloud) <= 16

    kicks = [float(np.linalg.norm(v - velocity)) for _, _, v in cloud]

    assert max(kicks) <= 200.0 + 1e-6 and min(kicks) >= 20.0 - 1e-6

    # The same seed, the same cloud.
    again = debris.fragments(position, velocity, 1_000.0, 20_000.0, seed=7)

    assert all(np.allclose(a[2], b[2]) for a, b in zip(cloud, again))


def test_passes_and_hits_in_straight_lines():

    # 5 km off, closing at 10 m/s, missing by 300 m.
    relative = np.array([5_000.0, 300.0, 0.0])
    closing = np.array([-10.0, 0.0, 0.0])

    when, miss = debris.closest_approach(relative, closing, 1_000.0)

    assert when == pytest.approx(500.0) and miss == pytest.approx(300.0)
    assert debris.closest_approach(relative, closing, 100.0) is None

    # Through each other in one step at 8 km/s: caught.
    now = np.array([-50.0, 2.0, 0.0])
    velocity = np.array([-8_000.0, 0.0, 0.0])

    assert debris.swept_hit(now, velocity, 1.0 / 60.0, 10.0)
    assert not debris.swept_hit(now + np.array([0.0, 50.0, 0.0]), velocity, 1.0 / 60.0, 10.0)
