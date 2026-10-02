import math

import numpy as np
import pytest

from planet.solar import solar_time, sun_direction


EQUATOR = np.array([0.0, 0.0, 1.0])


def test_noon_puts_the_sun_overhead_at_equinox():

    sun = sun_direction(EQUATOR, 12.0, 0.0)

    np.testing.assert_allclose(sun, EQUATOR, atol=1e-12)

    hour, declination, elevation = solar_time(EQUATOR, sun)

    assert hour == pytest.approx(12.0)
    assert declination == pytest.approx(0.0)
    assert elevation == pytest.approx(math.pi / 2)


def test_six_and_eighteen_are_on_the_horizon():

    for hour in (6.0, 18.0):

        _, _, elevation = solar_time(EQUATOR, sun_direction(EQUATOR, hour, 0.0))

        assert elevation == pytest.approx(0.0, abs=1e-12)


def test_midnight_is_below_the_horizon():

    _, _, elevation = solar_time(EQUATOR, sun_direction(EQUATOR, 0.0, 0.0))

    assert elevation == pytest.approx(-math.pi / 2)


@pytest.mark.parametrize("hour", [0.25, 7.5, 12.0, 19.9, 23.75])
@pytest.mark.parametrize("declination", [-0.4, 0.0, 0.3])
def test_round_trip(hour, declination):

    point = np.array([0.3, 0.5, -0.8])
    point /= np.linalg.norm(point)

    found_hour, found_declination, _ = solar_time(point, sun_direction(point, hour, declination))

    assert found_hour == pytest.approx(hour)
    assert found_declination == pytest.approx(declination)


def test_summer_noon_is_higher_in_the_north():

    # 45 N at noon: sun higher with positive declination.
    point = np.array([0.0, math.sin(math.radians(45.0)), math.cos(math.radians(45.0))])

    _, _, summer = solar_time(point, sun_direction(point, 12.0, math.radians(23.44)))
    _, _, winter = solar_time(point, sun_direction(point, 12.0, math.radians(-23.44)))

    assert math.degrees(summer) == pytest.approx(90.0 - 45.0 + 23.44)
    assert math.degrees(winter) == pytest.approx(90.0 - 45.0 - 23.44)
