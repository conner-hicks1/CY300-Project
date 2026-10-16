import numpy as np
import pytest

from planet.shape import direction_of
from planet.surface import (
    equator_to_pole_drop,
    prevailing_wind,
    surface_precipitation,
    surface_temperature,
    wetness
)


def at(latitude, longitude=0.0):

    return direction_of(latitude, longitude)[None, :]


def test_drop_shrinks_under_thick_air():

    airless = equator_to_pole_drop(-20.0, 0.0)
    earth = equator_to_pole_drop(15.0, 1.0)
    venus = equator_to_pole_drop(464.0, 92.0)

    assert airless == pytest.approx(60.0)
    assert 35.0 < earth < 55.0
    assert venus < 15.0


def test_poles_colder_and_heights_colder():

    equator = surface_temperature(at(0.0), [0.0])[0]
    pole = surface_temperature(at(89.0), [0.0])[0]

    assert equator > 25.0 and pole < -15.0

    mountain = surface_temperature(at(0.0), [4_000.0], lapse_rate=6.5)[0]

    assert mountain == pytest.approx(equator - 26.0)

    # Seas below sea level are not warmed by their depth.
    assert surface_temperature(at(0.0), [-3_000.0])[0] == pytest.approx(equator)


def test_rain_belts():

    moisture = np.full(1, 0.5)

    tropics = surface_precipitation(at(0.0), moisture)[0]
    subtropics = surface_precipitation(at(25.0), moisture)[0]
    storms = surface_precipitation(at(50.0), moisture)[0]
    polar = surface_precipitation(at(85.0), moisture)[0]

    assert tropics > storms > subtropics
    assert polar < storms

    assert wetness(tropics, 27.0) > 1.2


def test_winds_are_tangent_and_by_belt():

    directions = np.concatenate([at(10.0, 40.0), at(45.0, -70.0), at(-45.0, 120.0)])

    wind = prevailing_wind(directions)

    np.testing.assert_allclose(np.einsum("ij,ij->i", wind, directions), 0.0, atol=1e-12)

    # Trade winds blow west, the westerlies east.
    east = np.cross([0.0, 1.0, 0.0], directions)
    east /= np.linalg.norm(east, axis=1, keepdims=True)

    along = np.einsum("ij,ij->i", wind, east)

    assert along[0] < 0.0 < along[1] and along[2] > 0.0
