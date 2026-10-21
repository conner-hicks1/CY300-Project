import math

import numpy as np
import pytest

from vessel import kepler


MU = 3.986004418e14
R = 6.371e6


def _at(periapsis, e, nu):
    """A state on a conic: periapsis radius, eccentricity, true anomaly."""

    p = periapsis * (1.0 + e)

    r = p / (1.0 + e * math.cos(nu))

    h = math.sqrt(MU * p)

    return (
        np.array([r * math.cos(nu), r * math.sin(nu), 0.0]),
        np.array([-MU / h * math.sin(nu), MU / h * (e + math.cos(nu)), 0.0]),
    )


def _energy(r, v):

    return float(v @ v) / 2.0 - MU / float(np.linalg.norm(r))


@pytest.mark.parametrize("e", [0.0, 0.3, 0.9, 0.97, 0.995, 0.9995, 1.3, 3.0])
@pytest.mark.parametrize("nu", [0.0, 1.0, 2.5, -2.0])
def test_every_conic_is_followed_exactly(e, nu):

    # (On a hyperbola, only where it goes.)
    if e >= 1.0 and abs(nu) > math.acos(-1.0 / e) - 0.1:
        pytest.skip("not on this hyperbola")

    position, velocity = _at(R + 300_000.0, e, nu)

    period = 2.0 * math.pi * math.sqrt((R + 300_000.0) ** 3 / (1.0 - e) ** 3 / MU) if e < 1.0 else 3.0 * 86_400.0

    times = np.linspace(-0.4, 1.0, 301) * min(period, 30.0 * 86_400.0)

    rs, vs = kepler.propagate_many(position, velocity, MU, times)

    assert np.isfinite(rs).all() and np.isfinite(vs).all()

    energy = _energy(position, velocity)
    momentum = float(np.cross(position, velocity)[2])

    for r, v in zip(rs, vs):

        assert _energy(r, v) == pytest.approx(energy, rel=1e-7, abs=1e-3)
        assert float(np.cross(r, v)[2]) == pytest.approx(momentum, rel=1e-8)

    # One state at a time: the same.
    for k in (17, 150, 290):

        r, v = kepler.propagate(position, velocity, MU, float(times[k]))

        np.testing.assert_allclose(r, rs[k], rtol=1e-8, atol=1e-3)
        np.testing.assert_allclose(v, vs[k], rtol=1e-8, atol=1e-6)


def test_a_transfer_to_the_moon_goes_where_it_should():

    # From low orbit, a burn to the Moon's distance (e ~ 0.97):
    # out to apoapsis in half a period, back to where it began
    # in a whole.
    periapsis = R + 200_000.0
    apoapsis = 384_400_000.0

    a = 0.5 * (periapsis + apoapsis)

    speed = math.sqrt(MU * (2.0 / periapsis - 1.0 / a))

    position = np.array([periapsis, 0.0, 0.0])
    velocity = np.array([0.0, speed, 0.0])

    period = 2.0 * math.pi * math.sqrt(a ** 3 / MU)

    far, _ = kepler.propagate(position, velocity, MU, 0.5 * period)

    assert float(np.linalg.norm(far)) == pytest.approx(apoapsis, rel=1e-6)

    back, _ = kepler.propagate(position, velocity, MU, period)

    np.testing.assert_allclose(back, position, atol=50.0)

    # A metre a second more: a little farther, not somewhere
    # else entirely.
    nudged, _ = kepler.propagate(position, velocity + np.array([0.0, 1.0, 0.0]), MU, 0.5 * period)

    assert abs(float(np.linalg.norm(nudged)) - apoapsis) < 0.02 * apoapsis
