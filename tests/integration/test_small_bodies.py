import numpy as np
import pytest

from programme import fleet

from tests.integration.harness import Game


pytestmark = pytest.mark.integration


def _readout(game: Game):

    app = game.app

    return app.vessel_system.readout(app.scene, app.orbit_system, app.planet_system)


@pytest.fixture
def lander(play):

    game: Game = play(sandbox=True, design="mariner")

    for _ in range(6):
        game.app.stage_vessel(game.vessel)

    game.vessel.vessel.throttle = 0.0

    return game


def test_a_small_moon_can_be_landed_on_and_orbited(lander):

    game = lander

    app = game.app

    phobos = fleet.body_profiles(app.scene)["phobos"]

    # Its sphere is bigger than it (Laplace's alone: 7 km,
    # within its 11 km).
    sphere = app.vessel_system.soi_bodies(app.scene)[phobos]

    assert sphere.soi > 3.0 * sphere.radius

    game.land_at(phobos, 10.0, 20.0)

    game.step(30)

    readout = _readout(game)

    # On Phobos itself (not at Earth's radius from its
    # center), at Phobos.
    assert 8_000.0 < float(np.linalg.norm(game.vessel.state.position)) < 15_000.0
    assert readout.situation == "Landed" and readout.body == "Phobos"

    # A few km up round it: it stays round it, and its
    # apoapsis and periapsis keep still (from the mean radius,
    # not the lumpy ground under it).
    game.put_in_orbit(body=phobos, altitude=6_000.0)

    game.step(5)

    before = _readout(game)

    game.warp(100.0)

    game.step(600)

    after = _readout(game)

    assert after.body == "Phobos" and after.situation == "Orbiting"
    assert after.apoapsis == pytest.approx(before.apoapsis, abs=1.0) and after.periapsis == pytest.approx(before.periapsis, abs=1.0)


def test_up_on_a_comet_is_square_to_its_lobes(lander):

    game = lander

    app = game.app

    comet = fleet.body_profiles(app.scene)["churyumov_gerasimenko"]

    vessels = app.vessel_system

    figure = vessels.figure(app.scene, comet)

    assert figure is not None and not figure.spheroid

    game.land_at(comet, 0.0, 0.0)

    game.step(3)

    state = game.vessel.state

    environment = vessels.environment(app.scene, comet, app.orbit_system, app.planet_system)

    assert environment.level is not None

    # Gravity pulls down the level, at the strength of the
    # comet's.
    from vessel.physics import gravity, up_at

    pull = gravity(environment, state.position)

    up = up_at(environment, state.position)

    assert float(pull @ up) < 0.0 and np.linalg.norm(np.cross(pull, up)) < 1e-9 * np.linalg.norm(pull)

    # Far off, straight out from the center again.
    far = state.position / np.linalg.norm(state.position) * 100_000.0

    np.testing.assert_allclose(up_at(environment, far), far / np.linalg.norm(far), atol=1e-12)
