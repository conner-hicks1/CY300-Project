import numpy as np
import pytest

from programme import fleet
from vessel.debris import FRAGMENT
from vessel.physics import VesselState
from vessel.vessel import PartState, Vessel

from tests.integration.harness import Game


pytestmark = pytest.mark.integration


def _in_orbit(game: Game, altitude: float = 500_000.0):

    game.step(5)

    component = game.vessel

    game.stage()

    game.put_in_orbit(altitude=altitude)

    game.step(5)

    return component


def test_a_stage_left_in_orbit_stays_there(play):
    """Pathfinder in a 500 km orbit drops its stages: they stay in orbit as debris (past the old range and age), saved with the fleet."""

    game: Game = play()

    component = _in_orbit(game)

    game.stage()
    game.step(60)

    game.stage()
    game.step(60)

    vessels = game.app.vessel_system

    kept = vessels.orbital_debris(game.app.scene, component.body)

    assert len(kept) >= 2

    # A day on: still there (they would be gone after 10 min
    # falling stages).
    start = game.clock

    game.run_until(lambda: game.clock - start > 86_400.0, 2_000, every=lambda: game.warp(10_000.0))

    assert len(vessels.orbital_debris(game.app.scene, component.body)) >= 2

    records = fleet.snapshot(game.app.scene, vessels, game.app.science_system, game.app.orbit_system.time)

    assert sum(1 for record in records if record.get("debris")) >= 2


def test_a_fragment_at_a_kilometre_a_second_shatters_a_vessel(play):

    game: Game = play()

    component = _in_orbit(game)

    state = component.state

    along = state.velocity / np.linalg.norm(state.velocity)

    # A fragment 300 m behind catching up at 1 km/s (faster:
    # its orbit still clears the air).
    piece = Vessel("Fragment", [PartState(FRAGMENT, None)])
    piece._restack()

    up = state.position / np.linalg.norm(state.position)

    fragment_state = VesselState(
        position=state.position - along * 300.0 + up * 3.0,
        velocity=state.velocity + along * 1_000.0,
        orientation=np.asarray(state.orientation).copy(),
        angular_velocity=np.zeros(3)
    )

    vessels = game.app.vessel_system

    before = len(vessels.orbital_debris(game.app.scene))

    vessels.add_debris(game.app.scene, piece, fragment_state, component.body, game.app.orbit_system, orbital=True)

    game.step(60)

    assert component.state.destroyed and "broke up" in component.state.fell

    # Its fragments now in orbit with the rest.
    assert len(vessels.orbital_debris(game.app.scene)) > before + 2


def test_a_close_pass_is_warned_of(play):

    game: Game = play()

    component = _in_orbit(game)

    state = component.state

    along = state.velocity / np.linalg.norm(state.velocity)

    side = np.cross(along, state.position / np.linalg.norm(state.position))

    piece = Vessel("Fragment", [PartState(FRAGMENT, None)])
    piece._restack()

    # 4 km ahead, closing at 10 m/s, to pass 400 m to the side.
    game.app.vessel_system.add_debris(game.app.scene, piece, VesselState(
        position=state.position + along * 4_000.0 + side * 400.0,
        velocity=state.velocity - along * 10.0,
        orientation=np.asarray(state.orientation).copy(),
        angular_velocity=np.zeros(3)
    ), component.body, game.app.orbit_system, orbital=True)

    game.step(3)

    readout = game.app.vessel_system.readout(game.app.scene, game.app.orbit_system, game.app.planet_system)

    assert readout.conjunction is not None

    seconds, miss = readout.conjunction

    assert 350.0 < seconds < 450.0 and miss < 1_000.0
