import numpy as np
import pytest

from vessel.parts import load_design

from tests.integration.harness import Game


pytestmark = pytest.mark.integration


def _two_in_orbit(game: Game, closing: float):
    """Two Pathfinders in the same orbit, the flown one just behind the other's tail, closing at `closing` m/s."""

    game.step(5)

    ahead = game.vessel

    game.stage()

    game.put_in_orbit(ahead, altitude=400_000.0)

    game.app._launch_design(load_design("pathfinder", game.app.parts))

    game.step(5)

    behind = game.vessel

    assert behind is not ahead

    game.stage()

    # Nose to tail: the one behind's nose 4 m short of the
    # tail of the one ahead, both pointing along the orbit.
    along = ahead.state.velocity / np.linalg.norm(ahead.state.velocity)

    behind.body = ahead.body
    behind.state.position = ahead.state.position - along * (behind.vessel.height + 4.0)
    behind.state.velocity = ahead.state.velocity + along * closing
    behind.state.orientation = ahead.state.orientation.copy()
    behind.state.angular_velocity = np.zeros(3)
    behind.state.landed = False
    behind.vessel.throttle = 0.0
    behind.controls.sas = False

    return ahead, behind


def test_vessels_bump_and_part(play):
    """Closing at 3 m/s nose to tail: they touch, bounce apart, both whole."""

    game: Game = play()

    ahead, behind = _two_in_orbit(game, 3.0)

    along = ahead.state.velocity / np.linalg.norm(ahead.state.velocity)

    game.step(240)

    assert not ahead.state.destroyed and not behind.state.destroyed

    # Now opening, the one behind slower than the one ahead.
    assert float((ahead.state.velocity - behind.state.velocity) @ along) > 0.2

    # Never through each other.
    gap = float((ahead.state.position - behind.state.position) @ along)

    assert gap > behind.vessel.height

    assert any(notice.startswith("Bumped") for notice in game.notices())


def test_a_hard_hit_breaks_them_up(play):
    """Closing at 30 m/s: the parts that meet break."""

    game: Game = play()

    ahead, behind = _two_in_orbit(game, 30.0)

    game.step(120)

    assert behind.state.destroyed or ahead.state.destroyed
    assert "broke up" in (behind.state.fell + ahead.state.fell)
