import pytest

from systems.comms_system import crew_names
from vessel import life

from tests.integration.harness import Game


pytestmark = pytest.mark.integration


def test_a_crew_left_too_long_without_power_is_lost(play):
    """A crewed Pathfinder in orbit: named at launch; its capsule's batteries last a day and a half, then a day dark: lost."""

    game: Game = play()

    game.step(5)

    component = game.vessel

    corps = game.app.programme_system.career.corps

    game.stage()

    names = crew_names(component.vessel)

    assert len(names) == 3
    assert all(corps.named(name).status == "flying" for name in names)

    game.put_in_orbit(altitude=400_000.0)

    game.step(5)

    status = component.life

    assert status is not None and status.crew == 3 and status.days == pytest.approx(14.0, abs=0.2)
    assert 0.4 < status.rate < 1.0

    start = game.clock

    game.run_until(lambda: all(corps.named(name).status == "lost" for name in names), 30_000, every=lambda: game.warp(10_000.0))

    days = (game.clock - start) / life.DAY

    # Batteries (~1.4 days), then a day without power.
    assert 2.0 < days < 3.5

    assert crew_names(component.vessel) == []
    assert any("without power" in notice for notice in game.notices())

    # Each lost its dose so far.
    assert all(corps.named(name).dose > 0.5 for name in names)


def test_a_revert_brings_the_crew_back(play):

    game: Game = play()

    game.step(5)

    corps = game.app.programme_system.career.corps

    game.stage()

    names = crew_names(game.vessel.vessel)

    assert names

    game.step(30)

    game.app.revert()

    game.step(10)

    assert all(corps.named(name).status == "ready" for name in names)
