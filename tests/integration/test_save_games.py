import numpy as np
import pytest

from programme import saves

from tests.integration.harness import Game


pytestmark = pytest.mark.integration


def test_quickload_goes_back_to_the_quicksave(play):

    game: Game = play()

    career = game.app.programme_system.career

    game.step(5)

    game.stage()

    game.put_in_orbit(altitude=300_000.0)

    game.step(30)

    funds = career.funds
    name = game.vessel.vessel.name
    where = game.vessel.state.position.copy()
    clock = game.clock

    assert game.app.quicksave()
    assert saves.kept(game.app.saves, saves.QUICKSAVE) is not None

    # Played on: half an hour along, funds spent.
    game.warp(1_000.0)

    game.run_until(lambda: game.clock > clock + 1_800.0, 2_000)

    game.warp(1.0)

    game.app.programme_system.career.funds -= 50.0

    assert np.linalg.norm(game.vessel.state.position - where) > 1_000.0

    assert game.app.quickload()

    game.step(2)

    # Back: the career as it was, flying the same vessel from
    # where it was.
    assert game.app.programme_system.career.funds == pytest.approx(funds)
    assert game.vessel.vessel.name == name and game.vessel.paid
    assert game.clock == pytest.approx(clock, abs=5.0)
    assert np.linalg.norm(game.vessel.state.position - where) < 50_000.0
    assert "Quickloaded" in game.notices()


def test_no_quicksave_no_quickload(play):

    game: Game = play()

    game.step(2)

    funds = game.app.programme_system.career.funds

    assert not game.app.quickload()
    assert game.app.programme_system.career.funds == funds
    assert any("No quicksave" in notice for notice in game.notices())


def test_the_title_screen_starts_and_loads_games(play, tmp_path):

    game: Game = play()

    app = game.app

    app.title.directory = tmp_path / "slots"

    game.step(2)

    app.show_title()

    game.step(2)

    assert app.time_warp.paused

    actions = app._title_actions()

    # A new sandbox: no budget, its own folder.
    actions.new("Moon base", True)

    game.step(5)

    assert not app.title.open and app.sandbox and app.slot.name == "Moon base"
    assert app.saves == app.slot.folder

    # A career, a launch, saved.
    actions.new("Career", False)

    game.step(5)

    assert not app.sandbox

    game.stage()

    game.step(5)

    app.show_title()

    listed = saves.slots(app.title.directory)

    assert [slot.name for slot in listed] == ["Career", "Moon base"]
    assert listed[0].summary["vessels"] == 1 and listed[0].summary["funds"] < 400.0

    # Back to the sandbox, then the career as left.
    actions.play(listed[1], None)

    game.step(5)

    assert app.sandbox and not any(component.paid for _, component in game.vessels())

    actions.play(saves.find("Career", app.title.directory), None)

    game.step(5)

    assert not app.sandbox and any(component.paid for _, component in game.vessels())
