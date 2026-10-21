import time

import numpy as np
import pytest

from core.key_codes import Key

from game.lessons import LESSONS

from tests.integration.harness import Game


pytestmark = pytest.mark.integration


def _look(game: Game, frames: int = 3):
    """Frames, flight school looking at each (it looks a few times a second)."""

    for _ in range(frames):

        game.app._guided = time.monotonic() - 1.0

        game.step()


def _start(game: Game, lesson_id: str):

    game.app.start_lesson(lesson_id)

    _look(game)

    return game.app.school


def test_every_lesson_starts_where_it_should(play):

    game: Game = play()

    app = game.app

    game.step(3)

    for lesson in LESSONS:

        school = _start(game, lesson.id)

        assert school.active and school.lesson is lesson and not school.lost, lesson.id
        assert app.sandbox and app.saves is None

        vessel = game.vessel

        assert not vessel.state.destroyed, lesson.id

        # Every lesson's vessel can be flown: crewed, or a probe
        # with its link home.
        assert app.link.connected or app.link.crewed, lesson.id

        body = app.science_system.body_id(app.scene, vessel.body)

        if lesson.setup == "pad":
            assert vessel.state.landed and body == "earth"
        elif lesson.setup in ("orbit", "docking", "power"):
            assert body == "earth" and app.vessel_system.readout(app.scene, app.orbit_system, app.planet_system).situation == "Orbiting", lesson.id
        elif lesson.setup == "landing":
            assert body == "moon" and not vessel.state.landed
        elif lesson.setup == "comms":
            assert body == "mars" and app.link.delay > 60.0


def test_the_docking_lesson(play):

    game: Game = play()

    app = game.app

    game.step(3)

    school = _start(game, "docking")

    assert school.step.title == "The target"

    station = app._lesson_station

    distance = np.linalg.norm(app.scene.get_component(station, type(game.vessel)).state.position - game.vessel.state.position)

    assert 100.0 < distance < 150.0

    app.map_view.target = ("vessel", station)

    _look(game)

    assert school.step.title == "Thrusters"

    game.hold(Key.UP)

    _look(game)

    game.let_go(Key.UP)

    assert school.step.title == "Close in"


def test_the_power_lesson(play):

    game: Game = play()

    game.step(3)

    school = _start(game, "power")

    assert game.vessel.power.share == pytest.approx(0.15, abs=0.02) and game.vessel.power.panels == "folded"

    # (A probe: it obeys over the link.)
    assert game.app.link.connected

    school.skip()

    game.press(Key.U)

    _look(game, 5)

    assert school.step.title == "Charging"


def test_the_landing_lesson_starts_over(play):

    game: Game = play()

    app = game.app

    game.step(3)

    school = _start(game, "landing")

    game.press(Key.T)

    _look(game)

    # Lost: started over, a fresh lander high up again.
    game.vessel.state.destroyed = True

    _look(game)

    assert school.lost

    app.lesson_again()

    _look(game)

    assert school.active and not school.lost and school.index == 0
    assert not game.vessel.state.destroyed and not game.vessel.state.landed
