import time

import pytest

from core.key_codes import Key

from tests.integration.harness import Game


pytestmark = pytest.mark.integration


def _settle(game: Game, frames: int = 4):
    """A few frames, a guide's look between each (it looks a few times a second)."""

    for _ in range(frames):

        game.app._guided = time.monotonic() - 1.0

        game.step()


def test_flight_school_follows_the_first_steps(play):

    game: Game = play()

    app = game.app

    game.step(5)

    app.start_flight_school()

    assert app.school.active and app.sandbox and app.saves is None

    _settle(game)

    assert app.school.step.title == "Full throttle"

    game.press(Key.Z)

    _settle(game)

    assert app.school.step.title == "Light the engines"

    game.stage()

    _settle(game)

    game.press(Key.T)

    _settle(game)

    assert app.school.step.title == "Climb"

    game.run_until(lambda: app.school.step.title != "Climb", 3_000, every=lambda: setattr(app, "_guided", 0.0))

    assert app.school.step.title == "Gravity turn"

    # Leaving: the title screen.
    app.leave_flight_school()

    assert not app.school.active and app.title.open


def test_a_hint_on_the_pad(play):

    game: Game = play()

    app = game.app

    # (Headless: none seen, none kept.)
    assert app.hints.path is None

    _settle(game)

    assert app.hints.showing is not None and app.hints.showing[0].id == "pad"
