import pytest

from core.input import Input

from tests.integration.harness import Game


@pytest.fixture
def play(tmp_path):
    """Start a headless game (Game's arguments); closed after the test. Skips without OpenGL 4.5."""

    games = []

    def start(**options) -> Game:

        Input.clear()

        try:
            game = Game(tmp_path / f"saves{len(games)}", **options)
        except RuntimeError as error:
            pytest.skip(f"no OpenGL window: {error}")

        games.append(game)

        return game

    yield start

    for game in games:
        game.close()

    Input.clear()
