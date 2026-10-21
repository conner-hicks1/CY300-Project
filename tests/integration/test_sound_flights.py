import pytest

from core.key_codes import Key

from tests.integration.harness import Game


pytestmark = pytest.mark.integration


class Recorder:
    """A sound engine that only notes what it is asked (headless runs make no sound)."""

    mixer = object()

    def __init__(self):

        self.scenes = []
        self.played = []

    def update(self, scene):
        self.scenes.append(scene)

    def play(self, name, level=1.0, pan=0.0, alert=False):
        self.played.append(name)

    def notice(self, tone):
        pass

    def close(self):
        pass


def test_the_flight_is_heard(play):

    game: Game = play()

    app = game.app

    sound = app.sound = Recorder()

    game.step(5)

    # On the pad: no music (a launch to come), the ground's
    # gravel under it.
    assert sound.scenes[-1].music == 0.0 and sound.scenes[-1].surface == "gravel"

    game.press(Key.Z)

    game.stage()

    game.step(5)

    assert "stage" in sound.played and sound.scenes[-1].thrust > 0.0

    # Coasting in orbit: the music.
    game.put_in_orbit(altitude=300_000.0)

    game.vessel.vessel.throttle = 0.0

    game.step(5)

    assert sound.scenes[-1].music > 0.0 and sound.scenes[-1].density == 0.0

    # The title screen: its music.
    app.show_title()

    game.step(2)

    assert sound.scenes[-1].paused and sound.scenes[-1].music == app.TITLE_MUSIC
