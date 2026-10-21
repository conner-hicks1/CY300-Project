import numpy as np
import pytest

from core.key_codes import Key

from tests.integration.harness import Game


pytestmark = pytest.mark.integration


# Uneven frames, as a busy laptop plays (the game clamps a
# frame at 0.1 s).
UNEVEN = (0.02, 0.085)


def _readout(game: Game):

    app = game.app

    return app.vessel_system.readout(app.scene, app.orbit_system, app.planet_system)


@pytest.mark.parametrize("seed", [1, 3])
def test_a_launch_over_streaming_terrain_at_an_uneven_frame_rate(play, seed):
    """
    Finer terrain tiles coming in under a climbing rocket once
    put the ground above it: a touchdown, on the pad's slope a
    fall. (Without that fix, these seeds lost the rocket at
    lift-off.)
    """

    game: Game = play(terrain=True)

    # The terrain round the pad coming in; then frames as a
    # busy laptop plays them, from the launch on.
    game.step(200)

    game.uneven(UNEVEN, seed)

    game.press(Key.Z)

    game.stage()

    heights = []

    for _ in range(30):

        game.step(10)

        state = game.vessel.state

        assert not state.destroyed, state.fell or "lost"

        heights.append(_readout(game).height)

    assert heights[-1] > 200.0 and heights[-1] > heights[len(heights) // 2]


def test_standing_on_the_moon_while_its_terrain_comes_in(play):

    game: Game = play(design="mariner", terrain=True, frame_times=UNEVEN)

    for _ in range(6):
        game.app.stage_vessel(game.vessel)

    game.vessel.vessel.throttle = 0.0

    moon = game.body("Moon")

    game.land_at(moon, 2.0, -10.0)

    state = game.vessel.state

    def on_the_moon():
        # (In its own turning frame: landed, it turns with it.)
        return game.app.orbit_system.states[moon][1].T @ state.position

    where = on_the_moon()

    game.step(300)

    assert state.landed and not state.destroyed

    # (Settled onto finer ground at most a few metres.)
    assert float(np.linalg.norm(on_the_moon() - where)) < 20.0
