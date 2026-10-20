import numpy as np
import pytest

from core.key_codes import Key

from vessel import power

from tests.integration.harness import Game


pytestmark = pytest.mark.integration


def test_a_rover_drives_off(play):
    """The rover on the pad: W held, it drives off over the real ground (a probe: the keys through the link), staying on its wheels."""

    game: Game = play(design="rover")

    game.step(10)

    component = game.vessel

    assert component.state.landed

    start = component.state.position.copy()

    game.hold(Key.W)

    game.step(600)

    game.let_go(Key.W)

    state = component.state

    assert state.landed and not state.toppled and not state.destroyed

    assert float(np.linalg.norm(state.position - start)) > 15.0

    readout = game.app.vessel_system.readout(game.app.scene, game.app.orbit_system, game.app.planet_system)

    assert readout.driving is not None and readout.driving > 2.0

    # Its wheels turned with it.
    assert abs(state.wheel_turn) > 10.0


def test_a_helicopter_set_down_on_mars_flies(play):
    """Mariner Helicopter landed on Mars: stage 6 sets the helicopter down beside it; full throttle, it lifts off in the thin air."""

    game: Game = play(design="mariner_helicopter")

    game.step(5)

    lander = game.vessel

    for _ in range(6):
        game.app.stage_vessel(lander)

    game.step(5)

    mars = game.body("Mars")

    game.app.vessel_system._height = 0.0

    game.land_at(mars, -18.4, 77.5, lander)

    game.step(5)

    power.set_panels(lander.vessel, True)

    count = len(game.vessels())

    # Stage 6: the helicopter's separator.
    game.app.stage_vessel(lander)

    game.step(5)

    assert len(game.vessels()) == count + 1

    helicopter = next(component for _, component in game.vessels() if any(part.part.id == "helicopter_body" for part in component.vessel.parts))

    assert helicopter.state.landed
    assert not any(part.part.kind == "decoupler" for part in helicopter.vessel.parts)

    # Beside the lander, not in it.
    gap = float(np.linalg.norm(helicopter.state.position - lander.state.position))

    assert 1.0 < gap < 6.0

    # Fly it.
    entity = next(entity for entity, component in game.vessels() if component is helicopter)

    game.app.vessel_system.switch(game.app.scene, entity, game.app.orbit_system, game.app.planet_system)

    game.step(5)

    ground = float(np.linalg.norm(helicopter.state.position))

    helicopter.vessel.throttle = 1.0

    game.step(240)

    assert not helicopter.state.destroyed

    assert float(np.linalg.norm(helicopter.state.position)) - ground > 2.0
