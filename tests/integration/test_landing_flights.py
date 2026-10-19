import numpy as np
import pytest

from core.key_codes import Key

from math3d import quaternion

from vessel import landing

from tests.integration.harness import Game


pytestmark = pytest.mark.integration


def _drop(game: Game, component, moon, height: float, speed: float, tilt: float = 0.0):
    """Hold the lander `height` m over the ground, coming down at `speed`, its stack `tilt` deg off upright."""

    # (The ground found exactly: the last height asked for
    # coarse.)
    game.app.vessel_system._height = 0.0

    game.land_at(moon, -20.0, 30.0, component)

    state = component.state

    up = state.position / np.linalg.norm(state.position)

    side = np.cross(up, [0.0, 0.0, 1.0])
    side /= np.linalg.norm(side)

    axis = up * np.cos(np.radians(tilt)) + side * np.sin(np.radians(tilt))

    surface = np.cross(game.app.vessel_system._body_physics(game.app.scene, moon, game.app.orbit_system).spin, state.position)

    state.landed = False
    state.position = state.position + up * height
    state.velocity = surface - up * speed
    state.orientation = quaternion.look_rotation(np.cross(axis, side), axis)
    component.vessel.throttle = 0.0
    component.controls.sas = False


def test_a_lander_stands_on_its_legs_and_one_tilted_falls_over(play):
    """Mariner's lander on the Moon's real ground: legs down, a gentle touchdown stands on the slope; tilted in, it lies on its side and cannot lift off."""

    game: Game = play(design="mariner")

    game.step(5)

    component = game.vessel

    for _ in range(6):
        game.app.stage_vessel(component)

    moon = game.body("Moon")

    # Legs down (a probe: through the link, from the pad).
    game.press(Key.G)

    game.run_until(lambda: landing.leg_state(component.vessel) == "down", 600)

    _drop(game, component, moon, 3.0, 1.5)

    game.run_until(lambda: component.state.landed or component.state.destroyed, 600)

    state = component.state

    assert state.landed and not state.toppled and not state.destroyed

    readout = game.app.vessel_system.readout(game.app.scene, game.app.orbit_system, game.app.planet_system)

    assert readout.slope is not None and readout.slope < readout.standing and readout.legs == "down"

    # Standing square to the ground: tilted off vertical by
    # the slope.
    up = state.position / np.linalg.norm(state.position)

    assert np.degrees(np.arccos(np.clip(state.up @ up, -1.0, 1.0))) == pytest.approx(readout.slope, abs=1.0)

    # Again, tilted 60 deg: over.
    _drop(game, component, moon, 3.0, 1.5, tilt=60.0)

    game.run_until(lambda: component.state.landed or component.state.destroyed, 600)

    assert state.landed and state.toppled and not state.destroyed
    assert any("fell over" in text for text in game.notices())

    # Full throttle: it stays where it lies.
    component.vessel.throttle = 1.0

    game.step(60)

    assert state.landed and state.toppled
