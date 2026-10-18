import numpy as np
import pytest

from core.key_codes import Key

from math3d import quaternion

from vessel import power

from tests.integration.harness import Game


pytestmark = pytest.mark.integration


def test_wings_keep_a_probe_alive_and_without_them_it_goes_deaf(play):
    """Mariner in a high orbit: U (a light time late) unfolds its wings; days at warp, charged. Folded, it runs flat and loses its link."""

    game: Game = play(design="mariner")

    game.step(5)

    component = game.vessel

    # Up to the lander alone.
    for _ in range(5):
        game.app.stage_vessel(component)

    # High (a ground station always in sight), the Sun in the
    # orbit's plane (the most shadow there is), the stack
    # pointed at it: the wings square on.
    center, _ = game.app.vessel_system.reference_state(game.app.orbit_system, component.body)

    to_sun = -np.asarray(center, dtype=np.float64) / np.linalg.norm(center)

    pole = np.cross(to_sun, [0.0, 0.0, 1.0])
    pole /= np.linalg.norm(pole)

    game.put_in_orbit(altitude=20_000_000.0, direction=to_sun, pole=pole)

    component.state.orientation = quaternion.look_rotation(pole, to_sun)

    game.step(5)

    assert game.app.link.connected

    status = component.power

    assert status is not None and status.powered and status.panels == "folded"
    assert status.capacity == pytest.approx(8_800.0)

    # U: the command goes up (a probe), the wings unfold.
    game.press(Key.U)

    game.run_until(lambda: power.panel_state(component.vessel) == "unfolding", 600)

    assert any("unfolding" in text for text in game.notices())

    game.run_until(lambda: power.panel_state(component.vessel) == "out", 2_000)

    # Three days at warp (each frame a step longer than an
    # orbit: the light averaged round it).
    game.warp(10_000.0)

    game.run_until(lambda: game.clock > 3.0 * 86_400.0, 20_000)

    status = component.power

    assert status.powered and status.share > 0.9

    # Folded (sent), and on into the dark: flat within days,
    # the link with it.
    game.warp(1.0)

    game.press(Key.U)

    game.run_until(lambda: power.panel_state(component.vessel) == "folded", 2_000)

    game.warp(10_000.0)

    game.run_until(lambda: not component.vessel.powered, 40_000)

    game.step(2)

    assert component.power.charge == 0.0
    assert not game.app.link.connected and game.app.link.reason == "No power"
    assert component.vessel.control_torque() == 0.0
    assert any("out of power" in text for text in game.notices())


def test_wings_tear_off_in_thick_fast_air(play):
    """Pathfinder Heavy, its wings out (crewed: at once), thrown through the air at 20 km: torn off."""

    game: Game = play(design="pathfinder_heavy")

    game.step(5)

    component = game.vessel

    game.press(Key.U)

    game.step(60)

    assert any(state.extending and state.extended > 0.1 for state in power.solar_parts(component.vessel))

    # 20 km up, 700 m/s sideways (the air ~0.09 kg/m3: ~20 kPa).
    game.put_in_orbit(altitude=20_000.0, direction=(1.0, 0.0, 0.0), pole=(0.0, 0.0, 1.0))

    component.state.velocity = component.state.velocity / np.linalg.norm(component.state.velocity) * 700.0

    game.step(30)

    assert power.panel_state(component.vessel) == "torn off"
    assert any("torn off" in text for text in game.notices())
