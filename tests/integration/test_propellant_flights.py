import numpy as np
import pytest

from math3d import quaternion

from vessel import power
from vessel.parts import StackEntry, VesselDesign

from tests.integration.harness import Game


pytestmark = pytest.mark.integration


def _hydrogen_stage(game: Game, cooled: bool):
    """A crewed hydrogen stage with solar power (and a cryocooler), launched and put high in sunlight."""

    parts = game.app.parts

    stack = [StackEntry(parts["capsule"], None), StackEntry(parts["solar_array_1m25"], None)]

    if cooled:
        stack.append(StackEntry(parts["cryocooler"], None))

    stack += [StackEntry(parts["tank_2m5_hydrolox_long"], None), StackEntry(parts["engine_hydrolox"], 0)]

    game.app._launch_design(VesselDesign("hydrogen", "Cooled" if cooled else "Bare", "", tuple(stack)))

    game.step(5)

    component = game.vessel

    # Launched (paid for): it boils from now on.
    game.app.stage_vessel(component)

    assert component.paid

    center, _ = game.app.vessel_system.reference_state(game.app.orbit_system, component.body)

    to_sun = -np.asarray(center, dtype=np.float64) / np.linalg.norm(center)

    pole = np.cross(to_sun, [0.0, 0.0, 1.0])
    pole /= np.linalg.norm(pole)

    game.put_in_orbit(component, altitude=20_000_000.0, direction=to_sun, pole=pole)

    # The stack at the Sun: the array's wings square on.
    component.state.orientation = quaternion.look_rotation(pole, to_sun)

    power.set_panels(component.vessel, True)

    game.step(5)

    return component


def test_hydrogen_boils_off_unless_a_cryocooler_keeps_it(play):

    game: Game = play()

    game.step(5)

    bare = _hydrogen_stage(game, cooled=False)

    tank = next(state for state in bare.vessel.parts if state.part.propellant_type == "hydrolox")

    full = tank.propellant

    readout = game.app.vessel_system.readout(game.app.scene, game.app.orbit_system, game.app.planet_system)

    assert readout.propellant_kind == "LH2/LOX"
    assert bare.power.boiloff == pytest.approx(full * 0.015, rel=0.1)

    start = game.clock

    game.warp(10_000.0)

    game.run_until(lambda: game.clock - start > 3.0 * 86_400.0, 20_000)

    game.warp(1.0)

    days = (game.clock - start) / 86_400.0

    lost = 1.0 - tank.propellant / full

    # ~1.5% a day in sunlight (a little shadow).
    assert 0.8 * 0.015 * days < lost < 1.05 * 0.015 * days

    cooled = _hydrogen_stage(game, cooled=True)

    tank = next(state for state in cooled.vessel.parts if state.part.propellant_type == "hydrolox")

    full = tank.propellant

    start = game.clock

    game.warp(10_000.0)

    game.run_until(lambda: game.clock - start > 3.0 * 86_400.0, 20_000)

    assert cooled.vessel.powered and cooled.power.boiloff == 0.0
    assert tank.propellant == pytest.approx(full)
