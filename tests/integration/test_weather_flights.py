import numpy as np
import pytest

from ecs.components import ClockComponent

from controllers.programme import mars_forecast
from planet import weather
from vessel import power

from tests.integration.harness import Game


pytestmark = pytest.mark.integration


def _storm_peak(start: float) -> float:
    """The first time after `start` a global dust storm is at its height."""

    now = start

    for _ in range(3_000):

        if any(storm.global_storm and strength >= 1.0 for storm, strength in weather.active(now)):
            return now

        now += 86_400.0

    raise AssertionError("no global storm within 8 years")


def test_a_lander_in_a_global_dust_storm(play):
    """Mariner's lander on Mars at the height of a global storm: the HUD and the programme see it, its sky is dark, dust gathers on its wings."""

    game: Game = play(design="mariner")

    game.step(5)

    component = game.vessel

    for _ in range(6):
        game.app.stage_vessel(component)

    game.step(5)

    clock = next(clock for _, clock in game.app.scene.registry.view_with(ClockComponent))

    clock.time = _storm_peak(clock.time)

    game.step(5)

    mars = game.body("Mars")

    game.app.vessel_system._height = 0.0

    game.land_at(mars, 0.0, 0.0, component)

    power.set_panels(component.vessel, True)
    power.move_panels(component.vessel, power.PANEL_TIME)

    game.step(10)

    readout = game.app.vessel_system.readout(game.app.scene, game.app.orbit_system, game.app.planet_system)

    sky = readout.weather

    assert sky is not None and sky.storm.startswith("Global") and sky.tau > 4.0
    assert sky.light < 0.2

    text, tone = mars_forecast(game.app.orbit_system.time)

    assert "global dust storm" in text and tone == "bad"

    # The season's air: the surface pressure follows the
    # factor.
    physics = game.app.vessel_system._body_physics(game.app.scene, mars, game.app.orbit_system)

    environment = game.app.vessel_system.environment(game.app.scene, mars, game.app.orbit_system, game.app.planet_system)

    radius = float(np.linalg.norm(component.state.position))

    plain = physics.column.pressure((radius - physics.radius) / 1000.0)

    assert environment.air(radius, component.state.position / radius)[0] == pytest.approx(plain * weather.pressure_factor(sky.longitude), rel=0.05)

    # Twenty sols of it (the warp kept up: running out of
    # power stops the clock): dust on the wings, the
    # batteries run down in the dark sky, as Opportunity's in
    # 2018.
    start = game.clock

    game.run_until(lambda: game.clock - start > 20 * weather.SOL, 20_000, every=lambda: game.warp(100_000.0))

    game.warp(1.0)

    assert component.vessel.panel_dust > 0.02
    assert component.power.share < 0.5
    assert component.state.landed and not component.state.destroyed
