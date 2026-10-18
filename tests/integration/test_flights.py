import math

import numpy as np
import pytest

from core.key_codes import Key

from science import survey

from vessel.sequence import parse

from tests.integration.harness import Game, angle_between


pytestmark = pytest.mark.integration


def test_home_from_orbit(play):
    """Plan "Come down", warp to it, burn it on the maneuver SAS, separate, come in on the shield, land under the parachute; recovered."""

    game: Game = play()

    game.step(5)

    game.stage()

    # The booster off, the upper stage lit, idling in orbit.
    game.stage()

    game.put_in_orbit(altitude=200_000.0)

    vessels = game.app.vessel_system

    component = game.vessel

    suggestions = vessels.suggested_burns(game.app.scene, game.app.orbit_system, game.app.planet_system)

    come_down = next(index for index, item in enumerate(suggestions) if item.label == "Come down")

    mu = vessels.soi_bodies(game.app.scene)[component.body].mu

    game.app._node_action(component, ("suggest", come_down), suggestions, mu)

    node = component.node

    assert node is not None and node.prograde < 0.0

    component.controls.sas = True
    component.controls.sas_mode = "maneuver"

    game.app._node_action(component, ("warp",), suggestions, mu)

    def burn_due():
        readout = game.app._node_readout(component)
        return readout is not None and readout[1] <= 0.0

    # Warping slows to arrive in time; then burn.
    game.run_until(burn_due, 20_000)

    assert angle_between(component.state.up, vessels.node_remaining(component)) < 5.0

    game.press(Key.Z)

    game.run_until(lambda: component.node is None, 5_000)

    assert component.vessel.throttle == 0.0

    periapsis = vessels.readout(game.app.scene, game.app.orbit_system, game.app.planet_system).periapsis

    assert 30_000.0 < periapsis < 60_000.0

    # The capsule off its stage, the parachute armed, tail
    # first into the air.
    game.step(10)
    game.stage()
    game.step(10)
    game.stage()

    component = game.vessel

    component.controls.sas_mode = "retrograde"

    peak = [0.0]

    def fly():

        readout = vessels.readout(game.app.scene, game.app.orbit_system, game.app.planet_system)

        if readout.heat is not None:
            peak[0] = max(peak[0], readout.heat[0])

        if readout.surface_speed > 0.0 and np.linalg.norm(component.state.position) - 6_371_000.0 > 140_000.0:
            game.warp(100.0)

    game.run_until(lambda: component.state.landed or component.state.destroyed, 60_000, every=fly)

    assert component.state.landed and not component.state.destroyed

    # The shield took the heat, well within its limit.
    assert 1_500.0 < peak[0] < 3_300.0

    # Recovered: taken away, a fresh rocket on the pad.
    game.run_until(lambda: game.vessel is not component, 2_000)

    assert game.vessel.state.landed and not game.vessel.paid


def test_a_probe_lands_itself_on_the_moon(play):
    """Mariner in low lunar orbit: commands a light time late, then the landing program, out of contact behind the Moon."""

    game: Game = play(design="mariner")

    game.step(5)

    component = game.vessel

    # Up to the lander alone, its parachute armed.
    for _ in range(5):
        game.app.stage_vessel(component)

    moon = game.body("Moon")

    _, frame = game.app.orbit_system.states[moon]

    game.put_in_orbit(body=moon, altitude=40_000.0, direction=frame @ survey.direction_of(10.0, 20.0), pole=frame @ survey.direction_of(0.0, 110.0))

    game.step(5)

    link = game.app.link

    assert link.connected and not link.crewed and 1.0 < link.delay < 1.4

    # A key: there a light time later.
    sent = game.clock

    game.press(Key.Z)

    game.run_until(lambda: component.vessel.throttle > 0.5, 600)

    assert game.clock - sent == pytest.approx(link.delay, abs=0.1)

    game.press(Key.X)

    game.run_until(lambda: component.vessel.throttle == 0.0, 600)

    vessels = game.app.vessel_system

    suggestions = vessels.suggested_burns(game.app.scene, game.app.orbit_system, game.app.planet_system)

    come_down = next(index for index, item in enumerate(suggestions) if item.label == "Come down")

    game.app._node_action(component, ("suggest", come_down), suggestions, vessels.soi_bodies(game.app.scene)[moon].mu)

    text = open("assets/programs/mariner_moon_landing.seq", encoding="utf-8").read()

    # (Separated already.)
    text = text.replace("stage\nwait 3 s\nstage\nwait 1 s\nstage\n", "stage\n")

    game.app._uplink(component, "program", parse(text))

    def coast():
        computer = component.computer
        if computer is not None and not computer.done:
            altitude = np.linalg.norm(component.state.position) - 1_737_400.0
            game.warp(100.0 if altitude > 15_000.0 else 5.0)

    game.run_until(lambda: component.state.landed or component.state.destroyed, 40_000, every=coast)

    assert component.state.landed and not component.state.destroyed

    log = [text for _, text in component.computer.log]

    assert "Braking" in log and "Burn done" in log


@pytest.mark.parametrize("port", ["bottom", "side"])
def test_a_ferry_docks_to_the_station_and_undocks(play, port):

    game: Game = play(design="station_core")

    game.step(5)

    vessels = game.app.vessel_system

    core = game.vessel

    while core.vessel.next_stage <= 3:
        game.app.stage_vessel(core)

    game.put_in_orbit(altitude=300_000.0, direction=(1.0, 0.0, 0.0), pole=(0.0, 1.0, 0.0))

    core.controls.sas = True
    core.controls.sas_mode = "hold"

    core_entity = game.vessel_entity

    game.launch_design("ferry")

    game.stage()

    ferry = game.vessel

    game.app.stage_vessel(ferry)

    # 40 m out from the station's port, 6 m off its axis.
    _, (point, face) = next(item for item in vessels.port_points(core) if item[0].end == port)

    aside = core.state.up if port == "side" else np.cross(face, [0.0, 1.0, 0.0]) / np.linalg.norm(np.cross(face, [0.0, 1.0, 0.0]))

    ferry.body = core.body
    ferry.state.position = point + face * 40.0 + aside * 6.0
    ferry.state.velocity = core.state.velocity.copy()
    ferry.state.landed = False
    ferry.vessel.throttle = 0.0
    ferry.state.angular_velocity = np.zeros(3)

    game.app.map_view.target = ("vessel", core_entity)

    game.press(Key.B)

    assert game.app._docking == core_entity

    game.run_until(lambda: len(game.vessels()) == 1, 30_000)

    joined = game.vessel

    assert joined.vessel.name == "Station Core"

    from vessel import docking

    assert docking.modules(joined.vessel) == 2

    game.run_until(lambda: game.app.programme_system.career.outcomes.get("first_docking") == "done", 300)

    game.app.undock()

    names = sorted(component.vessel.name for _, component in game.vessels())

    assert names == ["Ferry", "Station Core"]

    game.step(120)

    a, b = (component for _, component in game.vessels())

    assert np.linalg.norm(a.state.position - b.state.position) > 10.0


def test_the_fleet_goes_along_and_relays(play):
    """A vessel left in lunar orbit stays through a revert; it relays a lander on the far side."""

    game: Game = play(design="mariner")

    game.step(5)

    relay = game.vessel

    game.stage()

    relay_entity = game.vessel_entity

    moon = game.body("Moon")

    earth = game.app.vessel_system.launch_body

    to_earth = game.app.orbit_system.states[earth][0] - game.app.orbit_system.states[moon][0]
    to_earth /= np.linalg.norm(to_earth)

    side = np.cross(to_earth, [0.0, 1.0, 0.0])
    side /= np.linalg.norm(side)

    # High beside the Moon: Earth and the far side both in view.
    game.put_in_orbit(relay, body=moon, altitude=8_000_000.0, direction=side * 0.97 - to_earth * 0.25, pole=(0.0, 1.0, 0.0))

    # Another on the pad, launched, then reverted: the one in
    # orbit stays.
    game.launch_design("mariner")

    game.stage()

    game.app.revert()

    game.step(5)

    names = [component.vessel.name for _, component in game.vessels()]

    assert sorted(names) == ["Mariner", "Mariner"]

    lander = game.vessel

    assert lander is not relay and not lander.paid

    game.stage()

    # (A new scene: the Moon's entity, and where Earth is now.)
    moon = game.body("Moon")
    earth = game.app.vessel_system.launch_body

    to_earth = game.app.orbit_system.states[earth][0] - game.app.orbit_system.states[moon][0]
    to_earth /= np.linalg.norm(to_earth)

    _, frame = game.app.orbit_system.states[moon]

    far = -(frame.T @ to_earth)

    latitude, longitude = survey.latitude_longitude(far)

    game.land_at(moon, float(latitude[0]), float(longitude[0]))

    game.step(5)

    link = game.app.link

    assert link.connected and "via Mariner" in link.station, link


def test_mapping_the_moon(play):

    game: Game = play(design="explorer")

    game.step(5)

    game.stage()

    moon = game.body("Moon")

    game.put_in_orbit(body=moon, altitude=120_000.0, direction=(1.0, 0.0, 0.0), pole=(0.0, 0.0, 1.0))

    game.step(5)

    science = game.app.science_system

    mask = science.record.surveys["moon"]

    game.warp(100_000.0)

    game.step(400)

    assert mask.coverage() > 0.15

    assert science.record.points > 0.0

    # The Moon's sharp heights wait for its survey; home's do not.
    assert survey.mask_for("moon").gating and not survey.mask_for("earth").gating
