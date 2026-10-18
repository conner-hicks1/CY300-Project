import math

from types import SimpleNamespace

import numpy as np
import pytest

from systems.power_system import PowerSystem, _behind, _rotate
from vessel import kepler, power
from vessel.parts import load_design, load_parts, parse_part
from vessel.physics import VesselState
from vessel.records import vessel_from_json, vessel_to_json
from vessel.sequence import SequenceError, parse
from vessel.vessel import Vessel


EARTH_RADIUS = 6_371_000.0
EARTH_MU = 3.986004418e14


@pytest.fixture(scope="module")
def parts():

    return load_parts()


@pytest.fixture
def mariner(parts):

    return Vessel.from_design(load_design("mariner", parts))


def test_parts_carry_power(parts):

    wing = parts["solar_wing"]

    assert wing.kind == "power" and wing.solar.wings == 1 and wing.solar.power == 700.0
    assert parts["rtg"].generator == 110.0
    assert parts["probe_core"].battery > 0.0 and parts["probe_core"].power_draw > 0.0
    assert parts["survey_camera"].experiment.energy == 50.0
    assert parts["isru_plant"].isru.power > 0.0

    part = parse_part({
        "id": "panel", "kind": "power", "mass": 5,
        "shape": {"radius_top": 0.1, "radius_bottom": 0.1, "height": 0.2},
        "solar": {"power": 100, "length": 1, "width": 0.5}
    })

    assert part.solar.wings == 2


def test_a_design_starts_full(mariner):

    assert mariner.charge == pytest.approx(mariner.charge_capacity) == pytest.approx(8_800.0)
    assert mariner.powered


def test_the_sun_fades_with_distance():

    assert power.strength(power.AU) == pytest.approx(1.0)
    assert power.strength(1.524 * power.AU) == pytest.approx(0.43, abs=0.01)
    assert power.strength(5.203 * power.AU) == pytest.approx(0.037, abs=0.001)


def test_wings_face_the_sun_about_their_axis(mariner):

    wings = power.solar_parts(mariner)

    # Folded: nothing.
    assert power.solar_power(mariner, np.array([0.0, 0.0, 1.0]), 1.0) == 0.0

    for state in wings:
        state.extending = True

    power.move_panels(mariner, power.PANEL_TIME)

    assert power.panel_state(mariner) == "out"

    # Two wings straight out along +X and -X: the Sun across
    # them (along the stack, or sideways on Z), full; along
    # them (X), nothing.
    assert power.solar_power(mariner, np.array([0.0, 1.0, 0.0]), 1.0) == pytest.approx(1_400.0)
    assert power.solar_power(mariner, np.array([0.0, 0.0, 1.0]), 1.0) == pytest.approx(1_400.0)
    assert power.solar_power(mariner, np.array([1.0, 0.0, 0.0]), 1.0) == pytest.approx(0.0, abs=1e-6)

    half = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)

    assert power.solar_power(mariner, half, 0.43) == pytest.approx(1_400.0 * 0.43 / math.sqrt(2.0))


def test_wings_fold_and_tear(mariner):

    assert power.toggle_panels(mariner) is True

    power.move_panels(mariner, 0.5 * power.PANEL_TIME)

    assert power.panel_state(mariner) == "unfolding"

    # Slow, thin air: fine; a launch's: torn off.
    assert power.tear_panels(mariner, 100.0) == []
    assert power.tear_panels(mariner, 20_000.0) == ["Solar wing"]
    assert power.panel_state(mariner) == "torn off"
    assert power.toggle_panels(mariner) is None
    assert power.solar_power(mariner, np.array([0.0, 1.0, 0.0]), 1.0) == 0.0


def test_folded_wings_survive_the_air(mariner):

    assert power.tear_panels(mariner, 50_000.0) == []
    assert power.panel_state(mariner) == "folded"


def test_shadows():

    sun = np.zeros(3)

    planet = (np.array([power.AU, 0.0, 0.0]), EARTH_RADIUS)

    behind = np.array([power.AU + 7_000_000.0, 0.0, 0.0])
    beside = np.array([power.AU, 7_000_000.0, 0.0])

    assert power.shadowed(behind, sun, [planet])
    assert not power.shadowed(beside, sun, [planet])

    # A cylinder behind the body, none in front.
    assert _behind(behind, planet[0], EARTH_RADIUS, np.array([-1.0, 0.0, 0.0]))
    assert not _behind(np.array([power.AU - 7_000_000.0, 0.0, 0.0]), planet[0], EARTH_RADIUS, np.array([-1.0, 0.0, 0.0]))

    turned = _rotate(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 0.5 * math.pi)

    np.testing.assert_allclose(turned, [0.0, 1.0, 0.0], atol=1e-12)


def test_the_air_dims_the_light():

    assert power.through_air(0.0, 0.1) == 1.0
    assert power.through_air(1.0, 1.0) == pytest.approx(0.88, abs=0.01)

    # Low sun, more air in the way; Venus's 92 bar: dark.
    assert power.through_air(1.0, 0.2) < power.through_air(1.0, 1.0)
    assert power.through_air(92.0, 1.0) < 1e-4


def test_batteries_run_down_and_power_goes(mariner):

    draw = power.base_draw(mariner)

    assert draw == pytest.approx(30.0 + 40.0 + 60.0)

    # An hour in the dark.
    status = power.settle(mariner, 0.0, 3600.0)

    assert status.charge == pytest.approx(8_800.0 - draw)
    assert status.net == pytest.approx(-draw) and status.powered

    # Long enough to run out.
    status = power.settle(mariner, 0.0, 400_000.0)

    assert status.charge == 0.0 and not status.powered and not mariner.powered

    # No power: no reaction wheels.
    assert mariner.control_torque() == 0.0

    # Sunlight: back, charging.
    status = power.settle(mariner, 1_400.0, 60.0)

    assert status.powered and mariner.powered and status.charge > 0.0
    assert mariner.control_torque() > 0.0

    # Full: it stays full.
    status = power.settle(mariner, 1_400.0, 1e6)

    assert status.charge == pytest.approx(mariner.charge_capacity)


def test_wheels_draw_as_they_work(mariner):

    idle = power.base_draw(mariner)

    mariner.wheel_use = 1.0

    assert power.base_draw(mariner) == pytest.approx(idle + 60.0)


def test_a_vessel_without_batteries(parts):

    vessel = Vessel("Probe", [])

    status = power.settle(vessel, 0.0, 10.0)

    assert status.powered and status.share == 1.0


def test_experiments_and_transmissions_spend_charge(mariner):

    assert power.take(mariner, 100.0)
    assert mariner.charge == pytest.approx(8_700.0)

    assert not power.take(mariner, 1e6)
    assert mariner.charge == pytest.approx(8_700.0)


def test_the_plant_takes_only_what_is_spare(parts):

    vessel = Vessel.from_design(load_design("mariner", parts))

    # A plant aboard, wanting 20 kW.
    from vessel.vessel import PartState

    vessel.parts.append(PartState(parts["isru_plant"], None))

    vessel.isru_wanted = True

    capacity = vessel.charge_capacity

    # Down at the reserve, making 10 kW over the draw: half.
    power.set_charge(vessel, power.ISRU_RESERVE * capacity)

    status = power.settle(vessel, 10_000.0 + power.base_draw(vessel), 3600.0)

    assert vessel.isru_share == pytest.approx(0.5)
    assert status.isru == pytest.approx(10_000.0)
    assert vessel.charge == pytest.approx(power.ISRU_RESERVE * capacity)

    # Nothing spare: it waits; the batteries are not drained
    # below the reserve.
    power.settle(vessel, 0.0, 3600.0)

    assert vessel.isru_share == 0.0
    assert vessel.powered

    # Not running: no share asked.
    vessel.isru_wanted = False

    power.settle(vessel, 0.0, 1.0)

    assert vessel.isru_share == 1.0


def test_power_is_saved(parts, mariner):

    power.take(mariner, 1_000.0)

    wing = power.solar_parts(mariner)[0]

    wing.extending = True
    wing.extended = 0.4

    loaded = vessel_from_json(vessel_to_json(mariner), parts)

    assert loaded.charge == pytest.approx(mariner.charge)

    wing = power.solar_parts(loaded)[0]

    assert wing.extending and wing.extended == pytest.approx(0.4)

    # Saved before there was power: full.
    data = vessel_to_json(mariner)

    for raw in data["parts"]:
        raw.pop("charge")

    assert vessel_from_json(data, parts).charge == pytest.approx(mariner.charge_capacity)


def test_the_budget(mariner):

    budget = power.budget(mariner, 1.524 * power.AU)

    assert budget.solar == pytest.approx(1_400.0 * 0.43, rel=0.01)
    assert budget.dark_hours == pytest.approx(8_800.0 / 130.0)

    rtg_only = power.PowerBudget(0.0, 0.0, 110.0, 100.0, 0.0)

    assert rtg_only.dark_hours is None


def test_the_computer_folds_wings_and_waits_for_charge():

    program = parse("panels out\npanels in\nwings fold\nwait until charge > 25%")

    assert [statement.value for statement in program.statements[:3]] == [True, False, False]

    with pytest.raises(SequenceError):
        parse("panels sideways")


def _orbiter(parts):

    vessel = Vessel.from_design(load_design("mariner", parts))

    power.set_panels(vessel, True)
    power.move_panels(vessel, power.PANEL_TIME)

    return vessel


def test_low_orbit_spends_a_third_in_shadow(parts):

    vessel = _orbiter(parts)

    radius = EARTH_RADIUS + 400_000.0

    # The Earth a little way along +X from the Sun; the orbit
    # in the XY plane, so the shadow crosses it.
    state = VesselState(
        position=np.array([radius, 0.0, 0.0]),
        velocity=np.array([0.0, kepler.circular_speed(EARTH_MU, radius), 0.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        angular_velocity=np.zeros(3)
    )

    physics = SimpleNamespace(radius=EARTH_RADIUS, mu=EARTH_MU, spin=np.array([0.0, 0.0, 7.29e-5]), column=None)

    center = np.array([power.AU, 0.0, 0.0])

    system = PowerSystem()

    period = 2.0 * math.pi * math.sqrt(radius ** 3 / EARTH_MU)

    sunlight, light, _, flux = system._sunlight(vessel, state, physics, center, [], EARTH_RADIUS, 3.0 * period)

    # A cylindrical shadow: asin(R / r) / pi of the orbit.
    shadow = math.asin(EARTH_RADIUS / radius) / math.pi

    assert sunlight == pytest.approx(1.0 - shadow, abs=0.04)
    assert flux == pytest.approx(1.0, abs=1e-3)

    # The Sun along -X, along the wings (out along X): nothing.
    # Turned a quarter about the stack, they catch it all.
    assert light == pytest.approx(0.0, abs=1.0)

    state.orientation = np.array([0.0, math.sin(math.pi / 4.0), 0.0, math.cos(math.pi / 4.0)])

    _, light, _, _ = system._sunlight(vessel, state, physics, center, [], EARTH_RADIUS, 3.0 * period)

    assert light == pytest.approx(1_400.0 * (1.0 - shadow), rel=0.06)

    # A short step on the night side (away from the Sun, +X):
    # dark.
    state.position = np.array([radius, 0.0, 0.0])

    sunlight, light, sun_local, _ = system._sunlight(vessel, state, physics, center, [], EARTH_RADIUS, 1.0)

    assert sunlight == 0.0 and light == 0.0 and sun_local is None


def test_a_landed_day_averages_half(parts):

    vessel = _orbiter(parts)

    state = VesselState(
        position=np.array([0.0, EARTH_RADIUS, 0.0]),
        velocity=np.zeros(3),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        angular_velocity=np.zeros(3),
        landed=True
    )

    physics = SimpleNamespace(radius=EARTH_RADIUS, mu=EARTH_MU, spin=np.array([0.0, 0.0, 7.29e-5]), column=None)

    center = np.array([power.AU, 0.0, 0.0])

    sunlight, _, _, _ = PowerSystem()._sunlight(vessel, state, physics, center, [], EARTH_RADIUS, 3.0 * 86_400.0)

    # On the equator: half the day lit.
    assert sunlight == pytest.approx(0.5, abs=0.05)
