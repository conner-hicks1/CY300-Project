import math

import pytest

from programme.resources import Resource, products
from systems.programme_system import fill_tanks, move_propellant
from vessel import propellants
from vessel.design import DesignDraft
from vessel.parts import PartError, StackEntry, VesselDesign, load_parts, parse_part
from vessel.vessel import Vessel


@pytest.fixture(scope="module")
def parts():

    return load_parts()


def vessel_of(parts, *ids):
    """A stack, top down; engines fire at stage 0."""

    stack = tuple(StackEntry(parts[part_id], 0 if parts[part_id].kind == "engine" else None) for part_id in ids)

    return Vessel.from_design(VesselDesign("test", "Test", "", stack))


def test_every_tank_and_engine_names_a_propellant(parts):

    for part in parts.values():

        if part.propellant:
            assert part.propellant_type in propellants.PROPELLANTS

        if part.engine is not None:
            assert part.engine.propellant in propellants.PROPELLANTS

    assert parts["tank_2m5_long"].propellant_type == "kerolox"
    assert parts["engine_hydrolox"].engine.propellant == "hydrolox"

    # Light hydrogen: an 8 m tank holds about a third of
    # kerosene's mass.
    assert parts["tank_2m5_hydrolox_long"].propellant < 0.4 * parts["tank_2m5_long"].propellant

    with pytest.raises(PartError, match="propellant_type"):
        parse_part({
            "id": "x", "kind": "tank", "mass": 10, "propellant": 100, "propellant_type": "custard",
            "shape": {"radius_top": 1, "radius_bottom": 1, "height": 1}
        })


def test_an_engine_burns_only_its_own_kind(parts):

    # A hydrogen engine under a kerosene tank: nothing to burn.
    wrong = vessel_of(parts, "probe_core", "tank_2m5_long", "engine_hydrolox")

    wrong.stage()
    wrong.throttle = 1.0

    assert wrong.burning_engines() == []
    assert wrong.delta_v() == 0.0

    # With a hydrogen tank too: it burns that one alone.
    mixed = vessel_of(parts, "probe_core", "tank_2m5_long", "tank_2m5_hydrolox_long", "engine_hydrolox")

    mixed.stage()
    mixed.throttle = 1.0

    kerosene = mixed.parts[1].propellant

    mixed.burn(10.0)

    assert mixed.parts[1].propellant == kerosene
    assert mixed.parts[2].propellant < mixed.parts[2].part.propellant

    assert mixed.section_kind() == "hydrolox"

    left, capacity = mixed.section_propellant()

    assert capacity == mixed.parts[2].part.propellant and left == mixed.parts[2].propellant

    # Its delta-v: the hydrogen only (455 s).
    expected = 455.0 * 9.80665 * math.log(mixed.mass / (mixed.mass - mixed.parts[2].propellant))

    assert mixed.delta_v() == pytest.approx(expected, rel=1e-6)


def test_hydrogen_boils_off_in_sunlight(parts):

    stage = vessel_of(parts, "probe_core", "tank_2m5_hydrolox_long", "engine_hydrolox")

    full = stage.parts[1].propellant

    rate = propellants.of("hydrolox").boil

    assert propellants.boil_rate(stage, 1.0, 1.0) == pytest.approx(full * rate)

    lost = propellants.boil(stage, 1.0, 1.0, 86_400.0)

    assert lost == pytest.approx(full * rate, rel=1e-6)

    # In shadow, and far out: much less.
    assert propellants.boil_rate(stage, 0.0, 1.0) == pytest.approx(stage.parts[1].propellant * rate * propellants.DARK_SHARE)
    assert propellants.boil_rate(stage, 1.0, 0.04) < 0.3 * propellants.boil_rate(stage, 1.0, 1.0)

    # A long step: what is left of what is left.
    left = stage.parts[1].propellant

    propellants.boil(stage, 1.0, 1.0, 10 * 86_400.0)

    assert stage.parts[1].propellant == pytest.approx(left * (1.0 - rate) ** 10)


def test_storable_keeps_and_insulation_helps(parts):

    lander = vessel_of(parts, "probe_core", "tank_1m25_hypergolic", "engine_hypergolic")

    assert propellants.boil(lander, 1.0, 1.0, 100 * 86_400.0) == 0.0

    plain = vessel_of(parts, "probe_core", "tank_2m5_hydrolox_long", "engine_hydrolox")
    wrapped = vessel_of(parts, "probe_core", "tank_2m5_hydrolox_insulated", "engine_hydrolox")

    assert propellants.boil_rate(wrapped, 1.0, 1.0) == pytest.approx(0.3 * propellants.boil_rate(plain, 1.0, 1.0))


def test_a_cryocooler_keeps_its_tonnes_cold_while_powered(parts):

    stage = vessel_of(parts, "probe_core", "cryocooler", "tank_2m5_hydrolox_long", "engine_hydrolox")

    full = stage.parts[2].propellant

    cooled = propellants.boil_rate(stage, 1.0, 1.0)

    # 15 t kept cold: the 13.4 t tank boils nothing.
    assert full < 15_000.0 and cooled == 0.0

    stage.powered = False

    assert propellants.boil_rate(stage, 1.0, 1.0) == pytest.approx(full * propellants.of("hydrolox").boil)


def test_plants_make_hydrogen_from_ice_and_methane_from_air(parts):

    made = products([Resource("ice", 1.5, "ice"), Resource("air", 1.0, "air")], 600.0, 250.0)

    assert made == {"hydrolox": 900.0, "methalox": 250.0}

    depot = vessel_of(parts, "probe_core", "tank_2m5_hydrolox_short", "tank_2m5_methalox_medium", "engine_methalox")

    for state in depot.parts:
        state.propellant = 0.0

    assert fill_tanks(depot, 1_000.0, "methalox") == pytest.approx(1_000.0)
    assert depot.parts[1].propellant == 0.0 and depot.parts[2].propellant == pytest.approx(1_000.0)


def test_propellant_moves_tank_to_tank_of_its_kind(parts):

    giver = vessel_of(parts, "probe_core", "tank_2m5_long", "tank_2m5_hydrolox_short", "engine_vacuum")
    taker = vessel_of(parts, "probe_core", "tank_2m5_hydrolox_long", "engine_hydrolox")

    for state in taker.parts:
        state.propellant = 0.0

    moved = move_propellant(giver, taker, 100_000.0)

    # Only the hydrogen goes: the taker has no kerosene tanks.
    assert moved == pytest.approx(giver.parts[2].part.propellant)
    assert giver.parts[1].propellant == giver.parts[1].part.propellant
    assert giver.parts[2].propellant == pytest.approx(0.0)


def test_the_editor_warns_of_an_engine_without_its_propellant(parts):

    design = VesselDesign("x", "X", "", (
        StackEntry(parts["probe_core"], None),
        StackEntry(parts["tank_2m5_long"], None),
        StackEntry(parts["engine_hydrolox"], 0),
    ))

    warnings = DesignDraft.from_design(design).warnings()

    assert any("burns LH2/LOX" in warning for warning in warnings)
