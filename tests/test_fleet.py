import numpy as np
import pytest

from programme import fleet
from systems.programme_system import drain_tanks, move_propellant
from vessel.maneuver import ManeuverNode
from vessel.parts import load_design, load_parts
from vessel.physics import VesselState, upright
from vessel.records import node_from_json, node_to_json, state_from_json, state_to_json, vessel_from_json, vessel_to_json
from vessel.vessel import Vessel


@pytest.fixture(scope="module")
def parts():

    return load_parts()


def test_a_vessel_in_flight_round_trips(parts):

    vessel = Vessel.from_design(load_design("pathfinder_heavy", parts))

    # Flown a while: staged, burned, a part hot.
    vessel.stage()
    vessel.stage()
    vessel.throttle = 1.0
    vessel.burn(30.0)
    vessel.parts[-1].temperature = 1_234.0

    again = vessel_from_json(vessel_to_json(vessel), parts)

    assert again.name == vessel.name and again.next_stage == vessel.next_stage
    assert [state.part.id for state in again.parts] == [state.part.id for state in vessel.parts]
    assert again.mass == pytest.approx(vessel.mass)
    assert again.delta_v() == pytest.approx(vessel.delta_v())
    assert again.parts[-1].temperature == 1_234.0

    # Heights come back from the stack.
    assert [state.bottom for state in again.parts] == pytest.approx([state.bottom for state in vessel.parts])

    # The radial groups too.
    assert [(state.group, state.count) for state in again.parts] == [(state.group, state.count) for state in vessel.parts]


def test_state_and_node_round_trip():

    state = VesselState(
        np.array([7.0e6, 1.0, 2.0]),
        np.array([0.0, 7_500.0, 3.0]),
        upright(np.array([1.0, 0.0, 0.0])),
        np.array([0.01, 0.0, 0.0]),
        landed=False,
        thrust_dv=np.array([1.0, 2.0, 3.0])
    )

    again = state_from_json(state_to_json(state))

    np.testing.assert_allclose(again.position, state.position)
    np.testing.assert_allclose(again.velocity, state.velocity)
    np.testing.assert_allclose(again.orientation, state.orientation)
    np.testing.assert_allclose(again.thrust_dv, state.thrust_dv)

    node = ManeuverNode(1_000.0, 50.0, -2.0, 1.0, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))

    back = node_from_json(node_to_json(node))

    assert back.components == node.components and back.time == node.time
    np.testing.assert_allclose(back.vector, node.vector)

    assert node_from_json(node_to_json(None)) is None


def test_fleet_file(tmp_path):

    path = tmp_path / "fleet.json"

    fleet.save([{"vessel": {"name": "x", "parts": []}, "body": "moon"}], path)

    assert fleet.load(path)[0]["body"] == "moon"

    # Nothing saved, or broken: empty.
    assert fleet.load(tmp_path / "none.json") == []

    (tmp_path / "broken.json").write_text("[", encoding="utf-8")

    assert fleet.load(tmp_path / "broken.json") == []


def test_moving_propellant(parts):

    depot = Vessel.from_design(load_design("pathfinder", parts))
    ship = Vessel.from_design(load_design("mariner", parts))

    for state in ship.parts:
        state.propellant = 0.0

    before = depot.propellant

    moved = move_propellant(depot, ship, 2_000.0)

    assert moved == pytest.approx(2_000.0)
    assert depot.propellant == pytest.approx(before - 2_000.0)
    assert ship.propellant == pytest.approx(2_000.0)

    # "All": what fits, of what there is (the depot runs dry
    # first here).
    room = ship.propellant_capacity - ship.propellant

    held = depot.propellant

    assert move_propellant(depot, ship, float("inf")) == pytest.approx(min(room, held))
    assert depot.propellant == pytest.approx(0.0, abs=1e-6)

    # The other way, what fits only.
    assert move_propellant(ship, depot, float("inf")) == pytest.approx(depot.propellant_capacity)

    # Nothing left to give: nothing moves.
    empty = Vessel.from_design(load_design("pathfinder", parts))

    drain_tanks(empty, float("inf"))

    assert empty.propellant == pytest.approx(0.0)
    assert move_propellant(empty, depot, 1_000.0) == 0.0
