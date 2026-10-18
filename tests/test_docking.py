import math

import numpy as np
import pytest

from math3d import quaternion
from programme.career import Campaign, Career, Flight, Mission
from vessel import docking
from vessel.parts import load_design, load_parts
from vessel.physics import Controls, Environment, VesselState, rcs_push, upright
from vessel.vessel import Vessel


@pytest.fixture(scope="module")
def parts():

    return load_parts()


def station(parts) -> Vessel:

    vessel = Vessel.from_design(load_design("station_core", parts))

    # In orbit: everything below the habitat dropped.
    while vessel.stage() is not None:
        pass

    vessel.dropped.clear()

    return vessel


def ferry(parts) -> Vessel:

    vessel = Vessel.from_design(load_design("ferry", parts))

    # Booster gone, the upper stage lit.
    vessel.stage()
    vessel.stage()
    vessel.dropped.clear()

    return vessel


def test_free_ports(parts):

    core = station(parts)

    assert [port.end for port in docking.free_ports(core)] == ["top", "bottom", "side", "side"]

    ship = ferry(parts)

    assert [port.end for port in docking.free_ports(ship)] == ["top"]


def test_bottom_to_top_keeps_both_upright(parts):

    core = station(parts)
    ship = ferry(parts)

    core_mass, ship_mass = core.mass, ship.mass

    joined, below, rise = docking.merge(core, "bottom", ship, "top")

    assert below and rise == pytest.approx(ship.height)
    assert joined.mass == pytest.approx(core_mass + ship_mass)
    assert docking.modules(joined) == 2

    # The station's parts first, the ferry's after, none turned.
    assert joined.parts[0].part.id == "docking_port_2m5"
    assert not any(state.flipped for state in joined.parts)

    # The joint: the station's bottom port then the ferry's top one.
    at = docking.joint(joined)

    assert joined.parts[at].part.id == "docking_port" and joined.parts[at - 1].part.id == "docking_port_2m5"

    # The ferry's stages come after the station's.
    assert all(state.stage is None or state.stage >= joined.next_stage for state in joined.parts if not state.ignited)

    # Its engine still fires (not turned over), from tanks it shares.
    assert joined.burning_engines()

    # And apart again.
    upper, lower, lift, upper_turned, lower_turned = docking.split(joined, at)

    assert [state.part.id for state in upper.parts] == [state.part.id for state in core.parts]
    assert [state.part.id for state in lower.parts] == [state.part.id for state in ship.parts]
    assert not upper_turned and not lower_turned
    assert lift == pytest.approx(ship.height)
    assert docking.free_ports(upper) and docking.free_ports(lower)
    assert docking.modules(upper) == docking.modules(lower) == 1


def test_top_to_top_turns_one_over(parts):

    core = station(parts)
    ship = ferry(parts)

    joined, below, rise = docking.merge(core, "top", ship, "top")

    assert not below and rise == 0.0

    # The ferry above, upside down: its engine pointing the
    # wrong way does not fire.
    flipped = [state for state in joined.parts if state.flipped]

    assert len(flipped) == len(ship.parts)
    assert joined.parts[0].part.id == ship.parts[-1].part.id
    assert not joined.burning_engines()

    upper, lower, _, upper_turned, lower_turned = docking.split(joined, docking.joint(joined))

    # The ferry, split off, turns back its own way up.
    assert upper_turned and not lower_turned
    assert not any(state.flipped for state in upper.parts)
    assert upper.burning_engines()


def test_ports_and_capture(parts):

    ship = ferry(parts)

    orientation = upright(np.array([1.0, 0.0, 0.0]))

    point, facing = docking.port_point(ship, np.array([100.0, 0.0, 0.0]), orientation, "top")

    np.testing.assert_allclose(facing, [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(point, [100.0 + ship.height, 0.0, 0.0], atol=1e-6)

    target = (point + np.array([0.5, 0.0, 0.0]), -facing)

    assert docking.can_capture((point, facing), np.zeros(3), target, np.array([0.2, 0.0, 0.0]))

    # Too fast, too far, side on: no.
    assert not docking.can_capture((point, facing), np.zeros(3), target, np.array([1.0, 0.0, 0.0]))
    assert not docking.can_capture((point, facing), np.zeros(3), (point + np.array([2.0, 0, 0]), -facing), np.zeros(3))
    assert not docking.can_capture((point, facing), np.zeros(3), (target[0], np.array([0.0, 1.0, 0.0])), np.zeros(3))


def test_the_approach_flies_in(parts):
    """Free space, two vessels: the guidance brings the ports together and stops them there."""

    ship = ferry(parts)
    core = station(parts)

    target_point = np.array([0.0, 0.0, 0.0])
    target_face = np.array([1.0, 0.0, 0.0])

    # The ferry 40 m out, 6 m off the axis, its nose wrong.
    position = np.array([40.0, 6.0, 0.0])
    velocity = np.zeros(3)
    orientation = upright(np.array([0.0, 1.0, 0.0]))

    acceleration = ship.rcs_thrust / ship.mass

    dt = 0.1

    for step in range(int(1800 / dt)):

        own = docking.port_point(ship, position, orientation, "top")

        if docking.can_capture(own, velocity, (target_point, target_face), np.zeros(3)):
            break

        command = docking.docking_command(own, velocity, orientation, "top", (target_point, target_face), np.zeros(3), acceleration)

        # Pointing: straight to it (the SAS turns it; here at once).
        orientation = upright(command.up)

        controls = Controls(translate=command.translate)

        state = VesselState(position, velocity, orientation, np.zeros(3))

        velocity = velocity + rcs_push(ship, state, controls) * dt
        position = position + velocity * dt

    else:
        pytest.fail("never docked")

    assert step * dt < 1800.0


def test_docking_missions():

    campaign = Campaign((Mission("dock", "Dock", "dock", value=2), Mission("station", "Station", "dock", value=3, crew=True)))

    career = Career(funds=0.0)

    flight = Flight(body="earth", situation="Orbiting", crewed=True, docked=2)

    assert [mission.id for mission in career.check(campaign, __import__("datetime").date(2027, 1, 1), flight, 0.0, {}, set())] == ["dock"]

    three = Flight(body="earth", situation="Orbiting", crewed=True, docked=3)

    assert [mission.id for mission in career.check(campaign, __import__("datetime").date(2027, 1, 1), three, 0.0, {}, set())] == ["station"]


def test_names_come_back(parts):

    core = station(parts)
    ship = ferry(parts)

    # The ferry docks to the station: still the station.
    joined, _, _ = docking.merge(ship, "top", core, "bottom")

    assert joined.name == "Station Core"

    upper, lower, *_ = docking.split(joined, docking.joint(joined))

    assert upper.name == "Station Core" and lower.name == "Ferry"


def test_a_module_on_a_side_port(parts):

    core = station(parts)
    ship = ferry(parts)

    sides = [port for port in docking.free_ports(core) if port.end == "side"]

    assert [port.copy for port in sides] == [0, 1]

    # The side port faces straight out: +X for the first copy.
    point, facing = docking.port_face(core, np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), sides[0])

    np.testing.assert_allclose(facing, [1.0, 0.0, 0.0], atol=1e-9)
    assert point[0] == pytest.approx(sides[0].state.distance + sides[0].state.part.height)

    assert docking.can_hang(ship) == ""

    mass = core.mass + ship.mass

    ship_ids = [state.part.id for state in ship.parts]

    joined = docking.attach_side(core, sides[0], ship, "top")

    assert joined.mass == pytest.approx(mass)
    assert docking.modules(joined) == 2 and joined.name == "Station Core"

    # The ferry hangs straight out, its top (the port) in, the
    # rest outward one after another; its engine does not fire.
    module = [state for state in joined.parts if state.anchor]

    assert [state.part.id for state in module] == ship_ids
    assert all(state.outward == -1 for state in module)
    assert all(b.distance > a.distance for a, b in zip(module, module[1:]))
    assert not joined.burning_engines()

    # The other side port is still free; the used one not.
    assert [(port.end, port.copy) for port in docking.free_ports(joined) if port.end == "side"] == [("side", 1)]

    detached = docking.detach_side(joined)

    assert [state.part.id for state in detached.guest.parts] == ship_ids
    assert detached.guest.name == "Ferry" and detached.host.name == "Station Core"
    assert detached.sign == -1
    np.testing.assert_allclose(detached.out, [1.0, 0.0, 0.0], atol=1e-9)

    # Its stages back: the engine fires again once staged.
    assert any(state.stage is not None for state in detached.guest.parts)
    assert not any(state.outward or state.anchor for state in detached.guest.parts)

    assert [(port.end, port.copy) for port in docking.free_ports(detached.host) if port.end == "side"] == [("side", 0), ("side", 1)]


def test_only_plain_stacks_hang_off_side_ports(parts):

    mariner = Vessel.from_design(load_design("mariner", parts))

    assert docking.can_hang(mariner)


def test_thrusters_burn_propellant(parts):

    ship = ferry(parts)

    state = VesselState(np.array([7.0e6, 0.0, 0.0]), np.zeros(3), upright(np.array([1.0, 0.0, 0.0])), np.zeros(3))

    before = ship.propellant

    push = rcs_push(ship, state, Controls(translate=np.array([1.0, 0.0, 0.0])), 10.0)

    assert np.linalg.norm(push) > 0.0

    # F / (Isp g0) for 10 s.
    from vessel.physics import RCS_ISP
    from vessel.vessel import G0

    assert before - ship.propellant == pytest.approx(ship.rcs_thrust / (RCS_ISP * G0) * 10.0, rel=1e-6)

    # Dry: no push.
    ship.draw_propellant(float("inf"))

    assert np.linalg.norm(rcs_push(ship, state, Controls(translate=np.array([1.0, 0.0, 0.0])), 1.0)) == 0.0


def test_an_undocked_module_keeps_its_staging(parts):

    core = station(parts)
    ship = ferry(parts)

    before = ship.next_stage
    lit = [state.ignited for state in ship.parts]

    side = next(port for port in docking.free_ports(core) if port.end == "side")

    joined = docking.attach_side(core, side, ship, "top")

    detached = docking.detach_side(joined)

    assert detached.guest.next_stage == before
    assert [state.ignited for state in detached.guest.parts] == lit

    # Its upper engine burns again as it did.
    assert detached.guest.burning_engines()
