import numpy as np
import pytest

from systems.comms_system import SPEED_OF_LIGHT, antenna_range, blocked_by, crew
from vessel.parts import load_design, load_parts
from vessel.sequence import (
    FlightComputer,
    FlightContext,
    SequenceError,
    hover_throttle,
    list_programs,
    parse,
    parse_condition,
    parse_duration
)
from vessel.vessel import Vessel


# =========================================================
# Parsing
# =========================================================

def test_durations():

    assert parse_duration("30 s") == 30.0
    assert parse_duration("5 min") == 300.0
    assert parse_duration("1:30") == 90.0
    assert parse_duration("1:02:03") == 3_723.0
    assert parse_duration("2h") == 7_200.0
    assert parse_duration("1.5 d") == 129_600.0
    assert parse_duration("45") == 45.0

    with pytest.raises(SequenceError):
        parse_duration("soon")


def test_statements():

    program = parse(
        "# a comment\n"
        "throttle full\n"
        "throttle 60%   # trailing comment\n"
        "throttle 0.25\n"
        "sas Retrograde\n"
        "hover\n"
        "descend 3 m/s\n"
        "stage\n"
        "wait 10 s\n"
        "wait until altitude < 12 km\n"
        "at periapsis - 1 min\n"
        "at node\n"
        "log Hello, World\n"
        "burn\n"
    )

    kinds = [(statement.kind, statement.value) for statement in program.statements]

    assert kinds[:7] == [
        ("throttle", 1.0), ("throttle", 0.6), ("throttle", 0.25), ("sas", "retrograde"),
        ("hover", None), ("descend", 3.0), ("stage", None)
    ]

    assert program.statements[7].value == 10.0
    assert str(program.statements[8].value) == "altitude < 12000"

    at = program.statements[9]

    assert at.kind == "at" and at.value == "periapsis" and at.offset == -60.0

    # The log keeps its case.
    assert program.statements[11].value == "Hello, World"

    # Line numbers are the source's.
    assert program.statements[0].line == 2


@pytest.mark.parametrize("source", [
    "fly to mars",
    "throttle 150%",
    "sas sideways",
    "wait until altitude",
    "wait until depth < 3",
    "wait until altitude < 3 furlongs",
    "at lunch",
    "repeat 2\nscience",
    "end",
    "descend fast",
])
def test_errors_name_the_line(source):

    with pytest.raises(SequenceError) as error:
        parse("# fine\n" + source)

    assert error.value.line >= 2


def test_conditions():

    class Here(FlightContext):

        def __init__(self, **values):
            self.values = values

        def quantity(self, name):
            return self.values.get(name)

        def situation(self):
            return self.values.get("situation", "")

    condition = parse_condition("altitude < 10 km and vertical < 0 or landed")

    assert condition.holds(Here(altitude=5_000.0, vertical=-3.0))
    assert not condition.holds(Here(altitude=5_000.0, vertical=3.0))
    assert condition.holds(Here(altitude=50.0e3, vertical=3.0, situation="Landed"))

    assert parse_condition("not landed").holds(Here(situation="Flying"))
    assert parse_condition("fuel < 10%").holds(Here(fuel=0.05))
    assert parse_condition("speed >= 1.5 km/s").holds(Here(speed=1_500.0))

    # Unknown now (no apoapsis, escaping): not true.
    assert not parse_condition("apoapsis > 0").holds(Here())


# =========================================================
# Running
# =========================================================

class Probe(FlightContext):
    """A pretend vessel for the computer."""

    def __init__(self):

        self.clock = 0.0
        self.values = {}
        self.situation_now = "Orbiting"
        self.actions = []
        self.node = None            # (time, start)
        self.events = {}

    def time(self):
        return self.clock

    def quantity(self, name):
        return self.values.get(name)

    def situation(self):
        return self.situation_now

    def event_time(self, name):
        return self.events.get(name)

    def has_node(self):
        return self.node is not None

    def node_start(self):
        return self.node[1] if self.node is not None else None

    def set_throttle(self, value):
        self.actions.append(("throttle", value))

    def hover(self, sink_speed):
        self.actions.append(("hover", sink_speed))

    def stage(self):
        self.actions.append(("stage",))

    def sas(self, word):
        self.actions.append(("sas", word))

    def science(self):
        self.actions.append(("science",))

    def transmit(self):
        self.actions.append(("transmit",))


def test_waits_and_conditions():

    computer = FlightComputer(parse(
        "sas retrograde\n"
        "wait 30 s\n"
        "stage\n"
        "wait until altitude < 1 km\n"
        "descend 2 m/s\n"
        "wait until landed\n"
        "throttle off\n"
    ))

    probe = Probe()

    probe.values["altitude"] = 5_000.0

    computer.tick(probe)

    assert probe.actions == [("sas", "retrograde")]
    assert computer.deadline == 30.0 and computer.current.kind == "wait"

    probe.clock = 29.0
    computer.tick(probe)

    assert ("stage",) not in probe.actions

    probe.clock = 30.0
    computer.tick(probe)

    assert probe.actions[-1] == ("stage",) and computer.current.kind == "until"

    probe.values["altitude"] = 900.0
    computer.tick(probe)

    assert probe.actions[-1] == ("hover", 2.0)

    probe.situation_now = "Landed"
    computer.tick(probe)

    assert probe.actions[-2:] == [("hover", None), ("throttle", 0.0)]
    assert computer.done and computer.log[-1][1] == "Done"


def test_at_an_event_and_repeat():

    computer = FlightComputer(parse(
        "repeat 3\n"
        "  science\n"
        "  wait 1 h\n"
        "end\n"
        "at periapsis - 10 s\n"
        "log there\n"
        "at apoapsis\n"
    ))

    probe = Probe()
    probe.events["periapsis"] = 20_000.0

    for hour in range(4):
        probe.clock = hour * 3600.0
        computer.tick(probe)

    assert probe.actions.count(("science",)) == 3

    # Then waiting for 10 s before the periapsis.
    assert computer.current.kind == "at" and computer.deadline == pytest.approx(19_990.0)

    probe.clock = 19_990.0
    computer.tick(probe)

    # No apoapsis (escaping): noted, skipped, done.
    assert any("there" == text for _, text in computer.log)
    assert any("no apoapsis" in text for _, text in computer.log)
    assert computer.done


def test_burn_flies_the_node():

    computer = FlightComputer(parse("burn\nlog after"))

    probe = Probe()
    probe.node = (100.0, 90.0)

    computer.tick(probe)

    # Pointing, waiting for the burn's start.
    assert ("sas", "maneuver") in probe.actions and computer.deadline == 90.0
    assert ("throttle", 1.0) not in probe.actions

    probe.clock = 90.0
    computer.tick(probe)

    assert probe.actions[-1] == ("throttle", 1.0) and not computer.done

    # The node is done (the game removes it).
    probe.node = None
    probe.clock = 101.0
    computer.tick(probe)

    assert probe.actions[-1] == ("throttle", 0.0) and computer.done

    # Without a node: skipped, noted.
    lone = FlightComputer(parse("burn"))

    lone.tick(Probe())

    assert lone.done and any("no maneuver node" in text for _, text in lone.log)


def test_a_loop_without_waits_cannot_hang():

    computer = FlightComputer(parse("repeat 1000000\n  log spin\nend"))

    probe = Probe()

    computer.tick(probe)

    assert not computer.done and len(computer.log) <= FlightComputer.STEPS


def test_stop():

    computer = FlightComputer(parse("descend 3\nwait 1 d"))

    probe = Probe()

    computer.tick(probe)
    computer.stop(probe)

    assert computer.done and probe.actions[-1] == ("hover", None) and computer.log[-1][1] == "Stopped"


def test_hover_throttle():

    # Holding the weight: thrust = m g at 0 sinking speed.
    assert hover_throttle(0.0, 0.0, 1.62, 3_000.0, 120_000.0) == pytest.approx(3_000.0 * 1.62 / 120_000.0)

    # Falling too fast: more; rising: none.
    assert hover_throttle(2.0, -20.0, 1.62, 3_000.0, 120_000.0) > hover_throttle(2.0, -2.0, 1.62, 3_000.0, 120_000.0)
    assert hover_throttle(2.0, 10.0, 1.62, 3_000.0, 120_000.0) == 0.0

    # Tilted: more to hold the same; on its side: nothing.
    assert hover_throttle(0.0, 0.0, 1.62, 3_000.0, 120_000.0, 0.5) == pytest.approx(2.0 * 3_000.0 * 1.62 / 120_000.0)
    assert hover_throttle(0.0, 0.0, 1.62, 3_000.0, 120_000.0, 0.0) == 0.0


def test_example_programs_parse():

    programs = [item for item in list_programs() if not item[2]]

    assert len(programs) >= 4

    for name, path, _ in programs:

        with open(path, encoding="utf-8") as handle:
            program = parse(handle.read())

        assert program.statements, name


# =========================================================
# Links
# =========================================================

def test_line_of_sight():

    start = np.array([0.0, 0.0, 0.0])
    end = np.array([10.0, 0.0, 0.0])

    assert blocked_by(start, end, np.array([5.0, 0.5, 0.0]), 1.0)
    assert not blocked_by(start, end, np.array([5.0, 2.0, 0.0]), 1.0)

    # Beyond either end: not between them.
    assert not blocked_by(start, end, np.array([12.0, 0.0, 0.0]), 1.0)

    # The light time to the Moon.
    assert 384_400_000.0 / SPEED_OF_LIGHT == pytest.approx(1.28, abs=0.01)


def test_crew_and_antennas():

    parts = load_parts()

    pathfinder = Vessel.from_design(load_design("pathfinder", parts))
    mariner = Vessel.from_design(load_design("mariner", parts))

    assert crew(pathfinder) == 3 and crew(mariner) == 0

    # The capsule's antenna reaches the Moon; Mariner's dish the outer planets.
    assert antenna_range(pathfinder) >= 4.0e8
    assert antenna_range(mariner) >= 1.0e12


def test_time_left_for_the_warp():

    probe = Probe()

    computer = FlightComputer(parse("wait 1 h\nwait until brake < 2 s\nwait until landed"))

    computer.tick(probe)

    assert computer.time_left(probe) == 3600.0

    probe.clock = 3600.0
    probe.values["brake"] = 50.0

    computer.tick(probe)

    assert computer.time_left(probe) == pytest.approx(48.0)

    probe.values["brake"] = 1.0

    computer.tick(probe)

    # Waiting to land: no telling.
    assert computer.time_left(probe) is None
