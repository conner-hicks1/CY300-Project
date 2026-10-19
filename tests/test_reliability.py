import math

from types import SimpleNamespace

import pytest

from programme.career import Career, load_campaign
from systems.programme_system import ProgrammeSystem
from vessel import reliability
from vessel.parts import load_design, load_parts
from vessel.records import vessel_from_json, vessel_to_json
from vessel.reliability import Experience, Failures, Record
from vessel.vessel import Vessel


class Always:
    """A die that always rolls `value`."""

    def __init__(self, value: float):
        self.value = value

    def random(self) -> float:
        return self.value


def test_new_parts_fail_more_than_flown_ones():

    new = Experience()
    flown = Experience(seconds=6_000.0, ignitions=30)

    assert reliability.ignition_failure(new) == pytest.approx(0.042)
    assert reliability.ignition_failure(flown) < 0.01

    # A 300 s burn: ~3% new, a fraction of a percent flown.
    assert 1.0 - math.exp(-reliability.burn_rate(new) * 300.0) == pytest.approx(0.0303, abs=0.001)
    assert 1.0 - math.exp(-reliability.burn_rate(flown) * 300.0) < 0.004

    assert reliability.mission_reliability(new, 300.0) < 0.94 < reliability.mission_reliability(flown, 300.0)

    assert reliability.chute_failure(Experience(ignitions=30)) < reliability.chute_failure(new)


def test_flying_teaches_either_way():

    record = Record()

    failures = Failures(record, seed=1)

    failures.random = Always(0.999)

    assert failures.lights("engine_small", 2)
    assert failures.burns("engine_small", 100.0, 2)
    assert failures.opens("parachute_main")

    experience = record.of("engine_small")

    assert experience.ignitions == 2 and experience.seconds == 200.0 and experience.failures == 0

    failures.random = Always(0.0)

    assert not failures.lights("engine_small")
    assert not failures.burns("engine_small", 1.0)
    assert record.of("engine_small").failures == 2

    # Off (a sandbox, tests): never fails, still learns.
    failures.enabled = False

    assert failures.lights("engine_small") and failures.burns("engine_small", 1.0)
    assert record.of("engine_small").ignitions == 4


def test_the_career_keeps_its_flight_data(tmp_path):

    career = Career(funds=10.0)

    career.reliability.of("engine_booster").seconds = 1_234.0
    career.reliability.of("engine_booster").ignitions = 5

    path = tmp_path / "career.json"

    career.save(path)

    loaded = Career.load(load_campaign(), path)

    assert loaded.reliability.of("engine_booster").seconds == 1_234.0
    assert loaded.reliability.of("engine_booster").ignitions == 5


def test_a_failed_engine_will_not_light_or_burn():

    parts = load_parts()

    rocket = Vessel.from_design(load_design("pathfinder", parts))

    booster = rocket.parts[-1]

    assert booster.part.engine is not None

    booster.failed = True

    rocket.stage()

    assert not booster.ignited and rocket.burning_engines() == []

    # Kept with the vessel.
    loaded = vessel_from_json(vessel_to_json(rocket), parts)

    assert loaded.parts[-1].failed


def test_staging_rolls_for_each_engine_and_parachute():

    parts = load_parts()

    programme = ProgrammeSystem(load_campaign(), Career(funds=1_000.0), None)

    programme.failures.random = Always(0.0)

    rocket = Vessel.from_design(load_design("pathfinder", parts))

    component = SimpleNamespace(vessel=rocket)

    stage = rocket.stage()

    notices = programme.staged(component, stage)

    assert notices == ["Engine failure: the kerosene booster engine did not light"]
    assert rocket.parts[-1].failed and rocket.burning_engines() == []

    # A sandbox: nothing fails.
    sandbox = ProgrammeSystem(load_campaign(), Career(funds=0.0), None, sandbox=True)

    sandbox.failures.random = Always(0.0)

    rocket = Vessel.from_design(load_design("pathfinder", parts))

    stage = rocket.stage()

    assert sandbox.staged(SimpleNamespace(vessel=rocket), stage) == []
    assert rocket.burning_engines()
