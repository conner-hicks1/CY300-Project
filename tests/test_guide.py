from dataclasses import replace

from game.guide import FIRST_ORBIT, HINT_SECONDS, FlightSchool, Hints, Moment


def test_flight_school_goes_step_by_step_to_orbit():

    school = FlightSchool()

    school.start()

    moment = Moment()

    school.update(moment)

    assert school.step.title == "Full throttle"

    # Done out of order: waits for its own step.
    moment = replace(moment, sas=True)

    school.update(moment)

    assert school.index == 0

    flight = [
        dict(throttle=1.0),
        dict(staged=True, thrust=2e6, situation="Flying"),
        # (T already on: passes straight through.)
        dict(height=1_200.0),
        dict(tilt=10.0),
        dict(sas_mode="prograde"),
        dict(apoapsis=82_000.0, situation="Sub-orbital"),
        dict(throttle=0.0, thrust=0.0),
        dict(map_open=True),
        dict(node=True, node_in=600.0),
        dict(sas_mode="maneuver"),
        dict(node_in=10.0),
        dict(situation="Orbiting", node=False, node_in=None),
    ]

    seen = []

    for change in flight:

        moment = replace(moment, **change)

        school.update(moment)

        seen.append(school.index)

    assert seen == sorted(seen) and school.finished and school.index == len(FIRST_ORBIT)


def test_a_lost_rocket_starts_the_lesson_over():

    school = FlightSchool()

    school.start()

    moment = Moment(throttle=1.0, staged=True, thrust=1e6, sas=True, situation="Flying", height=2_000.0)

    school.update(moment)

    assert school.index > 3

    school.update(replace(moment, situation="Destroyed"))

    assert school.lost

    # Reverted: on the pad again.
    school.update(Moment())

    assert school.index == 0 and not school.lost

    # Skipping past a step.
    school.skip()

    assert school.step.title == "Light the engines"


def test_hints_come_once_one_at_a_time(tmp_path):

    hints = Hints(tmp_path / "hints.json")

    pad = Moment()

    hints.update(pad, 0.1)

    assert hints.showing[0].id == "pad"

    # Something else meanwhile: waits its turn.
    hints.update(replace(pad, map_open=True), 0.1)

    assert hints.showing[0].id == "pad"

    hints.update(pad, HINT_SECONDS)

    assert hints.showing is None

    hints.update(replace(pad, map_open=True), 0.1)

    assert hints.showing[0].id == "map"

    hints.dismiss()

    # Never again; kept.
    hints.update(replace(pad, map_open=True), 0.1)

    assert hints.showing is None

    again = Hints(tmp_path / "hints.json")

    assert {"pad", "map"} <= again.seen

    again.update(pad, 0.1)

    assert again.showing is None

    # Off: none; reset: from the start.
    again.set_enabled(False)

    again.reset()

    again.update(pad, 0.1)

    assert again.showing is None and not Hints(tmp_path / "hints.json").enabled

    again.set_enabled(True)

    again.update(pad, 0.1)

    assert again.showing[0].id == "pad"


def test_hints_for_the_flight():

    hints = Hints(None)

    def first(**changes):

        hints.seen.clear()
        hints.showing = None

        hints.update(Moment(**changes), 0.1)

        return hints.showing[0].id if hints.showing else None

    flying = dict(staged=True, situation="Flying", thrust=1e6, air=True)

    assert first(**flying, height=6_000.0, tilt=0.0) == "gravity turn"
    assert first(**flying, height=6_000.0, tilt=20.0) is None
    assert first(staged=True, situation="Flying", stage_dry=True) == "stage dry"
    assert first(staged=True, situation="Sub-orbital", thrust=1e6, air_top=70_000.0, apoapsis=80_000.0) == "apoapsis out of the air"
    assert first(staged=True, situation="Flying", air=True, vertical_speed=-80.0, height=8_000.0, parachute="") == "parachute"
    assert first(staged=True, situation="Flying", air=True, vertical_speed=-80.0, height=8_000.0, parachute="none") is None
    assert first(staged=True, situation="Flying", legs="up", height=1_000.0, vertical_speed=-10.0) == "legs"
    assert first(staged=True, situation="Destroyed") == "lost"
    assert first(staged=True, situation="Flying", heat=0.8) == "heat"
