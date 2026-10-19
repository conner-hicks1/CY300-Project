import pytest

from types import SimpleNamespace

from programme.career import Career, load_campaign
from programme.crew import Corps
from systems.comms_system import crew, crew_names
from systems.programme_system import ProgrammeSystem
from vessel import life
from vessel.parts import load_design, load_parts
from vessel.vessel import Vessel


AU = life.AU


def test_a_new_corps():

    corps = Corps.founded()

    names = [astronaut.name for astronaut in corps.astronauts]

    assert len(names) == len(set(names)) == 7
    assert {astronaut.role for astronaut in corps.astronauts} == {"pilot", "engineer", "scientist"}

    # The same every time.
    assert names == [astronaut.name for astronaut in Corps.founded().astronauts]

    again = Corps.from_json(corps.to_json())

    assert [a.name for a in again.astronauts] == names and again.hired == corps.hired


def test_seats_are_filled_pilot_engineer_scientist_first():

    corps = Corps.founded()

    chosen = corps.choose(3)

    assert [astronaut.role for astronaut in chosen] == ["pilot", "engineer", "scientist"]

    corps.launch([a.name for a in chosen], "Pathfinder", 0.0)

    assert all(a.status == "flying" for a in chosen)
    assert len(corps.ready()) == 4

    # Home after ten days: ready, a flight more; one past the
    # dose limit retires.
    chosen[0].dose = 1_200.0

    retired = corps.home([a.name for a in chosen], 10 * 86_400.0, life.CAREER_LIMIT)

    assert retired == [chosen[0].name]
    assert chosen[1].status == "ready" and chosen[1].flights == 1 and chosen[1].days == pytest.approx(10.0)

    assert corps.lose([chosen[2].name]) == [chosen[2].name] and chosen[2].status == "lost"


def test_doses_where_they_are():

    def at(body, height, radius=6_371_000.0, landed=False, pressure=0.0, sun=AU, shielding=1.0):
        return life.Place(body, radius + height, radius, landed, pressure, sun, shielding)

    assert life.background_rate(at("Earth", 0.0, pressure=1.0)) < 0.05
    # The ISS: half a mSv to most of one a day.
    assert 0.5 <= life.background_rate(at("Earth", 400_000.0)) <= 0.8

    # The inner belt's heart: cross it fast.
    assert life.background_rate(at("Earth", 0.7 * 6_371_000.0)) > 50.0

    assert life.background_rate(at("Earth", 60 * 6_371_000.0)) == pytest.approx(life.DEEP_SPACE)
    assert life.background_rate(at("Moon", 0.0, 1_737_400.0, landed=True)) == pytest.approx(1.4)
    assert life.background_rate(at("Mars", 0.0, 3_389_500.0, landed=True, pressure=0.006)) == pytest.approx(0.7)

    # Europa's orbit: thousands; Callisto: little more than
    # deep space.
    assert life.background_rate(at("Europa", 0.0, 1_560_800.0)) > 5_000.0
    assert life.background_rate(at("Callisto", 0.0, 2_410_300.0)) < 5.0

    # A habitat's walls halve it.
    shielded, _ = life.dose_rate(at("Moon", 0.0, 1_737_400.0, landed=True, shielding=0.5), 0.0)

    assert shielded <= 1.4 * 0.5 + 0.7


def test_solar_storms_come_some_months_and_the_magnetosphere_turns_them():

    months = [life.storm_of(month) for month in range(0, 600)]

    storms = [storm for storm in months if storm is not None]

    assert 0.06 < len(storms) / len(months) < 0.2
    assert storms[0] == life.storm_of(next(m for m, s in enumerate(months) if s is not None))

    storm = storms[0]

    peak_time = storm.start + 3600.0

    assert life.storm_rate(peak_time) > 50.0

    deep = life.Place("Earth", 60 * 6_371_000.0, 6_371_000.0, False, 0.0, AU)
    low = life.Place("Earth", 6_771_000.0, 6_371_000.0, False, 0.0, AU)

    assert life.storm_exposure(deep) == pytest.approx(1.0)
    assert life.storm_exposure(low) < 0.05

    # Farther from the Sun, weaker.
    mars = life.Place("Mars", 3_789_500.0, 3_389_500.0, False, 0.0, 1.52 * AU)

    assert life.storm_exposure(mars) == pytest.approx(1.0 / 1.52 ** 2)


def test_supplies_run_down_slower_recycled():

    assert life.days_left(210.0, 3, 0.0) == pytest.approx(14.0)
    assert life.days_left(3_600.0, 4, 0.6) == pytest.approx(450.0)

    assert life.use(210.0, 3, 0.0, 86_400.0) == pytest.approx(195.0)
    assert life.use(10.0, 3, 0.0, 86_400.0) == 0.0


def test_a_launch_fills_the_seats_by_name():

    parts = load_parts()

    programme = ProgrammeSystem(load_campaign(), Career(funds=1_000.0), None)

    rocket = Vessel.from_design(load_design("pathfinder", parts))

    # On the pad: every seat counts.
    assert crew(rocket) == 3 and crew_names(rocket) == []

    component = SimpleNamespace(vessel=rocket, paid=False)

    assert programme.pay_launch(component)

    names = crew_names(rocket)

    assert len(names) == 3 and crew(rocket) == 3
    assert all(programme.career.corps.named(name).status == "flying" for name in names)

    # Lost with it.
    programme.crew_lost(names, "with the Pathfinder")

    assert all(programme.career.corps.named(name).status == "lost" for name in names)


def test_no_astronauts_no_crewed_launch():

    parts = load_parts()

    programme = ProgrammeSystem(load_campaign(), Career(funds=1_000.0), None)

    for astronaut in programme.career.corps.astronauts:
        astronaut.status = "lost"

    component = SimpleNamespace(vessel=Vessel.from_design(load_design("pathfinder", parts)), paid=False)

    assert not programme.pay_launch(component)
    assert any("No astronauts" in news for news in programme.news)

    assert programme.hire("pilot") == "" and programme.career.funds == pytest.approx(995.0)
    assert programme.pay_launch(component)
