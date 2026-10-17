from datetime import date

import pytest

from programme.career import (
    CREW_LOST,
    DEADLINE_MISSED,
    RECOVERY,
    Campaign,
    Career,
    Flight,
    Mission,
    Reward,
    load_campaign
)
from programme.resources import production, resources_at
from systems.programme_system import fill_tanks
from vessel.parts import load_design, load_parts
from vessel.vessel import Vessel


TODAY = date(2026, 10, 17)


def flight(**fields) -> Flight:

    values = dict(body="earth", situation="Landed", crewed=True, site="", home="earth", landed_on=frozenset())
    values.update(fields)

    return Flight(**values)


def check(career, campaign, flight_now=None, today=TODAY, science=0.0, mapped=None, sampled=None):

    return career.check(campaign, today, flight_now, science, mapped or {}, sampled or set())


def test_the_campaign_loads():

    campaign = load_campaign()

    ids = [mission.id for mission in campaign.missions]

    assert len(ids) == len(set(ids)) >= 10
    assert "first_orbit" in ids and "mars_samples" in ids

    # Real dates on the calendar.
    assert campaign.mission("mars_soi").deadline == date(2029, 12, 31)

    # Every site named exists.
    from science.sites import load_sites

    catalog = load_sites()

    for mission in campaign.missions:
        if mission.site:
            assert any(site.id == mission.site for site in catalog.of(mission.body)), mission.id


def test_missions_in_order():

    campaign = Campaign((
        Mission("orbit", "Orbit", "orbit", body="earth", reward=Reward(80.0, 10.0)),
        Mission("moon", "Moon", "soi", body="moon", requires=("orbit",), reward=Reward(100.0)),
    ))

    career = Career(funds=0.0)

    assert career.state(campaign.missions[1], TODAY) == "locked"

    # Landed on the pad: not orbiting.
    assert check(career, campaign, flight()) == []

    done = check(career, campaign, flight(situation="Orbiting"))

    assert [mission.id for mission in done] == ["orbit"]
    assert career.funds == 80.0 and career.reputation == 10.0

    # Once only.
    assert check(career, campaign, flight(situation="Orbiting")) == []

    assert career.state(campaign.missions[1], TODAY) == "open"

    assert [mission.id for mission in check(career, campaign, flight(body="moon", situation="Flyby"))] == ["moon"]


def test_landing_sites_crews_and_returns():

    campaign = Campaign((
        Mission("pole", "Pole", "land", body="moon", site="lunar_south_pole"),
        Mission("crew_home", "Crew and home", "return", body="moon", crew=True),
    ))

    career = Career(funds=0.0)

    # Landed on the Moon, not at the pole.
    assert check(career, campaign, flight(body="moon", situation="Landed")) == []

    assert [m.id for m in check(career, campaign, flight(body="moon", situation="Landed", site="lunar_south_pole"))] == ["pole"]

    # Home after landing there, but uncrewed: no.
    assert check(career, campaign, flight(crewed=False, landed_on=frozenset({"moon"}))) == []

    assert [m.id for m in check(career, campaign, flight(landed_on=frozenset({"moon"})))] == ["crew_home"]


def test_science_map_samples_isru():

    campaign = Campaign((
        Mission("science", "Science", "science", value=20.0),
        Mission("map", "Map", "map", body="moon", value=0.5),
        Mission("samples", "Samples", "sample", body="mars"),
        Mission("isru", "Fuel", "isru", value=1000.0),
    ))

    career = Career(funds=0.0)

    assert check(career, campaign, science=19.0, mapped={"moon": 0.49}) == []

    done = check(career, campaign, science=25.0, mapped={"moon": 0.5}, sampled={"mars"})

    assert {m.id for m in done} == {"science", "map", "samples"}

    career.made = 1_000.0

    assert [m.id for m in check(career, campaign)] == ["isru"]


def test_deadlines_on_the_calendar():

    campaign = Campaign((
        Mission("mars", "Mars", "soi", body="mars", deadline=date(2029, 12, 31)),
        Mission("later", "Later", "soi", body="venus", opens=date(2030, 1, 1)),
    ))

    career = Career(funds=0.0)

    assert career.state(campaign.missions[1], TODAY) == "waiting"

    check(career, campaign, today=date(2030, 1, 2))

    assert career.outcomes["mars"] == "failed" and career.reputation == -DEADLINE_MISSED

    # Too late now, even arriving.
    assert check(career, campaign, flight(body="mars", situation="Flyby"), today=date(2030, 1, 3)) == []

    assert career.state(campaign.missions[1], date(2030, 1, 2)) == "open"


def test_the_monthly_budget():

    campaign = Campaign((), 400.0, 25.0, 0.25)

    career = Career(funds=400.0, reputation=40.0)

    # The first look sets the month.
    assert career.pay_budget(campaign, date(2026, 10, 17)) == 0.0

    assert career.pay_budget(campaign, date(2026, 10, 30)) == 0.0

    # Three months later (a window's wait at warp).
    assert career.pay_budget(campaign, date(2027, 1, 2)) == pytest.approx(3 * 35.0)

    assert career.funds == pytest.approx(505.0)

    # The clock going back (a revert) pays nothing.
    assert career.pay_budget(campaign, date(2026, 12, 1)) == 0.0


def test_launch_recovery_and_crew():

    career = Career(funds=100.0)

    assert career.can_afford(96.0) and not career.can_afford(100.5)

    career.launch(96.0)

    assert career.funds == pytest.approx(4.0)

    assert career.recover(50.0) == pytest.approx(50.0 * RECOVERY)

    career.crew_lost()

    assert career.reputation == -CREW_LOST and career.news


def test_saving(tmp_path):

    campaign = load_campaign()

    career = Career(funds=123.0, reputation=7.0, outcomes={"first_orbit": "done"}, paid_through=(2026, 10), made=5.0, clock=8.4e8)

    path = tmp_path / "career.json"

    career.save(path)

    again = Career.load(campaign, path)

    assert (again.funds, again.reputation, again.outcomes, again.paid_through, again.made, again.clock) == (123.0, 7.0, {"first_orbit": "done"}, (2026, 10), 5.0, 8.4e8)

    # Nothing saved: the campaign's start.
    assert Career.load(campaign, tmp_path / "none.json").funds == campaign.start_funds


def test_costs():

    parts = load_parts()

    pathfinder = Vessel.from_design(load_design("pathfinder", parts))

    assert pathfinder.cost == pytest.approx(sum(state.part.cost * state.count for state in pathfinder.parts))

    # Affordable a few times over from the start.
    assert 50.0 < pathfinder.cost < load_campaign().start_funds / 3.0

    assert all(part.cost > 0.0 for part in parts.values())


# =========================================================
# Making propellant
# =========================================================

def test_where_there_is_something_to_make_it_from():

    assert resources_at("moon", -89.0, "lunar_south_pole")[0].richness > 1.0
    assert resources_at("moon", -85.0)[0].kind == "ice"
    assert resources_at("moon", 0.0) == []

    # Mars: its air anywhere, ice at the caps.
    assert [item.kind for item in resources_at("mars", 0.0, has_air=True)] == ["air"]
    assert {item.kind for item in resources_at("mars", 80.0, has_air=True)} == {"ice", "air"}

    assert resources_at("europa", 10.0)[0].kind == "ice"

    assert production(resources_at("mars", 80.0, has_air=True), 600.0, 250.0) == 600.0
    assert production(resources_at("mars", 0.0, has_air=True), 600.0, 250.0) == 250.0
    assert production([], 600.0, 250.0) == 0.0


def test_filling_the_tanks():

    parts = load_parts()

    vessel = Vessel.from_design(load_design("mariner", parts))

    for state in vessel.parts:
        state.propellant = 0.0

    capacity = vessel.propellant_capacity

    assert fill_tanks(vessel, 1_000.0) == pytest.approx(1_000.0)
    assert vessel.propellant == pytest.approx(1_000.0)

    # Evenly by room: every tank the same share full.
    shares = {round(state.propellant / state.part.propellant, 9) for state in vessel.parts if state.part.propellant}

    assert len(shares) == 1

    # Full: no more than the room.
    assert fill_tanks(vessel, 1.0e9) == pytest.approx(capacity - 1_000.0)
    assert fill_tanks(vessel, 10.0) == 0.0

    assert parts["isru_plant"].isru.rate_ice > parts["isru_plant"].isru.rate_air > 0.0


def test_a_sandbox_launches_free(parts=None):

    from types import SimpleNamespace

    from systems.programme_system import ProgrammeSystem

    vessel = Vessel.from_design(load_design("pathfinder", load_parts()))

    sandbox = ProgrammeSystem(load_campaign(), Career(funds=0.0), None, sandbox=True)

    component = SimpleNamespace(paid=False, vessel=vessel)

    assert sandbox.pay_launch(component) and component.paid
    assert sandbox.career.funds == 0.0 and not sandbox.news

    # A career with nothing in the bank: no launch.
    career = ProgrammeSystem(load_campaign(), Career(funds=0.0), None)

    poor = SimpleNamespace(paid=False, vessel=vessel)

    assert not career.pay_launch(poor) and not poor.paid and career.news
