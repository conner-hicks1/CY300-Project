from datetime import date

import numpy as np
import pytest

from programme import events
from programme.career import Campaign, Career, Flight, Mission, Reward
from vessel import kepler


AU = 1.495978707e11
SUN_RADIUS = 6.957e8
MOON_RADIUS = 1_737_400.0


def test_apophis_passes_inside_the_geostationary_ring():

    position, velocity = events.apophis_perigee_state()

    assert float(np.linalg.norm(position)) == pytest.approx(38_017_000.0)

    # 7.4 km/s at its closest; square to its position.
    assert float(np.linalg.norm(velocity)) == pytest.approx(7_420.0, abs=20.0)
    assert abs(float(position @ velocity)) < 1e-3 * float(np.linalg.norm(position) * np.linalg.norm(velocity))

    # Leaving on a hyperbola (not bound to the Earth).
    assert kepler.orbit_shape(position, velocity, events.MU_EARTH).period is None

    assert float(np.linalg.norm(events.apophis_position(events.APOPHIS_TIME))) == pytest.approx(38_017_000.0, rel=1e-6)

    # A day out: hundreds of thousands of km.
    for hours in (-24.0, 24.0):
        assert 4.5e8 < float(np.linalg.norm(events.apophis_position(events.APOPHIS_TIME + hours * 3600.0))) < 6e8

    assert events.apophis_near(events.APOPHIS_TIME - 30 * 86_400.0)
    assert not events.apophis_near(events.APOPHIS_TIME - 60 * 86_400.0)

    assert len(events.apophis_path(events.APOPHIS_TIME)) > 50


def test_how_much_of_the_sun_a_body_hides():

    sun = np.zeros(3)

    where = np.array([AU, 0.0, 0.0])

    def moon_at(offset):
        return np.array([AU - 384_400_000.0, offset, 0.0])

    # Centered: the Moon (a little smaller from here) hides
    # most of it; to one side, part; far off, none.
    centered = events.sun_coverage(where, sun, SUN_RADIUS, moon_at(0.0), MOON_RADIUS)

    assert 0.9 < centered < 1.0

    partial = events.sun_coverage(where, sun, SUN_RADIUS, moon_at(MOON_RADIUS), MOON_RADIUS)

    assert 0.2 < partial < centered

    assert events.sun_coverage(where, sun, SUN_RADIUS, moon_at(1e7), MOON_RADIUS) == 0.0

    # Closer to the Moon, it covers the Sun whole.
    near = np.array([AU - 384_400_000.0 + 300_000_000.0, 0.0, 0.0])

    assert events.sun_coverage(near, sun, SUN_RADIUS, moon_at(0.0), MOON_RADIUS) == 1.0

    # Behind the vessel (away from the Sun): nothing.
    assert events.sun_coverage(where, sun, SUN_RADIUS, np.array([AU + 384_400_000.0, 0.0, 0.0]), MOON_RADIUS) == 0.0


def _campaign():

    return Campaign((
        Mission("eclipse", "Eclipse", "eclipse", body="moon", value=0.9, optional=True,
                opens=date(2027, 8, 2), deadline=date(2027, 8, 2), reward=Reward(80.0, 20.0, 15.0)),
        Mission("lunar", "Lunar", "eclipse", body="earth", value=0.99, landed="moon", optional=True,
                opens=date(2028, 12, 31), deadline=date(2028, 12, 31), reward=Reward(120.0, 20.0)),
        Mission("apophis", "Apophis", "apophis", value=2000.0, opens=date(2029, 4, 6), deadline=date(2029, 4, 20),
                optional=True, reward=Reward(300.0, 40.0)),
    ))


def test_an_eclipse_is_met_in_its_shadow_on_its_day():

    campaign = _campaign()

    career = Career(funds=0.0)

    shadowed = Flight(body="earth", situation="Landed", crewed=False, eclipsed={"moon": 0.97})

    # Not before the day.
    assert career.events(campaign, date(2027, 8, 1), shadowed) == []

    done = career.events(campaign, date(2027, 8, 2), shadowed)

    assert [mission.id for mission in done] == ["eclipse"] and career.funds == 80.0

    # A lunar eclipse: only landed on the Moon.
    orbiting = Flight(body="moon", situation="Orbiting", crewed=False, eclipsed={"earth": 1.0})
    landed = Flight(body="moon", situation="Landed", crewed=False, eclipsed={"earth": 1.0})

    assert career.events(campaign, date(2028, 12, 31), orbiting) == []
    assert [mission.id for mission in career.events(campaign, date(2028, 12, 31), landed)] == ["lunar"]

    # Apophis: within 2,000 km.
    assert career.events(campaign, date(2029, 4, 13), Flight("earth", "Orbiting", False, apophis=2_500_000.0)) == []
    assert career.events(campaign, date(2029, 4, 13), Flight("earth", "Orbiting", False, apophis=1_500_000.0))


def test_a_missed_event_costs_no_reputation():

    career = Career(funds=0.0, reputation=10.0)

    career.check(_campaign(), date(2027, 8, 3), None, 0.0, {}, set())

    assert career.outcomes["eclipse"] == "failed"
    assert career.reputation == 10.0
    assert any(news.startswith("Passed:") for news in career.news)
