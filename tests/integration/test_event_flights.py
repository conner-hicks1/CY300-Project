from datetime import datetime, timezone

import numpy as np
import pytest

from ecs.components import ClockComponent

from planet.orbits import seconds_since_j2000
from programme import events

from tests.integration.harness import Game


pytestmark = pytest.mark.integration


def _set_clock(game: Game, moment: datetime):

    clock = next(clock for _, clock in game.app.scene.registry.view_with(ClockComponent))

    clock.time = seconds_since_j2000(moment)

    game.step(3)


def test_the_moon_covers_the_sun_over_luxor_on_2_august_2027(play):
    """The real ephemerides: at the century's longest eclipse, standing in Luxor, the Moon hides the Sun; hours later, nothing; the mission is done."""

    game: Game = play()

    game.step(5)

    component = game.vessel

    earth = game.body("Earth")

    # Greatest eclipse: 10:07 UTC, near Luxor.
    _set_clock(game, datetime(2027, 8, 2, 10, 7, 0, tzinfo=timezone.utc))

    game.app.vessel_system._height = 0.0

    game.land_at(earth, 25.69, 32.64, component)

    game.step(3)

    programme = game.app.programme_system

    eclipsed, _ = programme._events_seen(game.app, component, always=True)

    assert eclipsed.get("moon", 0.0) > 0.95

    assert programme.career.outcomes.get("eclipse_2027") == "done"
    assert any("great eclipse" in notice for notice in game.notices())

    # Three hours later: the Sun whole again.
    _set_clock(game, datetime(2027, 8, 2, 13, 30, 0, tzinfo=timezone.utc))

    game.app.vessel_system._height = 0.0

    game.land_at(earth, 25.69, 32.64, component)

    game.step(3)

    assert programme._events_seen(game.app, component, always=True)[0].get("moon", 0.0) == 0.0


def test_apophis_passes_close(play):
    """Near its pass, the vessel where Apophis is: the mission is done."""

    game: Game = play()

    game.step(5)

    component = game.vessel

    # (It waits on reaching orbit first.)
    game.app.programme_system.career.outcomes["first_orbit"] = "done"

    # A while before its pass, the vessel where it will be.
    _set_clock(game, datetime(2029, 4, 13, 21, 40, 0, tzinfo=timezone.utc))

    now = game.app.orbit_system.time

    there = events.apophis_position(now)

    game.put_in_orbit(component, altitude=float(np.linalg.norm(there)) - 6_371_000.0, direction=there / np.linalg.norm(there), pole=(0.0, 1.0, 0.0))

    game.step(3)

    _, apophis = game.app.programme_system._events_seen(game.app, component, always=True)

    assert apophis is not None and apophis < 50_000.0

    assert game.app.programme_system.career.outcomes.get("apophis_2029") == "done"
