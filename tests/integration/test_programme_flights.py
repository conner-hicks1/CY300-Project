import pytest

from ecs.components import ClockComponent

from tests.integration.harness import Game


pytestmark = pytest.mark.integration


def test_a_launch_is_paid_for_and_orbit_pays(play):

    game: Game = play()

    career = game.app.programme_system.career

    start = career.funds

    game.step(10)

    assert game.vessel.state.landed and not game.vessel.paid

    game.stage()

    assert game.vessel.paid
    assert career.funds == pytest.approx(start - game.vessel.vessel.cost)

    game.put_in_orbit(altitude=200_000.0)

    game.run_until(lambda: career.outcomes.get("first_orbit") == "done", 600)

    assert career.funds == pytest.approx(start - game.vessel.vessel.cost + 80.0)


def test_no_funds_no_launch(play):

    game: Game = play()

    game.app.programme_system.career.funds = 5.0

    game.step(5)

    game.stage()

    assert not game.vessel.paid and game.vessel.vessel.next_stage == 0
    assert any("Not enough funds" in notice and "enough on 01 " in notice for notice in game.notices())


def test_a_revert_keeps_the_calendar(play):

    game: Game = play()

    game.step(5)

    game.stage()

    game.put_in_orbit()

    # A month on.
    clock = next(component for _, component in game.app.scene.registry.view_with(ClockComponent))

    clock.time += 30 * 86_400.0

    game.step(5)

    before = game.clock

    game.app.revert()

    game.step(2)

    # Not back, not on to the next morning: where it was.
    assert before <= game.clock < before + 60.0


def test_the_sandbox_launches_free_and_travels(play):

    game: Game = play(sandbox=True, body="moon")

    assert game.app.vessel_system.body_name(game.app.scene, game.vessel.body) == "Moon"

    game.step(5)

    game.stage()

    assert game.vessel.paid and game.app.programme_system.career.funds == 0.0


def test_a_new_career_researches_before_it_flies_a_probe(play):
    """A new career: Mariner waits on its research; science earned buys it; then it launches, and the career keeps it."""

    import json

    game: Game = play(design="mariner", researched=False)

    programme = game.app.programme_system

    record = game.app.science_system.record

    start = programme.career.funds

    game.step(5)

    game.app.stage_vessel(game.vessel)

    game.step(2)

    assert not game.vessel.paid and programme.career.funds == start
    assert any("Not researched yet" in notice for notice in game.notices())

    # Science from earlier flights; the nodes Mariner needs,
    # in the tree's order.
    record.points = 400.0

    needed = ["side_boosters", "small_rockets", "instruments", "electrics", "probes", "survey", "deep_space", "storable"]

    for node_id in needed:
        assert programme.research(node_id, record) == ""

    spent = sum(programme.tree.node(node_id).cost for node_id in needed)

    assert programme.science_to_spend(record) == pytest.approx(400.0 - spent)
    assert programme.unresearched(game.vessel.vessel) == []

    game.app.stage_vessel(game.vessel)

    assert game.vessel.paid

    programme.save()

    saved = json.loads(programme.path.read_text(encoding="utf-8"))

    assert set(needed) <= set(saved["researched"]) and saved["research_spent"] == pytest.approx(spent)


def test_an_engine_fails_in_flight_and_the_programme_learns(play):
    """Failures on: the booster lights, burns (flight data grows), then a failure shuts it down; the HUD and the notices say so."""

    from core.key_codes import Key

    from tests.test_reliability import Always

    game: Game = play(failures=True)

    programme = game.app.programme_system

    assert programme.failures.enabled

    programme.failures.random = Always(0.999)

    game.step(5)

    game.press(Key.Z)
    game.stage()

    game.step(120)

    booster = game.vessel.vessel.parts[-1]

    assert booster.ignited and not booster.failed

    flown = programme.career.reliability.of(booster.part.id)

    assert flown.ignitions == 1 and flown.seconds > 1.0

    # The next roll fails.
    programme.failures.random = Always(0.0)

    game.step(2)

    assert booster.failed and game.vessel.vessel.burning_engines() == []
    assert any("Engine failure" in notice for notice in game.notices())

    readout = game.app.vessel_system.readout(game.app.scene, game.app.orbit_system, game.app.planet_system)

    assert booster.part.name in readout.failed
