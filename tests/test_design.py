import json

import pytest

from vessel.design import (
    DesignDraft,
    list_designs,
    open_design,
    save_draft,
    slug
)
from vessel.parts import PartError, load_design, load_parts
from vessel.vessel import Vessel


@pytest.fixture(scope="module")
def parts():

    return load_parts()


@pytest.fixture
def pathfinder(parts):

    return DesignDraft.from_design(load_design("pathfinder", parts))


def ids(draft):

    return [entry.part.id for entry in draft.entries]


def stages(draft):

    return [entry.stage for entry in draft.entries]


# =========================================================
# Editing
# =========================================================

def test_insert_move_remove(parts):

    draft = DesignDraft("Test")

    assert draft.insert(0, parts["capsule"]) == 0
    assert draft.insert(99, parts["engine_small"]) == 1

    # Between them.
    assert draft.insert(1, parts["tank_1m25_long"]) == 1

    assert ids(draft) == ["capsule", "tank_1m25_long", "engine_small"]

    assert draft.move(2, -1) == 1
    assert ids(draft) == ["capsule", "engine_small", "tank_1m25_long"]

    # Off the ends: nothing moves.
    assert draft.move(0, -1) == 0
    assert draft.move(2, 1) == 2

    draft.remove(1)

    assert ids(draft) == ["capsule", "tank_1m25_long"]

    draft.remove(7)

    assert len(draft.entries) == 2


def test_only_staged_kinds_take_stages(parts):

    draft = DesignDraft("Test")

    draft.insert(0, parts["capsule"])
    draft.insert(1, parts["engine_small"])

    draft.set_stage(0, 3)
    draft.set_stage(1, -2)

    assert stages(draft) == [None, 0]


def test_auto_stage_reproduces_pathfinder(pathfinder):

    designed = stages(pathfinder)

    pathfinder.auto_stage()

    assert stages(pathfinder) == designed


def test_auto_stage_single_stage_and_no_engine(parts):

    draft = DesignDraft("Hopper")

    for part in ("probe_core", "tank_1m25_short", "engine_small"):
        draft.insert(99, parts[part])

    draft.auto_stage()

    assert stages(draft) == [None, None, 0]

    # A separator under a capsule with no engines: its
    # own stage 0, then the parachute.
    lander = DesignDraft("Drop")

    for part in ("parachute_main", "capsule", "decoupler_2m5", "tank_2m5_short"):
        lander.insert(99, parts[part])

    lander.auto_stage()

    assert stages(lander) == [1, None, 0, None]


# =========================================================
# Checking
# =========================================================

def test_problems_and_warnings(parts):

    draft = DesignDraft("Test")

    assert draft.problems()

    draft.insert(0, parts["tank_1m25_short"])
    draft.insert(1, parts["engine_small"])

    problems = draft.problems()

    assert any("command" in problem for problem in problems)
    assert any("no stage" in problem for problem in problems)

    with pytest.raises(PartError):
        draft.to_design(parts)

    draft.insert(0, parts["probe_core"])
    draft.auto_stage()

    assert draft.problems() == []

    design = draft.to_design(parts)

    assert design.name == "Test" and len(design.stack) == 3

    # An engine with nothing to burn.
    dry = DesignDraft("Dry")

    for part in ("probe_core", "engine_small"):
        dry.insert(99, parts[part])

    assert any("no tank" in warning for warning in dry.warnings())

    # Coming home without a heat shield.
    home = DesignDraft("Home")

    for part in ("parachute_main", "capsule", "decoupler_2m5", "tank_2m5_short", "engine_booster"):
        home.insert(99, parts[part])

    assert any("heat shield" in warning for warning in home.warnings())

    home.insert(2, parts["heat_shield_2m5"])

    assert not any("heat shield" in warning for warning in home.warnings())


def test_stage_stats_add_up(parts, pathfinder):

    vessel = Vessel.from_design(pathfinder.to_design(parts))

    stats = vessel.stage_stats(1.01325, 9.80665)

    assert [stage.stage for stage in stats] == [0, 1, 2, 3]

    assert sum(stage.delta_v_vacuum for stage in stats) == pytest.approx(vessel.delta_v(), rel=1e-9)

    # Sea level costs the booster ~10% (Isp 285 vs 315).
    assert stats[0].delta_v_surface == pytest.approx(stats[0].delta_v_vacuum * 285.0 / 315.0, rel=1e-9)

    assert stats[0].twr_surface == pytest.approx(1.27, abs=0.01)
    assert stats[0].burn_time == pytest.approx(168.5, abs=0.5)

    # Separating and the parachute: no delta-v.
    assert stats[2].delta_v_vacuum == 0.0 and stats[3].engines == 0

    # Nothing about the vessel itself changed.
    assert vessel.next_stage == 0 and vessel.mass == pytest.approx(159_970.0)


# =========================================================
# Files
# =========================================================

def test_save_list_open_round_trip(tmp_path, parts, pathfinder):

    pathfinder.name = "My Rocket!"

    path = save_draft(pathfinder, tmp_path / "user")

    assert path.name == "my_rocket.json"
    assert json.loads(path.read_text())["name"] == "My Rocket!"

    found = list_designs(tmp_path / "user", tmp_path / "missing")

    assert [(design.id, design.built_in) for design in found] == [("my_rocket", False)]

    reopened = open_design(found[0], parts)

    assert ids(reopened) == ids(pathfinder) and stages(reopened) == stages(pathfinder)

    # The built-in designs are listed too.
    assert any(design.id == "pathfinder" and design.built_in for design in list_designs(tmp_path / "user"))


def test_slug():

    assert slug("Pathfinder Mk II") == "pathfinder_mk_ii"
    assert slug("***") == "vessel"


# =========================================================
# Radial Groups
# =========================================================

def booster_draft(parts):

    draft = DesignDraft("Twin")

    for part in ("probe_core", "tank_2m5_long", "engine_booster"):
        draft.insert(99, parts[part])

    group = draft.attach_group(1, 2, parts["decoupler_radial"])

    path = draft.add_after(group, parts["nose_cone_1m25"])
    path = draft.add_after(path, parts["tank_1m25_xl"])
    draft.add_after(path, parts["engine_medium"])

    draft.auto_stage()

    return draft


def test_attach_and_fill_a_group(parts):

    draft = booster_draft(parts)

    group = draft.get((1, 0))

    assert group.symmetry == 2 and group.decoupler.part.id == "decoupler_radial"
    assert [entry.part.id for entry in group.entries] == ["nose_cone_1m25", "tank_1m25_xl", "engine_medium"]

    # Rows: core, the group, its separator and parts.
    assert [path for path, _ in draft.rows()] == [
        (0,), (1,), (1, 0), (1, 0, "separator"), (1, 0, 0), (1, 0, 1), (1, 0, 2), (2,)
    ]

    # A second separator is refused; reordering stays in the group.
    assert draft.add_after((1, 0), parts["decoupler_radial"]) is None
    assert draft.move((1, 0, 2), -1) == (1, 0, 1)
    assert draft.get((1, 0, 1)).part.id == "engine_medium"

    draft.set_symmetry((1, 0), 4)
    draft.set_offset((1, 0), -3.0)

    assert group.symmetry == 4 and group.offset == 0.0


def test_auto_stage_lights_boosters_with_the_core(parts):

    draft = booster_draft(parts)

    assert draft.get((2,)).stage == 0
    assert draft.get((1, 0, 2)).stage == 0
    assert draft.get((1, 0, "separator")).stage == 1

    assert draft.problems() == []


def test_empty_group_and_missing_separator(parts):

    draft = booster_draft(parts)

    draft.attach_group(1, 3, None)

    assert any("no parts" in problem for problem in draft.problems())

    draft.add_after((1, 1), parts["tank_1m25_short"])

    assert draft.problems() == []
    assert any("no separator" in warning for warning in draft.warnings())

    draft.remove((1, 1))

    assert len(draft.get((1,)).radial) == 1


def test_radial_round_trip(tmp_path, parts):

    draft = booster_draft(parts)

    design = draft.to_design(parts)

    assert design.stack[1].radial[0].symmetry == 2

    vessel = Vessel.from_design(design)

    assert sum(1 for state in vessel.parts if state.radial) == 4

    path = save_draft(draft, tmp_path)

    reopened = open_design(next(d for d in list_designs(tmp_path, tmp_path / "none")), parts)

    assert reopened.to_json() == draft.to_json()

    # The built-in Heavy opens, and auto staging keeps its stages.
    heavy = DesignDraft.from_design(load_design("pathfinder_heavy", parts))

    before = heavy.to_json()

    heavy.auto_stage()

    assert heavy.to_json() == before
