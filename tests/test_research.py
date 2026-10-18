import json

from types import SimpleNamespace

import pytest

from programme import research
from programme.career import Career, load_campaign
from programme.research import TechError, load_tree
from science.record import ScienceRecord
from systems.programme_system import ProgrammeSystem
from vessel.parts import load_design, load_parts
from vessel.vessel import Vessel


@pytest.fixture(scope="module")
def parts():

    return load_parts()


@pytest.fixture(scope="module")
def tree(parts):

    return load_tree(parts=parts)


def test_every_part_is_in_the_tree_once(tree, parts):

    assert all(tree.node_of(part_id) is not None for part_id in parts)

    # Basic rocketry is free: Pathfinder from the start.
    assert tree.free == {"basic_rocketry"}

    locked = research.locked_parts(tree, set())

    pathfinder = load_design("pathfinder", parts)

    assert not any(entry.part.id in locked for entry in pathfinder.stack)

    assert "probe_core" in locked and locked["probe_core"].id == "probes"


def test_columns_follow_what_comes_first(tree):

    assert tree.depth("basic_rocketry") == 0
    assert tree.depth("probes") == 2
    assert tree.depth("isru") == 4


def test_researching_needs_what_comes_first_and_the_science(tree):

    researched = set()

    probes = tree.node("probes")

    assert research.state(tree, probes, researched) == "locked"
    assert "needs" in research.research(tree, "probes", researched, 1_000.0)

    assert research.research(tree, "electrics", researched, 5.0).startswith("Electrics needs 12 science")
    assert research.research(tree, "electrics", researched, 12.0) == ""
    assert research.research(tree, "small_rockets", researched, 10.0) == ""

    assert research.state(tree, probes, researched) == "open"
    assert research.research(tree, "probes", researched, 25.0) == ""
    assert research.state(tree, probes, researched) == "known"
    assert "known already" in research.research(tree, "probes", researched, 100.0)

    assert "probe_core" not in research.locked_parts(tree, researched)


def _write(tmp_path, nodes):

    path = tmp_path / "tech.json"

    path.write_text(json.dumps({"nodes": nodes}), encoding="utf-8")

    return path


def test_a_bad_tree_is_refused(tmp_path, parts):

    with pytest.raises(TechError, match="loops"):
        load_tree(_write(tmp_path, [
            {"id": "a", "requires": ["b"]},
            {"id": "b", "requires": ["a"]},
        ]))

    with pytest.raises(TechError, match="not a node"):
        load_tree(_write(tmp_path, [{"id": "a", "requires": ["nowhere"]}]))

    with pytest.raises(TechError, match="both"):
        load_tree(_write(tmp_path, [{"id": "a", "parts": ["capsule"]}, {"id": "b", "parts": ["capsule"]}]))

    with pytest.raises(TechError, match="catalog"):
        load_tree(_write(tmp_path, [{"id": "a", "parts": ["warp_drive"]}]), parts=parts)


def test_the_career_keeps_its_research(tmp_path):

    career = Career(funds=100.0, researched={"electrics", "probes"}, research_spent=37.0)

    path = tmp_path / "career.json"

    career.save(path)

    loaded = Career.load(load_campaign(), path)

    assert loaded.researched == {"electrics", "probes"} and loaded.research_spent == 37.0

    # An older save: nothing researched.
    data = json.loads(path.read_text(encoding="utf-8"))

    del data["researched"], data["research_spent"]

    path.write_text(json.dumps(data), encoding="utf-8")

    assert Career.load(load_campaign(), path).researched == set()


def _programme(tree, sandbox=False):

    return ProgrammeSystem(load_campaign(), Career(funds=1_000.0), None, sandbox=sandbox, tree=tree)


def test_the_programme_spends_science_and_keeps_what_was_earned(tree):

    programme = _programme(tree)

    record = ScienceRecord(points=30.0)

    assert programme.science_to_spend(record) == 30.0

    assert programme.research("electrics", record) == ""
    assert programme.science_to_spend(record) == pytest.approx(18.0)

    # The science banked (for the missions) is all of it.
    assert record.points == 30.0

    assert "needs" in programme.research("deep_space", record)
    assert any("Electrics" in news for news in programme.news)


def test_a_vessel_waits_on_its_research(tree, parts):

    programme = _programme(tree)

    mariner = SimpleNamespace(vessel=Vessel.from_design(load_design("mariner", parts)), paid=False)

    assert not programme.pay_launch(mariner)
    assert not mariner.paid and programme.career.funds == 1_000.0
    assert any("Not researched yet" in news for news in programme.news)

    programme.career.researched = {node.id for node in tree.nodes}

    assert programme.unresearched(mariner.vessel) == []
    assert programme.pay_launch(mariner) and mariner.paid

    # Pathfinder: from the start.
    pathfinder = SimpleNamespace(vessel=Vessel.from_design(load_design("pathfinder", parts)), paid=False)

    assert _programme(tree).pay_launch(pathfinder)


def test_a_sandbox_knows_everything(tree, parts):

    programme = _programme(tree, sandbox=True)

    assert programme.locked_parts() == {}

    probe = SimpleNamespace(vessel=Vessel.from_design(load_design("mariner", parts)), paid=False)

    assert programme.pay_launch(probe)


@pytest.mark.parametrize("design_id", ["pathfinder", "pathfinder_heavy", "ferry", "explorer", "mariner", "station_core"])
def test_every_built_in_design_can_be_researched(tree, parts, design_id):

    design = load_design(design_id, parts)

    vessel = Vessel.from_design(design)

    needed = {tree.node_of(state.part.id).id for state in vessel.parts}

    # With what they need and what that needs, nothing is
    # locked.
    known = set()

    def add(node_id):
        if node_id not in known:
            known.add(node_id)
            for other in tree.node(node_id).requires:
                add(other)

    for node_id in needed:
        add(node_id)

    locked = research.locked_parts(tree, known)

    assert not any(state.part.id in locked for state in vessel.parts)
