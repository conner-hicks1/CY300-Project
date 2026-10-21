import json

from programme import upgrades
from programme.career import Career, load_campaign
from programme.research import load_tree


def test_a_career_from_before_the_tech_tree_keeps_every_part(tmp_path):

    tree = load_tree()

    old = {"format": 1, "funds": 120.0}

    new = upgrades.upgrade("career", old)

    assert new["format"] == upgrades.FORMATS["career"]
    assert set(new["researched"]) == {node.id for node in tree.nodes}

    # (The data read is left as it was.)
    assert "researched" not in old

    # Saved since by the tree's first version, nothing
    # researched yet having flown parts it holds back: the
    # same.
    flown = {"format": 1, "funds": 1.0, "researched": [], "reliability": {"engine_hydrolox": {"seconds": 10.0}}}

    assert set(upgrades.upgrade("career", flown)["researched"]) == {node.id for node in tree.nodes}


def test_a_career_from_the_tree_on_is_left_alone():

    for data in (
        {"format": 1, "funds": 1.0, "researched": [], "reliability": {"engine_booster": {}}},
        {"format": 1, "funds": 1.0, "researched": ["storable"], "reliability": {"engine_hydrolox": {}}},
        {"format": 1, "funds": 1.0, "researched": [], "research_spent": 5.0, "reliability": {"engine_hydrolox": {}}},
    ):

        assert upgrades.upgrade("career", data)["researched"] == data["researched"]


def test_formats_come_and_go(tmp_path):

    path = tmp_path / "career.json"

    path.write_text(json.dumps({"format": 1, "funds": 50.0}), encoding="utf-8")

    career = Career.load(load_campaign(), path)

    assert career.funds == 50.0 and len(career.researched) > 5

    career.save(path)

    assert json.loads(path.read_text(encoding="utf-8"))["format"] == upgrades.FORMATS["career"]

    # From a newer game: read as it is.
    newer = {"format": 99, "funds": 3.0, "researched": []}

    assert upgrades.upgrade("career", newer) == newer

    # No format: the first.
    assert upgrades.upgrade("fleet", {"vessels": []})["format"] == upgrades.FORMATS["fleet"]
