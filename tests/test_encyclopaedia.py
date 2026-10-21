from game.encyclopaedia import SECTIONS, entries
from planet.bodies import load_presets
from programme.research import load_tree
from vessel.parts import load_parts


def test_every_part_body_and_article_has_an_entry():

    parts = load_parts()
    presets = load_presets()

    found = entries(parts, presets, load_tree())

    by_section = {section: [entry for entry in found if entry.section == section] for section in SECTIONS}

    assert len(by_section["Parts"]) == len(parts) and len(by_section["Bodies"]) == len(presets)
    assert len(by_section["How it works"]) >= 15

    # Every entry says something.
    assert all(entry.title and (entry.text or entry.facts) for entry in found)

    moon = next(entry for entry in found if entry.id == "moon")

    assert dict(moon.facts)["Orbits"].startswith("Earth") and dict(moon.facts)["Air"] == "none"

    wing = next(entry for entry in found if entry.id == "solar_wing")

    assert "W at 1 AU" in dict(wing.facts)["Solar"] and dict(wing.facts)["Research"] != "known from the start"


def test_it_is_searched_by_every_word():

    found = entries(load_parts(), load_presets(), load_tree())

    hits = [entry.id for entry in found if entry.matches("light delay")]

    assert "comms" in hits

    assert not [entry for entry in found if entry.matches("zzzz nothing")]
