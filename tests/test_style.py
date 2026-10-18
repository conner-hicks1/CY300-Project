from game import style
from game.hud import CONTROLS, notice_tone


def test_tones_fall_back_to_text():

    assert style.tone(None) == style.TONES["text"]
    assert style.tone("no such tone") == style.TONES["text"]
    assert style.tone("good") == style.TONES["good"]


def test_every_situation_has_a_tone():

    for situation, name in style.SITUATIONS.items():
        assert name in style.TONES, situation


def test_notices_are_toned_by_what_they_say():

    assert notice_tone("Pathfinder burned up coming in") == "bad"
    assert notice_tone("No link (Below every station's horizon)") == "warn"
    assert notice_tone("Mission done: Reach orbit (+$80M)") == "good"
    assert notice_tone("Stage 1: separated, engines lit") == "text"


def test_the_controls_list_has_f1():

    keys = [keys for keys, _ in CONTROLS]

    assert len(set(keys)) == len(keys)
    assert any("F1" in key for key in keys)
