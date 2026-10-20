import json

from programme import saves
from game.title_screen import ago, describe


def test_games_are_made_found_and_listed_last_played_first(tmp_path):

    first = saves.create("My Career", False, tmp_path)
    second = saves.create("my career", True, tmp_path)

    # Two folders, though the names differ only in case.
    assert first.folder != second.folder and first.folder.name == "my-career"

    first.played = 100.0
    second.played = 200.0

    saves.write(first)
    saves.write(second)

    listed = saves.slots(tmp_path)

    assert [slot.folder for slot in listed] == [second.folder, first.folder]
    assert listed[0].sandbox and not listed[1].sandbox

    assert saves.find("MY CAREER", tmp_path) is not None
    assert saves.find("Nobody", tmp_path) is None

    # A folder that is not a game is passed over.
    (tmp_path / "stray").mkdir()

    assert len(saves.slots(tmp_path)) == 2

    saves.delete(first)

    assert [slot.name for slot in saves.slots(tmp_path)] == ["my career"]


def test_unused_names_count_up(tmp_path):

    assert saves.unused_name(False, tmp_path) == "Career"

    saves.create("Career", False, tmp_path)
    saves.create("Career 2", False, tmp_path)

    assert saves.unused_name(False, tmp_path) == "Career 3"
    assert saves.unused_name(True, tmp_path) == "Sandbox"


def test_a_quicksave_is_gone_back_to(tmp_path):

    slot = saves.create("Career", False, tmp_path)

    folder = slot.folder

    (folder / "career.json").write_text('{"funds": 1}', encoding="utf-8")
    (folder / "fleet.json").write_text('{"vessels": []}', encoding="utf-8")

    assert slot.kept(saves.QUICKSAVE) is None
    assert not saves.restore(folder, saves.QUICKSAVE)

    saves.keep(folder, saves.QUICKSAVE)

    assert slot.kept(saves.QUICKSAVE) is not None

    # Played on: changed, and a record made since.
    (folder / "career.json").write_text('{"funds": 2}', encoding="utf-8")
    (folder / "science.json").write_text("{}", encoding="utf-8")

    assert saves.restore(folder, saves.QUICKSAVE)

    assert json.loads((folder / "career.json").read_text(encoding="utf-8")) == {"funds": 1}
    assert (folder / "fleet.json").is_file() and not (folder / "science.json").exists()

    # Kept again: replaced.
    (folder / "career.json").write_text('{"funds": 3}', encoding="utf-8")

    saves.keep(folder, saves.QUICKSAVE)
    saves.restore(folder, saves.QUICKSAVE)

    assert json.loads((folder / "career.json").read_text(encoding="utf-8")) == {"funds": 3}

    # The slot's own file untouched.
    assert saves.find("Career", tmp_path) is not None


def test_old_saves_become_games_once(tmp_path):

    old = tmp_path / "data"

    (old / "programme").mkdir(parents=True)
    (old / "science").mkdir()

    (old / "programme" / "career.json").write_text('{"funds": 7}', encoding="utf-8")
    (old / "science" / "progress.json").write_text("{}", encoding="utf-8")
    (old / "programme" / "sandbox_fleet.json").write_text('{"vessels": []}', encoding="utf-8")

    directory = tmp_path / "saves"

    made = saves.migrate(directory, old)

    assert {slot.name: slot.sandbox for slot in made} == {"Career": False, "Sandbox": True}

    career = saves.find("Career", directory)

    assert json.loads((career.folder / "career.json").read_text(encoding="utf-8")) == {"funds": 7}
    assert (career.folder / "science.json").is_file() and not (career.folder / "fleet.json").exists()

    sandbox = saves.find("Sandbox", directory)

    assert (sandbox.folder / "fleet.json").is_file()

    # The old files stay; a second time, nothing.
    assert (old / "programme" / "career.json").is_file()
    assert saves.migrate(directory, old) == []


def test_nothing_old_nothing_made(tmp_path):

    assert saves.migrate(tmp_path / "saves", tmp_path / "data") == []
    assert saves.slots(tmp_path / "saves") == []


def test_a_game_in_a_line():

    slot = saves.Slot(folder=None, name="Career", summary={"clock": 846_000_000.0, "funds": 312.5, "vessels": 1, "science": 42.0})

    line = describe(slot)

    assert "2026" in line and "$312.5M" in line and "1 vessel in flight" in line and "42 science" in line

    slot.sandbox = True

    assert "$" not in describe(slot)

    assert ago(1000.0, 1000.0 + 30.0) == "just now"
    assert ago(1000.0, 1000.0 + 120.0) == "2 minutes ago"
    assert ago(0.0, 86_400.0 * 1.5) == "1 day ago"
