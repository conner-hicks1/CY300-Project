import json
import re
import shutil
import time

from dataclasses import dataclass, field
from pathlib import Path

from core.logger import Logger

from programme import upgrades


# =========================================================
# Saved Games
# =========================================================
#
# Each game in a folder of its own under data/saves/:
#
#   slot.json      its name, career or sandbox, when it was
#                  made and last played, a summary for the
#                  title screen (date, funds, vessels...)
#   career.json    the programme (programme/career.py; none
#                  in a sandbox)
#   science.json   the science record (science/record.py)
#   fleet.json     the vessels in flight (programme/fleet.py)
#
# The game writes those as it goes (and every few minutes);
# besides them:
#
#   quicksave/     a copy at a moment chosen (F5), gone back
#                  to with F9
#   autosave/      a copy every few minutes of play, in case
#
# A folder of saves from before slots (data/programme/,
# data/science/) becomes two slots, "Career" and "Sandbox",
# the first time (copied: the old files stay).

SAVES_DIRECTORY = Path("data/saves")

FILES = ("career.json", "science.json", "fleet.json")

QUICKSAVE = "quicksave"
AUTOSAVE = "autosave"

SLOT_FILE = "slot.json"


@dataclass(slots=True)
class Slot:

    folder: Path
    name: str
    sandbox: bool = False
    created: float = 0.0            # s, wall clock
    played: float = 0.0             # s, wall clock
    summary: dict = field(default_factory=dict)

    def kept(
        self,
        kind: str
    ) -> float | None:
        """When the quicksave / autosave was made (wall clock s), or None."""

        return kept(self.folder, kind)

    def to_json(self) -> dict:

        return {
            "format": upgrades.FORMATS["slot"],
            "name": self.name,
            "sandbox": self.sandbox,
            "created": self.created,
            "played": self.played,
            "summary": self.summary,
        }

    @classmethod
    def from_json(
        cls,
        folder: Path,
        data: dict
    ) -> "Slot":

        return cls(
            folder=Path(folder),
            name=str(data.get("name") or Path(folder).name),
            sandbox=bool(data.get("sandbox", False)),
            created=float(data.get("created", 0.0)),
            played=float(data.get("played", 0.0)),
            summary=dict(data.get("summary") or {})
        )


def slots(
    directory: Path = SAVES_DIRECTORY
) -> list[Slot]:
    """Every saved game, the last played first."""

    found = []

    directory = Path(directory)

    if not directory.is_dir():
        return found

    for folder in directory.iterdir():

        try:
            data = json.loads((folder / SLOT_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue

        if isinstance(data, dict):
            found.append(Slot.from_json(folder, upgrades.upgrade("slot", data)))

    found.sort(key=lambda slot: slot.played, reverse=True)

    return found


def find(
    name: str,
    directory: Path = SAVES_DIRECTORY
) -> Slot | None:
    """A game by its name (any case)."""

    return next((slot for slot in slots(directory) if slot.name.casefold() == name.strip().casefold()), None)


def _folder_name(
    name: str
) -> str:

    return re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")[:40] or "game"


def create(
    name: str,
    sandbox: bool,
    directory: Path = SAVES_DIRECTORY
) -> Slot:
    """A new game (nothing in it yet: it starts as a new career or sandbox does)."""

    name = name.strip() or ("Sandbox" if sandbox else "Career")

    directory = Path(directory)

    base = _folder_name(name)

    folder = directory / base

    number = 2

    while folder.exists():

        folder = directory / f"{base}-{number}"

        number += 1

    now = time.time()

    slot = Slot(folder, name, sandbox, now, now)

    write(slot)

    return slot


def write(
    slot: Slot
):

    slot.folder.mkdir(parents=True, exist_ok=True)

    temporary = slot.folder / (SLOT_FILE + ".tmp")

    temporary.write_text(json.dumps(slot.to_json(), indent=1), encoding="utf-8")

    temporary.replace(slot.folder / SLOT_FILE)


def delete(
    slot: Slot
):
    """The game gone, with its quicksave and autosave."""

    shutil.rmtree(slot.folder, ignore_errors=True)


def unused_name(
    sandbox: bool,
    directory: Path = SAVES_DIRECTORY
) -> str:
    """'Career', 'Career 2'... (a name no game has)."""

    base = "Sandbox" if sandbox else "Career"

    taken = {slot.name.casefold() for slot in slots(directory)}

    name, number = base, 2

    while name.casefold() in taken:

        name = f"{base} {number}"

        number += 1

    return name


# ---------------------------------------------------------
# Quicksave and autosave
# ---------------------------------------------------------

def keep(
    folder: Path,
    kind: str
):
    """A copy of the game's files as they are now (into quicksave/ or autosave/)."""

    folder = Path(folder)

    target = folder / kind
    temporary = folder / (kind + ".tmp")

    shutil.rmtree(temporary, ignore_errors=True)

    temporary.mkdir(parents=True)

    for name in FILES:

        if (folder / name).is_file():
            shutil.copy2(folder / name, temporary / name)

    (temporary / "kept").write_text(str(time.time()), encoding="utf-8")

    shutil.rmtree(target, ignore_errors=True)

    temporary.replace(target)


def kept(
    folder: Path,
    kind: str
) -> float | None:

    try:
        return float((Path(folder) / kind / "kept").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def restore(
    folder: Path,
    kind: str
) -> bool:
    """The game's files put back as they were kept (a file not there then: none now). False if nothing was kept."""

    folder = Path(folder)

    source = folder / kind

    if kept(folder, kind) is None:
        return False

    for name in FILES:

        if (source / name).is_file():
            shutil.copy2(source / name, folder / name)
        else:
            (folder / name).unlink(missing_ok=True)

    return True


# ---------------------------------------------------------
# Saves from before slots
# ---------------------------------------------------------

def migrate(
    directory: Path = SAVES_DIRECTORY,
    old: Path = Path("data")
) -> list[Slot]:
    """The old saves (data/programme/, data/science/) as slots, once: when there are no slots yet."""

    directory = Path(directory)
    old = Path(old)

    if slots(directory):
        return []

    made = []

    for name, sandbox, files in (
        ("Career", False, {
            "career.json": old / "programme" / "career.json",
            "science.json": old / "science" / "progress.json",
            "fleet.json": old / "programme" / "fleet.json",
        }),
        ("Sandbox", True, {
            "science.json": old / "science" / "sandbox_progress.json",
            "fleet.json": old / "programme" / "sandbox_fleet.json",
        }),
    ):

        present = {target: source for target, source in files.items() if source.is_file()}

        if not present:
            continue

        slot = create(name, sandbox, directory)

        for target, source in present.items():
            shutil.copy2(source, slot.folder / target)

        # Played when they last were.
        slot.played = max(source.stat().st_mtime for source in present.values())

        write(slot)

        made.append(slot)

        Logger.info("[Saves] The old %s saves are now the game '%s' (%s).", name.lower(), slot.name, slot.folder)

    return made
