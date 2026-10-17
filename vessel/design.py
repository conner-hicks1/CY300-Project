import json
import re

from dataclasses import dataclass, field
from pathlib import Path

from vessel.parts import (
    VESSELS_DIRECTORY,
    PartDefinition,
    PartError,
    StackEntry,
    VesselDesign,
    parse_design
)


# =========================================================
# Vessel Designs Being Edited
# =========================================================
#
# The part editor's model: a stack of parts from the top
# down, each with the stage that fires it (engines,
# separators and parachutes; others none). Editing never
# fails: a draft may be incomplete (no command part yet);
# `problems` says what stops it flying, and `to_design`
# makes it a VesselDesign once it can.
#
# Designs are saved as JSON like the built-in ones
# (assets/vessels/): the player's go to data/vessels/ (not
# in the repository), and either kind can be opened.

USER_VESSELS_DIRECTORY = Path("data/vessels")

STAGED_KINDS = ("engine", "decoupler", "parachute")


@dataclass(slots=True)
class DraftEntry:

    part: PartDefinition
    stage: int | None = None


@dataclass(slots=True)
class DesignDraft:

    name: str = "Untitled"
    description: str = ""
    entries: list[DraftEntry] = field(default_factory=list)

    @classmethod
    def from_design(
        cls,
        design: VesselDesign
    ) -> "DesignDraft":

        return cls(
            design.name,
            design.description,
            [DraftEntry(entry.part, entry.stage) for entry in design.stack]
        )

    # -----------------------------------------------------
    # Editing
    # -----------------------------------------------------

    def insert(
        self,
        index: int,
        part: PartDefinition
    ) -> int:
        """Put a part at `index` (0 = the top; past the end = the bottom). Returns where it went."""

        index = min(max(index, 0), len(self.entries))

        self.entries.insert(index, DraftEntry(part))

        return index

    def remove(
        self,
        index: int
    ):

        if 0 <= index < len(self.entries):
            del self.entries[index]

    def move(
        self,
        index: int,
        offset: int
    ) -> int:
        """Swap a part up (-1) or down (+1) the stack. Returns where it is now."""

        target = index + offset

        if not (0 <= index < len(self.entries) and 0 <= target < len(self.entries)):
            return index

        self.entries[index], self.entries[target] = self.entries[target], self.entries[index]

        return target

    def set_stage(
        self,
        index: int,
        stage: int | None
    ):
        """A staged part's stage (>= 0); others stay unstaged."""

        entry = self.entries[index]

        if entry.part.kind not in STAGED_KINDS:
            return

        entry.stage = None if stage is None else max(int(stage), 0)

    def auto_stage(self):
        """
        Stages as a rocket fires them: the bottom section's
        engines first (0); then each separator from the
        bottom up, with the engines just above it; parachutes
        last.
        """

        for entry in self.entries:
            entry.stage = None

        # Sections between separators, from the bottom up,
        # each with the separator just below it (None for
        # the bottom one).
        sections: list[tuple[DraftEntry | None, list[DraftEntry]]] = []

        below_separator = None

        section: list[DraftEntry] = []

        for entry in reversed(self.entries):

            if entry.part.kind == "decoupler":

                sections.append((below_separator, section))

                below_separator = entry

                section = []

            else:

                section.append(entry)

        sections.append((below_separator, section))

        stage = 0

        for separator, members in sections:

            engines = [entry for entry in members if entry.part.kind == "engine"]

            if separator is not None:

                separator.stage = stage

                for engine in engines:
                    engine.stage = stage

                stage += 1

            elif engines:

                for engine in engines:
                    engine.stage = stage

                stage += 1

        for entry in self.entries:

            if entry.part.kind == "parachute":
                entry.stage = stage

    # -----------------------------------------------------
    # Checking
    # -----------------------------------------------------

    def problems(
        self
    ) -> list[str]:
        """What keeps the draft from flying (empty: it can)."""

        found = []

        if not self.entries:
            return ["Add some parts: a command part at least."]

        if not any(entry.part.kind == "command" for entry in self.entries):
            found.append("No command part: add a capsule or a probe core.")

        for index, entry in enumerate(self.entries):

            if entry.part.kind in STAGED_KINDS and entry.stage is None:
                found.append(f"{entry.part.name} (part {index + 1}) has no stage.")

        if not any(entry.part.kind == "engine" for entry in self.entries):
            found.append("No engine: it will not leave the ground.")

        return found

    def warnings(
        self
    ) -> list[str]:
        """What would fly, but probably badly."""

        found = []

        sections: list[list[DraftEntry]] = [[]]

        for entry in self.entries:

            if entry.part.kind == "decoupler":
                sections.append([])
            else:
                sections[-1].append(entry)

        for members in sections:

            engines = [entry for entry in members if entry.part.kind == "engine"]
            tanks = [entry for entry in members if entry.part.kind == "tank"]

            if engines and not tanks:
                found.append(f"{engines[0].part.name}: no tank in its section to burn from.")

        if self.entries and self.entries[0].part.kind == "decoupler":
            found.append("A separator at the very top drops nothing useful.")

        return found

    # -----------------------------------------------------
    # As a design
    # -----------------------------------------------------

    def stack(
        self
    ) -> tuple[StackEntry, ...]:

        return tuple(StackEntry(entry.part, entry.stage) for entry in self.entries)

    def preview(
        self
    ) -> VesselDesign:
        """As a design even if it cannot fly yet (to show it)."""

        return VesselDesign(slug(self.name), self.name, self.description, self.stack())

    def to_design(
        self,
        parts: dict[str, PartDefinition]
    ) -> VesselDesign:
        """The design, checked (PartError if it cannot fly)."""

        problems = self.problems()

        if problems:
            raise PartError(problems[0])

        return parse_design(self.to_json(), slug(self.name), parts)

    def to_json(
        self
    ) -> dict:

        return {
            "format": 1,
            "name": self.name,
            "description": self.description,
            "parts": [
                {"part": entry.part.id, **({"stage": entry.stage} if entry.stage is not None else {})}
                for entry in self.entries
            ],
        }


# =========================================================
# Files
# =========================================================

def slug(
    name: str
) -> str:
    """A file name from a vessel name."""

    text = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

    return text or "vessel"


def save_draft(
    draft: DesignDraft,
    directory: Path = USER_VESSELS_DIRECTORY
) -> Path:

    directory = Path(directory)

    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{slug(draft.name)}.json"

    path.write_text(json.dumps(draft.to_json(), indent=2) + "\n", encoding="utf-8")

    return path


@dataclass(frozen=True, slots=True)
class SavedDesign:

    id: str
    name: str
    path: Path
    built_in: bool


def list_designs(
    user_directory: Path = USER_VESSELS_DIRECTORY,
    built_in_directory: Path = VESSELS_DIRECTORY
) -> list[SavedDesign]:
    """Every design that can be opened: the built-in ones, then the player's."""

    found = []

    for directory, built_in in ((Path(built_in_directory), True), (Path(user_directory), False)):

        if not directory.is_dir():
            continue

        for path in sorted(directory.glob("*.json")):

            try:
                name = json.loads(path.read_text(encoding="utf-8")).get("name", path.stem)
            except (OSError, ValueError):
                continue

            found.append(SavedDesign(path.stem, str(name), path, built_in))

    return found


def open_design(
    saved: SavedDesign,
    parts: dict[str, PartDefinition]
) -> DesignDraft:
    """A saved design as a draft (PartError if a part is unknown)."""

    data = json.loads(saved.path.read_text(encoding="utf-8"))

    entries = []

    for index, raw in enumerate(data.get("parts", [])):

        part = parts.get(raw.get("part"))

        if part is None:
            raise PartError(f"{saved.path.name}: part {index} '{raw.get('part')}' is not in the catalog")

        stage = raw.get("stage")

        entries.append(DraftEntry(part, stage if isinstance(stage, int) and not isinstance(stage, bool) else None))

    return DesignDraft(str(data.get("name", saved.name)), str(data.get("description", "")), entries)
