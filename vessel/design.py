import json
import re

from dataclasses import dataclass, field
from pathlib import Path

from vessel.parts import (
    SYMMETRIES,
    VESSELS_DIRECTORY,
    PartDefinition,
    PartError,
    RadialGroup,
    StackEntry,
    VesselDesign,
    parse_design
)


# =========================================================
# Vessel Designs Being Edited
# =========================================================
#
# The part editor's model: a core stack of parts from the
# top down, each with the stage that fires it (engines,
# separators and parachutes; others none), and on any core
# part radial groups: a little stack of its own, repeated
# `symmetry` times round the core (side boosters), with a
# radial separator that drops them.
#
# Parts are addressed by paths:
#
#   (i,)               core part i
#   (i, g)             radial group g on core part i
#   (i, g, j)          part j of that group
#   (i, g, "separator") the group's separator
#
# Editing never fails: a draft may be incomplete (no
# command part yet); `problems` says what stops it flying,
# and `to_design` makes it a VesselDesign once it can.
#
# Designs are saved as JSON like the built-in ones
# (assets/vessels/): the player's go to data/vessels/ (not
# in the repository), and either kind can be opened.

USER_VESSELS_DIRECTORY = Path("data/vessels")

STAGED_KINDS = ("engine", "decoupler", "parachute")

SEPARATOR = "separator"


@dataclass(slots=True)
class DraftGroup:

    symmetry: int = 2
    offset: float = 0.0
    decoupler: "DraftEntry | None" = None
    entries: list["DraftEntry"] = field(default_factory=list)
    angle: float = 0.0              # deg round the core of its first copy


@dataclass(slots=True)
class DraftEntry:

    part: PartDefinition
    stage: int | None = None
    radial: list[DraftGroup] = field(default_factory=list)


def _entry_from(
    entry: StackEntry
) -> DraftEntry:

    return DraftEntry(
        entry.part,
        entry.stage,
        [
            DraftGroup(
                group.symmetry,
                group.offset,
                DraftEntry(group.decoupler.part, group.decoupler.stage) if group.decoupler is not None else None,
                [DraftEntry(member.part, member.stage) for member in group.stack],
                group.angle
            )
            for group in entry.radial
        ]
    )


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

        return cls(design.name, design.description, [_entry_from(entry) for entry in design.stack])

    # -----------------------------------------------------
    # Finding
    # -----------------------------------------------------

    def get(
        self,
        path: tuple
    ):
        """The DraftEntry (or DraftGroup, for a group path) at `path`; None if there is none."""

        try:

            entry = self.entries[path[0]]

            if len(path) == 1:
                return entry

            group = entry.radial[path[1]]

            if len(path) == 2:
                return group

            if path[2] == SEPARATOR:
                return group.decoupler

            return group.entries[path[2]]

        except (IndexError, TypeError):

            return None

    def rows(
        self
    ) -> list[tuple[tuple, int]]:
        """Every path in display order, with its depth (0 core, 1 group, 2 group part)."""

        found = []

        for i, entry in enumerate(self.entries):

            found.append(((i,), 0))

            for g, group in enumerate(entry.radial):

                found.append(((i, g), 1))

                if group.decoupler is not None:
                    found.append(((i, g, SEPARATOR), 2))

                for j in range(len(group.entries)):
                    found.append(((i, g, j), 2))

        return found

    def all_entries(
        self
    ):
        """Every DraftEntry: (path, entry, its group or None)."""

        for i, entry in enumerate(self.entries):

            yield (i,), entry, None

            for g, group in enumerate(entry.radial):

                if group.decoupler is not None:
                    yield (i, g, SEPARATOR), group.decoupler, group

                for j, member in enumerate(group.entries):
                    yield (i, g, j), member, group

    # -----------------------------------------------------
    # Editing
    # -----------------------------------------------------

    def insert(
        self,
        index: int,
        part: PartDefinition
    ) -> int:
        """Put a part in the core at `index` (0 = the top; past the end = the bottom). Returns where it went."""

        index = min(max(index, 0), len(self.entries))

        self.entries.insert(index, DraftEntry(part))

        return index

    def add_after(
        self,
        path: tuple | None,
        part: PartDefinition
    ) -> tuple | None:
        """
        Add a part next to the selection: below a core part,
        at the bottom of a selected group, below a group's
        part; nothing selected: the bottom of the core. A
        separator added to a group becomes the group's.
        Returns the new part's path (None: not added).
        """

        if path is None:
            return (self.insert(len(self.entries), part),)

        if len(path) == 1:
            return (self.insert(path[0] + 1, part),)

        group = self.get(path[:2])

        if group is None:
            return None

        if part.kind == "decoupler":

            if group.decoupler is not None:
                return None

            group.decoupler = DraftEntry(part)

            return (*path[:2], SEPARATOR)

        if len(path) == 3 and path[2] != SEPARATOR:
            position = path[2] + 1
        else:
            position = len(group.entries)

        group.entries.insert(position, DraftEntry(part))

        return (*path[:2], position)

    def attach_group(
        self,
        index: int,
        symmetry: int,
        separator: PartDefinition | None
    ) -> tuple:
        """A new radial group on core part `index` (with a separator if given). Returns its path."""

        entry = self.entries[index]

        entry.radial.append(DraftGroup(
            symmetry=symmetry if symmetry in SYMMETRIES else 2,
            decoupler=DraftEntry(separator) if separator is not None else None
        ))

        return (index, len(entry.radial) - 1)

    def set_symmetry(
        self,
        path: tuple,
        symmetry: int
    ):

        group = self.get(path[:2])

        if group is not None and symmetry in SYMMETRIES:
            group.symmetry = symmetry

    def set_offset(
        self,
        path: tuple,
        offset: float
    ):
        """A group's height above the bottom of the part it hangs on (m, 0 or more)."""

        group = self.get(path[:2])

        if group is not None:
            group.offset = max(float(offset), 0.0)

    def remove(
        self,
        target
    ):
        """Remove a part, or a whole group (an int: a core index)."""

        path = (target,) if isinstance(target, int) else tuple(target)

        if len(path) == 1:

            if 0 <= path[0] < len(self.entries):
                del self.entries[path[0]]

            return

        entry = self.get(path[:1])

        if entry is None or not (0 <= path[1] < len(entry.radial)):
            return

        if len(path) == 2:

            del entry.radial[path[1]]

            return

        group = entry.radial[path[1]]

        if path[2] == SEPARATOR:
            group.decoupler = None
        elif 0 <= path[2] < len(group.entries):
            del group.entries[path[2]]

    def move(
        self,
        target,
        offset: int
    ):
        """
        Swap a part up (-1) or down (+1) within its stack (an
        int: a core index, returning the new index; a path
        returns the new path).
        """

        if isinstance(target, int):
            return self._swap(self.entries, target, offset)

        path = tuple(target)

        if len(path) == 1:
            return (self._swap(self.entries, path[0], offset),)

        if len(path) == 3 and path[2] != SEPARATOR:

            group = self.get(path[:2])

            if group is not None:
                return (*path[:2], self._swap(group.entries, path[2], offset))

        return path

    @staticmethod
    def _swap(
        items: list,
        index: int,
        offset: int
    ) -> int:

        target = index + offset

        if not (0 <= index < len(items) and 0 <= target < len(items)):
            return index

        items[index], items[target] = items[target], items[index]

        return target

    def set_stage(
        self,
        target,
        stage: int | None
    ):
        """A staged part's stage (>= 0); others stay unstaged."""

        path = (target,) if isinstance(target, int) else tuple(target)

        entry = self.get(path)

        if not isinstance(entry, DraftEntry) or entry.part.kind not in STAGED_KINDS:
            return

        entry.stage = None if stage is None else max(int(stage), 0)

    def auto_stage(self):
        """
        Stages as a rocket fires them, from the bottom up: a
        section's engines (with the radial groups hung on
        it) light together, with the separator below the
        section; the radial groups' separators drop them at
        the next stage; parachutes last.
        """

        for _, entry, _ in self.all_entries():
            entry.stage = None

        # Core sections from the bottom up, each with the
        # separator just below it (None for the bottom one).
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

            groups = [group for member in members for group in member.radial]

            engines = [entry for entry in members if entry.part.kind == "engine"]
            engines += [entry for group in groups for entry in group.entries if entry.part.kind == "engine"]

            if separator is None and not engines:

                # (Radial separators with nothing lit still
                # need a stage.)
                dropped = [group.decoupler for group in groups if group.decoupler is not None]

                for decoupler in dropped:
                    decoupler.stage = stage

                if dropped:
                    stage += 1

                continue

            if separator is not None:
                separator.stage = stage

            for engine in engines:
                engine.stage = stage

            stage += 1

            dropped = [group.decoupler for group in groups if group.decoupler is not None]

            for decoupler in dropped:
                decoupler.stage = stage

            if dropped:
                stage += 1

        for _, entry, _ in self.all_entries():

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
            found.append("No command part in the core stack: add a capsule or a probe core.")

        for path, entry, _ in self.all_entries():

            if entry.part.kind in STAGED_KINDS and entry.stage is None:
                found.append(f"{entry.part.name} ({_describe(path)}) has no stage.")

        for i, entry in enumerate(self.entries):

            for g, group in enumerate(entry.radial):

                if not group.entries:
                    found.append(f"Radial group {g + 1} on {entry.part.name} has no parts: add some, or remove it.")

        if not any(entry.part.kind == "engine" for _, entry, _ in self.all_entries()):
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

        for entry in self.entries:

            for group in entry.radial:

                engines = [member for member in group.entries if member.part.kind == "engine"]
                tanks = [member for member in group.entries if member.part.kind == "tank"]

                if engines and not tanks:
                    found.append(f"Radial {engines[0].part.name}: no tank in its group to burn from.")

                if group.decoupler is None and (engines or tanks):
                    found.append(f"A radial group on {entry.part.name} has no separator: it stays on all the way.")

        if self.entries and self.entries[0].part.kind == "decoupler":
            found.append("A separator at the very top drops nothing useful.")

        # Meant to come home (a parachute), with nothing to take
        # the heat of coming in from orbit.
        if any(entry.part.kind == "parachute" for entry in self.entries) and not any(
            entry.part.max_temperature >= 2_500.0 for entry in self.entries
        ):
            found.append("No heat shield: coming back from orbit, the capsule burns up (put one under it).")

        return found

    # -----------------------------------------------------
    # As a design
    # -----------------------------------------------------

    def stack(
        self
    ) -> tuple[StackEntry, ...]:

        return tuple(
            StackEntry(
                entry.part,
                entry.stage,
                tuple(
                    RadialGroup(
                        group.symmetry,
                        group.offset,
                        StackEntry(group.decoupler.part, group.decoupler.stage) if group.decoupler is not None else None,
                        tuple(StackEntry(member.part, member.stage) for member in group.entries),
                        group.angle
                    )
                    for group in entry.radial
                    if group.entries
                )
            )
            for entry in self.entries
        )

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

        def entry_json(entry: DraftEntry) -> dict:

            data = {"part": entry.part.id}

            if entry.stage is not None:
                data["stage"] = entry.stage

            return data

        def core_json(entry: DraftEntry) -> dict:

            data = entry_json(entry)

            groups = [group for group in entry.radial if group.entries]

            if groups:
                data["radial"] = [
                    {
                        "symmetry": group.symmetry,
                        "offset": group.offset,
                        **({"angle": group.angle} if group.angle else {}),
                        **({"decoupler": entry_json(group.decoupler)} if group.decoupler is not None else {}),
                        "parts": [entry_json(member) for member in group.entries],
                    }
                    for group in groups
                ]

            return data

        return {
            "format": 1,
            "name": self.name,
            "description": self.description,
            "parts": [core_json(entry) for entry in self.entries],
        }


def _describe(
    path: tuple
) -> str:

    if len(path) == 1:
        return f"part {path[0] + 1}"

    if path[2] == SEPARATOR:
        return f"separator of radial group {path[1] + 1} on part {path[0] + 1}"

    return f"radial group {path[1] + 1} on part {path[0] + 1}"


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

    def entry_from(raw: dict, where: str) -> DraftEntry:

        part = parts.get(raw.get("part"))

        if part is None:
            raise PartError(f"{saved.path.name}: {where} '{raw.get('part')}' is not in the catalog")

        stage = raw.get("stage")

        return DraftEntry(part, stage if isinstance(stage, int) and not isinstance(stage, bool) else None)

    entries = []

    for index, raw in enumerate(data.get("parts", [])):

        entry = entry_from(raw, f"part {index}")

        for number, group in enumerate(raw.get("radial", []) or []):

            where = f"part {index} radial group {number}"

            symmetry = group.get("symmetry", 2)

            entry.radial.append(DraftGroup(
                symmetry=symmetry if symmetry in SYMMETRIES else 2,
                offset=float(group.get("offset", 0.0) or 0.0),
                angle=float(group.get("angle", 0.0) or 0.0),
                decoupler=entry_from(group["decoupler"], f"{where} separator") if group.get("decoupler") else None,
                entries=[entry_from(member, f"{where} part {j}") for j, member in enumerate(group.get("parts", []))]
            ))

        entries.append(entry)

    return DesignDraft(str(data.get("name", saved.name)), str(data.get("description", "")), entries)
