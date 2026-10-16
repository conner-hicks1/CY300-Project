import json

from dataclasses import dataclass
from pathlib import Path


# =========================================================
# Parts
# =========================================================
#
# The catalog of parts vessels are built from
# (assets/parts/parts.json) and vessel designs: parts
# stacked from the top down with the stage that fires each
# (assets/vessels/*.json). Units are SI: kg, m, N, s.

PARTS_PATH = Path("assets/parts/parts.json")
VESSELS_DIRECTORY = Path("assets/vessels")

KINDS = ("command", "tank", "engine", "decoupler")


class PartError(ValueError):
    """A part or vessel file that cannot be used."""


@dataclass(frozen=True, slots=True)
class EngineSpec:

    thrust_vacuum: float            # N
    isp_vacuum: float               # s
    isp_sea_level: float            # s, at 1 atm
    gimbal: float                   # deg the nozzle can swing


@dataclass(frozen=True, slots=True)
class PartDefinition:

    id: str
    name: str
    kind: str
    mass: float                     # kg, without propellant
    radius_top: float               # m
    radius_bottom: float
    height: float
    color: tuple[float, float, float]
    metallic: float = 0.0
    roughness: float = 0.5
    propellant: float = 0.0         # kg a full tank holds
    engine: EngineSpec | None = None
    reaction_wheel_torque: float = 0.0
    crash_speed: float = 8.0        # m/s it survives hitting the ground at

    @property
    def radius(
        self
    ) -> float:

        return max(self.radius_top, self.radius_bottom)


@dataclass(frozen=True, slots=True)
class StackEntry:

    part: PartDefinition
    stage: int | None               # fires (engine, separator) at this stage


@dataclass(frozen=True, slots=True)
class VesselDesign:

    id: str
    name: str
    description: str
    stack: tuple[StackEntry, ...]   # from the top down


# =========================================================
# Loading
# =========================================================

def _number(
    data: dict,
    key: str,
    where: str,
    default: float | None = None,
    minimum: float = 0.0
) -> float:

    value = data.get(key, default)

    if value is None:
        raise PartError(f"{where}: '{key}' is required")

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PartError(f"{where}: '{key}' must be a number")

    if value < minimum:
        raise PartError(f"{where}: '{key}' must be at least {minimum}")

    return float(value)


def parse_part(
    data: dict
) -> PartDefinition:

    part_id = data.get("id")

    if not isinstance(part_id, str) or not part_id:
        raise PartError("a part needs an 'id'")

    where = f"part '{part_id}'"

    kind = data.get("kind")

    if kind not in KINDS:
        raise PartError(f"{where}: kind must be one of {', '.join(KINDS)}")

    shape = data.get("shape") or {}

    engine = None

    if kind == "engine":

        raw = data.get("engine")

        if not isinstance(raw, dict):
            raise PartError(f"{where}: an engine needs an 'engine' section")

        engine = EngineSpec(
            thrust_vacuum=_number(raw, "thrust_vacuum", where, minimum=1.0),
            isp_vacuum=_number(raw, "isp_vacuum", where, minimum=1.0),
            isp_sea_level=_number(raw, "isp_sea_level", where),
            gimbal=_number(raw, "gimbal", where, default=0.0)
        )

    color = data.get("color", [0.7, 0.7, 0.7])

    if not (isinstance(color, list) and len(color) == 3 and all(isinstance(c, (int, float)) for c in color)):
        raise PartError(f"{where}: color must be [r, g, b]")

    return PartDefinition(
        id=part_id,
        name=str(data.get("name", part_id)),
        kind=kind,
        mass=_number(data, "mass", where, minimum=1.0),
        radius_top=_number(shape, "radius_top", where),
        radius_bottom=_number(shape, "radius_bottom", where),
        height=_number(shape, "height", where, minimum=0.01),
        color=tuple(float(c) for c in color),
        metallic=_number(data, "metallic", where, default=0.0),
        roughness=_number(data, "roughness", where, default=0.5),
        propellant=_number(data, "propellant", where, default=0.0),
        engine=engine,
        reaction_wheel_torque=_number(data, "reaction_wheel_torque", where, default=0.0),
        crash_speed=_number(data, "crash_speed", where, default=8.0, minimum=0.1)
    )


def load_parts(
    path: Path = PARTS_PATH
) -> dict[str, PartDefinition]:

    data = json.loads(Path(path).read_text(encoding="utf-8"))

    parts = {}

    for raw in data.get("parts", []):

        part = parse_part(raw)

        if part.id in parts:
            raise PartError(f"part '{part.id}' is defined twice")

        parts[part.id] = part

    return parts


def parse_design(
    data: dict,
    design_id: str,
    parts: dict[str, PartDefinition]
) -> VesselDesign:

    where = f"vessel '{design_id}'"

    raw_stack = data.get("parts")

    if not isinstance(raw_stack, list) or not raw_stack:
        raise PartError(f"{where}: needs a list of parts")

    stack = []

    for index, entry in enumerate(raw_stack):

        part = parts.get(entry.get("part"))

        if part is None:
            raise PartError(f"{where}: part {index} '{entry.get('part')}' is not in the catalog")

        stage = entry.get("stage")

        if stage is not None and (isinstance(stage, bool) or not isinstance(stage, int) or stage < 0):
            raise PartError(f"{where}: part {index} stage must be a whole number >= 0")

        if stage is not None and part.kind not in ("engine", "decoupler"):
            raise PartError(f"{where}: part {index} ({part.id}) has nothing to fire at a stage")

        stack.append(StackEntry(part, stage))

    if not any(entry.part.kind == "command" for entry in stack):
        raise PartError(f"{where}: needs a command part")

    return VesselDesign(
        id=design_id,
        name=str(data.get("name", design_id)),
        description=str(data.get("description", "")),
        stack=tuple(stack)
    )


def load_design(
    design_id: str,
    parts: dict[str, PartDefinition],
    directory: Path = VESSELS_DIRECTORY
) -> VesselDesign:

    path = Path(directory) / f"{design_id}.json"

    return parse_design(json.loads(path.read_text(encoding="utf-8")), design_id, parts)
