import json

from dataclasses import dataclass
from pathlib import Path

from vessel.propellants import DEFAULT as DEFAULT_PROPELLANT, PROPELLANTS


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

KINDS = ("command", "tank", "engine", "decoupler", "parachute", "science", "antenna", "utility", "structure", "power", "legs", "wheels", "rotor")

# Where an experiment works (science/record.py).
SITUATIONS = ("landed", "flying", "space_low", "space_high")


class PartError(ValueError):
    """A part or vessel file that cannot be used."""


@dataclass(frozen=True, slots=True)
class EngineSpec:

    thrust_vacuum: float            # N
    isp_vacuum: float               # s
    isp_sea_level: float            # s, at 1 atm
    gimbal: float                   # deg the nozzle can swing
    propellant: str = DEFAULT_PROPELLANT   # what it burns (vessel/propellants.py)


@dataclass(frozen=True, slots=True)
class ParachuteSpec:

    area: float                     # m^2, the open canopy
    drag_coefficient: float
    max_speed: float                # m/s through the air it can open at
    min_pressure: float             # bar it needs to open in


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """Science a part can do (science/record.py, systems/science_system.py)."""

    id: str
    name: str
    situations: tuple[str, ...]
    value: float                    # science of one subject, before the body and region
    needs_air: bool = False         # samples the air (only where there is some)
    maps: bool = False              # maps the ground below from orbit (science/survey.py)
    energy: float = 10.0            # Wh a run takes (vessel/power.py)


@dataclass(frozen=True, slots=True)
class IsruSpec:
    """A propellant plant (programme/resources.py): kg a day from ice, from the air."""

    rate_ice: float
    rate_air: float
    power: float = 0.0              # W it draws making it (vessel/power.py)


@dataclass(frozen=True, slots=True)
class LegSpec:
    """
    A landing leg (vessel/landing.py), on a radial group: down,
    its pad stands `span` m further out than its mount, level
    with the vessel's bottom.
    """

    span: float


@dataclass(frozen=True, slots=True)
class WheelSpec:
    """A wheel (vessel/rover.py): its top speed (m/s) and the W it draws at it; on a radial group, its axle across the vessel."""

    speed: float
    power: float


@dataclass(frozen=True, slots=True)
class RotorSpec:
    """A rotor (vessel/rover.py): N of lift at full throttle per kg/m^3 of air, the W it draws then."""

    lift: float
    power: float


@dataclass(frozen=True, slots=True)
class SolarSpec:
    """
    Solar wings (vessel/power.py): W at 1 AU with the sun
    square on; each copy's wings out along one axis (a stack
    part's: both sides; a radial copy's: straight out), each
    `length` x `width` m, turning about that axis to follow
    the sun.
    """

    power: float
    length: float
    width: float
    wings: int = 2


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
    propellant_type: str = DEFAULT_PROPELLANT   # of which (vessel/propellants.py)
    insulation: float = 1.0         # its boil-off, of a plain tank's
    cooling: float = 0.0            # kg of cryogenic propellant a cryocooler keeps cold (powered)
    engine: EngineSpec | None = None
    parachute: ParachuteSpec | None = None
    reaction_wheel_torque: float = 0.0
    crash_speed: float = 8.0        # m/s it survives hitting the ground at
    max_temperature: float = 1400.0  # K its skin stands (vessel/heating.py)
    experiment: ExperimentSpec | None = None
    crew: int = 0                   # people aboard (crewed vessels fly without a link)
    antenna_range: float = 0.0      # m it reaches home from (systems/comms_system.py)
    cost: float = 0.0               # $ millions (programme/career.py)
    isru: IsruSpec | None = None
    docking: bool = False           # a docking port (vessel/docking.py)
    side: bool = False              # a side port: on a radial group, facing straight out
    rcs_thrust: float = 0.0         # N of translation thrust each way (docking)

    # Power (vessel/power.py): Wh stored, solar wings, W made
    # all the time (an RTG, a reactor), W drawn all the time
    # (avionics, life support, a camera, a transmitter), W
    # its reaction wheels draw turning hard.
    battery: float = 0.0
    solar: SolarSpec | None = None
    generator: float = 0.0
    power_draw: float = 0.0
    wheel_power: float = 0.0

    # A landing leg (vessel/landing.py).
    legs: LegSpec | None = None

    # Life (vessel/life.py): kg of supplies it holds, the
    # share of them its life support recycles (powered), its
    # shielding (the radiation that gets through: 1, all).
    supplies: float = 0.0
    recycling: float = 0.0
    shielding: float = 1.0

    # Wheels and rotors (vessel/rover.py).
    wheels: WheelSpec | None = None
    rotor: RotorSpec | None = None

    @property
    def radius(
        self
    ) -> float:

        return max(self.radius_top, self.radius_bottom)


@dataclass(frozen=True, slots=True)
class RadialGroup:
    """
    Parts on the side of a stack part, `symmetry` copies
    evenly round it: a little stack of their own (top down),
    its bottom `offset` m above the bottom of the part it
    hangs on; a radial separator, if any, drops every copy
    when its stage fires.
    """

    symmetry: int
    offset: float
    decoupler: "StackEntry | None"
    stack: tuple["StackEntry", ...]

    # Where round the core the first copy stands (deg; the
    # others evenly after it): groups apart on one part.
    angle: float = 0.0


@dataclass(frozen=True, slots=True)
class StackEntry:

    part: PartDefinition
    stage: int | None               # fires (engine, separator) at this stage
    radial: tuple[RadialGroup, ...] = ()


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


def _propellant(
    data: dict,
    where: str,
    key: str = "propellant"
) -> str:

    kind = data.get(key, DEFAULT_PROPELLANT)

    if kind not in PROPELLANTS:
        raise PartError(f"{where}: '{key}' must be one of {', '.join(PROPELLANTS)}")

    return kind


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
            gimbal=_number(raw, "gimbal", where, default=0.0),
            propellant=_propellant(raw, where)
        )

    parachute = None

    if kind == "parachute":

        raw = data.get("parachute")

        if not isinstance(raw, dict):
            raise PartError(f"{where}: a parachute needs a 'parachute' section")

        parachute = ParachuteSpec(
            area=_number(raw, "area", where, minimum=0.1),
            drag_coefficient=_number(raw, "drag_coefficient", where, minimum=0.01),
            max_speed=_number(raw, "max_speed", where, minimum=1.0),
            min_pressure=_number(raw, "min_pressure", where, default=0.0)
        )

    experiment = None

    raw = data.get("experiment")

    if raw is not None:

        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
            raise PartError(f"{where}: an experiment needs an 'id'")

        situations = raw.get("situations", list(SITUATIONS))

        if not isinstance(situations, list) or not situations or any(item not in SITUATIONS for item in situations):
            raise PartError(f"{where}: experiment situations must be some of {', '.join(SITUATIONS)}")

        experiment = ExperimentSpec(
            id=raw["id"],
            name=str(raw.get("name", raw["id"])),
            situations=tuple(situations),
            value=_number(raw, "value", where, default=10.0),
            needs_air=bool(raw.get("needs_air", False)),
            maps=bool(raw.get("maps", False)),
            energy=_number(raw, "energy", where, default=10.0)
        )

    isru = None

    if data.get("isru") is not None:

        raw = data["isru"]

        if not isinstance(raw, dict):
            raise PartError(f"{where}: 'isru' must be an object")

        isru = IsruSpec(
            rate_ice=_number(raw, "rate_ice", where, default=0.0),
            rate_air=_number(raw, "rate_air", where, default=0.0),
            power=_number(raw, "power", where, default=0.0)
        )

    solar = None

    if data.get("solar") is not None:

        raw = data["solar"]

        if not isinstance(raw, dict):
            raise PartError(f"{where}: 'solar' must be an object")

        solar = SolarSpec(
            power=_number(raw, "power", where, minimum=1.0),
            length=_number(raw, "length", where, minimum=0.1),
            width=_number(raw, "width", where, minimum=0.1),
            wings=int(_number(raw, "wings", where, default=2.0, minimum=1.0))
        )

    wheels = None

    if data.get("wheels") is not None:

        raw = data["wheels"]

        if not isinstance(raw, dict):
            raise PartError(f"{where}: 'wheels' must be an object")

        wheels = WheelSpec(speed=_number(raw, "speed", where, minimum=0.1), power=_number(raw, "power", where, default=0.0))

    rotor = None

    if data.get("rotor") is not None:

        raw = data["rotor"]

        if not isinstance(raw, dict):
            raise PartError(f"{where}: 'rotor' must be an object")

        rotor = RotorSpec(lift=_number(raw, "lift", where, minimum=1.0), power=_number(raw, "power", where, default=0.0))

    legs = None

    if data.get("legs") is not None:

        raw = data["legs"]

        if not isinstance(raw, dict):
            raise PartError(f"{where}: 'legs' must be an object")

        legs = LegSpec(span=_number(raw, "span", where, minimum=0.1))

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
        propellant_type=_propellant(data, where, "propellant_type"),
        insulation=_number(data, "insulation", where, default=1.0),
        cooling=_number(data, "cooling", where, default=0.0),
        engine=engine,
        parachute=parachute,
        reaction_wheel_torque=_number(data, "reaction_wheel_torque", where, default=0.0),
        crash_speed=_number(data, "crash_speed", where, default=8.0, minimum=0.1),
        max_temperature=_number(data, "max_temperature", where, default=1400.0, minimum=300.0),
        experiment=experiment,
        crew=int(_number(data, "crew", where, default=0.0)),
        antenna_range=_number(data, "antenna_range", where, default=0.0),
        cost=_number(data, "cost", where, default=0.0),
        isru=isru,
        docking=bool(data.get("docking", False)),
        side=bool(data.get("side", False)),
        rcs_thrust=_number(data, "rcs_thrust", where, default=0.0),
        battery=_number(data, "battery", where, default=0.0),
        solar=solar,
        generator=_number(data, "generator", where, default=0.0),
        power_draw=_number(data, "power_draw", where, default=0.0),
        wheel_power=_number(data, "wheel_power", where, default=0.0),
        legs=legs,
        supplies=_number(data, "supplies", where, default=0.0),
        recycling=min(_number(data, "recycling", where, default=0.0), 0.95),
        shielding=_number(data, "shielding", where, default=1.0),
        wheels=wheels,
        rotor=rotor
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


SYMMETRIES = (1, 2, 3, 4, 6, 8)


def _parse_entry(
    entry: dict,
    where: str,
    parts: dict[str, PartDefinition],
    radial_allowed: bool
) -> StackEntry:

    if not isinstance(entry, dict):
        raise PartError(f"{where}: must be an object")

    part = parts.get(entry.get("part"))

    if part is None:
        raise PartError(f"{where} '{entry.get('part')}' is not in the catalog")

    stage = entry.get("stage")

    if stage is not None and (isinstance(stage, bool) or not isinstance(stage, int) or stage < 0):
        raise PartError(f"{where}: stage must be a whole number >= 0")

    if stage is not None and part.kind not in ("engine", "decoupler", "parachute"):
        raise PartError(f"{where} ({part.id}) has nothing to fire at a stage")

    groups = []

    for number, raw in enumerate(entry.get("radial", []) or []):

        if not radial_allowed:
            raise PartError(f"{where}: radial parts cannot carry radial parts of their own")

        place = f"{where} radial group {number}"

        if not isinstance(raw, dict):
            raise PartError(f"{place}: must be an object")

        symmetry = raw.get("symmetry", 2)

        if symmetry not in SYMMETRIES:
            raise PartError(f"{place}: symmetry must be one of {', '.join(map(str, SYMMETRIES))}")

        offset = raw.get("offset", 0.0)

        if isinstance(offset, bool) or not isinstance(offset, (int, float)):
            raise PartError(f"{place}: offset must be a number (m)")

        angle = raw.get("angle", 0.0)

        if isinstance(angle, bool) or not isinstance(angle, (int, float)):
            raise PartError(f"{place}: angle must be a number (deg)")

        decoupler = None

        if raw.get("decoupler") is not None:

            decoupler = _parse_entry(raw["decoupler"], f"{place} separator", parts, radial_allowed=False)

            if decoupler.part.kind != "decoupler":
                raise PartError(f"{place}: its separator must be a stage separator")

        members = tuple(
            _parse_entry(member, f"{place} part {i}", parts, radial_allowed=False)
            for i, member in enumerate(raw.get("parts", []))
        )

        if not members:
            raise PartError(f"{place}: needs parts")

        if any(member.part.kind == "decoupler" for member in members):
            raise PartError(f"{place}: separators go in its 'decoupler', not its parts")

        groups.append(RadialGroup(int(symmetry), float(offset), decoupler, members, float(angle)))

    return StackEntry(part, stage, tuple(groups))


def parse_design(
    data: dict,
    design_id: str,
    parts: dict[str, PartDefinition]
) -> VesselDesign:

    where = f"vessel '{design_id}'"

    raw_stack = data.get("parts")

    if not isinstance(raw_stack, list) or not raw_stack:
        raise PartError(f"{where}: needs a list of parts")

    stack = [
        _parse_entry(entry, f"{where}: part {index}", parts, radial_allowed=True)
        for index, entry in enumerate(raw_stack)
    ]

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
