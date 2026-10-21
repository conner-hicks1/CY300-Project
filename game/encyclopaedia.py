import json

from dataclasses import dataclass, field
from pathlib import Path

from imgui_bundle import imgui

from game import style
from game.bindings import keys

from vessel import propellants


# =========================================================
# Encyclopaedia
# =========================================================
#
# In the game (J, the pause menu, the title screen):
#
#   How it works   short articles on the game's mechanics
#                  (assets/encyclopaedia/mechanics.json)
#   Parts          every part, from its own data: what it is,
#                  its mass, cost, engine, tank, power, crew,
#                  limits, the research it waits on
#   Bodies         every body, from its profile: its
#                  description, size, gravity, day, orbit,
#                  air, temperature
#
# Searchable by title and text. Made from the data each time
# the game starts: never out of date with it.

ARTICLES_PATH = Path("assets/encyclopaedia/mechanics.json")

SECTIONS = ("How it works", "Parts", "Bodies")


@dataclass(frozen=True, slots=True)
class Entry:

    section: str
    group: str
    id: str
    title: str
    text: str
    facts: tuple[tuple[str, str], ...] = field(default=())

    def matches(
        self,
        words: str
    ) -> bool:

        text = f"{self.title} {self.group} {self.text}".casefold()

        return all(word in text for word in words.casefold().split())


# ---------------------------------------------------------
# Entries
# ---------------------------------------------------------

def articles(
    path: Path = ARTICLES_PATH
) -> list[Entry]:

    data = json.loads(Path(path).read_text(encoding="utf-8"))

    return [
        Entry(SECTIONS[0], str(item.get("group", "")), str(item["id"]), str(item["title"]), str(item["text"]))
        for item in data["articles"]
    ]


KINDS = {
    "command": "Command",
    "tank": "Tanks",
    "engine": "Engines",
    "decoupler": "Structure",
    "structural": "Structure",
    "adapter": "Structure",
    "nose": "Structure",
    "parachute": "Recovery",
    "heat_shield": "Recovery",
}


def _mass(
    kg: float
) -> str:

    return f"{kg / 1000.0:,.2f} t" if kg >= 1000.0 else f"{kg:,.0f} kg"


def _range(
    metres: float
) -> str:

    if metres >= 1.496e11 * 0.1:
        return f"{metres / 1.496e11:,.1f} AU"

    return f"{metres / 1000.0:,.0f} km"


def part_entry(
    part,
    node=None
) -> Entry:
    """A part, said from its data (node: the tech node it waits on, or None: known from the start)."""

    facts = [
        ("Mass", _mass(part.mass)),
        ("Cost", f"${part.cost:,.1f}M"),
        ("Size", f"{2.0 * part.radius:,.2f} m wide, {part.height:,.2f} m tall"),
    ]

    said = []

    if part.engine is not None:

        engine = part.engine

        burns = propellants.of(engine.propellant).name

        facts += [
            ("Thrust", f"{engine.thrust_vacuum / 1000.0:,.0f} kN in vacuum"),
            ("Efficiency", f"Isp {engine.isp_vacuum:,.0f} s in vacuum, {engine.isp_sea_level:,.0f} s at sea level"),
            ("Burns", burns),
        ]

        said.append(f"An engine burning {burns}, steered {engine.gimbal:,.0f} degrees either way.")

    if part.propellant > 0.0:

        facts.append(("Holds", f"{_mass(part.propellant)} of {propellants.of(part.propellant_type).name}"))

        said.append("A tank: the engines below it burn from it." + (" Insulated: it keeps cryogenic propellant longer." if part.insulation < 1.0 else ""))

    if part.parachute is not None:
        said.append("A parachute: armed by staging, it opens low and slow enough.")

    if part.crew:

        facts.append(("Crew", f"{part.crew}"))

        said.append(f"Seats {part.crew}: a crew flies it at once, wherever the link is.")

    if part.supplies:
        facts.append(("Supplies", _mass(part.supplies)))

    if part.antenna_range:

        facts.append(("Antenna", f"reaches home from {_range(part.antenna_range)}"))

    if part.battery:
        facts.append(("Battery", f"{part.battery:,.0f} Wh"))

    if part.solar is not None:

        facts.append(("Solar", f"{part.solar.power:,.0f} W at 1 AU, sunlit"))

        said.append("Solar wings: U unfolds them; they turn to the Sun.")

    if part.generator:
        facts.append(("Makes", f"{part.generator:,.0f} W, always"))

    if part.power_draw:
        facts.append(("Draws", f"{part.power_draw:,.0f} W"))

    if part.reaction_wheel_torque:
        facts.append(("Reaction wheels", f"{part.reaction_wheel_torque / 1000.0:,.1f} kN m"))

    if part.rcs_thrust:
        facts.append(("RCS", f"{part.rcs_thrust:,.0f} N each way"))

    if part.docking:
        said.append("A docking port" + (" on the side of a radial group." if part.side else " at the end of a stack."))

    if part.experiment is not None:
        said.append("An instrument: R runs it, for science in each new situation.")

    if part.isru is not None:
        said.append("An ISRU plant: makes propellant from ice or air where it lands.")

    if part.legs is not None:
        said.append("Landing legs: G lowers them; a wider stance stands on steeper ground.")

    if part.wheels is not None:
        said.append("Wheels: W / S drive, A / D steer, on power.")

    if part.rotor is not None:
        said.append("A rotor: lift in air, set with the throttle.")

    facts += [
        ("Survives", f"{part.crash_speed:,.0f} m/s at the ground, {part.max_temperature:,.0f} K"),
        ("Research", node.name if node is not None else "known from the start"),
    ]

    group = KINDS.get(part.kind, part.kind.replace("_", " ").capitalize())

    return Entry(SECTIONS[1], group, part.id, part.name, " ".join(said), tuple(facts))


def body_entry(
    profile,
    parent_name: str = ""
) -> Entry:
    """A body, said from its profile."""

    facts = [
        ("Radius", f"{profile.radius_km:,.0f} km"),
        ("Gravity", f"{profile.surface_gravity:,.2f} m/s^2 ({profile.surface_gravity / 9.80665:,.2f} g)"),
        ("Escape", f"{profile.escape_velocity / 1000.0:,.2f} km/s"),
        ("Day", f"{profile.solar_day_hours:,.1f} h" if abs(profile.solar_day_hours) < 1_000.0 else f"{profile.solar_day_hours / 24.0:,.0f} days"),
        ("Temperature", f"{profile.mean_temperature_c:,.0f} C on average"),
    ]

    if parent_name:
        facts.append(("Orbits", f"{parent_name}, {profile.parent_distance_km:,.0f} km out, every {profile.period_days:,.1f} days" if profile.parent_distance_km else parent_name))
    else:
        facts.append(("Orbits", f"the Sun at {profile.distance_au:,.2f} AU, every {profile.period_days / 365.25:,.2f} years"))

    air = profile.atmosphere

    if air is not None and air.surface_pressure_bar > 0.0:

        gases = ", ".join(f"{gas} {share * 100.0:,.0f}%" for gas, share in air.composition[:3])

        facts.append(("Air", f"{air.surface_pressure_bar:,.3g} bar at the surface: {gases}"))

    else:
        facts.append(("Air", "none"))

    group = {"planet": "Planets", "moon": "Moons", "dwarf": "Dwarf planets"}.get(profile.kind, profile.kind.replace("_", " ").capitalize() + "s")

    return Entry(SECTIONS[2], group, profile.id, profile.name, profile.description, tuple(facts))


def entries(
    parts: dict,
    presets: dict,
    tree=None,
    path: Path = ARTICLES_PATH
) -> list[Entry]:
    """Every entry: the articles, the parts, the bodies."""

    waits = {}

    if tree is not None:
        for node in tree.nodes:
            if node.id not in tree.free:
                for part_id in node.parts:
                    waits[part_id] = node

    found = articles(path)

    found += sorted((part_entry(part, waits.get(part.id)) for part in parts.values()), key=lambda entry: (entry.group, entry.title))

    # (A profile names what it orbits: "Earth", "Sun".)
    found += [body_entry(profile, profile.orbits if profile.orbits.casefold() != "sun" else "") for profile in presets.values()]

    return found


# ---------------------------------------------------------
# The window
# ---------------------------------------------------------

class Encyclopaedia:

    def __init__(
        self,
        entries_of=None
    ):
        """entries_of(): the entries, made the first time it opens."""

        self._entries_of = entries_of

        self.entries: list[Entry] | None = None

        self.open = False

        self.section = SECTIONS[0]

        self.search = ""

        self.chosen: Entry | None = None

    def toggle(self):

        self.open = not self.open

    def show(
        self,
        entry_id: str | None = None
    ):

        self.open = True

        if entry_id is not None:

            self._ensure()

            self.chosen = next((entry for entry in self.entries if entry.id == entry_id), self.chosen)

            if self.chosen is not None:
                self.section = self.chosen.section

    def _ensure(self):

        if self.entries is None:
            self.entries = self._entries_of() if self._entries_of is not None else []

    def draw(self):

        if not self.open:
            return

        self._ensure()

        viewport = imgui.get_main_viewport()

        imgui.set_next_window_size((840.0, min(620.0, viewport.work_size.y * 0.9)), imgui.Cond_.first_use_ever)
        imgui.set_next_window_pos(
            (viewport.work_pos.x + viewport.work_size.x * 0.5, viewport.work_pos.y + viewport.work_size.y * 0.5),
            imgui.Cond_.first_use_ever,
            (0.5, 0.5)
        )

        opened, keep = imgui.begin("Encyclopaedia (J)###encyclopaedia", True, imgui.WindowFlags_.no_collapse.value | imgui.WindowFlags_.no_docking.value)

        if not keep:
            self.open = False

        if opened:

            imgui.set_next_item_width(260.0)

            _, self.search = imgui.input_text_with_hint("##search", "Search...", self.search)

            imgui.same_line()

            for section in SECTIONS:

                imgui.same_line()

                if imgui.radio_button(section, self.section == section):
                    self.section = section

            shown = [entry for entry in self.entries if (entry.section == self.section or self.search) and (not self.search or entry.matches(self.search))]

            if imgui.begin_child("entries", (260.0, 0.0), imgui.ChildFlags_.borders.value):

                group = None

                for entry in shown:

                    heading = f"{entry.group}" if not self.search else f"{entry.section}: {entry.group}"

                    if heading != group:

                        group = heading

                        style.heading(group)

                    if imgui.selectable(f"{entry.title}##{entry.section}{entry.id}", self.chosen is entry)[0]:
                        self.chosen = entry

                if not shown:
                    imgui.text_disabled("Nothing found.")

            imgui.end_child()

            imgui.same_line()

            if imgui.begin_child("entry", (0.0, 0.0), imgui.ChildFlags_.borders.value):
                self._draw_entry(self.chosen or (shown[0] if shown else None))

            imgui.end_child()

        imgui.end()

    @staticmethod
    def _draw_entry(
        entry: Entry | None
    ):

        if entry is None:
            return

        style.heading(f"{entry.section}  ·  {entry.group}", "accent")

        with style.font("bold", 24.0):
            imgui.text(entry.title)

        imgui.push_text_wrap_pos(0.0)

        for paragraph in entry.text.split("\n\n"):

            if paragraph.strip():

                imgui.text_wrapped(keys(paragraph.strip()))

                imgui.dummy((0.0, 4.0))

        imgui.pop_text_wrap_pos()

        if entry.facts and imgui.begin_table("facts", 2, imgui.TableFlags_.row_bg.value):

            imgui.table_setup_column("label", imgui.TableColumnFlags_.width_fixed.value, style.px(120.0))
            imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch.value)

            for label, value in entry.facts:

                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_colored(style.tone("muted"), label)
                imgui.table_next_column()
                imgui.text_wrapped(value)

            imgui.end_table()
