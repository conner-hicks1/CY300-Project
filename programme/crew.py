import random

from dataclasses import dataclass, field


# =========================================================
# The Astronaut Corps
# =========================================================
#
# The career's astronauts, by name: each a pilot, an engineer
# or a scientist, with the flights and days flown and the
# radiation dose (mSv, vessel/life.py) built up. At a crewed
# launch the corps fills the seats (a pilot first, then an
# engineer, then a scientist, then whoever is ready); coming
# home they are ready again (past the career dose limit,
# retired); lost with their vessel, lost. A new career has
# seven; more can be hired.
#
#   pilot      flies (any crewed vessel needs one aboard, or
#              any astronaut at all)
#   engineer   aboard, halves the chance of a part failing
#              (vessel/reliability.py)
#   scientist  aboard, a quarter more science from each
#              experiment run
#
# Their names are made up.

ROLES = ("pilot", "engineer", "scientist")

HIRE_COST = 5.0                     # $ millions

FIRST = (
    "Ada", "Lin", "Tomas", "Priya", "Jonah", "Mei", "Samir", "Ines", "Kofi", "Hana",
    "Mateo", "Yara", "Oskar", "Leila", "Ravi", "Noor", "Elias", "Sofia", "Kenji", "Amara",
)

LAST = (
    "Okafor", "Moreau", "Reyes", "Nair", "Kessler", "Tanaka", "Haddad", "Larsen", "Mensah", "Sato",
    "Duarte", "Farouk", "Lindqvist", "Rahimi", "Iyer", "Qureshi", "Varga", "Castell", "Ishida", "Achebe",
)


@dataclass(slots=True)
class Astronaut:

    name: str
    role: str
    flights: int = 0
    days: float = 0.0               # in space, all flights
    dose: float = 0.0               # mSv, all flights
    status: str = "ready"           # "ready", "flying", "lost", "retired"
    vessel: str = ""                # the vessel flown, when flying
    launched: float = 0.0           # clock s of the launch, when flying


@dataclass(slots=True)
class Corps:

    astronauts: list = field(default_factory=list)

    # Names hired so far (each new name drawn after them).
    hired: int = 0

    @classmethod
    def founded(cls) -> "Corps":
        """A new career's seven: three pilots, two engineers, two scientists (drawn the same each time)."""

        corps = cls()

        for role in ("pilot", "pilot", "engineer", "engineer", "scientist", "scientist", "pilot"):
            corps.hire(role)

        return corps

    def hire(
        self,
        role: str
    ) -> Astronaut:

        rng = random.Random(self.hired * 7_919 + 11)

        taken = {astronaut.name for astronaut in self.astronauts}

        name = ""

        for _ in range(400):

            name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"

            if name not in taken:
                break

        self.hired += 1

        astronaut = Astronaut(name, role if role in ROLES else "pilot")

        self.astronauts.append(astronaut)

        return astronaut

    def named(
        self,
        name: str
    ) -> Astronaut | None:

        return next((astronaut for astronaut in self.astronauts if astronaut.name == name), None)

    def ready(self) -> list[Astronaut]:

        return [astronaut for astronaut in self.astronauts if astronaut.status == "ready"]

    def choose(
        self,
        seats: int
    ) -> list[Astronaut]:
        """Who flies, for so many seats: a pilot, an engineer, a scientist, then the others ready (fewest flights first)."""

        ready = sorted(self.ready(), key=lambda astronaut: (astronaut.flights, astronaut.name))

        chosen = []

        for role in ROLES:

            match = next((astronaut for astronaut in ready if astronaut.role == role and astronaut not in chosen), None)

            if match is not None and len(chosen) < seats:
                chosen.append(match)

        for astronaut in ready:

            if len(chosen) >= seats:
                break

            if astronaut not in chosen:
                chosen.append(astronaut)

        return chosen

    def launch(
        self,
        names: list[str],
        vessel: str,
        now: float
    ):

        for name in names:

            astronaut = self.named(name)

            if astronaut is not None:
                astronaut.status = "flying"
                astronaut.vessel = vessel
                astronaut.launched = now

    def home(
        self,
        names: list[str],
        now: float,
        limit: float
    ) -> list[str]:
        """Back home: ready again (or retired, past the dose limit). Who retired."""

        retired = []

        for name in names:

            astronaut = self.named(name)

            if astronaut is None or astronaut.status != "flying":
                continue

            astronaut.flights += 1
            astronaut.days += max(now - astronaut.launched, 0.0) / 86_400.0
            astronaut.vessel = ""

            if astronaut.dose > limit:
                astronaut.status = "retired"
                retired.append(name)
            else:
                astronaut.status = "ready"

        return retired

    def stand_down(
        self,
        names: list[str]
    ):
        """A flight called off (a revert): ready again, as if it never went."""

        for name in names:

            astronaut = self.named(name)

            if astronaut is not None and astronaut.status == "flying":
                astronaut.status = "ready"
                astronaut.vessel = ""

    def lose(
        self,
        names: list[str]
    ) -> list[str]:
        """Lost: who was."""

        lost = []

        for name in names:

            astronaut = self.named(name)

            if astronaut is not None and astronaut.status not in ("lost",):
                astronaut.status = "lost"
                astronaut.vessel = ""
                lost.append(name)

        return lost

    def to_json(self) -> dict:

        return {
            "hired": self.hired,
            "astronauts": [
                {
                    "name": a.name, "role": a.role, "flights": a.flights, "days": a.days, "dose": a.dose,
                    "status": a.status, "vessel": a.vessel, "launched": a.launched,
                }
                for a in self.astronauts
            ],
        }

    @classmethod
    def from_json(
        cls,
        data: dict | None
    ) -> "Corps":

        if not data:
            return cls.founded()

        corps = cls(hired=int(data.get("hired", 0)))

        for raw in data.get("astronauts", []):
            corps.astronauts.append(Astronaut(
                str(raw["name"]), str(raw.get("role", "pilot")), int(raw.get("flights", 0)),
                float(raw.get("days", 0.0)), float(raw.get("dose", 0.0)), str(raw.get("status", "ready")),
                str(raw.get("vessel", "")), float(raw.get("launched", 0.0))
            ))

        return corps
