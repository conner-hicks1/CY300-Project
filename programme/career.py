import json


from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from programme.crew import Corps
from programme import upgrades
from vessel.reliability import Record


# =========================================================
# The Space Programme
# =========================================================
#
# Money, reputation and a campaign on the real calendar.
#
#   funds        $ millions: a launch pays for the vessel
#                (its parts' cost, at liftoff); landing back
#                home recovers RECOVERY of what came back
#   budget       every calendar month: a base, more with
#                reputation (time warp spends months fast:
#                waiting for a window costs nothing but the
#                chance)
#   reputation   from missions done; lost with a crew, and
#                for a mission past its deadline
#   missions     assets/programme/missions.json: reach orbit,
#                the Moon, land at its south pole, Mars by the
#                2028 window, bring back samples... each with
#                what it pays, what must come first, and maybe
#                a real date it must be done by
#   research     the tech tree's nodes known (programme/
#                research.py), and the science spent on them
#
# The game reports what happens (a Flight snapshot each
# frame, events); the career checks the missions and keeps
# the books. Saved in data/programme/career.json.

MISSIONS_PATH = Path("assets/programme/missions.json")
CAREER_PATH = Path("data/programme/career.json")

KINDS = ("orbit", "soi", "land", "return", "map", "science", "sample", "isru", "dock", "eclipse", "apophis")

# Missions for a moment on the calendar (programme/events.py):
# checked every frame, not just now and then.
EVENT_KINDS = ("eclipse", "apophis")

# Of the parts' cost, back when landed home.
RECOVERY = 0.8

# Reputation lost with a crew, and for a missed deadline.
CREW_LOST = 25.0
DEADLINE_MISSED = 10.0


@dataclass(frozen=True, slots=True)
class Reward:

    funds: float = 0.0
    reputation: float = 0.0
    science: float = 0.0


@dataclass(frozen=True, slots=True)
class Mission:

    id: str
    title: str
    kind: str
    about: str = ""
    body: str = ""
    site: str = ""
    crew: bool = False
    value: float = 0.0
    reward: Reward = Reward()
    requires: tuple[str, ...] = ()
    opens: date | None = None
    deadline: date | None = None

    # Missed, no reputation lost (a chance, not a promise).
    optional: bool = False

    # Met only landed on this body (an eclipse seen from it).
    landed: str = ""


@dataclass(frozen=True, slots=True)
class Campaign:

    missions: tuple[Mission, ...]
    start_funds: float = 400.0
    monthly: float = 25.0
    per_reputation: float = 0.25

    def mission(self, mission_id: str) -> Mission | None:

        return next((mission for mission in self.missions if mission.id == mission_id), None)


def _date(
    text,
    where: str
) -> date | None:

    if text is None:
        return None

    try:
        return date.fromisoformat(str(text))
    except ValueError as error:
        raise ValueError(f"{where}: dates are YYYY-MM-DD") from error


def load_campaign(
    path: Path = MISSIONS_PATH
) -> Campaign:

    data = json.loads(Path(path).read_text(encoding="utf-8"))

    missions = []

    for raw in data.get("missions", []):

        where = f"mission '{raw.get('id')}'"

        if raw.get("kind") not in KINDS:
            raise ValueError(f"{where}: kind must be one of {', '.join(KINDS)}")

        reward = raw.get("reward", {})

        missions.append(Mission(
            id=str(raw["id"]),
            title=str(raw.get("title", raw["id"])),
            kind=raw["kind"],
            about=str(raw.get("about", "")),
            body=str(raw.get("body", "")),
            site=str(raw.get("site", "")),
            crew=bool(raw.get("crew", False)),
            value=float(raw.get("value", 0.0)),
            reward=Reward(float(reward.get("funds", 0.0)), float(reward.get("reputation", 0.0)), float(reward.get("science", 0.0))),
            requires=tuple(raw.get("requires", [])),
            opens=_date(raw.get("opens"), where),
            deadline=_date(raw.get("deadline"), where),
            optional=bool(raw.get("optional", False)),
            landed=str(raw.get("landed", ""))
        ))

    ids = {mission.id for mission in missions}

    for mission in missions:
        for need in mission.requires:
            if need not in ids:
                raise ValueError(f"mission '{mission.id}': requires unknown '{need}'")

    budget = data.get("budget", {})

    return Campaign(
        tuple(missions),
        float(data.get("start_funds", 400.0)),
        float(budget.get("monthly", 25.0)),
        float(budget.get("per_reputation", 0.25))
    )


@dataclass(frozen=True, slots=True)
class Flight:
    """What the vessel is doing now, for the missions."""

    body: str                       # profile id of what it flies around
    situation: str                  # the readout's ("Orbiting", "Landed"...)
    crewed: bool
    site: str = ""                  # the science site it is in
    home: str = ""                  # the launch body's id
    # Bodies landed on this flight (since it left home).
    landed_on: frozenset = frozenset()

    # Vessels docked in one (1: just itself).
    docked: int = 1

    # Real events (programme/events.py): body id -> the share
    # of the Sun it hides from the vessel now; the vessel's
    # distance from Apophis (m), near its pass.
    eclipsed: dict = field(default_factory=dict)
    apophis: float | None = None


@dataclass(slots=True)
class Career:

    funds: float
    reputation: float = 0.0

    # Mission id -> "done" or "failed".
    outcomes: dict = field(default_factory=dict)

    # The last month budgeted ((year, month)), or None.
    paid_through: tuple | None = None

    # Propellant made away from home (kg), all flights.
    made: float = 0.0

    # The latest clock time reached (s since J2000): new
    # scenes start from it.
    clock: float = 0.0

    # Tech nodes researched (programme/research.py), and the
    # science spent on them.
    researched: set = field(default_factory=set)
    research_spent: float = 0.0

    # Flight data: what each part has flown (vessel/
    # reliability.py Record).
    reliability: Record = field(default_factory=Record)

    # The astronauts (programme/crew.py).
    corps: Corps = field(default_factory=Corps.founded)

    # Notices for the HUD (taken by the game).
    news: list = field(default_factory=list)

    # ---------------------------------------------------------
    # The books
    # ---------------------------------------------------------

    def budget(
        self,
        campaign: Campaign
    ) -> float:
        """$M a month now."""

        return campaign.monthly + campaign.per_reputation * max(self.reputation, 0.0)

    def pay_budget(
        self,
        campaign: Campaign,
        today: date
    ) -> float:
        """Every calendar month begun since the last: its budget. Returns what came in."""

        month = (today.year, today.month)

        if self.paid_through is None:

            self.paid_through = month

            return 0.0

        if month <= self.paid_through:
            return 0.0

        months = (month[0] - self.paid_through[0]) * 12 + (month[1] - self.paid_through[1])

        income = months * self.budget(campaign)

        self.funds += income
        self.paid_through = month

        return income

    def can_afford(
        self,
        cost: float
    ) -> bool:

        return self.funds >= cost - 1e-9

    def launch(
        self,
        cost: float
    ):

        self.funds -= cost

    def recover(
        self,
        value: float
    ) -> float:
        """Landed home: RECOVERY of what came back."""

        refund = value * RECOVERY

        self.funds += refund

        return refund

    def crew_lost(self):

        self.reputation -= CREW_LOST

        self.news.append(f"Crew lost: reputation -{CREW_LOST:.0f}")

    # ---------------------------------------------------------
    # Missions
    # ---------------------------------------------------------

    def state(
        self,
        mission: Mission,
        today: date
    ) -> str:
        """"done", "failed", "locked" (needs others first), "waiting" (opens later) or "open"."""

        outcome = self.outcomes.get(mission.id)

        if outcome is not None:
            return outcome

        if any(self.outcomes.get(need) != "done" for need in mission.requires):
            return "locked"

        if mission.opens is not None and today < mission.opens:
            return "waiting"

        return "open"

    def check(
        self,
        campaign: Campaign,
        today: date,
        flight: Flight | None,
        science_points: float,
        mapped: dict,
        sampled: set
    ) -> list[Mission]:
        """
        Missions done now (paid, noted); missed deadlines
        failed. mapped: body -> share mapped; sampled: bodies
        whose surface samples have been recovered (or sent).
        """

        done = []

        for mission in campaign.missions:

            if self.state(mission, today) != "open":
                continue

            if mission.deadline is not None and today > mission.deadline:

                self.outcomes[mission.id] = "failed"

                if mission.optional:

                    self.news.append(f"Passed: {mission.title}")

                else:

                    self.reputation -= DEADLINE_MISSED

                    self.news.append(f"Missed: {mission.title} (reputation -{DEADLINE_MISSED:.0f})")

                continue

            if self._met(mission, flight, science_points, mapped, sampled):

                self.complete(mission)

                done.append(mission)

        return done

    def complete(
        self,
        mission: Mission
    ):
        """A mission done: paid, noted."""

        self.outcomes[mission.id] = "done"

        self.funds += mission.reward.funds
        self.reputation += mission.reward.reputation

        self.news.append(f"Mission done: {mission.title} (+${mission.reward.funds:,.0f}M)")

    def events(
        self,
        campaign: Campaign,
        today: date,
        flight: Flight | None
    ) -> list[Mission]:
        """The real events' missions met now (checked every frame: an eclipse is soon over)."""

        done = []

        for mission in campaign.missions:

            if mission.kind not in EVENT_KINDS or self.state(mission, today) != "open":
                continue

            if mission.deadline is not None and today > mission.deadline:
                continue

            if self._met(mission, flight, 0.0, {}, set()):

                self.complete(mission)

                done.append(mission)

        return done

    def _met(
        self,
        mission: Mission,
        flight: Flight | None,
        science_points: float,
        mapped: dict,
        sampled: set
    ) -> bool:

        kind = mission.kind

        if kind == "science":
            return science_points >= mission.value

        if kind == "map":
            return mapped.get(mission.body, 0.0) >= mission.value

        if kind == "sample":
            return mission.body in sampled

        if kind == "isru":
            return self.made >= mission.value

        if flight is None or (mission.crew and not flight.crewed):
            return False

        if kind == "soi":
            return flight.body == mission.body

        if kind == "orbit":
            return flight.body == mission.body and flight.situation == "Orbiting"

        if kind == "land":
            return (
                flight.body == mission.body
                and flight.situation == "Landed"
                and (not mission.site or flight.site == mission.site)
            )

        if kind == "eclipse":

            # Covered enough by its body (and landed where it
            # says: a lander in the Earth's shadow).
            return flight.eclipsed.get(mission.body, 0.0) >= mission.value and (
                not mission.landed or (flight.body == mission.landed and flight.situation == "Landed")
            )

        if kind == "apophis":
            return flight.apophis is not None and flight.apophis <= mission.value * 1000.0

        if kind == "dock":
            return (
                flight.docked >= mission.value
                and flight.situation == "Orbiting"
                and (not mission.body or flight.body == mission.body)
            )

        if kind == "return":
            return flight.situation == "Landed" and flight.body == flight.home and mission.body in flight.landed_on

        return False

    # ---------------------------------------------------------
    # Saving
    # ---------------------------------------------------------

    def to_json(self) -> dict:

        return {
            "format": upgrades.FORMATS["career"],
            "funds": self.funds,
            "reputation": self.reputation,
            "outcomes": self.outcomes,
            "paid_through": list(self.paid_through) if self.paid_through else None,
            "made": self.made,
            "clock": self.clock,
            "researched": sorted(self.researched),
            "research_spent": self.research_spent,
            "reliability": self.reliability.to_json(),
            "corps": self.corps.to_json(),
        }

    @classmethod
    def from_json(
        cls,
        data: dict
    ) -> "Career":

        paid = data.get("paid_through")

        return cls(
            funds=float(data["funds"]),
            reputation=float(data.get("reputation", 0.0)),
            outcomes={str(key): str(value) for key, value in data.get("outcomes", {}).items()},
            paid_through=tuple(paid) if paid else None,
            made=float(data.get("made", 0.0)),
            clock=float(data.get("clock", 0.0)),
            researched={str(item) for item in data.get("researched", [])},
            research_spent=float(data.get("research_spent", 0.0)),
            reliability=Record.from_json(data.get("reliability")),
            corps=Corps.from_json(data.get("corps"))
        )

    def save(
        self,
        path: Path = CAREER_PATH
    ):

        path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        temporary = path.with_suffix(".tmp")

        temporary.write_text(json.dumps(self.to_json(), indent=1), encoding="utf-8")

        temporary.replace(path)

    @classmethod
    def load(
        cls,
        campaign: Campaign,
        path: Path = CAREER_PATH
    ) -> "Career":
        """The saved career, or a new one with the campaign's starting funds."""

        try:
            return cls.from_json(upgrades.upgrade("career", json.loads(Path(path).read_text(encoding="utf-8"))))
        except (OSError, ValueError, KeyError, TypeError):
            return cls(funds=campaign.start_funds)


def today_of(
    moment: datetime
) -> date:

    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)

    return moment.astimezone(timezone.utc).date()
