import json
import math

from dataclasses import dataclass, field
from pathlib import Path

from science.survey import SurveyMask


# =========================================================
# Science Record
# =========================================================
#
# What experiments find and what it is worth.
#
# A result is about a subject: one experiment, on one body,
# in one situation, in one region (or anywhere, for those
# that do not care where: high above the body). A subject
# is worth at most
#
#   cap = experiment value x body factor x region value
#
# and each result brought home banks KEEP of what is left of
# it (75%, then 75% of the rest...): doing it again is worth
# less each time. Transmitted instead of recovered, half
# that.
#
# The player's record (science points, how much of each
# subject is done, the survey masks) is saved in
# data/science/progress.json.

PROGRESS_PATH = Path("data/science/progress.json")

SITUATIONS = ("landed", "flying", "space_low", "space_high")

SITUATION_NAMES = {
    "landed": "landed",
    "flying": "flying",
    "space_low": "in space near",
    "space_high": "in space high over",
}

# Where the region matters (else the whole body is one).
REGIONAL = ("landed", "flying", "space_low")

# Below this many body radii above the surface: near.
SPACE_LOW_RADII = 0.25

KEEP = 0.75

TRANSMIT_SHARE = 0.5

# Points for each percent of a body mapped (x its factor).
MAPPING_POINTS_PER_PERCENT = 1.0


def situation_of(
    landed: bool,
    in_air: bool,
    altitude: float,
    radius: float
) -> str:

    if landed:
        return "landed"

    if in_air:
        return "flying"

    return "space_low" if altitude < SPACE_LOW_RADII * radius else "space_high"


def subject_key(
    experiment: str,
    body: str,
    situation: str,
    region: str
) -> str:

    return f"{experiment}@{body}:{situation}:{region}"


@dataclass(slots=True)
class ScienceResult:
    """One experiment's data, carried on the vessel until recovered or transmitted."""

    subject: str
    title: str                      # "Crew report, landed at Gale crater, Mars"
    cap: float                      # what the subject is worth in all


@dataclass(slots=True)
class ScienceRecord:

    points: float = 0.0

    # Subject key -> share of it done (0..1).
    subjects: dict = field(default_factory=dict)

    # Body id -> share of it mapped, credited so far (0..1).
    mapped: dict = field(default_factory=dict)

    # Body id -> SurveyMask.
    surveys: dict = field(default_factory=dict)

    def worth(
        self,
        result: ScienceResult,
        transmitted: bool = False
    ) -> float:
        """Points the result would bank now."""

        done = self.subjects.get(result.subject, 0.0)

        return result.cap * (1.0 - done) * KEEP * (TRANSMIT_SHARE if transmitted else 1.0)

    def bank(
        self,
        result: ScienceResult,
        transmitted: bool = False
    ) -> float:
        """Count the result; returns the points it gave."""

        gained = self.worth(result, transmitted)

        if result.cap > 0.0:
            self.subjects[result.subject] = min(1.0, self.subjects.get(result.subject, 0.0) + gained / result.cap)

        self.points += gained

        return gained

    def survey(
        self,
        body: str,
        gating: bool = True
    ) -> SurveyMask:

        mask = self.surveys.get(body)

        if mask is None:
            mask = self.surveys[body] = SurveyMask(gating=gating)

        mask.gating = gating

        return mask

    def credit_mapping(
        self,
        body: str,
        factor: float
    ) -> float:
        """Points for the body's newly mapped share (whole percents)."""

        mask = self.surveys.get(body)

        if mask is None:
            return 0.0

        covered = math.floor(mask.coverage() * 100.0) / 100.0

        credited = self.mapped.get(body, 0.0)

        if covered <= credited:
            return 0.0

        gained = (covered - credited) * 100.0 * MAPPING_POINTS_PER_PERCENT * factor

        self.mapped[body] = covered
        self.points += gained

        return gained

    # -----------------------------------------------------
    # Saving
    # -----------------------------------------------------

    def to_json(
        self
    ) -> dict:

        return {
            "format": 1,
            "points": self.points,
            "subjects": self.subjects,
            "mapped": self.mapped,
            # (Only those begun: an empty one is 8 KB of nothing.)
            "surveys": {body: mask.to_text() for body, mask in self.surveys.items() if mask.cells.any()},
        }

    @classmethod
    def from_json(
        cls,
        data: dict
    ) -> "ScienceRecord":

        return cls(
            points=float(data.get("points", 0.0)),
            subjects={str(key): float(value) for key, value in data.get("subjects", {}).items()},
            mapped={str(key): float(value) for key, value in data.get("mapped", {}).items()},
            surveys={str(body): SurveyMask.from_text(text) for body, text in data.get("surveys", {}).items()}
        )

    def save(
        self,
        path: Path = PROGRESS_PATH
    ):

        path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        temporary = path.with_suffix(".tmp")

        temporary.write_text(json.dumps(self.to_json(), indent=1), encoding="utf-8")

        temporary.replace(path)

    @classmethod
    def load(
        cls,
        path: Path = PROGRESS_PATH
    ) -> "ScienceRecord":
        """The saved record, or a new one (none saved, or unreadable)."""

        try:
            return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError):
            return cls()
