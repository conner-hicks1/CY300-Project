import math
import random

from dataclasses import dataclass, field


# =========================================================
# Reliability
# =========================================================
#
# A new engine is not yet trusted: it may fail to light, or
# shut down in the middle of a burn; a new parachute may not
# open. Every flight teaches the programme its parts: each
# second an engine model burns, each time it lights, each
# time a parachute opens is flight data, and the more there
# is the rarer failures get (a few percent for the first
# flights, a few tenths of a percent once well flown). Test
# flights are worth flying.
#
#   ignition   P(fails) = 4% / (1 + e) + 0.2%
#   burning    a failure rate: 3% over a 300 s burn / (1 + e),
#              + 0.03% a 300 s
#   parachute  P(fails to open) = 3% / (1 + e) + 0.2%
#
# e is the experience with that part: an engine's seconds
# burned / 600 plus its ignitions / 3; a parachute's
# openings / 3. A failed engine is dead (it will not light
# again); a failed parachute stays packed. A sandbox has no
# failures.

IGNITION_NEW = 0.04
IGNITION_FLOOR = 0.002

BURN_NEW = 0.03 / 300.0             # per s
BURN_FLOOR = 0.0003 / 300.0

CHUTE_NEW = 0.03
CHUTE_FLOOR = 0.002

SECONDS_PER_LESSON = 600.0
IGNITIONS_PER_LESSON = 3.0
OPENINGS_PER_LESSON = 3.0


@dataclass(slots=True)
class Experience:
    """What the programme has flown of one part."""

    seconds: float = 0.0            # burned (engines; every copy)
    ignitions: int = 0              # lit (engines), opened (parachutes)
    failures: int = 0

    def lessons(
        self,
        parachute: bool = False
    ) -> float:

        if parachute:
            return self.ignitions / OPENINGS_PER_LESSON

        return self.seconds / SECONDS_PER_LESSON + self.ignitions / IGNITIONS_PER_LESSON


@dataclass(slots=True)
class Record:
    """The programme's flight data, part id -> Experience (saved with the career)."""

    parts: dict = field(default_factory=dict)

    def of(
        self,
        part_id: str
    ) -> Experience:

        found = self.parts.get(part_id)

        if found is None:
            found = self.parts[part_id] = Experience()

        return found

    def to_json(self) -> dict:

        return {
            part_id: {"seconds": item.seconds, "ignitions": item.ignitions, "failures": item.failures}
            for part_id, item in self.parts.items()
        }

    @classmethod
    def from_json(
        cls,
        data: dict | None
    ) -> "Record":

        record = cls()

        for part_id, raw in (data or {}).items():
            record.parts[str(part_id)] = Experience(
                float(raw.get("seconds", 0.0)),
                int(raw.get("ignitions", 0)),
                int(raw.get("failures", 0))
            )

        return record


def ignition_failure(
    experience: Experience
) -> float:

    return IGNITION_NEW / (1.0 + experience.lessons()) + IGNITION_FLOOR


def burn_rate(
    experience: Experience
) -> float:
    """Failures a second burning."""

    return BURN_NEW / (1.0 + experience.lessons()) + BURN_FLOOR


def chute_failure(
    experience: Experience
) -> float:

    return CHUTE_NEW / (1.0 + experience.lessons(parachute=True)) + CHUTE_FLOOR


def mission_reliability(
    experience: Experience,
    seconds: float
) -> float:
    """0..1: the chance an engine lights and burns `seconds` without failing."""

    return (1.0 - ignition_failure(experience)) * math.exp(-burn_rate(experience) * seconds)


class Failures:
    """Rolls the dice (a seeded generator: a run can be replayed)."""

    def __init__(
        self,
        record: Record,
        seed: int | None = None,
        enabled: bool = True
    ):

        self.record = record
        self.random = random.Random(seed)
        self.enabled = enabled

    def lights(
        self,
        part_id: str,
        copies: int = 1,
        scale: float = 1.0
    ) -> bool:
        """An engine (all its copies) lighting: True if it does (learned from either way). scale: on the odds (an engineer aboard: 0.5)."""

        experience = self.record.of(part_id)

        failed = self.enabled and any(self.random.random() < ignition_failure(experience) * scale for _ in range(copies))

        experience.ignitions += copies

        if failed:
            experience.failures += 1

        return not failed

    def opens(
        self,
        part_id: str,
        copies: int = 1,
        scale: float = 1.0
    ) -> bool:
        """A parachute arming: True if it will open."""

        experience = self.record.of(part_id)

        failed = self.enabled and any(self.random.random() < chute_failure(experience) * scale for _ in range(copies))

        experience.ignitions += copies

        if failed:
            experience.failures += 1

        return not failed

    def burns(
        self,
        part_id: str,
        seconds: float,
        copies: int = 1,
        scale: float = 1.0
    ) -> bool:
        """An engine burning `seconds` more: True if it keeps going (and its seconds are learned)."""

        experience = self.record.of(part_id)

        chance = 1.0 - math.exp(-burn_rate(experience) * seconds * copies * scale)

        experience.seconds += seconds * copies

        if self.enabled and self.random.random() < chance:

            experience.failures += 1

            return False

        return True
