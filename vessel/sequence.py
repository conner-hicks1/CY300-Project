import re

from dataclasses import dataclass, field
from pathlib import Path


# =========================================================
# Flight Computer Sequences
# =========================================================
#
# A probe minutes of light away cannot be flown by hand: it
# is sent a sequence and flies it on its own. One statement
# a line ("#" starts a comment; case does not matter):
#
#   throttle full | off | 0.6 | 60%    set the throttle
#   hover                              throttle to hold the weight
#   descend 3 m/s                      throttle to sink at that
#                                      speed (landing)
#   stage                              fire the next stage
#   sas on | off | hold | prograde | retrograde | maneuver
#   science                            run the experiments
#   transmit                           send the results home
#   burn                               fly the maneuver node: point
#                                      along it, wait for its start,
#                                      full throttle until done
#   log <text>                         a line in the computer's log
#   wait 30 s | 5 min | 1:30 | 2 h | 3 d
#   wait until <condition>
#   at node | periapsis | apoapsis [+ or - <duration>]
#   repeat <n> ... end                 (nested allowed)
#
# Conditions: <quantity> <op> <number>[unit], or a
# situation (landed, flying, orbiting, sub-orbital,
# escaping), joined by "and" / "or" ("and" binds tighter),
# "not" before one. Quantities:
#
#   altitude  m above the ground      apoapsis, periapsis  m above sea level
#   speed     m/s over the ground     vertical  m/s, up positive
#   orbital   m/s                     pressure  bar outside
#   fuel      share left (0-1, or %)  heat      the hottest part's share of its limit
#   brake     s until a landing burn must start (suicide burn)
#   node      s until the maneuver node
#
# Units: m, km, m/s, km/s, s, min, h, d, %, bar.
#
# The computer (FlightComputer) runs on the vessel against
# a context (FlightContext) the game provides: what it can
# read, what it can do. Waits on the clock ("wait", "at")
# have a deadline the time warp holds back for.

class SequenceError(ValueError):
    """A line that is not a statement: (line number, message)."""

    def __init__(self, line: int, message: str):

        super().__init__(f"line {line}: {message}")

        self.line = line
        self.message = message


QUANTITIES = (
    "altitude", "apoapsis", "periapsis", "speed", "vertical", "orbital",
    "pressure", "fuel", "heat", "brake", "node"
)

SITUATIONS = {
    "landed": "Landed",
    "flying": "Flying",
    "orbiting": "Orbiting",
    "sub-orbital": "Sub-orbital",
    "suborbital": "Sub-orbital",
    "escaping": "Escaping",
    "flyby": "Flyby",
}

SAS_WORDS = ("on", "off", "hold", "prograde", "retrograde", "maneuver")

EVENTS = ("node", "periapsis", "apoapsis")

_UNITS = {
    "": 1.0, "m": 1.0, "km": 1000.0,
    "m/s": 1.0, "km/s": 1000.0,
    "s": 1.0, "sec": 1.0, "min": 60.0, "h": 3600.0, "hr": 3600.0, "d": 86_400.0, "day": 86_400.0, "days": 86_400.0,
    "%": 0.01, "bar": 1.0,
}

_DURATION_UNITS = {"s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
                   "m": 60.0, "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
                   "h": 3600.0, "hr": 3600.0, "hour": 3600.0, "hours": 3600.0,
                   "d": 86_400.0, "day": 86_400.0, "days": 86_400.0}

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?"


def parse_duration(
    text: str,
    line: int = 0
) -> float:
    """'30 s', '5 min', '1:30', '1:02:03', '2h', '90' (s) -> seconds."""

    text = text.strip().lower()

    if re.fullmatch(r"\d+(:\d{1,2}){1,2}", text):

        total = 0.0

        for part in text.split(":"):
            total = total * 60.0 + float(part)

        return total

    match = re.fullmatch(rf"({_NUMBER})\s*([a-z]*)", text)

    if match is None or match.group(2) not in _DURATION_UNITS and match.group(2) != "":
        raise SequenceError(line, f"not a duration: '{text}' (say 30 s, 5 min, 1:30, 2 h)")

    return float(match.group(1)) * (_DURATION_UNITS[match.group(2)] if match.group(2) else 1.0)


# ---------------------------------------------------------
# Conditions
# ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Compare:

    quantity: str
    op: str
    value: float

    def holds(self, context) -> bool:

        now = context.quantity(self.quantity)

        if now is None:
            return False

        return {
            "<": now < self.value,
            "<=": now <= self.value,
            ">": now > self.value,
            ">=": now >= self.value,
            "=": abs(now - self.value) < 1e-9,
        }[self.op]

    def __str__(self) -> str:
        return f"{self.quantity} {self.op} {self.value:g}"


@dataclass(frozen=True, slots=True)
class Situation:

    name: str                       # as the readout says it

    def holds(self, context) -> bool:
        return context.situation() == self.name

    def __str__(self) -> str:
        return self.name.lower()


@dataclass(frozen=True, slots=True)
class Not:

    inner: object

    def holds(self, context) -> bool:
        return not self.inner.holds(context)

    def __str__(self) -> str:
        return f"not {self.inner}"


@dataclass(frozen=True, slots=True)
class All:

    parts: tuple

    def holds(self, context) -> bool:
        return all(part.holds(context) for part in self.parts)

    def __str__(self) -> str:
        return " and ".join(str(part) for part in self.parts)


@dataclass(frozen=True, slots=True)
class Any:

    parts: tuple

    def holds(self, context) -> bool:
        return any(part.holds(context) for part in self.parts)

    def __str__(self) -> str:
        return " or ".join(str(part) for part in self.parts)


def parse_condition(
    text: str,
    line: int = 0
):

    text = text.strip().lower()

    if not text:
        raise SequenceError(line, "a condition is needed (e.g. altitude < 10 km)")

    alternatives = [part.strip() for part in re.split(r"\s+or\s+", text)]

    parsed = []

    for alternative in alternatives:

        terms = [part.strip() for part in re.split(r"\s+and\s+", alternative)]

        parsed.append(All(tuple(_atom(term, line) for term in terms)) if len(terms) > 1 else _atom(terms[0], line))

    return Any(tuple(parsed)) if len(parsed) > 1 else parsed[0]


def _atom(
    text: str,
    line: int
):

    if text.startswith("not "):
        return Not(_atom(text[4:].strip(), line))

    if text in SITUATIONS:
        return Situation(SITUATIONS[text])

    match = re.fullmatch(rf"([a-z]+)\s*(<=|>=|<|>|=)\s*({_NUMBER})\s*([a-z/%]*)", text)

    if match is None:
        raise SequenceError(line, f"not a condition: '{text}' (e.g. altitude < 10 km, landed)")

    quantity, op, number, unit = match.groups()

    if quantity not in QUANTITIES:
        raise SequenceError(line, f"unknown quantity '{quantity}' (one of {', '.join(QUANTITIES)})")

    if unit not in _UNITS:
        raise SequenceError(line, f"unknown unit '{unit}'")

    # (Minutes are "min": a bare "m" is meters.)
    return Compare(quantity, op, float(number) * _UNITS[unit])


# ---------------------------------------------------------
# Statements
# ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Statement:

    line: int                       # 1-based source line
    kind: str                       # throttle, hover, descend, stage, sas, science, transmit,
                                    # burn, log, wait, until, at, repeat, end
    value: object = None            # its argument
    offset: float = 0.0             # "at": s after the event (negative: before)
    target: int = -1                # repeat: its end; end: its repeat
    text: str = ""                  # the source, trimmed


@dataclass(slots=True)
class Program:

    statements: list[Statement]
    source: str = ""


def parse(
    source: str
) -> Program:
    """The program, or SequenceError at the first bad line."""

    statements: list[Statement] = []

    open_repeats: list[int] = []

    for number, raw in enumerate(source.splitlines(), start=1):

        text = raw.split("#", 1)[0].strip()

        if not text:
            continue

        words = text.lower().split()

        head, rest = words[0], " ".join(words[1:])

        def add(kind, value=None, offset=0.0):
            statements.append(Statement(number, kind, value, offset, -1, text))

        if head == "throttle":

            if rest in ("full", "max"):
                add("throttle", 1.0)
            elif rest in ("off", "cut", "zero"):
                add("throttle", 0.0)
            else:

                match = re.fullmatch(rf"({_NUMBER})\s*(%?)", rest)

                if match is None:
                    raise SequenceError(number, "throttle takes full, off, a share (0.6) or a percent (60%)")

                value = float(match.group(1)) * (0.01 if match.group(2) else 1.0)

                if not 0.0 <= value <= 1.0:
                    raise SequenceError(number, "throttle must be between 0 and 1 (0% and 100%)")

                add("throttle", value)

        elif head == "hover" and not rest:
            add("hover")

        elif head == "descend":

            match = re.fullmatch(rf"(?:at\s+)?({_NUMBER})\s*(m/s)?", rest)

            if match is None or float(match.group(1)) < 0.0:
                raise SequenceError(number, "descend takes a sinking speed: descend 3 m/s")

            add("descend", float(match.group(1)))

        elif head == "stage" and not rest:
            add("stage")

        elif head == "sas":

            if rest not in SAS_WORDS:
                raise SequenceError(number, f"sas takes one of {', '.join(SAS_WORDS)}")

            add("sas", rest)

        elif head in ("science", "transmit", "burn") and not rest:
            add(head)

        elif head == "log":
            add("log", raw.split("#", 1)[0].strip()[3:].strip())

        elif head == "wait":

            if words[1:2] == ["until"]:
                add("until", parse_condition(" ".join(words[2:]), number))
            elif rest:
                add("wait", parse_duration(rest, number))
            else:
                raise SequenceError(number, "wait takes a duration (wait 30 s) or a condition (wait until landed)")

        elif head == "at":

            match = re.fullmatch(r"([a-z]+)\s*(?:([+-])\s*(.+))?", rest)

            if match is None or match.group(1) not in EVENTS:
                raise SequenceError(number, f"at takes one of {', '.join(EVENTS)}, and maybe - 30 s")

            offset = 0.0

            if match.group(2):
                offset = parse_duration(match.group(3), number) * (1.0 if match.group(2) == "+" else -1.0)

            add("at", match.group(1), offset)

        elif head == "repeat":

            match = re.fullmatch(r"(\d+)(?:\s*times)?", rest)

            if match is None or int(match.group(1)) < 1:
                raise SequenceError(number, "repeat takes how many times: repeat 3")

            open_repeats.append(len(statements))

            add("repeat", int(match.group(1)))

        elif head == "end" and not rest:

            if not open_repeats:
                raise SequenceError(number, "end without a repeat")

            start = open_repeats.pop()

            statements[start] = Statement(
                statements[start].line, "repeat", statements[start].value, 0.0, len(statements), statements[start].text
            )

            statements.append(Statement(number, "end", None, 0.0, start, text))

        else:

            raise SequenceError(number, f"unknown statement '{text}'")

    if open_repeats:
        raise SequenceError(statements[open_repeats[-1]].line, "repeat without an end")

    return Program(statements, source)


# ---------------------------------------------------------
# Running
# ---------------------------------------------------------

class FlightContext:
    """
    What the computer can read and do (the game fills it in).
    Every method may be overridden; these do nothing.
    """

    def time(self) -> float:
        return 0.0

    def quantity(self, name: str) -> float | None:
        return None

    def situation(self) -> str:
        return ""

    def event_time(self, name: str) -> float | None:
        """Clock time of the next node / periapsis / apoapsis, or None."""
        return None

    def node_start(self) -> float | None:
        """Clock time the node's burn should start (half its burn before it), or None: no node."""
        return None

    def has_node(self) -> bool:
        return False

    def set_throttle(self, value: float): ...

    def hover(self, sink_speed: float | None): ...

    def stage(self): ...

    def sas(self, word: str): ...

    def science(self): ...

    def transmit(self): ...


@dataclass(slots=True)
class FlightComputer:

    program: Program

    # The statement next (or now waiting).
    index: int = 0

    # Waiting on the clock until (or None).
    deadline: float | None = None

    # A repeat's runs left, by its index.
    counts: dict = field(default_factory=dict)

    # "burn": where it is (None, "point", "burning").
    burning: str | None = None

    log: list = field(default_factory=list)

    done: bool = False

    # Statements at most one tick (a loop with no wait cannot
    # hang the game).
    STEPS = 64

    @property
    def current(
        self
    ) -> Statement | None:

        statements = self.program.statements

        return statements[self.index] if not self.done and self.index < len(statements) else None

    def time_left(
        self,
        context: FlightContext
    ) -> float | None:
        """
        s (on the clock) until what it waits for can happen:
        a deadline, or a condition on a time ("brake < 2 s",
        "node < 1 min") counting down. None: no telling.
        """

        if self.deadline is not None:
            return self.deadline - context.time()

        statement = self.current

        if statement is None or statement.kind != "until":
            return None

        gaps = []

        def visit(condition):

            if isinstance(condition, Compare):

                if condition.quantity in ("brake", "node") and condition.op in ("<", "<="):

                    now = context.quantity(condition.quantity)

                    if now is not None:
                        gaps.append(now - condition.value)

            elif isinstance(condition, (All, Any)):

                for part in condition.parts:
                    visit(part)

        visit(statement.value)

        return min(gaps) if gaps else None

    def note(
        self,
        context: FlightContext,
        text: str
    ):

        self.log.append((context.time(), text))

        del self.log[:-200]

    def stop(
        self,
        context: FlightContext,
        why: str = "Stopped"
    ):

        if not self.done:

            self.done = True
            self.deadline = None

            context.hover(None)

            self.note(context, why)

    def tick(
        self,
        context: FlightContext
    ):
        """Run what can run now; stop at the first wait still waiting."""

        for _ in range(self.STEPS):

            statement = self.current

            if statement is None:

                if not self.done:
                    self.done = True
                    self.note(context, "Done")

                return

            if not self._step(statement, context):
                return

        # (Still busy: the rest next tick.)

    def _step(
        self,
        statement: Statement,
        context: FlightContext
    ) -> bool:
        """Carry out one statement; False while it waits."""

        kind = statement.kind

        now = context.time()

        if kind == "throttle":

            context.hover(None)
            context.set_throttle(statement.value)

        elif kind == "hover":

            context.hover(0.0)

        elif kind == "descend":

            context.hover(statement.value)

        elif kind == "stage":

            context.stage()

        elif kind == "sas":

            context.sas(statement.value)

        elif kind == "science":

            context.science()

        elif kind == "transmit":

            context.transmit()

        elif kind == "log":

            self.note(context, statement.value)

        elif kind == "wait":

            if self.deadline is None:
                self.deadline = now + statement.value

            if now < self.deadline:
                return False

            self.deadline = None

        elif kind == "until":

            if not statement.value.holds(context):
                return False

        elif kind == "at":

            if self.deadline is None:

                when = context.event_time(statement.value)

                if when is None:

                    self.note(context, f"Line {statement.line}: no {statement.value} to wait for; skipped")

                    self.index += 1

                    return True

                self.deadline = when + statement.offset

            if now < self.deadline:
                return False

            self.deadline = None

        elif kind == "burn":

            return self._burn(statement, context, now)

        elif kind == "repeat":

            self.counts.setdefault(self.index, statement.value)

        elif kind == "end":

            start = statement.target

            left = self.counts.get(start, 1) - 1

            if left > 0:

                self.counts[start] = left

                self.index = start + 1

                return True

            self.counts.pop(start, None)

        self.index += 1

        return True

    def _burn(
        self,
        statement: Statement,
        context: FlightContext,
        now: float
    ) -> bool:

        if self.burning is None:

            if not context.has_node():

                self.note(context, f"Line {statement.line}: no maneuver node to burn; skipped")

                self.index += 1

                return True

            context.sas("maneuver")

            self.burning = "point"

        if self.burning == "point":

            start = context.node_start()

            if start is None:

                # Gone before it began.
                self.burning = None
                self.index += 1

                return True

            if now < start:

                self.deadline = start

                return False

            self.deadline = None

            context.hover(None)
            context.set_throttle(1.0)

            self.burning = "burning"

            self.note(context, "Burning")

        if context.has_node():
            return False

        # The node is done (the game cuts the throttle too).
        context.set_throttle(0.0)

        self.note(context, "Burn done")

        self.burning = None
        self.index += 1

        return True


def hover_throttle(
    sink_speed: float,
    vertical_speed: float,
    gravity: float,
    mass: float,
    max_thrust: float,
    up_share: float = 1.0
) -> float:
    """
    The throttle that holds a sinking speed (m/s, 0: hover):
    weight, plus a push to close the speed's error in ~1 s,
    by the thrust pointing up (up_share: the nose's cosine
    to the vertical).
    """

    if max_thrust <= 0.0 or up_share <= 0.05:
        return 0.0

    wanted = gravity + (-sink_speed - vertical_speed) * 1.0

    return min(max(wanted * mass / (max_thrust * up_share), 0.0), 1.0)



# ---------------------------------------------------------
# Program files
# ---------------------------------------------------------
#
# Built-in examples in assets/programs/, the player's in
# data/programs/ (not in the repository); "name.seq" files.

PROGRAMS_DIRECTORY = Path("assets/programs")
USER_PROGRAMS_DIRECTORY = Path("data/programs")


def list_programs(
    built_in: Path = PROGRAMS_DIRECTORY,
    user: Path = USER_PROGRAMS_DIRECTORY
) -> list[tuple[str, str, bool]]:
    """(name, path, the player's own) of every program, the player's first."""

    found = []

    for directory, own in ((user, True), (built_in, False)):

        for path in sorted(Path(directory).glob("*.seq")):
            found.append((path.stem, str(path), own))

    return found


def save_program(
    name: str,
    text: str,
    user: Path = USER_PROGRAMS_DIRECTORY
) -> str:
    """Save as the player's own; returns the path."""

    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_") or "program"

    path = Path(user) / f"{safe}.seq"

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(text, encoding="utf-8")

    return str(path)
