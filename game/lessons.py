from dataclasses import dataclass

from game.guide import FIRST_ORBIT, Step


# =========================================================
# Flight School's Lessons
# =========================================================
#
# Each a short flight, set up where it starts (setup: the
# application puts the vessels there, controllers/lessons.py)
# and done step by step as game/guide.py's Moment shows it.
# A step with no test is read and gone on from (Next).
#
#   first orbit      from the pad to a round orbit
#   maneuver nodes   plan a burn on the map, fly it
#   docking          close on a station with the thrusters,
#                    dock
#   landing          come down on the Moon on the engine
#   power            a probe's battery low: the wings out
#   comms            a probe at Mars: light delay, the
#                    flight computer

@dataclass(frozen=True, slots=True)
class Lesson:

    id: str
    title: str
    about: str
    setup: str                      # where it starts (controllers/lessons.py)
    steps: tuple[Step, ...]
    done: str                       # said at the end


def _next(
    title: str,
    text: str
) -> Step:
    """A step to read, then Next."""

    return Step(title, text, None)


LESSONS: tuple[Lesson, ...] = (
    Lesson(
        "first_orbit",
        "First orbit",
        "From the launch pad to a round orbit round the Earth: throttle, staging, the gravity turn, the map, the circularizing burn.",
        "pad",
        FIRST_ORBIT,
        "You are in orbit. From here: {P} shows the programme's missions, {K} the science up here ({R} runs experiments), "
        "and the map's Navigation panel finds transfers to the Moon and the planets."
    ),
    Lesson(
        "maneuver",
        "Maneuver nodes",
        "Plan a burn on the map, see the orbit it makes, point at it and fly it: an upper stage in low Earth orbit.",
        "orbit",
        (
            Step("The map", "Press {M} for the map: your orbit round the Earth, 250 km up.", lambda m: m.map_open),
            Step("A node", "In the Maneuver panel (right), click At Ap: a node, a burn planned, at the apoapsis.", lambda m: m.node),
            Step(
                "Shape the burn",
                "Drag the yellow prograde handle (or click +10 by Prograde) to about 100 m/s: the dashed orbit is the one the burn makes, its far side higher.",
                lambda m: m.node_dv is not None and m.node_dv >= 80.0
            ),
            Step("Point at it", "Press {F} until stability assist says MANEUVER: the stage turns to the burn's direction.", lambda m: m.sas and m.sas_mode == "maneuver"),
            Step("Wait for it", "Time warp ({PERIOD}) towards the node: the warp slows as the burn nears. The HUD counts down to its start.", lambda m: m.node_in is not None and m.node_in <= 15.0),
            Step("Burn", "At zero, {Z} for full throttle; {X} when the HUD's m/s left is near 0.", lambda m: m.node_dv is not None and m.node_dv <= 2.0),
        ),
        "The orbit is what you planned. The same way: Circularize, capture at another body, come down; the Navigation panel plans the burns that leave for the Moon and the planets."
    ),
    Lesson(
        "docking",
        "Docking",
        "A ferry 120 m from a station in orbit: target it, close in on the thrusters, dock.",
        "docking",
        (
            Step("The target", "Press {L} for the vessels in flight, and Target the Station Core: the HUD shows how far it is and how fast you close.", lambda m: m.target),
            Step("Thrusters", "The arrows, {H} and {N} move the ferry without turning it (RCS): try one.", lambda m: m.translating),
            Step(
                "Close in",
                "Come within 30 m of the station's port, slowly: under 1 m/s, the thrusters the other way to stop.",
                lambda m: m.target_distance is not None and m.target_distance <= 30.0
            ),
            Step("Dock", "Press {B}: the ferry lines its port up with the station's and docks (or fly it in yourself).", lambda m: m.docked),
        ),
        "Docked: one vessel now. Propellant moves between docked tanks (the Vessels list); the Vessels list undocks them too."
    ),
    Lesson(
        "landing",
        "Landing on the Moon",
        "A lander 10 km over the Moon, falling: point it back, legs down, burn at the last moment, touch down softly.",
        "landing",
        (
            Step("Point it back", "Press {T} for stability assist, then {F} until it says RETROGRADE: the engine faces the way you fall.", lambda m: m.sas and m.sas_mode == "retrograde"),
            Step("Legs", "Press {G}: the landing legs come down.", lambda m: m.legs in ("lowering", "down")),
            Step(
                "The landing burn",
                "The HUD counts down to the burn that stops you at the ground (Brake in...). At BRAKE NOW (a beep), {Z} for full throttle.",
                lambda m: m.thrust > 0.0
            ),
            Step(
                "Touch down",
                "Ease the throttle ({LEFT_SHIFT} / {LEFT_CONTROL}) to come down slower and slower: under 3 m/s at the ground, upright, {X} once down.",
                lambda m: m.situation == "Landed"
            ),
        ),
        "Down on the Moon. {R} runs the experiments here; {K} shows what more there is to learn."
    ),
    Lesson(
        "power",
        "Power",
        "A probe in orbit, its battery low and its solar wings folded.",
        "power",
        (
            _next("Power", "The HUD's POWER: the battery's charge, and the net (made less drawn). Below zero the battery drains; empty, the probe has no wheels, no link, no computer."),
            Step("Wings out", "Press {U}: the solar wings unfold (fold them for a burn through the air: it tears them off).", lambda m: m.panels in ("unfolding", "out")),
            Step("Charging", "Sunlit, the net is above zero: warp ({PERIOD}) until the battery is half full.", lambda m: m.charge >= 0.5),
            _next("Night", "In a planet's shadow (half of a low orbit) the wings make nothing and the battery carries the probe. Farther from the Sun they make less: an RTG or a reactor makes power anywhere."),
        ),
        "Every part's draw and every wing's output is in the part editor's power budget, with the hours a battery lasts in the dark."
    ),
    Lesson(
        "comms",
        "Far from home",
        "A probe orbiting Mars: commands take minutes to reach it; the flight computer flies for you.",
        "comms",
        (
            _next("Light delay", "The HUD's link line: the ground station and the light time one way. Every command you give takes that long to arrive (a crew aboard flies at once)."),
            Step("A command", "Press {T} for stability assist: it is sent; warp ({PERIOD}) until it arrives and the HUD shows SAS on.", lambda m: m.sas),
            Step("The flight computer", "Press {C}: the flight computer, which flies a sequence of commands on its own.", lambda m: m.computer_open),
            Step("Send a program", "Open a program... (an example will do), then Send: it goes at the speed of light and runs when it gets there.", lambda m: m.uplink or m.program),
        ),
        "A probe far away flies its sequences on its own: a landing on Mars cannot wait for a command from Earth."
    ),
)


def lesson(
    lesson_id: str
) -> Lesson:

    return next(item for item in LESSONS if item.id == lesson_id)
