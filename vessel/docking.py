import math

from dataclasses import dataclass

import numpy as np

from math3d import quaternion

from vessel.vessel import PartState, Vessel


# =========================================================
# Docking
# =========================================================
#
# Docking ports (parts with "docking") at the ends of a
# vessel's core stack, its top or its bottom, join it to
# another vessel's port: two vessels become one stack. The
# second's parts go on above or below the first's (turned
# over when the ports face the same way along the stacks:
# top to top, bottom to bottom); its stages come after the
# first's. The two ports stay, joined: undocking splits the
# stack between them again.
#
# Joined, the stacks share propellant between separators
# (a depot feeding a ship); a turned-over engine points
# the wrong way and does not fire.
#
# And the approach: guidance for an automatic docking
# (docking_command): the nose turned so the ports face each
# other, small translation thrusts (RCS) to come in along
# the other port's axis, slowing to a touch.

# Ports dock within (m) of each other, slower than (m/s),
# their faces within this of opposite (cosine).
CAPTURE_DISTANCE = 1.0
CAPTURE_SPEED = 0.6
CAPTURE_ALIGNMENT = 0.97


@dataclass(frozen=True, slots=True)
class Port:
    """A free docking port at one end of a vessel's core."""

    end: str                        # "top" or "bottom"
    index: int                      # in vessel.parts
    state: PartState


def free_ports(
    vessel: Vessel
) -> list[Port]:
    """Docking ports at the core's ends, not joined."""

    core = [(index, state) for index, state in enumerate(vessel.parts) if not state.radial]

    if not core:
        return []

    found = []

    first_index, first = core[0]
    last_index, last = core[-1]

    if first.part.docking and not first.joined:
        found.append(Port("top", first_index, first))

    if last.part.docking and not last.joined and last_index != first_index:
        found.append(Port("bottom", last_index, last))

    return found


def port_point(
    vessel: Vessel,
    position: np.ndarray,
    orientation: np.ndarray,
    end: str
) -> tuple[np.ndarray, np.ndarray]:
    """(where the port's face is, the way it faces): about the vessel's body, inertial."""

    up = quaternion.rotate_vector(orientation, np.array([0.0, 1.0, 0.0]))

    if end == "top":
        return np.asarray(position, dtype=np.float64) + up * vessel.height, up

    return np.asarray(position, dtype=np.float64), -up


def modules(
    vessel: Vessel
) -> int:
    """How many vessels docked make this one (joined pairs + 1)."""

    return sum(1 for state in vessel.parts if state.joined) // 2 + 1


def _turned_over(
    parts: list[PartState]
) -> list[PartState]:
    """The stack upside down: core parts in reverse, each still followed by its radial groups."""

    blocks: list[list[PartState]] = []

    for state in parts:

        if not state.radial or not blocks:
            blocks.append([state])
        else:
            blocks[-1].append(state)

    turned = []

    for block in reversed(blocks):

        for state in block:
            state.flipped = not state.flipped

        turned.extend(block)

    return turned


def merge(
    a: Vessel,
    a_end: str,
    b: Vessel,
    b_end: str
) -> tuple[Vessel, bool, float]:
    """
    Join b's port at b_end to a's port at a_end. Returns
    (the joined vessel, whether b's parts went below a's,
    how far (m) a's origin rose in the joined stack).
    """

    a_port = next(port for port in free_ports(a) if port.end == a_end)
    b_port = next(port for port in free_ports(b) if port.end == b_end)

    a_port.state.joined = True
    b_port.state.joined = True

    # Each part remembers the vessel it came with; the joined
    # one is named for the heavier (a ferry docks to a
    # station).
    for vessel in (a, b):
        for state in vessel.parts:
            state.origin = state.origin or vessel.name

    name = a.name if a.mass >= b.mass else b.name

    b_parts = list(b.parts)

    # Facing each other when one is a top and the other a
    # bottom; else b turns over.
    if a_end == b_end:
        b_parts = _turned_over(b_parts)

    # Its stages after a's.
    last = max((state.stage for state in a.parts if state.stage is not None), default=a.next_stage - 1)

    shift = max(last + 1, a.next_stage) - b.next_stage

    for state in b_parts:
        if state.stage is not None and state.stage >= b.next_stage:
            state.stage += shift

    # (Radial groups numbered apart.)
    groups = max((state.group for state in a.parts if state.group is not None), default=-1) + 1

    for state in b_parts:
        if state.group is not None:
            state.group += groups

    below = a_end == "bottom"

    height = b.height

    parts = list(a.parts) + b_parts if below else b_parts + list(a.parts)

    joined = Vessel(name, parts, a.next_stage, 0.0)

    joined._restack()

    return joined, below, height if below else 0.0


def joint(
    vessel: Vessel
) -> int | None:
    """Where to undock: the index of the lower of the last pair of joined ports next to each other."""

    core = [index for index, state in enumerate(vessel.parts) if not state.radial]

    for upper, lower in zip(reversed(core[:-1]), reversed(core[1:])):

        if vessel.parts[upper].joined and vessel.parts[lower].joined:
            return lower

    return None


def split(
    vessel: Vessel,
    at: int
) -> tuple[Vessel, Vessel, float, bool, bool]:
    """
    Undock at the joint above parts[at]: (the upper vessel,
    the lower, how far (m) above the old origin the upper's
    origin is, whether the upper / the lower turned back
    over: a half all turned over goes back its own way up).
    The joined ports come free.
    """

    upper_parts = vessel.parts[:at]
    lower_parts = vessel.parts[at:]

    upper_parts[_last_core(upper_parts)].joined = False
    lower_parts[0].joined = False

    lower = Vessel(_named(lower_parts, f"{vessel.name} (undocked)"), lower_parts, vessel.next_stage, 0.0)
    upper = Vessel(_named(upper_parts, vessel.name), upper_parts, vessel.next_stage, 0.0)

    lift = sum(state.part.height for state in lower_parts if not state.radial)

    turned = []

    # A half all turned over turns back (its own way up).
    for half in (upper, lower):

        flip = bool(half.parts) and all(state.flipped for state in half.parts)

        if flip:
            half.parts = _turned_over(half.parts)

        turned.append(flip)

    upper._restack()
    lower._restack()

    return upper, lower, lift, turned[0], turned[1]


def _named(
    parts: list[PartState],
    fallback: str
) -> str:
    """The name most of the parts' mass came with."""

    weights: dict[str, float] = {}

    for state in parts:
        if state.origin:
            weights[state.origin] = weights.get(state.origin, 0.0) + (state.part.mass + state.propellant) * state.count

    return max(weights, key=weights.get) if weights else fallback


def _last_core(
    parts: list[PartState]
) -> int:

    return max(index for index, state in enumerate(parts) if not state.radial)


# ---------------------------------------------------------
# The approach
# ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DockingCommand:

    up: np.ndarray                  # where to point the stack (inertial unit)
    translate: np.ndarray           # RCS command, vessel axes, -1..1 each
    distance: float                 # m between the port faces
    closing: float                  # m/s, toward each other positive
    misalignment: float             # deg between the faces and opposite
    lateral: float                  # m off the other port's axis


def docking_command(
    own_port: tuple[np.ndarray, np.ndarray],
    own_velocity: np.ndarray,
    own_orientation: np.ndarray,
    own_end: str,
    target_port: tuple[np.ndarray, np.ndarray],
    target_velocity: np.ndarray,
    rcs_acceleration: float
) -> DockingCommand:
    """
    One step of the approach. Points the nose so the ports
    face each other; translates to stand on the other port's
    axis, then comes in along it, slower the closer (0.1 m/s
    at the touch, at most 1 m/s), backing out first if
    beside or behind the port.
    """

    own_point, own_face = own_port
    target_point, target_face = target_port

    up = -target_face if own_end == "top" else target_face

    offset = own_point - target_point

    along = float(offset @ target_face)

    lateral = offset - target_face * along

    off_axis = float(np.linalg.norm(lateral))

    relative = np.asarray(own_velocity, dtype=np.float64) - np.asarray(target_velocity, dtype=np.float64)

    # No faster than the thrusters can stop from in what is
    # left (half their push, to spare): v = sqrt(a d).
    def stoppable(distance: float, most: float) -> float:
        return min(most, math.sqrt(max(rcs_acceleration, 1e-6) * max(distance, 0.0)))

    sideways = -lateral / off_axis * stoppable(off_axis, min(0.2 * off_axis, 2.0)) if off_axis > 1e-6 else np.zeros(3)

    if along < 2.0 and off_axis > 0.3:

        # Too close beside it: back out along the axis first.
        wanted = target_face * 0.5 + sideways

    else:

        centered = off_axis < max(0.3, 0.1 * along)

        approach = max(stoppable(along - CAPTURE_DISTANCE * 0.5, min(0.05 * along, 2.0)), 0.1) if centered else 0.0

        wanted = sideways - target_face * approach

    error = wanted - relative

    # Inertial -> the vessel's axes.
    local = quaternion.rotate_vector(quaternion.conjugate(own_orientation), error)

    gain = 2.0 / max(rcs_acceleration, 1e-6)

    translate = np.clip(local * gain, -1.0, 1.0)

    distance = float(np.linalg.norm(offset))

    closing = -float(relative @ (offset / distance)) if distance > 1e-9 else 0.0

    misalignment = math.degrees(math.acos(float(np.clip(-own_face @ target_face, -1.0, 1.0))))

    return DockingCommand(up / float(np.linalg.norm(up)), translate, distance, closing, misalignment, off_axis)


def can_capture(
    own_port: tuple[np.ndarray, np.ndarray],
    own_velocity: np.ndarray,
    target_port: tuple[np.ndarray, np.ndarray],
    target_velocity: np.ndarray
) -> bool:

    distance = float(np.linalg.norm(own_port[0] - target_port[0]))

    speed = float(np.linalg.norm(np.asarray(own_velocity) - np.asarray(target_velocity)))

    facing = float(-own_port[1] @ target_port[1])

    return distance <= CAPTURE_DISTANCE and speed <= CAPTURE_SPEED and facing >= CAPTURE_ALIGNMENT
