from dataclasses import dataclass

import numpy as np

from math3d import quaternion

from vessel.layout import copy_orientations, copy_positions
from vessel.vessel import Vessel


# =========================================================
# Collisions
# =========================================================
#
# Vessels are rigid: they bump, they do not pass through one
# another. Each part is a capsule along its own axis (a
# cylinder with rounded ends, as wide as the part), a flat
# part (a heat shield, a separator) a sphere; a radial
# group's copies one each. Docking ports are left out (their
# faces meet to dock), and so are solar wings and landing
# legs (thin, and the legs are for the ground).
#
# Touching, the two bodies get an impulse at the contact
# along its normal, enough to stop them closing and bounce
# them a little apart (RESTITUTION), with friction across it
# (FRICTION): each by its mass and its moments of inertia, so
# a glancing blow sets it turning. A landed vessel does not
# move (it is held by the ground). They are also pushed
# apart a share of the overlap each step (no sinking in).
# Closing faster than a part stands (its crash speed), the
# vessel that part is on breaks up.

RESTITUTION = 0.25
FRICTION = 0.4

# Of the overlap, the share pushed apart each step.
CORRECTION = 0.4


@dataclass(frozen=True, slots=True)
class Capsule:
    """A part's shape in the vessel's frame: the segment from a to b, swept by a sphere of `radius`."""

    a: np.ndarray
    b: np.ndarray
    radius: float
    crash_speed: float
    name: str


def shapes(
    vessel: Vessel
) -> list[Capsule]:

    found = []

    up = np.array([0.0, 1.0, 0.0])

    for state in vessel.parts:

        part = state.part

        if part.docking or part.solar is not None or part.legs is not None:
            continue

        radius = part.radius
        height = part.height

        for position, turned in zip(copy_positions(state), copy_orientations(state)):

            base = np.asarray(position, dtype=np.float64)

            axis = quaternion.rotate_vector(turned, up)

            # (A sideways part's position is its axis's point
            # at the core's skin: it reaches out from there.)
            if height >= 2.0 * radius:

                found.append(Capsule(base + axis * radius, base + axis * (height - radius), radius, part.crash_speed, part.name))

            else:

                middle = base + axis * (0.5 * height)

                found.append(Capsule(middle, middle, max(0.5 * height, 0.6 * radius), part.crash_speed, part.name))

    return found


def reach(
    capsules: list[Capsule]
) -> tuple[np.ndarray, float]:
    """A sphere round all of them: (center, radius), in the same frame."""

    if not capsules:
        return np.zeros(3), 0.0

    ends = np.array([point for capsule in capsules for point in (capsule.a, capsule.b)])

    center = 0.5 * (ends.min(axis=0) + ends.max(axis=0))

    radius = max(
        max(float(np.linalg.norm(capsule.a - center)), float(np.linalg.norm(capsule.b - center))) + capsule.radius
        for capsule in capsules
    )

    return center, radius


def closest_points(
    p1: np.ndarray,
    q1: np.ndarray,
    p2: np.ndarray,
    q2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The closest points of two segments (Ericson, Real-Time Collision Detection 5.1.9)."""

    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2

    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)

    if a <= 1e-12 and e <= 1e-12:
        return p1, p2

    if a <= 1e-12:
        return p1, p2 + d2 * min(max(f / e, 0.0), 1.0)

    c = float(d1 @ r)

    if e <= 1e-12:
        return p1 + d1 * min(max(-c / a, 0.0), 1.0), p2

    b = float(d1 @ d2)

    denominator = a * e - b * b

    s = min(max((b * f - c * e) / denominator, 0.0), 1.0) if denominator > 1e-12 else 0.0

    t = (b * s + f) / e

    if t < 0.0:
        t = 0.0
        s = min(max(-c / a, 0.0), 1.0)
    elif t > 1.0:
        t = 1.0
        s = min(max((b - c) / a, 0.0), 1.0)

    return p1 + d1 * s, p2 + d2 * t


@dataclass(slots=True)
class Body:
    """A vessel (or a falling stage) for the collision: where it is and how it moves, about the same body's center."""

    capsules: list                  # Capsule, in its own frame
    position: np.ndarray            # its origin
    rotation: np.ndarray            # its frame -> inertial (3x3)
    velocity: np.ndarray
    spin: np.ndarray                # rad/s, inertial
    mass: float
    inertia: tuple                  # (pitch / yaw, roll) kg m^2 about its center of mass
    center: float                   # its center of mass's height in its frame
    static: bool = False            # held (landed): it does not move

    def to_world(
        self,
        point: np.ndarray
    ) -> np.ndarray:

        return self.position + self.rotation @ point

    @property
    def com(
        self
    ) -> np.ndarray:

        return self.to_world(np.array([0.0, self.center, 0.0]))

    def inverse_inertia(self) -> np.ndarray:

        if self.static:
            return np.zeros((3, 3))

        pitch, roll = self.inertia

        local = np.diag([1.0 / max(pitch, 1e-9), 1.0 / max(roll, 1e-9), 1.0 / max(pitch, 1e-9)])

        return self.rotation @ local @ self.rotation.T

    def inverse_mass(self) -> float:

        return 0.0 if self.static else 1.0 / max(self.mass, 1e-9)

    def point_velocity(
        self,
        point: np.ndarray
    ) -> np.ndarray:

        return self.velocity + np.cross(self.spin, point - self.com)


@dataclass(frozen=True, slots=True)
class Contact:

    point: np.ndarray               # between the two surfaces
    normal: np.ndarray              # from B toward A
    depth: float                    # m of overlap
    crash_a: float                  # the touching parts' crash speeds
    crash_b: float
    part_a: str
    part_b: str


def contacts(
    a: Body,
    b: Body
) -> list[Contact]:
    """Where two bodies' shapes overlap (none: not touching)."""

    center_a, reach_a = reach(a.capsules)
    center_b, reach_b = reach(b.capsules)

    if float(np.linalg.norm(a.to_world(center_a) - b.to_world(center_b))) >= reach_a + reach_b:
        return []

    world_b = [(b.to_world(capsule.a), b.to_world(capsule.b), capsule) for capsule in b.capsules]

    found = []

    for capsule in a.capsules:

        p1, q1 = a.to_world(capsule.a), a.to_world(capsule.b)

        for p2, q2, other in world_b:

            near_a, near_b = closest_points(p1, q1, p2, q2)

            between = near_a - near_b

            distance = float(np.linalg.norm(between))

            depth = capsule.radius + other.radius - distance

            if depth <= 0.0:
                continue

            if distance > 1e-9:
                normal = between / distance
            else:
                normal = a.com - b.com
                normal = normal / max(float(np.linalg.norm(normal)), 1e-9)

            point = near_b + normal * (other.radius - 0.5 * depth)

            found.append(Contact(point, normal, depth, capsule.crash_speed, other.crash_speed, capsule.name, other.name))

    return found


@dataclass(frozen=True, slots=True)
class Impact:
    """What a collision did: the closing speed, and which of the two broke up (a, b)."""

    speed: float
    broke_a: bool
    broke_b: bool
    part_a: str
    part_b: str


def resolve(
    a: Body,
    b: Body,
    found: list[Contact]
) -> Impact | None:
    """
    Push two touching bodies apart: an impulse at each
    contact (its velocity along the normal stopped and
    bounced, friction across), and a share of the overlap
    corrected. Changes their velocities, spins, positions.
    """

    if not found or (a.static and b.static):
        return None

    hardest = 0.0
    worst = found[0]

    for contact in sorted(found, key=lambda item: -item.depth):

        n = contact.normal

        ra = contact.point - a.com
        rb = contact.point - b.com

        relative = a.point_velocity(contact.point) - b.point_velocity(contact.point)

        closing = float(relative @ n)

        if closing >= 0.0:
            continue

        if -closing > hardest:
            hardest = -closing
            worst = contact

        inv_a = a.inverse_inertia()
        inv_b = b.inverse_inertia()

        def effective(direction):
            return (
                a.inverse_mass() + b.inverse_mass()
                + float(direction @ np.cross(inv_a @ np.cross(ra, direction), ra))
                + float(direction @ np.cross(inv_b @ np.cross(rb, direction), rb))
            )

        push = -(1.0 + RESTITUTION) * closing / max(effective(n), 1e-12)

        impulse = n * push

        # Friction across the normal, at most mu times the push.
        across = relative - n * closing

        sliding = float(np.linalg.norm(across))

        if sliding > 1e-6:

            t = across / sliding

            rub = min(sliding / max(effective(t), 1e-12), FRICTION * push)

            impulse = impulse - t * rub

        _apply(a, impulse, ra)
        _apply(b, -impulse, rb)

    # The overlap: pushed apart, the lighter more.
    deepest = max(found, key=lambda item: item.depth)

    total = a.inverse_mass() + b.inverse_mass()

    if total > 0.0:

        shift = deepest.normal * (CORRECTION * deepest.depth / total)

        a.position = a.position + shift * a.inverse_mass()
        b.position = b.position - shift * b.inverse_mass()

    if hardest <= 0.0:
        return Impact(0.0, False, False, worst.part_a, worst.part_b)

    return Impact(hardest, hardest > worst.crash_a, hardest > worst.crash_b, worst.part_a, worst.part_b)


def _apply(
    body: Body,
    impulse: np.ndarray,
    arm: np.ndarray
):

    if body.static:
        return

    body.velocity = body.velocity + impulse * body.inverse_mass()
    body.spin = body.spin + body.inverse_inertia() @ np.cross(arm, impulse)


def body_of(
    vessel: Vessel,
    state,
    capsules: list | None = None
) -> Body:
    """A vessel's collision body from its flight state (state: vessel/physics.py VesselState)."""

    rotation = quaternion.to_matrix3(state.orientation)

    return Body(
        capsules=capsules if capsules is not None else shapes(vessel),
        position=np.asarray(state.position, dtype=np.float64).copy(),
        rotation=rotation,
        velocity=np.asarray(state.velocity, dtype=np.float64).copy(),
        spin=rotation @ np.asarray(state.angular_velocity, dtype=np.float64),
        mass=vessel.mass,
        inertia=vessel.moments_of_inertia,
        center=vessel.center_of_mass,
        static=bool(state.landed)
    )


def store(
    body: Body,
    state
):
    """Back into the flight state."""

    if body.static:
        return

    state.position = body.position
    state.velocity = body.velocity
    state.angular_velocity = body.rotation.T @ body.spin
