import math

from collections.abc import Hashable
from dataclasses import dataclass, field

import numpy as np

from vessel import kepler


# =========================================================
# Spheres of Influence (patched conics)
# =========================================================
#
# A vessel feels one body's gravity at a time: the body
# whose sphere of influence it is in (Laplace's radius,
# r = a (m / M)^(2/5): the Moon's ~66,000 km, Earth's
# ~925,000 km). Leaving it, the vessel goes on around the
# parent (the planet, or the Sun); entering a moon's, around
# the moon. At each change the state is re-expressed
# relative to the new body: r' = r + p_old - p_new,
# v' = v + u_old - u_new.
#
# Positions and velocities here are in the solar system's
# non-rotating frame, relative to whichever body is named.

# G M of the Sun (m^3 / s^2).
SUN_MU = 1.32712440018e20


def soi_radius(
    semi_major_axis: float,
    mass: float,
    parent_mass: float
) -> float:

    return semi_major_axis * (mass / parent_mass) ** 0.4


# A sphere at least this many of the body's radii: Laplace's
# makes a small moon's smaller than the moon itself (Phobos's
# 7 km, its radius 11 km; Deimos's, Enceladus's barely out of
# them), so that close to it - landed on it - a vessel would
# still feel only the planet. Within a few radii a small
# body's own pull is what matters.
MIN_SOI_RADII = 4.0


def sphere(
    semi_major_axis: float,
    mass: float,
    parent_mass: float,
    radius: float
) -> float:
    """A body's sphere of influence for the game: Laplace's, but never within a few of its radii."""

    return max(soi_radius(semi_major_axis, mass, parent_mass), MIN_SOI_RADII * radius)


@dataclass(frozen=True, slots=True)
class Body:
    """A body for patched conics. key: anything identifying it (an entity; None = the Sun)."""

    key: Hashable
    name: str
    mu: float
    radius: float
    soi: float                      # m (inf for the Sun)
    parent: Hashable                # key of what it orbits (None: the Sun)


def rebase(
    position,
    velocity,
    old_position,
    old_velocity,
    new_position,
    new_velocity
) -> tuple[np.ndarray, np.ndarray]:
    """A state relative to one body, made relative to another (both bodies' states in a shared frame)."""

    return (
        np.asarray(position, dtype=np.float64) + np.asarray(old_position) - np.asarray(new_position),
        np.asarray(velocity, dtype=np.float64) + np.asarray(old_velocity) - np.asarray(new_velocity),
    )


def transition(
    position,
    reference: Body,
    bodies: dict[Hashable, Body],
    child_positions: dict[Hashable, np.ndarray],
    velocity=None
) -> tuple[bool, Hashable]:
    """
    Where the vessel belongs now: (True, new body's key) once
    outside `reference`'s sphere (its parent; None is the
    Sun) or inside a child's (a moon's); (False, None) to
    stay. child_positions: the children's positions
    relative to `reference`. velocity: given, it leaves only
    moving outward (no flip-flopping on the boundary just
    after coming in).
    """

    position = np.asarray(position, dtype=np.float64)

    for key, where in child_positions.items():

        if float(np.linalg.norm(position - where)) < bodies[key].soi:
            return True, key

    outward = velocity is None or float(np.dot(position, velocity)) > 0.0

    if float(np.linalg.norm(position)) > reference.soi and outward:
        return True, reference.parent

    return False, None


# =========================================================
# Prediction (the map's orbit lines)
# =========================================================

@dataclass(slots=True)
class Patch:

    body: Hashable                  # relative to which body
    points: np.ndarray              # (n, 3) m, relative to it
    start: float                    # s from now
    end: str                        # "orbit" (closes), "escape", "encounter", "impact", "time"

    # Lowest point of this patch above the body's center (m).
    periapsis: float = math.inf

    # For "encounter" / "escape": the patch it leads into.
    next_body: Hashable = None

    # s from now of each point.
    times: np.ndarray | None = None


@dataclass(slots=True)
class Prediction:

    patches: list[Patch] = field(default_factory=list)

    @property
    def encounter(
        self
    ) -> Patch | None:
        """The patch inside another body's sphere, if the path gets there."""

        return self.patches[1] if len(self.patches) > 1 and self.patches[0].end == "encounter" else None


# Patches followed at most (now; into a moon's sphere; out
# of it again round the planet; into another).
MAX_PATCHES = 4


def predict(
    position,
    velocity,
    reference: Body,
    bodies: dict[Hashable, Body],
    child_states: dict[Hashable, tuple[np.ndarray, np.ndarray]],
    samples: int = 256,
    horizon: float = 30.0 * 86_400.0
) -> Prediction:
    """
    The vessel's path from now: its conic about `reference`
    (one revolution, or until it leaves the sphere, hits the
    body, or enters a child's sphere); into a child's sphere,
    its path there, and out of it again round `reference`,
    and on (a free return round the Moon, one moon then
    another), up to MAX_PATCHES patches. child_states:
    children's (position, velocity) relative to `reference`
    now; they move on their own conics meanwhile. Leaving
    `reference`'s own sphere ends it (the next body named).
    """

    prediction = Prediction()

    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)

    def children_at(t):
        return {key: kepler.propagate(*state, reference.mu, t) for key, state in child_states.items()}

    patch, state, child, taken = _follow(position, velocity, reference, bodies, child_states, samples, horizon)

    prediction.patches.append(patch)

    now = taken

    while len(prediction.patches) < MAX_PATCHES:

        if patch.end == "escape" and patch.body == reference.key:

            if reference.parent in bodies:
                patch.next_body = reference.parent

            break

        if patch.end != "encounter" or child is None or now >= horizon:
            break

        # Into the child's sphere: the vessel re-expressed
        # round it (where it is then).
        child_position, child_velocity = kepler.propagate(*child_states[child], reference.mu, now)

        inner, inner_state, _, inside = _follow(
            state[0] - child_position,
            state[1] - child_velocity,
            bodies[child],
            bodies,
            {},
            samples,
            horizon - now,
            start=now
        )

        prediction.patches.append(inner)

        if inner.end != "escape" or len(prediction.patches) >= MAX_PATCHES:
            break

        # Out again: round `reference`, from the edge of the
        # child's sphere on (the child not met again on the
        # way out of it).
        now += inside

        child_position, child_velocity = kepler.propagate(*child_states[child], reference.mu, now)

        inner.next_body = reference.key

        patch, state, left, taken = _follow(
            inner_state[0] + child_position,
            inner_state[1] + child_velocity,
            reference,
            bodies,
            children_at(now),
            samples,
            horizon - now,
            start=now,
            leaving=child
        )

        prediction.patches.append(patch)

        child = left

        now += taken

    return prediction


def _follow(
    position: np.ndarray,
    velocity: np.ndarray,
    reference: Body,
    bodies: dict[Hashable, Body],
    child_states: dict,
    samples: int,
    horizon: float,
    start: float = 0.0,
    leaving: Hashable = None
):
    """One conic, sampled, until something ends it. leaving: a child whose sphere it is on its way out of (met only once out)."""

    mu = reference.mu

    shape = kepler.orbit_shape(position, velocity, mu)

    if shape.period is not None and (shape.apoapsis is None or shape.apoapsis < reference.soi):

        span = min(shape.period, horizon)
        closes = shape.period <= horizon

    else:

        # Open (or leaving the sphere): just until it is out.
        closes = False

        out = kepler.time_to_radius(position, velocity, mu, reference.soi) if math.isfinite(reference.soi) else None

        span = min(out * 1.001, horizon) if out else horizon

    # Every sample at once (vectorized), then where the
    # first thing ends the patch.
    times = np.linspace(0.0, span, samples + 1)

    rs, vs = kepler.propagate_many(position, velocity, mu, times)

    # (Beyond a sample the solver could not reach - a path
    # through the center - nothing more.)
    broken = np.nonzero(~np.isfinite(rs).all(axis=1))[0]

    if len(broken):
        times, rs, vs = times[:broken[0]], rs[:broken[0]], vs[:broken[0]]
        samples = max(len(times) - 1, 0)

    distances = np.linalg.norm(rs, axis=1)

    ends = {}

    below = np.nonzero(distances[1:] < reference.radius)[0]

    if len(below):
        ends["impact"] = int(below[0]) + 1

    outside = np.nonzero(distances[1:] > reference.soi)[0]

    if len(outside):
        ends["escape"] = int(outside[0]) + 1

    hits = {}

    for key, (child_position, child_velocity) in child_states.items():

        where, _ = kepler.propagate_many(child_position, child_velocity, mu, times)

        within = np.linalg.norm(rs[1:] - where[1:], axis=1) < bodies[key].soi

        # (Still on the way out of it: only once out.)
        if key == leaving:

            out = np.nonzero(~within)[0]

            within[: out[0] if len(out) else len(within)] = False

        inside = np.nonzero(within)[0]

        if len(inside):
            hits[key] = int(inside[0]) + 1

    child = None

    if hits:

        child = min(hits, key=hits.get)

        ends["encounter"] = hits[child]

    if ends:

        end = min(ends, key=ends.get)

        last = ends[end]

        if end != "encounter":
            child = None

    else:

        end = "orbit" if closes else "time"

        last = samples

    points = rs[:last + 1]

    r, v = rs[last], vs[last]

    elapsed = float(times[last])

    lowest = float(distances[:last + 1].min())

    # The conic's own periapsis, when the path reaches it
    # (closed, or still closing in at the start); else the
    # lowest point sampled.
    approaching = float(np.dot(position, velocity)) < 0.0

    if end in ("orbit", "time") or (approaching and end in ("escape", "impact")):
        periapsis = min(lowest, shape.periapsis)
    else:
        periapsis = lowest

    patch = Patch(
        body=reference.key,
        points=points,
        start=start,
        end=end,
        periapsis=periapsis,
        next_body=child,
        times=start + times[:last + 1]
    )

    return patch, (r, v), child, elapsed
