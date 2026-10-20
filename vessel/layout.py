import math

import numpy as np

from math3d import quaternion


# =========================================================
# Where a Part's Copies Stand
# =========================================================
#
# In the vessel's frame (its origin the bottom of its lowest
# core part, +Y up the stack): for drawing the parts
# (systems/vessel_system.py) and for their collision shapes
# (vessel/collision.py).


def copy_positions(
    state
) -> list[tuple[float, float, float]]:
    """
    Where a part's copies stand in the vessel's frame: the
    core's on the axis; a radial group's evenly round it,
    the first on +X (to the sides as the camera first sees
    a vessel on the pad).
    """

    # Wheels: in pairs, left and right (+X / -X), four of them
    # front and back too; their axles across the vessel.
    if state.part.wheels is not None and state.outward:
        return [
            (side * state.distance, state.bottom, row * 0.75 * state.distance)
            for side, row in _wheel_places(state.count)
        ]

    if not state.radial:

        # (Turned over in a docked stack: its own bottom at
        # the top of its place.)
        return [(0.0, state.bottom + (state.part.height if state.flipped else 0.0), 0.0)]

    base = math.radians(state.angle)

    return [
        (
            state.distance * math.cos(base + 2.0 * math.pi * k / state.count),
            state.bottom,
            state.distance * math.sin(base + 2.0 * math.pi * k / state.count)
        )
        for k in range(state.count)
    ]


def copy_orientations(
    state
) -> list:
    """
    How a part's copies are turned in the vessel's frame:
    upright, or (sideways: side ports, modules docked to them)
    their own axis straight out from the core (or in).
    """

    if not state.outward:

        if state.flipped:
            return [quaternion.from_axis_angle(np.array([1.0, 0.0, 0.0]), 180.0)] * max(state.count, 1)

        return [np.array([0.0, 0.0, 0.0, 1.0])] * max(state.count, 1)

    if state.part.wheels is not None:
        return [
            quaternion.look_rotation(np.array([0.0, 1.0, 0.0]), np.array([side, 0.0, 0.0]))
            for side, _ in _wheel_places(state.count)
        ]

    base = math.radians(state.angle)

    turned = []

    for k in range(state.count):

        angle = base + 2.0 * math.pi * k / state.count

        out = np.array([math.cos(angle), 0.0, math.sin(angle)]) * state.outward

        turned.append(quaternion.look_rotation(np.array([0.0, 1.0, 0.0]), out))

    return turned


def _wheel_places(
    count: int
) -> list[tuple[float, float]]:
    """(side +1 / -1, row +1 front / -1 back / 0) for each wheel copy."""

    if count <= 2:
        return [(1.0, 0.0), (-1.0, 0.0)][:max(count, 1)]

    places = [(1.0, -1.0), (-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]

    return (places * (count // 4 + 1))[:count]
