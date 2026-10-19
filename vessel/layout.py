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

    base = math.radians(state.angle)

    turned = []

    for k in range(state.count):

        angle = base + 2.0 * math.pi * k / state.count

        out = np.array([math.cos(angle), 0.0, math.sin(angle)]) * state.outward

        turned.append(quaternion.look_rotation(np.array([0.0, 1.0, 0.0]), out))

    return turned
