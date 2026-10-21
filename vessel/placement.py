import numpy as np

from math3d import quaternion

from vessel import kepler


# =========================================================
# Putting a Vessel Somewhere
# =========================================================
#
# Straight into a place, not flown there: a circular orbit,
# or high over the ground falling. For the lessons (flight
# school: game/lessons.py) and the tests' headless games.

def circular_orbit(
    state,
    mu: float,
    radius: float,
    direction=(1.0, 0.0, 0.0),
    pole=(0.0, 1.0, 0.0)
):
    """On a circular orbit of this radius (m from the body's centre): at an inertial direction, round a pole; nose prograde."""

    out = np.asarray(direction, dtype=np.float64)
    out = out / np.linalg.norm(out)

    forward = np.cross(np.asarray(pole, dtype=np.float64), out)
    forward = forward / np.linalg.norm(forward)

    state.position = out * radius
    state.velocity = forward * kepler.circular_speed(mu, radius)
    state.landed = False
    state.angular_velocity = np.zeros(3)
    state.orientation = quaternion.look_rotation(-out, forward)


def coming_down(
    state,
    radius: float,
    spin,
    direction=(1.0, 0.0, 0.0),
    falling: float = 40.0,
    sideways: float = 0.0
):
    """Over the ground at this radius, falling (m/s) and moving sideways (m/s) over the turning surface; upright."""

    out = np.asarray(direction, dtype=np.float64)
    out = out / np.linalg.norm(out)

    side = np.cross([0.0, 1.0, 0.0], out)

    if float(np.linalg.norm(side)) < 1e-6:
        side = np.cross([1.0, 0.0, 0.0], out)

    side = side / np.linalg.norm(side)

    state.position = out * radius
    state.velocity = np.cross(np.asarray(spin, dtype=np.float64), state.position) - out * falling + side * sideways
    state.landed = False
    state.angular_velocity = np.zeros(3)
    state.orientation = quaternion.look_rotation(side, out)
