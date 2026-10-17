import numpy as np

from core.input import Input
from core.key_codes import Key

from vessel.physics import SAS_MODES


# =========================================================
# Piloting
# =========================================================
#
# The keys of most rocket games:
#
#   Shift / Ctrl    throttle up / down (full range in 1 s)
#   Z / X           full throttle / cut
#   W / S           pitch: nose forward / back
#   A / D           yaw: nose left / right
#   Q / E           roll
#   T               stability assist on / off
#   F               its mode: hold the attitude, point
#                   prograde, point retrograde
#   Space           next stage
#   arrows, H / N   translate (RCS, docking): forward / back,
#                   left / right; along the stack fore / aft
#
# Commands are relative to the vessel: at launch "forward"
# is east, the way a rocket turns to reach orbit.

THROTTLE_RATE = 1.0


def read_pilot(
    vessel,
    controls,
    delta_time: float
) -> bool:
    """
    Apply this frame's keys to the vessel and its controls.
    Returns True when Space asked for the next stage.
    """

    throttle = vessel.throttle

    if Input.is_key_down(Key.LEFT_SHIFT) or Input.is_key_down(Key.RIGHT_SHIFT):
        throttle += THROTTLE_RATE * delta_time

    if Input.is_key_down(Key.LEFT_CONTROL):
        throttle -= THROTTLE_RATE * delta_time

    if Input.is_key_pressed(Key.Z):
        throttle = 1.0

    if Input.is_key_pressed(Key.X):
        throttle = 0.0

    vessel.throttle = min(max(throttle, 0.0), 1.0)

    def axis(negative, positive):

        return float(Input.is_key_down(positive)) - float(Input.is_key_down(negative))

    # Positive pitch turns the nose toward +Z (back); W
    # pushes it forward (-Z). Positive yaw (about +Z) turns
    # it toward -X (left).
    controls.pitch = axis(Key.W, Key.S)
    controls.yaw = axis(Key.D, Key.A)
    controls.roll = axis(Key.Q, Key.E)

    # Forward is the vessel's -Z (as pitching forward).
    controls.translate = np.array([axis(Key.LEFT, Key.RIGHT), axis(Key.N, Key.H), axis(Key.UP, Key.DOWN)])

    if Input.is_key_pressed(Key.T):
        controls.sas = not controls.sas

    if Input.is_key_pressed(Key.F):

        controls.sas = True
        controls.sas_mode = SAS_MODES[(SAS_MODES.index(controls.sas_mode) + 1) % len(SAS_MODES)]

    return Input.is_key_pressed(Key.SPACE)


def release(
    controls
):
    """No steering (the pilot is busy: menu open, free camera)."""

    controls.pitch = controls.yaw = controls.roll = 0.0
    controls.translate = np.zeros(3)
