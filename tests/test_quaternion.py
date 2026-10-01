import math

import numpy as np
import pytest

from math3d import quaternion
from math3d.matrices import decompose_trs_quaternion
from math3d.transform import Transform


def euler_matrix(x, y, z):
    """Reference Rz @ Ry @ Rx built from elementary rotations."""

    x, y, z = (math.radians(a) for a in (x, y, z))

    rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])

    return rz @ ry @ rx


ANGLES = [(0, 0, 0), (10, 20, 30), (-45, 170, -60), (90, 0, 0), (0, 90, 0), (33, -71, 128)]


@pytest.mark.parametrize("angles", ANGLES)
def test_from_euler_matches_zyx_matrix(angles):

    assert np.allclose(quaternion.to_matrix3(quaternion.from_euler(angles)), euler_matrix(*angles), atol=1e-12)


@pytest.mark.parametrize("angles", ANGLES)
def test_matrix_round_trip(angles):

    q = quaternion.from_euler(angles)

    back = quaternion.from_matrix3(quaternion.to_matrix3(q))

    # q and -q are the same rotation.
    assert quaternion.angle_between(q, back) < 1e-6


@pytest.mark.parametrize("angles", ANGLES)
def test_to_euler_reproduces_the_rotation(angles):

    euler = quaternion.to_euler(quaternion.from_euler(angles))

    assert np.allclose(euler_matrix(*euler), euler_matrix(*angles), atol=1e-9)


def test_multiply_applies_right_operand_first():

    yaw = quaternion.from_axis_angle((0, 1, 0), 90)
    pitch = quaternion.from_axis_angle((1, 0, 0), 90)

    combined = quaternion.multiply(yaw, pitch)

    # Pitch first: forward (-Z) -> up (+Y); then yaw leaves +Y alone.
    assert np.allclose(quaternion.rotate_vector(combined, (0, 0, -1)), (0, 1, 0), atol=1e-12)


def test_look_rotation_points_forward_and_keeps_up():

    q = quaternion.look_rotation((1, -1, 0), (0, 1, 0))

    m = quaternion.to_matrix3(q)

    assert np.allclose(-m[:, 2], np.array([1, -1, 0]) / math.sqrt(2))

    # Local up is perpendicular to forward and leans toward world up.
    assert abs(np.dot(m[:, 1], m[:, 2])) < 1e-12
    assert m[1, 1] > 0.0


def test_look_rotation_straight_up_is_valid():

    q = quaternion.look_rotation((0, 1, 0), (0, 1, 0))

    assert np.allclose(-quaternion.to_matrix3(q)[:, 2], (0, 1, 0))


def test_slerp_halfway():

    a = quaternion.identity()
    b = quaternion.from_axis_angle((0, 1, 0), 90)

    assert quaternion.angle_between(quaternion.slerp(a, b, 0.5), quaternion.from_axis_angle((0, 1, 0), 45)) < 1e-9


# =========================================================
# Transform
# =========================================================

def test_typed_euler_angles_are_kept():

    transform = Transform(rotation=(0, 190, 0))

    # Not the equivalent (180, -10, 180).
    assert np.allclose(transform.rotation, (0, -170, 0))


def test_orientation_setter_updates_euler_view():

    transform = Transform()

    transform.orientation = quaternion.from_axis_angle((0, 1, 0), 30)

    assert np.allclose(transform.rotation, (0, 30, 0), atol=1e-9)
    assert np.allclose(transform.forward, (-0.5, 0, -math.sqrt(3) / 2), atol=1e-12)


def test_rotate_about_axis_world_and_local():

    pitched = Transform(rotation=(-90, 0, 0))   # forward = -Y

    world = Transform(rotation=(-90, 0, 0))
    world.rotate_about_axis((0, 1, 0), 90)       # world Y: spins around forward

    local = Transform(rotation=(-90, 0, 0))
    local.rotate_about_axis((0, 1, 0), 90, world_space=False)

    assert np.allclose(world.forward, pitched.forward, atol=1e-12)
    assert not np.allclose(local.forward, pitched.forward, atol=1e-6)


def test_planet_scale_positions_keep_millimetres():

    radius = 6_371_000.0

    a = Transform(position=(radius, 0, 0))
    b = Transform(position=(radius + 0.001, 0, 0))

    assert b.position[0] - a.position[0] == pytest.approx(0.001, abs=1e-6)
    assert b.matrix[0, 3] - a.matrix[0, 3] == pytest.approx(0.001, abs=1e-6)


def test_decompose_quaternion_round_trip():

    original = Transform(position=(1, 2, 3), rotation=(20, -40, 60), scale=(2, 1, 0.5))

    position, orientation, scale = decompose_trs_quaternion(original.matrix)

    rebuilt = Transform(position=position, orientation=orientation, scale=scale)

    assert np.allclose(rebuilt.matrix, original.matrix, atol=1e-9)


def test_rejects_bad_orientation():

    with pytest.raises(Exception):
        Transform(orientation=(0, 0, 0))
