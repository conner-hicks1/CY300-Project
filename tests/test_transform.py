import numpy as np
import pytest

from math3d.transform import Transform


def test_identity_matrix():

    assert np.allclose(Transform().matrix, np.identity(4))


def test_forward_follows_pitch_and_yaw():

    # Pitch -45 looks down, yaw 30 turns toward -X.
    forward = Transform(rotation=(-45.0, 30.0, 0.0)).forward

    assert np.allclose(forward, (-0.3535534, -0.7071068, -0.6123725), atol=1e-6)


def test_pitch_minus_90_points_down():

    assert np.allclose(Transform(rotation=(-90.0, 0.0, 0.0)).forward, (0.0, -1.0, 0.0), atol=1e-6)


def test_components_are_read_only():

    transform = Transform()

    with pytest.raises(ValueError):
        transform.position[0] = 1.0

    with pytest.raises(ValueError):
        transform.rotation[1] += 5.0


def test_setters_invalidate_cached_matrix():

    transform = Transform()
    before = transform.matrix

    transform.position = (1.0, 2.0, 3.0)

    assert transform.matrix is not before
    assert np.allclose(transform.matrix[:3, 3], (1.0, 2.0, 3.0))


def test_matrix_is_cached_between_changes():

    transform = Transform(position=(1.0, 0.0, 0.0))

    assert transform.matrix is transform.matrix


def test_translate_and_rotate_helpers():

    transform = Transform()

    transform.translate((1.0, 0.0, 0.0))
    transform.translate((0.0, 2.0, 0.0))
    transform.rotate((0.0, 10.0, 0.0))

    assert np.allclose(transform.position, (1.0, 2.0, 0.0))
    assert np.allclose(transform.rotation, (0.0, 10.0, 0.0))


@pytest.mark.parametrize(
    ("angle", "wrapped"),
    [
        (0.0, 0.0),
        (179.0, 179.0),
        (180.0, -180.0),
        (370.0, 10.0),
        (-190.0, 170.0),
        (7200.5, 0.5),
    ],
)
def test_rotation_wraps_to_half_open_range(angle, wrapped):

    transform = Transform(rotation=(0.0, angle, 0.0))

    assert transform.rotation[1] == pytest.approx(wrapped, abs=1e-3)


def test_wrapping_preserves_orientation():

    a = Transform(rotation=(10.0, 400.0, -30.0))
    b = Transform(rotation=(10.0, 40.0, -30.0))

    assert np.allclose(a.matrix, b.matrix, atol=1e-5)


def test_normal_matrix_non_uniform_scale():

    # Plane scaled like the floor: normals stay +Y.
    transform = Transform(scale=(10.0, 1.0, 10.0))

    normal = transform.normal_matrix @ np.array([0.0, 1.0, 0.0])

    assert np.allclose(normal / np.linalg.norm(normal), (0.0, 1.0, 0.0))


def test_normal_matrix_zero_scale_does_not_raise():

    transform = Transform(scale=(1.0, 0.0, 1.0))

    normal = transform.normal_matrix @ np.array([0.0, 1.0, 0.0])

    assert np.allclose(normal / np.linalg.norm(normal), (0.0, 1.0, 0.0))


def test_rejects_non_finite_values():

    with pytest.raises(Exception):
        Transform(position=(float("nan"), 0.0, 0.0))
