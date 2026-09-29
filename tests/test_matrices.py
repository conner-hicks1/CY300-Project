import numpy as np
import pytest

from math3d.camera import Camera
from math3d.matrices import (
    look_at,
    normal_matrix,
    orthographic,
    perspective
)


def test_look_at_maps_eye_to_origin_and_target_to_minus_z():

    view = look_at((1.0, 2.0, 3.0), (1.0, 2.0, 0.0), (0.0, 1.0, 0.0))

    eye = view @ np.array([1.0, 2.0, 3.0, 1.0])
    target = view @ np.array([1.0, 2.0, 0.0, 1.0])

    assert np.allclose(eye[:3], 0.0, atol=1e-6)
    assert np.allclose(target[:3], (0.0, 0.0, -3.0), atol=1e-6)


def test_look_at_rejects_parallel_up():

    with pytest.raises(Exception):
        look_at((0.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))


def test_camera_uses_shared_helpers():

    camera = Camera(position=(0.0, 1.0, 5.0), target=(0.0, 0.0, 0.0), fov=60.0, aspect_ratio=1.5)

    assert np.allclose(camera.view_matrix, look_at((0.0, 1.0, 5.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    assert np.allclose(camera.projection_matrix, perspective(60.0, 1.5, 0.1, 100.0))


def test_perspective_maps_near_and_far_planes_to_ndc():

    projection = perspective(90.0, 1.0, 1.0, 10.0)

    near = projection @ np.array([0.0, 0.0, -1.0, 1.0])
    far = projection @ np.array([0.0, 0.0, -10.0, 1.0])

    assert near[2] / near[3] == pytest.approx(-1.0)
    assert far[2] / far[3] == pytest.approx(1.0)


def test_orthographic_maps_box_corners_to_ndc_cube():

    projection = orthographic(-2.0, 2.0, -1.0, 1.0, 0.5, 5.0)

    low = projection @ np.array([-2.0, -1.0, -0.5, 1.0])
    high = projection @ np.array([2.0, 1.0, -5.0, 1.0])

    assert np.allclose(low[:3], (-1.0, -1.0, -1.0))
    assert np.allclose(high[:3], (1.0, 1.0, 1.0))


def test_normal_matrix_matches_inverse_transpose_direction():

    rng = np.random.default_rng(7)

    for _ in range(20):

        model = np.identity(4)
        model[:3, :3] = rng.normal(size=(3, 3))

        expected = np.linalg.inv(model[:3, :3]).T
        actual = normal_matrix(model)

        # Same up to a positive scale factor.
        vector = rng.normal(size=3)

        a = expected @ vector
        b = actual @ vector

        assert np.allclose(a / np.linalg.norm(a), b / np.linalg.norm(b), atol=1e-4)


def test_normal_matrix_keeps_direction_for_mirroring():

    model = np.diag([-1.0, 1.0, 1.0, 1.0])

    normal = normal_matrix(model) @ np.array([1.0, 0.0, 0.0])

    assert normal[0] < 0.0


@pytest.mark.parametrize(
    ("position", "rotation", "scale"),
    [
        ((0, 0, 0), (0, 0, 0), (1, 1, 1)),
        ((1, -2, 3), (10, 20, 30), (2, 0.5, 3)),
        ((0, 1, 0), (-45, 170, -60), (1, 1, 1)),
        ((5, 5, 5), (0, 90, 0), (1, 2, 1)),
        ((0, 0, 0), (30, 0, 0), (0.25, 4, 1)),
    ],
)
def test_decompose_round_trips_through_transform(position, rotation, scale):

    from math3d.matrices import decompose_trs
    from math3d.transform import Transform

    original = Transform(position=position, rotation=rotation, scale=scale)

    p, r, s = decompose_trs(original.matrix)

    rebuilt = Transform(position=p, rotation=r, scale=s)

    # Euler angles are not unique; the matrices must match.
    assert np.allclose(rebuilt.matrix, original.matrix, atol=1e-4)
    assert np.allclose(s, scale, atol=1e-5)


def test_decompose_gimbal_lock():

    from math3d.matrices import decompose_trs
    from math3d.transform import Transform

    original = Transform(rotation=(25, 90, 40))

    p, r, s = decompose_trs(original.matrix)

    assert np.allclose(Transform(rotation=r).matrix, original.matrix, atol=1e-4)


def test_decompose_mirror_uses_negative_x_scale():

    from math3d.matrices import decompose_trs

    p, r, s = decompose_trs(np.diag([1.0, 1.0, -1.0, 1.0]))

    assert s[0] < 0.0
    assert np.prod(s) < 0.0
