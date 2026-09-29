import numpy as np
import pytest

from graphics.shadows import (
    cascade_splits,
    compute_cascades,
    frustum_slice_corners,
    spot_shadow
)
from math3d.camera import Camera


SUN = np.array([-0.35, -0.7, -0.61])
SUN = SUN / np.linalg.norm(SUN)


def camera_at(position, target=(0.0, 0.0, 0.0)):

    return Camera(position=position, target=target, fov=50.0, aspect_ratio=16 / 9, near=0.1, far=100.0)


def test_splits_increase_and_end_at_far():

    splits = cascade_splits(0.1, 30.0, 4, 0.75)

    assert splits == sorted(splits)
    assert splits[-1] == pytest.approx(30.0)


def test_split_lambda_extremes():

    uniform = cascade_splits(1.0, 9.0, 2, 0.0)
    logarithmic = cascade_splits(1.0, 9.0, 2, 1.0)

    assert uniform[0] == pytest.approx(5.0)
    assert logarithmic[0] == pytest.approx(3.0)


def test_each_cascade_contains_its_frustum_slice():

    camera = camera_at((2.0, 3.0, 8.0))

    cascades = compute_cascades(camera, SUN, 4, 30.0, 0.75, 2048)

    near = camera.near

    for cascade in cascades:

        corners = frustum_slice_corners(camera, near, cascade.split_far)

        clip = np.hstack((corners, np.ones((8, 1)))) @ cascade.matrix.T.astype(np.float64)
        ndc = clip[:, :3] / clip[:, 3:4]

        assert np.all(np.abs(ndc) <= 1.0 + 1e-4)

        near = cascade.split_far


def test_texel_size_grows_with_distance():

    cascades = compute_cascades(camera_at((0.0, 2.0, 6.0)), SUN, 4, 30.0, 0.75, 2048)

    sizes = [cascade.texel_world_size for cascade in cascades]

    assert sizes == sorted(sizes)


def test_small_camera_moves_keep_texel_grid_fixed():

    # A fixed world point must stay at the same sub-texel
    # position when the camera translates, or shadow edges
    # crawl ("shimmer").

    size = 2048
    point = np.array([0.3, 0.0, -0.7, 1.0])

    fractions = []

    for dx in (0.0, 0.013, 0.037):

        cascade = compute_cascades(camera_at((dx, 2.0, 6.0), (dx, 0.0, 0.0)), SUN, 4, 30.0, 0.75, size)[0]

        texel = (cascade.matrix.astype(np.float64) @ point)[:2] * (size / 2.0)

        fractions.append(texel - np.floor(texel))

    assert np.allclose(fractions[0], fractions[1], atol=1e-3)
    assert np.allclose(fractions[0], fractions[2], atol=1e-3)


def test_cascade_size_does_not_change_when_camera_turns():

    left = compute_cascades(camera_at((0, 2, 6), (-5, 0, 0)), SUN, 4, 30.0, 0.75, 2048)
    right = compute_cascades(camera_at((0, 2, 6), (5, 1, 0)), SUN, 4, 30.0, 0.75, 2048)

    for a, b in zip(left, right):
        assert a.texel_world_size == pytest.approx(b.texel_world_size)


def test_straight_down_sun_is_supported():

    cascades = compute_cascades(camera_at((0, 5, 5)), (0.0, -1.0, 0.0), 2, 20.0, 0.5, 1024)

    assert all(np.all(np.isfinite(c.matrix)) for c in cascades)


def test_spot_shadow_projects_cone_into_clip_space():

    shadow = spot_shadow((0.0, 3.0, 0.0), (0.0, -1.0, 0.0), 25.0, 8.0, 1024)

    inside_cone = np.array([0.5, 0.0, 0.3, 1.0])   # below the light, within 25 degrees
    behind = np.array([0.0, 5.0, 0.0, 1.0])

    clip = shadow.matrix.astype(np.float64) @ inside_cone
    ndc = clip[:3] / clip[3]

    assert np.all(np.abs(ndc) <= 1.0)
    assert (shadow.matrix.astype(np.float64) @ behind)[3] < 0.0

    assert shadow.texel_scale > 0.0
