import numpy as np
import pytest

from graphics.cubemap import cube_face_direction


def spec_lookup(direction):
    """
    OpenGL spec (section 8.13, cube map texture selection):
    pick the face by the major axis, then
    s = (sc / |ma| + 1) / 2, t = (tc / |ma| + 1) / 2.
    """

    x, y, z = direction
    ax, ay, az = abs(x), abs(y), abs(z)

    if ax >= ay and ax >= az:
        face, sc, tc, ma = (0, -z, -y, ax) if x > 0 else (1, z, -y, ax)
    elif ay >= az:
        face, sc, tc, ma = (2, x, z, ay) if y > 0 else (3, x, -z, ay)
    else:
        face, sc, tc, ma = (4, x, -y, az) if z > 0 else (5, -x, -y, az)

    return face, (sc / ma + 1) / 2, (tc / ma + 1) / 2


@pytest.mark.parametrize("face", range(6))
@pytest.mark.parametrize("u, v", [(0.0, 0.0), (0.5, -0.3), (-0.8, 0.9), (0.99, 0.99)])
def test_face_direction_round_trips_through_spec(face, u, v):

    direction = cube_face_direction(face, u, v)

    looked_up_face, s, t = spec_lookup(direction)

    assert looked_up_face == face

    # Rendering into a face: window x maps to s, window y
    # (row 0 at the bottom) maps to t.
    assert s == pytest.approx((u + 1) / 2)
    assert t == pytest.approx((v + 1) / 2)


def test_face_centers_are_axes():

    axes = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    for face, axis in enumerate(axes):
        assert np.allclose(cube_face_direction(face, 0.0, 0.0), axis)
