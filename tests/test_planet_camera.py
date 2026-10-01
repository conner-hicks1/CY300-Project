import numpy as np
import pytest

from ecs.components import CameraControllerComponent
from math3d import quaternion
from math3d.transform import Transform
from systems.camera_controller_system import (
    MAX_ORBIT_RATE,
    altitude,
    clamp_altitude,
    planet_look,
    planet_motion,
    planet_speed,
    planet_up
)


RADIUS = 6_371_000.0


def planet(**overrides):

    values = dict(
        planet_mode=True,
        planet_radius=RADIUS,
        movement_speed=5.0,
        min_altitude=2.0
    )

    values.update(overrides)

    return CameraControllerComponent(**values)


def axes(q):

    m = quaternion.to_matrix3(q)

    return -m[:, 2], m[:, 0], m[:, 1]  # forward, right, up


def test_up_points_away_from_center():

    up = planet_up((0.0, 0.0, RADIUS + 10.0), (0.0, 0.0, 0.0))

    np.testing.assert_allclose(up, (0.0, 0.0, 1.0))


def test_up_at_center_falls_back_to_world_y():

    np.testing.assert_allclose(planet_up((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)), (0.0, 1.0, 0.0))


@pytest.mark.parametrize("up", [
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.577, 0.577, -0.577),
])
def test_look_is_level_with_local_horizon(up):

    up = np.array(up) / np.linalg.norm(up)

    # Some forward with a slight roll relative to this up.
    forward = np.cross(up, (0.3, 0.2, 0.9))
    forward /= np.linalg.norm(forward)

    q = planet_look(forward, up, 0.0, 0.0, -89.0, 89.0)

    f, right, cam_up = axes(q)

    # No roll: right stays on the horizon.
    assert abs(np.dot(right, up)) < 1e-9
    assert np.dot(cam_up, up) > 0.0
    np.testing.assert_allclose(f, forward, atol=1e-9)


def test_yaw_turns_about_up_and_matches_flat_sign():

    up = np.array([0.0, 1.0, 0.0])

    # Positive yaw turns -Z toward -X, like Euler yaw in flat mode.
    q = planet_look((0.0, 0.0, -1.0), up, 90.0, 0.0, -89.0, 89.0)

    f, _, _ = axes(q)

    np.testing.assert_allclose(f, (-1.0, 0.0, 0.0), atol=1e-9)


def test_pitch_is_measured_from_horizon_and_clamped():

    up = np.array([1.0, 0.0, 0.0])

    q = planet_look((0.0, 0.0, -1.0), up, 0.0, 30.0, -89.0, 89.0)

    f, _, _ = axes(q)

    assert np.degrees(np.arcsin(np.dot(f, up))) == pytest.approx(30.0)

    q = planet_look(f, up, 0.0, 500.0, -80.0, 80.0)

    f, _, _ = axes(q)

    assert np.degrees(np.arcsin(np.dot(f, up))) == pytest.approx(80.0)


def test_relevel_after_moving_around_sphere():

    # Start at the north pole looking along +Z, then move to
    # the equator: the old forward now points straight down.
    q = planet_look((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), 0.0, 0.0, -89.0, 89.0)

    f, _, cam_up = axes(q)

    assert np.isfinite(f).all()
    assert np.dot(cam_up, (0.0, 0.0, 1.0)) >= 0.0


def test_altitude_and_speed_scaling():

    controller = planet(altitude_speed=0.5)

    position = (0.0, RADIUS + 1000.0, 0.0)

    assert altitude(controller, position) == pytest.approx(1000.0)
    assert planet_speed(controller, position) == pytest.approx(500.0)

    # Near the ground the base speed is the floor.
    assert planet_speed(controller, (0.0, RADIUS + 1.0, 0.0)) == pytest.approx(5.0)

    # Disabled scaling.
    assert planet_speed(planet(), position) == pytest.approx(5.0)


def test_clamp_altitude_pushes_out_radially():

    controller = planet()

    inside = np.array([RADIUS * 0.5, 0.0, 0.0])

    clamped = clamp_altitude(inside, controller)

    np.testing.assert_allclose(clamped, (RADIUS + 2.0, 0.0, 0.0))

    above = np.array([0.0, RADIUS + 100.0, 0.0])

    np.testing.assert_array_equal(clamp_altitude(above, controller), above)


def test_camera_relative_precision_at_planet_scale():

    # float64 transforms keep sub-millimetre steps at the surface.
    transform = Transform(position=(0.0, RADIUS + 2.0, 0.0))

    transform.translate((0.0005, 0.0, 0.0))

    assert transform.position[0] == pytest.approx(0.0005, abs=1e-9)


# =========================================================
# Planet Motion (WASD / Space / Shift)
# =========================================================

CENTER = np.zeros(3)


def move(position, forward, camera_up, forward_input=0.0, right_input=0.0, up_input=0.0, speed=100.0, dt=1.0):

    return planet_motion(
        position=position,
        center=CENTER,
        forward=forward,
        camera_up=camera_up,
        forward_input=forward_input,
        right_input=right_input,
        up_input=up_input,
        speed=speed,
        delta_time=dt
    )


def test_wasd_keeps_altitude():

    start = np.array([0.0, RADIUS + 500.0, 0.0])

    end, _ = move(start, (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), forward_input=1.0, right_input=1.0, speed=1000.0)

    assert np.linalg.norm(end) == pytest.approx(np.linalg.norm(start))
    assert np.linalg.norm(end - start) == pytest.approx(1000.0, rel=1e-3)


def test_forward_is_the_heading_even_looking_straight_down():

    # In orbit looking at the planet: forward = -up; the
    # top of the screen (camera up) points along -Z.
    start = np.array([0.0, 3.0 * RADIUS, 0.0])

    end, _ = move(start, (0.0, -1.0, 0.0), (0.0, 0.0, -1.0), forward_input=1.0, speed=1e5)

    # Moved toward -Z around the planet, not toward it.
    assert end[2] < 0.0
    assert np.linalg.norm(end) == pytest.approx(np.linalg.norm(start))


def test_pitch_does_not_change_heading():

    start = np.array([0.0, RADIUS + 10.0, 0.0])

    pitch = np.radians(-40.0)

    forward = (0.0, np.sin(pitch), -np.cos(pitch))
    camera_up = (0.0, np.cos(pitch), np.sin(pitch))

    end, _ = move(start, forward, camera_up, forward_input=1.0)
    level, _ = move(start, (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), forward_input=1.0)

    np.testing.assert_allclose(end, level, atol=1e-6)


def test_d_moves_right():

    start = np.array([0.0, RADIUS + 10.0, 0.0])

    end, _ = move(start, (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), right_input=1.0)

    assert end[0] > 0.0 and abs(end[2]) < 1e-6


def test_space_and_shift_move_along_zenith():

    start = np.array([RADIUS + 100.0, 0.0, 0.0])

    up_end, rotation = move(start, (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), up_input=1.0, speed=50.0)
    down_end, _ = move(start, (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), up_input=-1.0, speed=50.0)

    np.testing.assert_allclose(up_end, start + (50.0, 0.0, 0.0))
    np.testing.assert_allclose(down_end, start - (50.0, 0.0, 0.0))

    # No rotation for purely vertical moves.
    np.testing.assert_allclose(rotation, quaternion.identity())


def test_view_turns_with_the_orbit():

    # The returned rotation carries the camera around the
    # planet: what was "down" stays pointing at the center.
    start = np.array([0.0, 3.0 * RADIUS, 0.0])

    end, rotation = move(start, (0.0, -1.0, 0.0), (0.0, 0.0, -1.0), forward_input=1.0, speed=1e6)

    new_forward = quaternion.rotate_vector(rotation, (0.0, -1.0, 0.0))

    np.testing.assert_allclose(new_forward, -end / np.linalg.norm(end), atol=1e-9)


def test_orbit_rate_is_capped():

    start = np.array([0.0, 4.0 * RADIUS, 0.0])

    distance = 4.0 * RADIUS

    end, _ = move(start, (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), forward_input=1.0, speed=1e12)

    angle = np.arccos(np.dot(start, end) / distance ** 2)

    assert angle == pytest.approx(MAX_ORBIT_RATE, rel=1e-6)
