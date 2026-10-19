import numpy as np
import pytest

from vessel import collision
from vessel.collision import Body, Capsule, closest_points
from vessel.parts import load_design, load_parts
from vessel.vessel import Vessel


def test_closest_points_of_two_segments():

    a, b = closest_points(np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), np.array([1.0, 1.0, -1.0]), np.array([1.0, 1.0, 1.0]))

    np.testing.assert_allclose(a, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(b, [1.0, 1.0, 0.0])

    # Parallel, offset: some pair one apart.
    a, b = closest_points(np.zeros(3), np.array([0.0, 2.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([1.0, 2.0, 0.0]))

    assert float(np.linalg.norm(a - b)) == pytest.approx(1.0)

    # Points (spheres).
    a, b = closest_points(np.zeros(3), np.zeros(3), np.array([3.0, 4.0, 0.0]), np.array([3.0, 4.0, 0.0]))

    assert float(np.linalg.norm(a - b)) == pytest.approx(5.0)


def test_parts_become_capsules_without_their_ports():

    parts = load_parts()

    ferry = Vessel.from_design(load_design("ferry", parts))

    shapes = collision.shapes(ferry)

    names = {shape.name for shape in shapes}

    assert "Docking port, 1.25 m" not in names and len(shapes) == len(ferry.parts) - 1

    # A tall tank: a capsule inside its height, as wide as it.
    tank = next(shape for shape, state in zip(shapes, ferry.parts[1:]) if state.part.id == "tank_2m5_long")

    assert tank.radius == pytest.approx(1.25)
    assert float(np.linalg.norm(tank.b - tank.a)) == pytest.approx(8.0 - 2.5)

    # Side boosters: one capsule a copy, off the axis (two
    # tanks stacked on each of the two).
    heavy = Vessel.from_design(load_design("pathfinder_heavy", parts))

    boosters = [shape for shape in collision.shapes(heavy) if shape.name.startswith("Propellant tank, 1.25 m x 8")]

    assert len(boosters) == 4
    assert all(abs(float(shape.a[0])) > 1.5 or abs(float(shape.a[2])) > 1.5 for shape in boosters)


def _ball(position, velocity, mass=1_000.0, radius=1.0, crash=10.0, static=False):

    return Body(
        capsules=[Capsule(np.zeros(3), np.zeros(3), radius, crash, "ball")],
        position=np.asarray(position, dtype=np.float64),
        rotation=np.eye(3),
        velocity=np.asarray(velocity, dtype=np.float64),
        spin=np.zeros(3),
        mass=mass,
        inertia=(400.0, 400.0),
        center=0.0,
        static=static
    )


def test_a_gentle_bump_bounces_and_keeps_the_momentum():

    a = _ball([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    b = _ball([1.9, 0.0, 0.0], [-1.0, 0.0, 0.0], mass=3_000.0)

    found = collision.contacts(a, b)

    assert found and found[0].depth == pytest.approx(0.1)

    before = a.mass * a.velocity + b.mass * b.velocity

    impact = collision.resolve(a, b, found)

    assert impact.speed == pytest.approx(2.0) and not impact.broke_a and not impact.broke_b

    np.testing.assert_allclose(a.mass * a.velocity + b.mass * b.velocity, before, atol=1e-9)

    # Separating now, at a quarter of the speed they met at.
    assert float(b.velocity[0] - a.velocity[0]) == pytest.approx(2.0 * collision.RESTITUTION)

    # Pushed apart a little.
    assert float(b.position[0] - a.position[0]) > 1.9


def test_too_hard_breaks_them_up():

    a = _ball([0.0, 0.0, 0.0], [12.0, 0.0, 0.0], crash=8.0)
    b = _ball([1.9, 0.0, 0.0], [0.0, 0.0, 0.0], crash=20.0)

    impact = collision.resolve(a, b, collision.contacts(a, b))

    assert impact.broke_a and not impact.broke_b


def test_a_glancing_blow_sets_it_turning():

    # A long rod (a capsule along X) hit at its end.
    rod = Body(
        capsules=[Capsule(np.array([-5.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]), 0.5, 10.0, "rod")],
        position=np.zeros(3),
        rotation=np.eye(3),
        velocity=np.zeros(3),
        spin=np.zeros(3),
        mass=1_000.0,
        inertia=(8_000.0, 8_000.0),
        center=0.0
    )

    ball = _ball([4.5, 0.0, 1.4], [0.0, 0.0, -2.0])

    collision.resolve(ball, rod, collision.contacts(ball, rod))

    assert float(np.linalg.norm(rod.spin)) > 0.01
    assert rod.velocity[2] < 0.0


def test_a_landed_vessel_does_not_move():

    ground = _ball([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], static=True)
    falling = _ball([0.0, 1.9, 0.0], [0.0, -2.0, 0.0])

    collision.resolve(falling, ground, collision.contacts(falling, ground))

    np.testing.assert_allclose(ground.velocity, 0.0)
    np.testing.assert_allclose(ground.position, 0.0)

    assert falling.velocity[1] == pytest.approx(2.0 * collision.RESTITUTION)


def test_apart_they_do_not_touch():

    a = _ball([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    b = _ball([2.5, 0.0, 0.0], [-1.0, 0.0, 0.0])

    assert collision.contacts(a, b) == []
