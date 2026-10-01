from dataclasses import dataclass

import numpy as np

from graphics.draw_list import (
    DrawItem,
    build_commands,
    frustum_planes,
    group_by_material,
    prepare,
    spheres_in_frustum
)
from graphics.geometry_pool import PoolAllocation
from graphics.shadows import compute_cascades, to_render_space
from math3d.camera import Camera
from math3d.matrices import normal_matrix
from math3d.transform import Transform


@dataclass
class FakeMesh:

    allocation: PoolAllocation
    bounding_center: np.ndarray
    bounding_radius: float


def unit_mesh(base_vertex=0, first_index=0, index_count=36):

    return FakeMesh(
        allocation=PoolAllocation(base_vertex=base_vertex, vertex_count=24, first_index=first_index, index_count=index_count),
        bounding_center=np.zeros(3),
        bounding_radius=float(np.sqrt(3) / 2)
    )


def test_records_are_camera_relative_column_major():

    transform = Transform(position=(6_371_000.0, 2.0, 3.0), rotation=(10, 20, 30), scale=(2, 1, 0.5))
    origin = np.array([6_371_000.0, 0.0, 0.0])

    prepared = prepare([DrawItem(unit_mesh(), "m", transform.matrix)], origin)

    model = prepared.records[0, :16].reshape(4, 4).T
    normal = prepared.records[0, 16:].reshape(4, 4).T

    expected = np.array(transform.matrix)
    expected[:3, 3] -= origin

    assert np.allclose(model, expected, atol=1e-5)
    assert np.allclose(model[:3, 3], (0.0, 2.0, 3.0))
    assert np.allclose(normal[:3, :3], normal_matrix(transform.matrix), atol=1e-5)


def test_bounding_spheres_follow_transform():

    transform = Transform(position=(10.0, 0.0, 0.0), scale=(1, 4, 1))

    prepared = prepare([DrawItem(unit_mesh(), "m", transform.matrix)], np.zeros(3))

    assert np.allclose(prepared.centers[0], (10.0, 0.0, 0.0))
    assert np.isclose(prepared.radii[0], 4 * np.sqrt(3) / 2)


def test_camera_frustum_culling_with_reversed_infinite_projection():

    camera = Camera(position=(0.0, 0.0, 0.0), target=(0.0, 0.0, -1.0), fov=60.0, aspect_ratio=1.0, near=0.1)

    clip = camera.projection_matrix @ camera.view_rotation_matrix

    planes = frustum_planes(clip)

    centers = np.array([
        [0.0, 0.0, -5.0],          # in front
        [0.0, 0.0, 5.0],           # behind
        [100.0, 0.0, -5.0],        # far to the side
        [0.0, 0.0, -1e7],          # 10,000 km ahead: infinite far plane keeps it
        [0.0, 0.0, 0.5],           # just behind, but radius reaches in front
    ])

    radii = np.array([0.5, 0.5, 0.5, 1.0, 1.0])

    assert spheres_in_frustum(planes, centers, radii).tolist() == [True, False, False, True, True]


def test_cascade_culling_uses_render_space_matrix():

    camera = Camera(position=(50.0, 2.0, 6.0), target=(50.0, 0.0, 0.0), fov=50.0, aspect_ratio=1.0, near=0.1)

    sun = np.array([0.0, -1.0, -0.2])
    sun /= np.linalg.norm(sun)

    cascade = compute_cascades(camera, sun, 1, 20.0, 0.5, 1024)[0]

    planes = frustum_planes(to_render_space(cascade.matrix, camera.position))

    origin = camera.position

    near_object = np.array([50.0, 0.0, 0.0]) - origin
    far_away = np.array([500.0, 0.0, 0.0]) - origin

    assert spheres_in_frustum(planes, np.array([near_object, far_away]), np.array([1.0, 1.0])).tolist() == [True, False]


def test_commands_point_at_their_records():

    items = [
        DrawItem(unit_mesh(base_vertex=0, first_index=0), "a", np.identity(4)),
        DrawItem(unit_mesh(base_vertex=24, first_index=36), "b", np.identity(4)),
        DrawItem(unit_mesh(base_vertex=48, first_index=72, index_count=6), "a", np.identity(4)),
    ]

    prepared = prepare(items, np.zeros(3))

    commands = build_commands(prepared, np.array([2, 0]))

    assert commands.tolist() == [
        [6, 1, 72, 48, 2],
        [36, 1, 0, 0, 0],
    ]


def test_group_by_material_keeps_first_seen_order():

    materials = [object(), object()]

    items = [
        DrawItem(unit_mesh(), materials[0], np.identity(4)),
        DrawItem(unit_mesh(), materials[1], np.identity(4)),
        DrawItem(unit_mesh(), materials[0], np.identity(4)),
    ]

    groups = group_by_material(prepare(items, np.zeros(3)), np.array([0, 1, 2]))

    assert [g[0] for g in groups] == materials
    assert groups[0][1].tolist() == [0, 2]
    assert groups[1][1].tolist() == [1]


def test_empty_draw_list():

    prepared = prepare([], np.zeros(3))

    assert prepared.records.shape == (0, 32)
    assert build_commands(prepared, np.zeros(0, dtype=int)).shape == (0, 5)
