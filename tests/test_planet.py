import math
import time

import numpy as np
import pytest

from planet.chunk import build_chunk
from planet.cube_sphere import (
    FACE_NORMALS,
    FACE_U,
    FACE_V,
    ROOT_KEYS,
    ChunkKey,
    direction_to_face,
    edge_length,
    face_directions
)
from planet.lod import LodSelector, _above_horizon
from planet.noise import Perlin, fbm, octaves_for_spacing, ridged
from planet.spawn import fibonacci_sphere, find_spawn
from planet.terrain import Terrain, TerrainSettings


RADIUS = 6_371_000.0

SMALL = TerrainSettings(radius=RADIUS)


@pytest.fixture(scope="module")
def terrain():

    return Terrain(SMALL)


# =========================================================
# Noise
# =========================================================

def test_perlin_is_deterministic_and_seeded():

    points = np.random.default_rng(0).uniform(-50, 50, (500, 3))

    a = Perlin(3)(points)

    np.testing.assert_array_equal(a, Perlin(3)(points))

    assert not np.allclose(a, Perlin(4)(points))


def test_perlin_range_and_lattice_zeros():

    points = np.random.default_rng(1).uniform(-100, 100, (20_000, 3))

    values = Perlin(1)(points)

    assert np.abs(values).max() <= 1.1
    assert values.std() > 0.1

    # Gradient noise is zero at lattice points.
    lattice = np.random.default_rng(2).integers(-50, 50, (100, 3)).astype(float)

    np.testing.assert_allclose(Perlin(1)(lattice), 0.0, atol=1e-12)


def test_perlin_is_continuous():

    rng = np.random.default_rng(5)

    points = rng.uniform(-20, 20, (2000, 3))

    step = rng.normal(size=(2000, 3)) * 1e-5

    noise = Perlin(7)

    assert np.abs(noise(points + step) - noise(points)).max() < 1e-3


def test_fbm_and_ridged_ranges():

    points = np.random.default_rng(3).uniform(-10, 10, (5000, 3))

    noise = Perlin(2)

    assert np.abs(fbm(noise, points, 8)).max() <= 1.1

    r = ridged(noise, points, 8)

    assert r.min() >= 0.0 and r.max() <= 1.0


def test_octaves_follow_spacing():

    assert octaves_for_spacing(1000.0, 0.0, 10) == 10
    assert octaves_for_spacing(1000.0, 1000.0, 10) == 1

    counts = [octaves_for_spacing(1000.0, s, 20) for s in (100.0, 10.0, 1.0)]

    assert counts == sorted(counts) and counts[0] < counts[-1]


# =========================================================
# Cube-Sphere
# =========================================================

def test_face_bases_are_right_handed():

    np.testing.assert_allclose(np.cross(FACE_U, FACE_V), FACE_NORMALS)


def test_face_coordinates_round_trip():

    rng = np.random.default_rng(4)

    for face in range(6):

        a, b = rng.uniform(-0.99, 0.99, 2)

        direction = face_directions(face, a, b)

        assert np.linalg.norm(direction) == pytest.approx(1.0)

        found, fa, fb = direction_to_face(direction)

        assert (found, fa, fb) == (face, pytest.approx(a), pytest.approx(b))


def test_children_tile_parent():

    key = ChunkKey(2, 3, 5, 6)

    a0, a1, b0, b1 = key.bounds()

    children = key.children()

    assert all(child.parent == key for child in children)

    child_bounds = np.array([child.bounds() for child in children])

    assert child_bounds[:, 0].min() == pytest.approx(a0)
    assert child_bounds[:, 1].max() == pytest.approx(a1)
    assert child_bounds[:, 2].min() == pytest.approx(b0)
    assert child_bounds[:, 3].max() == pytest.approx(b1)

    area = sum((c[1] - c[0]) * (c[3] - c[2]) for c in child_bounds)

    assert area == pytest.approx((a1 - a0) * (b1 - b0))


def test_faces_meet_at_edges():

    # The +X face's a = -1 edge (u = -Z, so toward +Z) is
    # the +Z face's a = +1 edge.
    b = np.linspace(-1, 1, 9)

    np.testing.assert_allclose(
        face_directions(0, -1.0, b),
        face_directions(4, 1.0, b),
        atol=1e-12
    )


# =========================================================
# Chunks
# =========================================================

def test_chunk_mesh_layout(terrain):

    n = 9

    data = build_chunk(ChunkKey(4, 2, 1, 2), terrain, n)

    edge = 4 * (n - 1)

    assert data.mesh.vertex_count == n * n + edge

    # Grid quads + double-sided skirt quads.
    assert data.mesh.triangle_count == 2 * (n - 1) ** 2 + 4 * edge


def test_chunk_faces_outward_and_stays_above_sea(terrain):

    n = 9

    data = build_chunk(ChunkKey(2, 4, 7, 8), terrain, n)

    positions = data.mesh.positions[: n * n].astype(np.float64) + data.center

    radial = np.linalg.norm(positions, axis=1)

    assert radial.min() >= RADIUS - 1.0          # oceans sit at sea level

    triangles = data.mesh.indices[: 6 * (n - 1) ** 2].reshape(-1, 3)

    p0, p1, p2 = (positions[triangles[:, i]] for i in range(3))

    normals = np.cross(p1 - p0, p2 - p0)

    outward = np.einsum("ij,ij->i", normals, p0)

    assert (outward > 0).all()


def test_adjacent_chunks_share_edges(terrain):

    # Same depth, side by side: identical edge positions
    # and normals (no cracks, no lighting seams).
    n = 9

    left = build_chunk(ChunkKey(4, 3, 2, 5), terrain, n)
    right = build_chunk(ChunkKey(4, 3, 3, 5), terrain, n)

    def grid(data, attribute):

        values = getattr(data.mesh, attribute)[: n * n].astype(np.float64)

        return values.reshape(n, n, -1)

    left_edge = grid(left, "positions")[:, -1] + left.center
    right_edge = grid(right, "positions")[:, 0] + right.center

    np.testing.assert_allclose(left_edge, right_edge, atol=0.05)

    left_normals = grid(left, "normals")[:, -1]
    right_normals = grid(right, "normals")[:, 0]

    np.testing.assert_allclose(left_normals, right_normals, atol=1e-3)


def test_chunk_vertices_are_local(terrain):

    # Deep chunks keep small float32 coordinates.
    data = build_chunk(ChunkKey(0, 14, 9000, 7000), terrain, 9)

    assert np.abs(data.mesh.positions).max() < 10_000.0
    assert np.linalg.norm(data.center) == pytest.approx(RADIUS)


# =========================================================
# Level of Detail
# =========================================================

def selector(max_depth=10, split_factor=1.5):

    return LodSelector(RADIUS, SMALL.max_elevation, max_depth, split_factor)


def ancestors(key):

    while key.parent is not None:
        key = key.parent
        yield key


def test_far_camera_draws_six_roots():

    selection = selector().select(np.array([0.0, 0.0, 10 * RADIUS]), lambda key: True)

    # The far side is hidden by the horizon; the visible
    # faces draw at depth 0.
    assert all(key.depth == 0 for key in selection.draw)
    assert 1 <= len(selection.draw) <= 6


def test_near_camera_refines_toward_camera():

    camera = np.array([0.0, RADIUS + 1000.0, 0.0])

    selection = selector().select(camera, lambda key: True)

    draw = selection.draw

    assert max(key.depth for key in draw) == 10

    # No node is drawn together with one of its ancestors.
    drawn = set(draw)

    assert not any(parent in drawn for key in draw for parent in ancestors(key))

    # The deepest nodes are the ones under the camera.
    deepest = [key for key in draw if key.depth == 10]

    for key in deepest:

        a0, a1, b0, b1 = key.bounds()

        center = face_directions(key.face, (a0 + a1) / 2, (b0 + b1) / 2) * RADIUS

        assert np.linalg.norm(center - camera) < 5 * edge_length(RADIUS, 10)


def test_nothing_ready_requests_roots_first():

    selection = selector().select(np.array([0.0, RADIUS + 1000.0, 0.0]), lambda key: False)

    assert selection.draw == []

    # Visible roots (some are below the horizon) come
    # before anything finer.
    depths = [key.depth for key in selection.wanted]

    assert depths == sorted(depths)
    assert selection.wanted[0] in ROOT_KEYS


def test_parent_draws_until_all_children_ready():

    camera = np.array([0.0, RADIUS + 1000.0, 0.0])

    top = ChunkKey(2, 0, 0, 0)
    kids = top.children()

    # Three of four children ready: the parent still draws.
    ready = set(ROOT_KEYS) | set(kids[:3])

    selection = selector().select(camera, ready.__contains__)

    assert top in selection.draw
    assert kids[3] in selection.wanted
    assert not set(kids) & set(selection.draw)

    # All four: they replace it.
    ready.add(kids[3])

    selection = selector().select(camera, ready.__contains__)

    assert top not in selection.draw
    assert set(kids) <= set(selection.draw)


def test_horizon_hides_far_side():

    camera = np.array([0.0, RADIUS + 100.0, 0.0])

    points = np.array([
        [0.0, RADIUS, 50_000.0],     # nearby on the surface
        [0.0, -RADIUS, 0.0],         # antipode
        [RADIUS, 0.0, 0.0],          # 90 degrees around
    ])

    visible = _above_horizon(points, camera, RADIUS)

    assert visible.tolist() == [True, False, False]

    # A tall enough mountain peeks over the horizon.
    horizon = math.sqrt((RADIUS + 100.0) ** 2 - RADIUS ** 2)

    angle = horizon / RADIUS * 1.2

    peak = np.array([0.0, math.cos(angle), math.sin(angle)]) * (RADIUS + 9000.0)

    assert _above_horizon(peak[None], camera, RADIUS)[0]


def test_selection_is_fast_enough():

    sel = selector(max_depth=15)

    camera = np.array([0.0, RADIUS + 500.0, 0.0])

    sel.select(camera, lambda key: True)

    start = time.perf_counter()

    sel.select(camera, lambda key: True)

    assert time.perf_counter() - start < 0.05


# =========================================================
# Terrain and Spawn
# =========================================================

def test_terrain_is_deterministic_and_bounded(terrain):

    directions = fibonacci_sphere(4000)

    elevation = terrain.elevation(directions, 50_000.0)

    np.testing.assert_array_equal(elevation, Terrain(SMALL).elevation(directions, 50_000.0))

    assert elevation.max() <= SMALL.max_elevation

    # Both oceans and land.
    assert 0.2 < (elevation > 0).mean() < 0.9

    assert not np.allclose(
        elevation,
        Terrain(TerrainSettings(seed=2)).elevation(directions, 50_000.0)
    )


def test_moisture_is_in_range(terrain):

    moisture = terrain.moisture(fibonacci_sphere(1000))

    assert moisture.min() >= 0.0 and moisture.max() <= 1.0
    assert moisture.std() > 0.05


def test_chunk_carries_terrain_inputs(terrain):

    # Vertex color = (elevation, slope, precipitation),
    # uv.x = temperature, for the per-pixel shading.
    n = 9

    key = ChunkKey(2, 3, 4, 4)

    data = build_chunk(key, terrain, n)

    vertices = data.mesh.vertices[: n * n]

    elevation, slope, precipitation = vertices[:, 6], vertices[:, 7], vertices[:, 8]
    temperature = vertices[:, 9]

    positions = vertices[:, 0:3].astype(np.float64) + data.center

    directions = positions / np.linalg.norm(positions, axis=1, keepdims=True)

    expected = terrain.elevation(directions, edge_length(RADIUS, 3) / (n - 1))

    # Sea level clamps positions, not the stored elevation.
    np.testing.assert_allclose(elevation, expected, atol=60.0)

    assert slope.min() > 0.0 and slope.max() <= 1.0 + 1e-6

    expected_temperature, expected_precipitation = terrain.surface_climate(directions, elevation)

    np.testing.assert_allclose(temperature, expected_temperature, atol=0.05)
    np.testing.assert_allclose(precipitation, expected_precipitation, rtol=1e-4)

    assert precipitation.min() > 0.0
    assert -60.0 < temperature.min() and temperature.max() < 45.0


def test_spawn_is_on_land_facing_horizontally(terrain):

    spawn = find_spawn(terrain, samples=5000)

    assert spawn.elevation > 0.0
    assert np.linalg.norm(spawn.direction) == pytest.approx(1.0)
    assert np.linalg.norm(spawn.view_direction) == pytest.approx(1.0)
    assert abs(np.dot(spawn.direction, spawn.view_direction)) < 1e-9
