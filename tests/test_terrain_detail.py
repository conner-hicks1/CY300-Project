import math

from dataclasses import replace

import numpy as np
import pytest

from graphics.scatter import FLOATS, pack_layer, quaternions_from_up
from graphics.scatter_meshes import build_meshes
from graphics.shadows import terrain_shadow, terrain_shadow_radius
from graphics.uniform_blocks import TERRAIN_SHADOW_LAYER, LightingFrame, pack_lights_block
from graphics.lighting import DirectionalLight, LightEnvironment
from planet.bodies import components_for, load_presets
from planet.cube_sphere import ChunkKey, direction_to_face
from planet.erosion import WAVELENGTHS, gullies, relief_gradient
from planet.maps import DATASETS, load_detail_map, load_elevation_map
from planet.scatter import (
    BIOME_ROCK,
    LAYERS,
    SurfaceSettings,
    cell_depth,
    cells_around,
    generate_cell
)
from planet.shape import fibonacci_directions
from planet.terrain import Terrain, TerrainSettings
from systems.planet_system import terrain_settings_for


@pytest.fixture(scope="module")
def presets():

    return load_presets()


@pytest.fixture(scope="module")
def earth(presets):

    return Terrain(terrain_settings_for(components_for(presets["earth"]).planet))


# =========================================================
# Long Shadows
# =========================================================

def test_terrain_shadow_covers_the_land_in_view():

    radius = 6_371_000.0

    # On the ground: the hills around; from orbit: hundreds
    # of km; never more than 1,500 km.
    assert terrain_shadow_radius(2.0, radius) == pytest.approx(20_000.0)
    assert 500_000.0 < terrain_shadow_radius(400_000.0, radius) <= 1_500_000.0
    assert terrain_shadow_radius(1e9, radius) == pytest.approx(1_500_000.0)


def test_terrain_shadow_is_centered():

    center = np.array([1000.0, 2000.0, -500.0])

    shadow = terrain_shadow(center, (0.3, -0.9, 0.2), 50_000.0, 2048)

    clip = shadow.matrix.astype(np.float64) @ np.append(center, 1.0)

    # In the middle of the map (texel snapping moves it by
    # under a texel), in front of the light.
    assert abs(clip[0]) < 2.0 / 2048 and abs(clip[1]) < 2.0 / 2048
    assert -1.0 < clip[2] < 1.0

    assert shadow.texel_world_size == pytest.approx(100_000.0 / 2048)


def test_terrain_shadow_reaches_the_shader():

    lighting = LightEnvironment(directional=DirectionalLight(direction=(0.0, -1.0, 0.0)))

    shadow = terrain_shadow((0.0, 0.0, 0.0), (0.0, -1.0, 0.1), 30_000.0, 1024)

    data = np.frombuffer(pack_lights_block(lighting, LightingFrame(terrain_shadow=shadow)), dtype=np.float32)

    params = data[-4:]

    assert params[0] == pytest.approx(shadow.texel_world_size)
    assert params[1] == 1.0
    assert params[2] == TERRAIN_SHADOW_LAYER

    off = np.frombuffer(pack_lights_block(lighting, LightingFrame()), dtype=np.float32)

    assert off[-3] == 0.0


# =========================================================
# Carved Slopes
# =========================================================

def test_gullies_only_cut_and_follow_the_slope():

    directions = fibonacci_directions(4000)

    radius = 6_371_000.0

    # A steady slope everywhere (rise / run 0.3, tangent).
    reference = np.array([0.0, 1.0, 0.0])

    tangent = np.cross(directions, reference)

    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)

    cut = gullies(directions, radius, 0.3 * tangent, 10.0, 1.0, 7)

    assert np.all(cut <= 1e-9)
    assert cut.min() < -0.1 * WAVELENGTHS[0] * 0.5

    # Flat ground: nothing; no strength: nothing.
    assert np.all(gullies(directions, radius, np.zeros_like(directions), 10.0, 1.0, 7) == 0.0)
    assert np.all(gullies(directions, radius, 0.3 * tangent, 10.0, 0.0, 7) == 0.0)

    # Coarse samples skip what they cannot show.
    assert np.all(gullies(directions, radius, 0.3 * tangent, WAVELENGTHS[0], 1.0, 7) == 0.0)


def test_relief_gradient_of_a_tilted_plane():

    radius = 1_000_000.0

    directions = fibonacci_directions(200)

    # Height rising with z: gradient ~ 1 m per km along +z
    # (projected onto the tangent plane).
    gradient = relief_gradient(lambda d, spacing: d[:, 2] * radius * 0.001, directions, radius, 5_000.0)

    expected = np.array([0.0, 0.0, 0.001]) - directions * (directions[:, 2:3] * 0.001)

    np.testing.assert_allclose(gradient, expected, atol=2e-5)


def test_only_bodies_with_running_water_are_carved(presets):

    assert terrain_settings_for(components_for(presets["earth"]).planet).gullies == 1.0
    assert terrain_settings_for(components_for(presets["mars"]).planet).gullies == 0.5
    assert terrain_settings_for(components_for(presets["moon"]).planet).gullies == 0.0


def test_carving_changes_slopes_not_flats(earth):

    directions = fibonacci_directions(20_000)

    plain = Terrain(replace(earth.settings, gullies=0.0))

    carved = earth.elevation(directions, 100.0)
    smooth = plain.elevation(directions, 100.0)

    difference = carved - smooth

    assert np.all(difference <= 1e-6)
    assert difference.min() < -20.0

    # Coarse chunks are not carved (nothing they could show).
    np.testing.assert_allclose(earth.elevation(directions[:500], 50_000.0), plain.elevation(directions[:500], 50_000.0))


# =========================================================
# Sharper Maps
# =========================================================

@pytest.mark.parametrize("base, detail", [
    ("mola_megdr_16", "mola_megdr_64"),
    ("lola_ldem_16", "lola_ldem_64"),
    ("etopo1_5min", "etopo1_1min"),
])
def test_sharper_maps_agree_with_the_coarse(base, detail):

    assert DATASETS[base].detail == detail

    sharp = load_detail_map(detail)
    coarse = load_elevation_map(base)

    if sharp is None or coarse is None:
        pytest.skip("maps not downloaded (python tools/fetch_maps.py --sharp)")

    directions = fibonacci_directions(5_000)

    # Four times sharper.
    assert sharp.step == pytest.approx(1.0 / 64.0 if "64" in detail else 1.0 / 60.0)

    difference = sharp.sample(directions) - coarse.sample(directions)

    # The same surface: no offset, differences of detail.
    assert abs(float(difference.mean())) < 15.0
    assert float(np.sqrt(np.mean(difference ** 2))) < 150.0


def test_coarse_samples_keep_the_coarse_map():

    coarse = load_elevation_map("lola_ldem_16")

    if coarse is None or coarse.detail is None:
        pytest.skip("maps not downloaded")

    directions = fibonacci_directions(300)

    radius = 1_737_400.0

    resolution = coarse.resolution(radius)

    # At its own resolution: itself; finer: the sharper map.
    np.testing.assert_allclose(coarse.sample(directions, resolution, radius), coarse.sample(directions))

    np.testing.assert_allclose(
        coarse.sample(directions, 0.4 * resolution, radius),
        coarse.detail.sample(directions)
    )


# =========================================================
# Rocks, Trees, Grass
# =========================================================

EARTH_SURFACE = SurfaceSettings(life=True, has_liquid=True, mineral=False, frost_point=0.0, rock_color=BIOME_ROCK)


def _find(terrain, test, samples=20_000):

    directions = fibonacci_directions(samples)

    elevation = terrain.elevation(directions, 20_000.0)

    temperature, precipitation = terrain.surface_climate(directions, elevation)

    wetness = precipitation / (300.0 + 30.0 * np.maximum(temperature, 0.0))

    index = np.flatnonzero(test(elevation, temperature, wetness))

    assert len(index), "no such place"

    return directions[index[0]]


def _cell(terrain, direction, layer):

    depth = cell_depth(terrain.settings.radius, layer.cell)

    face, a, b = direction_to_face(direction)

    cells = 1 << depth

    return ChunkKey(face, depth, int((a + 1.0) * 0.5 * cells), int((b + 1.0) * 0.5 * cells))


def test_cells_around_the_camera():

    radius = 6_371_000.0

    direction = np.array([0.3, 0.8, 0.52])
    direction /= np.linalg.norm(direction)

    for layer in LAYERS:

        depth = cell_depth(radius, layer.cell)

        keys = cells_around(radius, direction, layer.reach, depth)

        # Cells about the layer's size, enough to cover its
        # reach, the camera's own among them.
        edge = math.pi * 0.5 * radius / (1 << depth)

        assert 0.6 * layer.cell < edge < 1.6 * layer.cell
        assert len(keys) >= math.pi * (layer.reach / edge) ** 2 * 0.8

        face, a, b = direction_to_face(direction)

        assert any(k.face == face for k in keys)


def test_forests_grow_trees(earth):

    forest = _find(earth, lambda e, t, w: (e > 100.0) & (w > 1.8) & (t > 6.0))

    trees = LAYERS[0]

    cell = generate_cell(earth, EARTH_SURFACE, trees, _cell(earth, forest, trees))

    edge = math.pi * 0.5 * earth.settings.radius / (1 << cell.key.depth)

    # Dense: hundreds per km^2.
    assert len(cell) > 100 * (edge / 1000.0) ** 2

    # Trees 5-38 m, standing on the ground (sunk ~1 m).
    assert cell.scale.min() >= 5.0 and cell.scale.max() <= 38.0

    directions = cell.positions / np.linalg.norm(cell.positions, axis=1, keepdims=True)

    # (On the body's shape: Earth is flattened.)
    ground = earth.settings.radius + earth.base_height(directions) + earth.elevation(directions, 10.0)

    height = np.linalg.norm(cell.positions, axis=1) - ground

    assert np.median(np.abs(height)) < 8.0

    # Deterministic.
    again = generate_cell(earth, EARTH_SURFACE, trees, cell.key)

    np.testing.assert_array_equal(cell.positions, again.positions)


def test_no_trees_without_life(earth):

    forest = _find(earth, lambda e, t, w: (e > 100.0) & (w > 1.8) & (t > 6.0))

    trees = LAYERS[0]

    lifeless = replace(EARTH_SURFACE, life=False)

    assert len(generate_cell(earth, lifeless, trees, _cell(earth, forest, trees))) == 0


def test_bare_worlds_have_rocks_and_nothing_else(presets):

    moon = Terrain(terrain_settings_for(components_for(presets["moon"]).planet))

    surface = SurfaceSettings(life=False, has_liquid=False, mineral=True, frost_point=-300.0, rock_color=(0.1, 0.1, 0.1))

    direction = np.array([0.0, 0.0, 1.0])

    counts = {layer.name: len(generate_cell(moon, surface, layer, _cell(moon, direction, layer))) for layer in LAYERS}

    assert counts["rocks"] > 0
    assert counts["trees"] == 0 and counts["grass"] == 0


def test_instances_stand_up():

    up = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]])

    q = quaternions_from_up(up, np.array([0.3, 1.0, -2.0, 0.5]))

    # Rotating local +y gives `up`.
    y = np.array([0.0, 1.0, 0.0])

    for quat, expected in zip(q, up):

        v = y + 2.0 * np.cross(quat[:3], np.cross(quat[:3], y) + quat[3] * y)

        np.testing.assert_allclose(v, expected, atol=1e-9)


def test_packing_sorts_into_blocks(earth):

    forest = _find(earth, lambda e, t, w: (e > 100.0) & (w > 1.8) & (t > 6.0))

    trees = LAYERS[0]

    cell = generate_cell(earth, EARTH_SURFACE, trees, _cell(earth, forest, trees))

    anchor = cell.positions.mean(axis=0)

    data = pack_layer([cell], anchor, 80.0)

    assert data.instances.shape == (len(cell), FLOATS)
    assert data.counts.sum() == len(cell)

    # Contiguous blocks, one kind each.
    np.testing.assert_array_equal(data.starts, np.concatenate(([0], np.cumsum(data.counts)[:-1])))

    # Small offsets from the anchor (float32 stays exact).
    assert np.abs(data.instances[:, :3]).max() < 2_000.0

    # Every instance inside its block's sphere.
    for start, count, center, radius in zip(data.starts, data.counts, data.centers, data.radii):

        offsets = data.instances[start:start + count, :3] - center

        assert np.linalg.norm(offsets, axis=1).max() <= radius + 1e-3


def test_scatter_meshes():

    meshes = build_meshes()

    # Every tree kind near and far, rocks and grass.
    for kind in range(4):
        assert (kind, "near") in meshes and (kind, "far") in meshes

    for (kind, detail), mesh in meshes.items():

        positions = mesh.vertices[:, :3]

        if kind < 4:
            # Trees: 1 unit tall, standing on 0.
            assert positions[:, 1].min() == pytest.approx(0.0, abs=1e-6)
            assert 0.85 < positions[:, 1].max() < 1.1

        # Far trees are cheap.
        if detail == "far":
            assert len(mesh.indices) // 3 <= 60
