import math

import numpy as np
import pytest

from planet.bodies import components_for, detail_depth, load_presets
from planet.chunk import build_chunk
from planet.cube_sphere import ChunkKey
from planet.lod import LodSelector
from planet.shape import (
    BodyShape,
    ShapeSettings,
    direction_of,
    fibonacci_directions,
    flat_basins,
    shape_settings,
    unflat_basins
)
from planet.terrain import Terrain, TerrainSettings, body_shape, oblate_offset
from systems.planet_system import planet_extent, terrain_settings_for


RADIUS = 100_000.0


@pytest.fixture(scope="module")
def presets():

    return load_presets()


# =========================================================
# Shapes
# =========================================================

def test_a_plain_sphere_has_no_shape():

    assert shape_settings() is None
    assert body_shape(TerrainSettings(radius=RADIUS)) is None


def test_flattening_matches_the_oblate_ellipsoid():

    shape = BodyShape(shape_settings(oblateness=0.1), RADIUS)

    directions = fibonacci_directions(500)

    np.testing.assert_allclose(shape.height(directions), oblate_offset(directions, RADIUS, 0.1), atol=1e-6)

    assert shape.spheroid


def test_triaxial_axes():

    shape = BodyShape(shape_settings(axes=(1.2, 0.6, 0.9)), RADIUS)

    axes = np.eye(3)

    np.testing.assert_allclose(RADIUS + shape.height(axes), (1.2 * RADIUS, 0.6 * RADIUS, 0.9 * RADIUS))

    assert not shape.spheroid


def test_two_lobes_make_one_connected_body():

    settings = shape_settings(
        axes=(1.0, 0.6, 1.2),
        main_center=(0.0, 0.0, -0.4),
        lobe_center=(0.2, 0.1, 1.1),
        lobe_axes=(0.8, 0.6, 0.7)
    )

    shape = BodyShape(settings, RADIUS)

    directions = fibonacci_directions(3000)

    r = 1.0 + shape.height(directions) / RADIUS

    assert np.all(np.isfinite(r)) and r.min() > 0.3

    # Toward the small lobe: out to its far side.
    toward = np.array([[0.2, 0.1, 1.1]]) / np.linalg.norm([0.2, 0.1, 1.1])

    assert 1.0 + shape.height(toward)[0] / RADIUS > 1.6

    # A surface without cliffs: neighbouring directions have
    # nearby radii (the neck is filled smoothly).
    spacing = math.sqrt(4.0 * math.pi / len(directions))

    nearest = np.argsort(directions @ directions.T, axis=1)[:, -2]

    assert np.max(np.abs(r - r[nearest])) < 30.0 * spacing


def test_basins():

    basins = [(-75.0, -59.0, 50_000.0, 4_000.0, 3_000.0)]

    assert unflat_basins(flat_basins(basins)) == [tuple(basins[0])]

    shape = BodyShape(shape_settings(basins=flat_basins(basins)), RADIUS)

    center = direction_of(-75.0, -59.0)[None, :]

    # Floor plus central peak; untouched far away.
    assert shape.height(center)[0] == pytest.approx(-4_000.0 + 3_000.0, abs=1.0)
    assert shape.height(-center)[0] == pytest.approx(0.0)


def test_bounds_hold_every_height():

    settings = shape_settings(axes=(1.1, 0.8, 1.2), lumpiness=0.08, seed=4)

    shape = BodyShape(settings, RADIUS)

    heights = shape.height(fibonacci_directions(20_000))

    assert shape.min_height <= heights.min()
    assert heights.max() <= shape.max_height
    assert shape.extent == pytest.approx(RADIUS + shape.max_height)


# =========================================================
# Terrain on a Shape
# =========================================================

def flat_terrain(shape):

    return Terrain(
        TerrainSettings(
            radius=RADIUS,
            has_liquid=False,
            continent_height=0.0,
            mountain_height=0.0,
            detail_height=0.0,
            shape=shape
        )
    )


def test_chunks_stand_on_the_shape():

    terrain = flat_terrain(ShapeSettings(axes=(1.3, 0.7, 1.0)))

    data = build_chunk(ChunkKey(0, 1, 1, 0), terrain, 9)

    points = data.mesh.positions[:81].astype(np.float64) + data.center

    directions = points / np.linalg.norm(points, axis=1, keepdims=True)

    np.testing.assert_allclose(
        np.linalg.norm(points, axis=1),
        RADIUS + terrain.base_height(directions),
        rtol=1e-6
    )

    # No relief: flat ground everywhere, though the shape's
    # surface is tilted from "away from the center".
    slope = data.mesh.vertices[:81, 7]     # color.g

    assert slope.min() > 0.99

    normals = data.mesh.normals[:81]

    assert np.einsum("ij,ij->i", normals, directions).min() < 0.95


def test_lod_measures_distance_to_the_shape():

    terrain = flat_terrain(ShapeSettings(axes=(2.0, 1.0, 1.0)))

    selector = LodSelector(
        RADIUS,
        terrain.max_elevation + terrain.base_bounds[1],
        8,
        1.5,
        terrain.min_elevation + terrain.base_bounds[0],
        shape=terrain.base_height
    )

    # Just above the tip of the long axis (x = 2 R): deep
    # detail there, though a sphere would be a radius away.
    selection = selector.select(np.array([2.02 * RADIUS, 0.0, 0.0]), lambda key: True)

    assert max(key.depth for key in selection.draw) >= 6


# =========================================================
# Bodies
# =========================================================

def test_small_bodies(presets):

    phobos = components_for(presets["phobos"]).planet

    # 27 x 22 x 18 km, long axis toward Mars (z).
    assert max(phobos.shape_axes) == pytest.approx(phobos.shape_axes[2])
    assert phobos.shape_axes[1] == min(phobos.shape_axes)

    assert planet_extent(phobos) > 1.1 * phobos.radius

    comet = components_for(presets["churyumov_gerasimenko"]).planet

    assert min(comet.lobe_axes) > 0.0

    vesta = components_for(presets["vesta"]).planet

    assert len(unflat_basins(vesta.basins)) == 2

    # Haumea: a stretched egg, with a ring.
    haumea = components_for(presets["haumea"])

    assert max(haumea.planet.shape_axes) / min(haumea.planet.shape_axes) > 1.9
    assert haumea.rings is not None

    assert components_for(presets["ceres"]).planet.oblateness == pytest.approx(0.075)


def test_detail_depth_follows_size():

    assert detail_depth(6_371_000.0) == 15
    assert detail_depth(1_650.0) < detail_depth(11_080.0) < 15


def test_shapes_reach_terrain_settings(presets):

    vesta = components_for(presets["vesta"]).planet

    settings = terrain_settings_for(vesta)

    assert settings.shape is not None

    terrain = Terrain(settings)

    low, high = terrain.base_bounds

    # Rheasilvia digs ~19 km into the south.
    assert low < -19_000.0 and high > 20_000.0
