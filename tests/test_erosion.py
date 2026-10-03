import dataclasses

import numpy as np
import pytest

from planet.bodies import components_for, erosion_settings, load_presets
from planet.chunk import build_chunk
from planet.cube_sphere import ChunkKey
from planet.dunes import DuneSettings, Dunes
from planet.hydrology import HydrologySettings, compute_hydrology, meander_offsets
from planet.sphere_grid import sphere_grid
from planet.tectonics import TectonicField, simulation_for
from planet.terrain import Terrain
from systems.climate_system import climate_settings_for, compute_climate
from systems.planet_system import terrain_settings_for
from systems.tectonics_system import tectonic_settings_for


@pytest.fixture(scope="module")
def presets():

    return load_presets()


def world(presets, body: str):
    """(terrain with climate and hydrology, climate field)."""

    parts = components_for(presets[body])

    simulation = simulation_for(tectonic_settings_for(parts.planet, parts.tectonics))

    field = TectonicField.from_state(simulation.grid, simulation.initial_state(), 1)

    settings = terrain_settings_for(parts.planet)

    climate, _ = compute_climate(
        climate_settings_for(parts.climate, parts.planet, parts.body),
        Terrain(settings, field)
    )

    return Terrain(settings, field, climate), climate


@pytest.fixture(scope="module")
def earth(presets):

    return world(presets, "earth")


# =========================================================
# Rivers
# =========================================================

def test_earth_has_rivers_lakes_and_deltas(earth):

    _, climate = earth

    hydrology = climate.hydrology

    assert hydrology is not None
    assert hydrology.river_count > 100
    assert len(hydrology.delta_radius) > 0
    assert 0.0 < hydrology.lake_fraction < 0.05

    # Wider rivers carry more water.
    order = np.argsort(hydrology.segment_discharge)

    assert hydrology.segment_width[order[-1]] > hydrology.segment_width[order[0]]


def test_rivers_flow_downhill_to_the_sea(earth):

    hydrology = earth[1].hydrology

    levels = hydrology.segment_levels

    # Every reach descends (lakes fill to just above their
    # spill point), ending at sea level at the coast.
    assert np.all(levels[:, 1] <= levels[:, 0] + 1e-6)
    assert np.any(levels[:, 1] == 0.0)


def test_glaciers_only_where_it_freezes(earth):

    terrain, climate = earth

    hydrology = climate.hydrology

    # At each reach's own height.
    temperature, _ = climate.surface(hydrology.segment_start, hydrology.segment_levels[:, 0])

    if hydrology.glacier_count:
        assert np.all(temperature[hydrology.segment_glacier] < 2.0)

    assert np.all(temperature[~hydrology.segment_glacier] > -3.0)


def test_terrain_carves_rivers_and_marks_water(earth):

    terrain, climate = earth

    hydrology = climate.hydrology

    # Along the biggest river: water, and a valley below the
    # uncarved ground.
    biggest = np.argmax(hydrology.segment_discharge)

    a, b = hydrology.segment_start[biggest], hydrology.segment_end[biggest]

    t = np.linspace(0.2, 0.8, 50)

    line = a[None, :] * (1.0 - t[:, None]) + b[None, :] * t[:, None]
    line /= np.linalg.norm(line, axis=1, keepdims=True)

    elevation, water = terrain.elevation_and_water(line, 500.0)

    assert water.max() > 0.5

    plain = Terrain(dataclasses.replace(terrain.settings, rivers=False), terrain.field, climate)

    ground = plain.elevation(line, 500.0)

    # On land rivers only cut down (deltas build up the sea
    # floor at the mouth).
    land = ground > 10.0

    assert land.any()
    assert np.all(elevation[land] <= ground[land] + 1e-6)


def test_dry_and_boiled_worlds_have_no_rivers(presets):

    _, mars = world(presets, "mars")

    assert mars.hydrology is None

    assert not erosion_settings(presets["moon"])["rivers"]
    assert erosion_settings(presets["titan"])["rivers"]


def test_meander_offsets_are_tangent():

    grid = sphere_grid(32)

    offsets = meander_offsets(grid.directions, grid.cell_angle)

    np.testing.assert_allclose(np.sum(offsets * grid.directions, axis=1), 0.0, atol=1e-12)

    size = np.linalg.norm(offsets, axis=1).mean() / grid.cell_angle

    assert 0.1 < size < 0.6


def test_chunks_carry_the_water_mask(earth):

    terrain, _ = earth

    key = ChunkKey(2, 4, 3, 3)

    data = build_chunk(key, terrain, 9)

    # Vertex layout: position, normal, color, uv, tangent.
    uv_y = data.mesh.vertices[:, 10].astype(np.float64)

    # uv.y = quadtree depth + 0.99 * water: the shader gets
    # the depth back with floor, the water with fract.
    np.testing.assert_array_equal(np.floor(uv_y), key.depth)

    water = (uv_y - key.depth) / 0.99

    assert water.min() >= 0.0 and water.max() <= 1.0 + 1e-6


# =========================================================
# Dunes
# =========================================================

def test_transverse_dunes_are_asymmetric():

    dunes = Dunes(DuneSettings(density=1.0, amplitude=50.0, wavelength=1_000.0), 3_390_000.0)

    grid = sphere_grid(48)

    cover = dunes.apply(grid.directions, 1e12)[1]

    center = grid.directions[np.argmax(cover)]

    # Across the crests (east-west at most latitudes).
    east = np.cross([0.0, 1.0, 0.0], center)
    east /= np.linalg.norm(east)

    s = np.linspace(-4_000.0, 4_000.0, 801)

    line = center[None, :] + s[:, None] * east[None, :] / 3_390_000.0
    line /= np.linalg.norm(line, axis=1, keepdims=True)

    height, _ = dunes.apply(line, 10.0)

    assert height.max() > 20.0
    assert height.min() >= 0.0

    # Steep slip faces: the steepest descent is much steeper
    # than the steepest climb, or vice versa.
    slope = np.diff(height)

    assert max(slope.max(), -slope.min()) > 2.0 * min(slope.max(), -slope.min())


def test_linear_dunes_stay_in_their_band():

    dunes = Dunes(DuneSettings(density=1.2, linear=True, max_latitude=30.0, wavelength=3_000.0), 2_574_700.0)

    grid = sphere_grid(32)

    cover = dunes.apply(grid.directions, 1e12)[1]

    latitude = np.degrees(np.arcsin(np.abs(grid.directions[:, 1])))

    assert cover[latitude > 31.0].max() == 0.0
    assert cover[latitude < 20.0].mean() > 0.5


def test_dunes_skip_when_too_small_to_see():

    dunes = Dunes(DuneSettings(density=1.0, wavelength=1_000.0), 3_390_000.0)

    directions = sphere_grid(16).directions

    height, cover = dunes.apply(directions, 5_000.0)

    assert not height.any()
    assert cover.any()


def test_bodies_get_their_dunes(presets):

    titan = erosion_settings(presets["titan"])
    mars = erosion_settings(presets["mars"])

    assert titan["dune_linear"] and titan["dune_max_latitude"] == 30.0
    assert not mars["dune_linear"] and mars["dune_darkening"] > 0.0

    # No air, no dunes.
    assert erosion_settings(presets["moon"])["dune_density"] == 0.0
    assert erosion_settings(presets["pluto"])["dune_density"] == 0.0


def test_dunes_bury_craters(presets):

    terrain, _ = world(presets, "mars")

    grid = sphere_grid(48)

    cover = terrain.dunes.apply(grid.directions, 1e12)[1]

    center = grid.directions[np.argmax(cover)]

    rng = np.random.default_rng(0)

    points = center + rng.normal(size=(2_000, 3)) * 0.002
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    with_dunes = terrain.elevation(points, 20.0)

    no_dunes = Terrain(dataclasses.replace(terrain.settings, dune_density=0.0), terrain.field, terrain.climate)

    # Under the dune field the craters are mostly gone: the
    # surface is smoother at scales beyond the dunes.
    assert np.std(with_dunes) < np.std(no_dunes.elevation(points, 20.0))
