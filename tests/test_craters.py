import dataclasses

import numpy as np
import pytest

from planet.bodies import components_for, crater_settings, load_presets
from planet.chunk import build_chunk
from planet.craters import (
    EJECTA,
    CraterSettings,
    Craters,
    crater_profile,
    cumulative_density
)
from planet.cube_sphere import ChunkKey
from planet.terrain import Terrain, TerrainSettings


MOON_RADIUS = 1_737_400.0


def random_directions(count: int, seed: int = 0) -> np.ndarray:

    v = np.random.default_rng(seed).normal(size=(count, 3))

    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def moon_craters():

    return Craters(CraterSettings(), MOON_RADIUS)


# =========================================================
# Physics
# =========================================================

def test_chronology():

    # Steady impacts for the last ~3 Gyr, a heavy
    # bombardment before.
    assert cumulative_density(1_000.0) == pytest.approx(8.4e-4, rel=0.01)
    assert cumulative_density(3_500.0) > 5.0 * cumulative_density(1_000.0)
    assert cumulative_density(4_400.0) > 100.0 * cumulative_density(3_500.0)
    assert cumulative_density(0.0) == 0.0


def test_profile_shape():

    x = np.linspace(0.0, 2.0, 401)

    simple = crater_profile(x, np.full_like(x, 1_000.0), 15_000.0)

    # Bowl 1/5 as deep as wide; rim at the edge; ejecta
    # thinning to nothing.
    assert simple[0] == pytest.approx(-200.0)
    assert simple[np.argmin(np.abs(x - 1.0))] == pytest.approx(0.18 * 200.0, rel=0.05)
    assert np.all(simple[x >= EJECTA] == 0.0)
    assert np.all(np.diff(simple[(x > 1.0) & (x < EJECTA)]) <= 0.0)

    # Complex craters are relatively shallower, with a
    # central peak.
    big = crater_profile(x, np.full_like(x, 100_000.0), 15_000.0)

    assert -big.min() < 0.1 * 100_000.0
    assert big[0] > big[np.argmin(np.abs(x - 0.4))]


# =========================================================
# Field
# =========================================================

def test_deterministic_and_continuous(moon_craters):

    directions = random_directions(500)

    np.testing.assert_array_equal(
        moon_craters.height(directions, 0.0, 4_000.0),
        Craters(CraterSettings(), MOON_RADIUS).height(directions, 0.0, 4_000.0)
    )

    # A fine line of samples (~0.4 m apart) crosses many
    # lattice cells: no jumps (no seams).
    t = np.linspace(0.0, 0.002, 4_000)

    line = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)

    heights = moon_craters.height(line, 0.0, 4_400.0)

    assert np.abs(heights).max() > 10.0
    assert np.abs(np.diff(heights)).max() < 5.0


def test_older_surfaces_have_more_craters(moon_craters):

    directions = random_directions(3_000)

    def cratered(age):
        return np.mean(np.abs(moon_craters.height(directions, 0.0, age)) > 1.0)

    young, middle, old = cratered(50.0), cratered(3_500.0), cratered(4_400.0)

    assert young < 0.01
    assert young < middle < old


def test_bounds(moon_craters):

    heights = moon_craters.height(random_directions(5_000), 0.0, 4_400.0)

    assert heights.min() >= -moon_craters.max_depth
    assert heights.max() <= moon_craters.max_rim


def test_air_and_weather():

    directions = random_directions(3_000)

    # Venus's air stops impactors smaller than ~3 km: no
    # small craters at close range.
    screened = Craters(CraterSettings(min_diameter=3_000.0), 6_000_000.0)

    assert min(screened.cells) * 0.3125 >= 3_000.0 * 0.5

    # Erosion caps the age that counts.
    eroded = Craters(CraterSettings(erosion_time=300.0), MOON_RADIUS)

    assert eroded.chance(4_400.0) == pytest.approx(eroded.chance(300.0))

    assert np.abs(eroded.height(directions, 0.0, 4_400.0)).max() < 1.0 + np.abs(
        Craters(CraterSettings(), MOON_RADIUS).height(directions, 0.0, 300.0)
    ).max()


def test_coarse_samples_skip_small_craters(moon_craters):

    directions = random_directions(1_000)

    fine = moon_craters.height(directions, 0.0, 4_000.0)
    coarse = moon_craters.height(directions, 20_000.0, 4_000.0)

    assert np.abs(coarse).max() > 0.0
    assert np.abs(fine - coarse).max() > 0.0


def test_rays_only_when_enabled():

    directions = random_directions(20_000, seed=3)

    rayed = Craters(CraterSettings(rays=True), MOON_RADIUS).brightness(directions, 4_000.0)
    plain = Craters(CraterSettings(rays=False), MOON_RADIUS).brightness(directions, 4_000.0)

    assert rayed.max() > 0.5
    assert 0.001 < np.mean(rayed > 0.1) < 0.2
    assert not plain.any()


def test_no_craters_when_disabled():

    craters = Craters(CraterSettings(density=0.0), MOON_RADIUS)

    assert not craters.enabled
    assert not craters.height(random_directions(100), 0.0, 4_000.0).any()


# =========================================================
# Terrain and Bodies
# =========================================================

def test_terrain_adds_craters_within_bounds():

    plain = TerrainSettings(radius=MOON_RADIUS, has_liquid=False, crater_density=0.0)
    cratered = dataclasses.replace(plain, crater_density=1.0, crater_rays=True)

    a, b = Terrain(plain), Terrain(cratered)

    key = ChunkKey(2, 8, 100, 100)

    da, db = build_chunk(key, a, 17), build_chunk(key, b, 17)

    assert not np.allclose(da.mesh.vertices[:, 6], db.mesh.vertices[:, 6])

    assert b.min_elevation <= db.min_elevation
    assert db.max_elevation <= b.max_elevation


def test_bodies_get_crater_physics():

    presets = load_presets()

    moon = crater_settings(presets["moon"])
    venus = crater_settings(presets["venus"])
    earth = crater_settings(presets["earth"])
    mars = crater_settings(presets["mars"])

    # Complex craters start smaller where gravity is stronger.
    assert moon["crater_transition"] == pytest.approx(15_000.0, rel=0.02)
    assert mars["crater_transition"] < moon["crater_transition"]

    # Air screens out small impactors.
    assert venus["crater_min_diameter"] == pytest.approx(3_000.0, rel=0.1)
    assert moon["crater_min_diameter"] == 0.0

    # Weather erases them; rays survive only without air.
    assert earth["crater_erosion"] > 0.0 and moon["crater_erosion"] == 0.0
    assert moon["crater_rays"] and not mars["crater_rays"]

    assert crater_settings(presets["jupiter"])["crater_density"] == 0.0

    # Bodies have their own seeds (their own crater layouts).
    assert components_for(presets["moon"]).planet.seed != components_for(presets["mercury"]).planet.seed
    assert components_for(presets["earth"]).planet.seed == 1
