import dataclasses

import numpy as np
import pytest

from planet.bodies import components_for, load_presets, volcano_settings
from planet.chunk import build_chunk
from planet.cube_sphere import ChunkKey
from planet.sphere_grid import sphere_grid
from planet.tectonics import TectonicField, TectonicSettings, simulation_for
from planet.terrain import Terrain, TerrainSettings
from planet.volcanoes import (
    SHIELD,
    SMALL_SHIELD,
    STRATO,
    VolcanoSettings,
    Volcanoes,
    max_volcano_height,
    volcano_profile
)
from systems.planet_system import terrain_settings_for
from systems.tectonics_system import tectonic_settings_for


def field_for(regime: str, radius: float = 3_390_000.0, resolution: int = 48) -> TectonicField:

    simulation = simulation_for(
        TectonicSettings(regime=regime, radius=radius, resolution=resolution, time_step=5.0)
    )

    return TectonicField.from_state(simulation.grid, simulation.initial_state(), 1)


@pytest.fixture(scope="module")
def mars_like():

    field = field_for("stagnant_lid", resolution=64)

    return Volcanoes(
        VolcanoSettings(seed=44, max_height=max_volcano_height(3.71)),
        3_390_000.0,
        field
    )


# =========================================================
# Physics
# =========================================================

def test_gravity_limits_height():

    assert max_volcano_height(9.81) == pytest.approx(10_000.0)
    assert max_volcano_height(3.71) == pytest.approx(26_400.0, rel=0.01)

    # Capped for the weakest gravity.
    assert max_volcano_height(1.0) == 27_000.0


@pytest.mark.parametrize("kind", [SHIELD, STRATO, SMALL_SHIELD])
def test_profiles(kind):

    x = np.linspace(0.0, 1.0, 201)

    height = np.full_like(x, 5_000.0)

    profile = volcano_profile(
        kind, x, height,
        np.full_like(x, 0.5),
        np.zeros_like(x),
        np.zeros(len(x), dtype=np.uint64),
        np.full_like(x, 0.3)
    )

    assert np.all(profile >= 0.0)
    assert profile.max() <= 5_000.0 + 1e-6
    assert profile[-1] == pytest.approx(0.0, abs=1.0)

    # A crater / caldera: the very summit is lower than the
    # rim around it.
    assert profile[0] < profile[:40].max()


def test_strato_is_steeper_than_shield():

    x = np.linspace(0.0, 1.0, 101)

    def mean_slope(kind, radius):

        h = volcano_profile(
            kind, x, np.full_like(x, 3_000.0), np.zeros_like(x), np.zeros_like(x),
            np.zeros(len(x), dtype=np.uint64), np.zeros_like(x)
        )

        return np.abs(np.diff(h)).sum() / radius

    # Same height; a stratovolcano is ~4x narrower.
    assert mean_slope(STRATO, 12_000.0) > 3.0 * mean_slope(SHIELD, 45_000.0)


def test_giant_gets_a_basal_cliff():

    x = np.linspace(0.85, 1.0, 61)

    def drop(stature):

        h = volcano_profile(
            SHIELD, x, np.full_like(x, 20_000.0), np.zeros_like(x), np.zeros_like(x),
            np.zeros(len(x), dtype=np.uint64), np.full_like(x, stature)
        )

        return np.abs(np.diff(h)).max()

    assert drop(0.9) > 3.0 * drop(0.2)


# =========================================================
# Placement
# =========================================================

def test_volcanoes_follow_activity(mars_like):

    field = mars_like.field

    directions = sphere_grid(64).directions

    heights = mars_like.height(directions, 0.0)

    activity = field.sample("activity", directions)

    on = heights > 100.0

    assert on.any()

    # Mostly on the volcanic provinces.
    assert np.mean(activity[on] > 0.02) > 0.8

    # Gravity's ceiling holds; giants reach most of it.
    assert heights.max() <= mars_like.settings.max_height + 1e-6
    assert heights.max() > 0.5 * mars_like.settings.max_height


def test_deterministic_and_continuous(mars_like):

    directions = sphere_grid(32).directions

    np.testing.assert_array_equal(
        mars_like.height(directions),
        Volcanoes(mars_like.settings, mars_like.radius, mars_like.field).height(directions)
    )

    # Across the tallest volcano, finely sampled: no jumps.
    grid = sphere_grid(64).directions

    summit = grid[np.argmax(mars_like.height(grid, 15_000.0))]

    side = np.cross(summit, [0.0, 1.0, 0.0])
    side /= np.linalg.norm(side)

    t = np.linspace(-0.15, 0.15, 6_000)

    line = summit[None, :] + t[:, None] * side[None, :]
    line /= np.linalg.norm(line, axis=1, keepdims=True)

    heights = mars_like.height(line, 0.0)

    assert heights.max() > 5_000.0

    # ~170 m apart; the basal cliff is the steepest part.
    assert np.abs(np.diff(heights)).max() < 1_500.0


def test_regimes_build_their_kinds():

    plates = Volcanoes(VolcanoSettings(), 6_371_000.0, field_for("plate_tectonics", 6_371_000.0))
    io = Volcanoes(VolcanoSettings(), 1_821_000.0, field_for("heat_pipe", 1_821_000.0))
    europa = Volcanoes(VolcanoSettings(), 1_561_000.0, field_for("ice_shell", 1_561_000.0))

    assert {o.kind for o in plates.octaves} == {SHIELD, STRATO, SMALL_SHIELD}
    assert {o.kind for o in io.octaves} == {SMALL_SHIELD}
    assert not europa.enabled

    # Small bodies' weak volcanism: small domes only.
    domes = Volcanoes(VolcanoSettings(volcanism=0.15), 1_737_400.0, field_for("stagnant_lid", 1_737_400.0))

    assert {o.kind for o in domes.octaves} == {SMALL_SHIELD}

    assert not Volcanoes(VolcanoSettings(), 1_000_000.0, None).enabled


# =========================================================
# Terrain and Bodies
# =========================================================

def test_terrain_includes_volcanoes_within_bounds(mars_like):

    settings = TerrainSettings(
        radius=3_390_000.0,
        has_liquid=False,
        seed=44,
        volcanism=1.0,
        volcano_max_height=max_volcano_height(3.71)
    )

    terrain = Terrain(settings, mars_like.field)

    assert terrain.volcanoes is not None

    plain = Terrain(dataclasses.replace(settings, volcanism=0.0), mars_like.field)

    assert plain.volcanoes is None
    assert terrain.max_elevation > plain.max_elevation

    for key in (ChunkKey(0, 1, 0, 0), ChunkKey(4, 3, 3, 4)):

        data = build_chunk(key, terrain, 9)

        assert data.max_elevation <= terrain.max_elevation
        assert terrain.min_elevation <= data.min_elevation


def test_bodies_get_volcanism():

    presets = load_presets()

    assert volcano_settings(presets["mars"])["volcanism"] == 1.0
    assert volcano_settings(presets["venus"])["volcanism"] == 1.0
    assert volcano_settings(presets["earth"])["volcanism"] == 1.0

    # Small rocky stagnant lids: domes only; icy: none.
    assert volcano_settings(presets["moon"])["volcanism"] < 0.5
    assert volcano_settings(presets["mercury"])["volcanism"] < 0.5
    assert volcano_settings(presets["europa"])["volcanism"] == 0.0
    assert volcano_settings(presets["callisto"])["volcanism"] == 0.0
    assert volcano_settings(presets["jupiter"])["volcanism"] == 0.0

    mars = components_for(presets["mars"]).planet

    assert mars.volcano_max_height == pytest.approx(26_400.0, rel=0.02)

    assert terrain_settings_for(mars).volcanism == 1.0
