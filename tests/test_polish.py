import math

import numpy as np
import pytest

from ecs.components import AtmosphereComponent
from graphics.atmosphere import AtmosphereParameters, pack_atmosphere_block
from planet.bodies import components_for, load_presets
from planet.craters import lattice_cells
from planet.noise import _GRADIENTS, Perlin


RADIUS = 6_051_800.0


# =========================================================
# Aerosol Layers (Venus's cloud deck)
# =========================================================

def deck(**overrides) -> AtmosphereParameters:

    values = dict(
        mie_scattering=3_000.0,
        mie_absorption=25.0,
        mie_scale_height=5_000.0,
        mie_layer_altitude=57_000.0,
        height=200_000.0
    )

    values.update(overrides)

    return AtmosphereParameters.from_components(RADIUS, AtmosphereComponent(**values))


def test_layer_density_peaks_at_its_altitude():

    p = deck()

    density = p.mie_density(np.array([0.0, 30.0, 57.0, 80.0]))

    assert density[2] == pytest.approx(1.0)
    assert density[0] < 1e-4 and density[1] < density[2] > density[3]

    # Ground-hugging aerosols are densest at the ground.
    ground = deck(mie_layer_altitude=0.0)

    assert ground.mie_density(np.array([0.0]))[0] == pytest.approx(1.0)


def test_layer_column_and_thickness():

    p = deck()

    # Both sides of the deck: about twice the scale height.
    assert p.mie_column == pytest.approx(10.0, rel=1e-3)
    assert deck(mie_layer_altitude=0.0).mie_column == pytest.approx(5.0)

    # Venus's clouds (tau ~30) make it optically thick.
    assert p.vertical_scattering_depth > 25.0
    assert p.thick_weight == 1.0


def test_layer_is_packed_for_the_shaders():

    v = np.frombuffer(
        pack_atmosphere_block(deck(), sun_direction=(0.0, 1.0, 0.0), sun_illuminance=(1.0, 1.0, 1.0)),
        dtype=np.float32
    ).reshape(27, 4)

    assert v[8, 3] == pytest.approx(57.0)


def test_venus_clouds_sit_high():

    venus = components_for(load_presets()["venus"]).atmosphere

    # The sulfuric-acid decks: 31-90 km, the thick one at
    # 47.5-57 km, all well inside the modeled air.
    decks = [venus.decks[i:i + 9] for i in range(0, 36, 9)]

    assert min(d[0] for d in decks) == pytest.approx(31_000.0)
    assert max(d[1] for d in decks) == pytest.approx(90_000.0)

    thickest = max(decks, key=lambda d: d[2])

    assert (thickest[0], thickest[1]) == pytest.approx((47_500.0, 57_000.0))
    assert venus.height > 90_000.0


def test_layer_clears_the_ground():

    p = deck()

    # Sunlight at the ground still crosses the deck, but the
    # air near the surface is free of cloud.
    near_ground = p.extinction(np.array([0.5]))[0]
    in_deck = p.extinction(np.array([57.0]))[0]

    assert np.all(in_deck > 100.0 * (near_ground - np.asarray(p.rayleigh_scattering)))


# =========================================================
# Lattice Cells
# =========================================================

@pytest.mark.parametrize("spread", [1e3, 1e7])
def test_dense_and_sorted_lattices_agree(spread):

    rng = np.random.default_rng(0)

    # Compact points (one chunk: dense path) and spread-out
    # ones (whole planet: sorted path).
    points = rng.normal(size=(500, 3)) * spread

    keys, cell_of, point = lattice_cells(points, 2_000.0)

    base = np.floor(points / 2_000.0 - 0.5).astype(np.int64)

    corners = np.array([[(c >> 0) & 1, (c >> 1) & 1, (c >> 2) & 1] for c in range(8)])

    expected = (base[:, None, :] + corners[None]).reshape(-1, 3)

    # Every pair points at its own cell; cells are distinct.
    np.testing.assert_array_equal(keys[cell_of], expected)
    np.testing.assert_array_equal(point, np.repeat(np.arange(500), 8))

    assert len(np.unique(keys, axis=0)) == len(keys)


# =========================================================
# Noise
# =========================================================

def reference_perlin(noise: Perlin, points: np.ndarray) -> np.ndarray:
    """The original formulation (row-gathered gradients)."""

    floor = np.floor(points)
    f = points - floor
    cell = floor.astype(np.int64) & 255
    perm = noise._perm
    gradient_of = _GRADIENTS[perm % 12]

    x, y, z = cell[:, 0], cell[:, 1], cell[:, 2]
    a = perm[x] + y
    b = perm[x + 1] + y
    aa, ab, ba, bb = perm[a] + z, perm[a + 1] + z, perm[b] + z, perm[b + 1] + z

    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    gx, gy, gz = fx - 1.0, fy - 1.0, fz - 1.0

    def corner(h, px, py, pz):
        g = gradient_of[h]
        return g[:, 0] * px + g[:, 1] * py + g[:, 2] * pz

    u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]

    x00 = corner(aa, fx, fy, fz) + ux * (corner(ba, gx, fy, fz) - corner(aa, fx, fy, fz))
    x10 = corner(ab, fx, gy, fz) + ux * (corner(bb, gx, gy, fz) - corner(ab, fx, gy, fz))
    x01 = corner(aa + 1, fx, fy, gz) + ux * (corner(ba + 1, gx, fy, gz) - corner(aa + 1, fx, fy, gz))
    x11 = corner(ab + 1, fx, gy, gz) + ux * (corner(bb + 1, gx, gy, gz) - corner(ab + 1, fx, gy, gz))

    y0 = x00 + uy * (x10 - x00)
    y1 = x01 + uy * (x11 - x01)

    return y0 + uz * (y1 - y0)


def test_faster_noise_is_unchanged():

    points = np.random.default_rng(1).normal(size=(2_000, 3)) * 40.0

    noise = Perlin(7)

    np.testing.assert_allclose(noise(points), reference_perlin(noise, points), atol=1e-12)


# =========================================================
# Terrain Sanity (every body)
# =========================================================

@pytest.mark.parametrize("body", ["earth", "mars", "venus", "moon", "titan", "io", "europa"])
def test_terrain_stays_in_a_sane_range(body):

    from planet.sphere_grid import sphere_grid
    from planet.tectonics import TectonicField, simulation_for
    from planet.terrain import Terrain
    from systems.climate_system import climate_settings_for, compute_climate
    from systems.planet_system import terrain_settings_for
    from systems.tectonics_system import tectonic_settings_for

    parts = components_for(load_presets()[body])

    simulation = simulation_for(tectonic_settings_for(parts.planet, parts.tectonics))

    field = TectonicField.from_state(simulation.grid, simulation.initial_state(), 1)

    settings = terrain_settings_for(parts.planet)

    climate, _ = compute_climate(
        climate_settings_for(parts.climate, parts.planet, parts.body),
        Terrain(settings, field)
    )

    terrain = Terrain(settings, field, climate)

    elevation = terrain.elevation(sphere_grid(96).directions, 5_000.0)

    # Deepest trenches and basins to the tallest volcanoes
    # (no runaway values from any process).
    assert elevation.min() > -20_000.0
    assert elevation.max() < 30_000.0

    if body == "earth":
        assert elevation.max() < 11_000.0
