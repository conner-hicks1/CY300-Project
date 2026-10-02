import json
import math

import numpy as np
import pytest

from ecs.components import AtmosphereComponent
from graphics.atmosphere import AtmosphereParameters
from planet.bodies import (
    ProfileError,
    components_for,
    load_presets,
    parse_profile,
    preset_groups
)
from planet.chunk import build_chunk
from planet.cube_sphere import ChunkKey
from planet.lod import LodSelector
from planet.terrain import Terrain, TerrainSettings
from systems.planet_system import terrain_settings_for


@pytest.fixture(scope="module")
def presets():

    return load_presets()


# =========================================================
# Loading
# =========================================================

def test_all_presets_load(presets):

    expected = {
        "mercury", "venus", "earth", "moon", "mars", "jupiter", "io", "europa",
        "ganymede", "callisto", "saturn", "enceladus", "titan", "uranus",
        "neptune", "triton", "pluto",
    }

    assert set(presets) == expected

    # Ordered outward: Mercury first, Pluto last; moons right
    # after their planet.
    order = list(presets)

    assert order[0] == "mercury" and order[-1] == "pluto"
    assert order.index("moon") == order.index("earth") + 1


def test_groups(presets):

    groups = preset_groups(presets)

    assert [p.id for p in groups["Dwarf planets"]] == ["pluto"]
    assert "titan" in [p.id for p in groups["Moons"]]
    assert "jupiter" in [p.id for p in groups["Planets"]]


def minimal_profile(**overrides) -> dict:

    data = json.loads(open("assets/bodies/moon.json", encoding="utf-8").read())

    for path, value in overrides.items():

        target = data

        *parents, key = path.split(".")

        for parent in parents:
            target = target[parent]

        target[key] = value

    return data


@pytest.mark.parametrize("path, value, message", [
    ("format", 99, "unsupported format"),
    ("kind", "comet-ish", "'kind' must be one of"),
    ("physical.radius_km", "big", "'radius_km' must be a number"),
    ("physical.bond_albedo", 1.5, "below 1"),
    ("surface.liquid", "milk", "'liquid' must be one of"),
    ("surface.colors.low", [2.0, 0.0, 0.0], "three numbers in 0..1"),
    ("surface.palette", "bands", "needs a 'bands' section"),
])
def test_validation(path, value, message):

    with pytest.raises(ProfileError, match=message):
        parse_profile(minimal_profile(**{path: value}), "test")


def test_unknown_gas_is_rejected():

    data = json.loads(open("assets/bodies/mars.json", encoding="utf-8").read())

    data["atmosphere"]["composition"]["Unobtainium"] = 0.1

    with pytest.raises(ProfileError, match="unknown gas"):
        parse_profile(data, "mars")


# =========================================================
# Derived Physics
# =========================================================

def test_earth_physics(presets):

    earth = presets["earth"]

    assert earth.surface_gravity == pytest.approx(9.82, abs=0.03)
    assert earth.escape_velocity == pytest.approx(11_186.0, rel=0.01)
    assert earth.sunlight == pytest.approx(1.0)
    assert earth.equilibrium_temperature_k == pytest.approx(255.0, abs=2.0)
    assert earth.greenhouse_warming_c == pytest.approx(33.0, abs=3.0)
    assert earth.scale_height_km == pytest.approx(8.4, abs=0.2)
    assert earth.sun_angular_radius_deg == pytest.approx(0.2666, abs=0.001)


def test_other_bodies(presets):

    assert presets["mars"].scale_height_km == pytest.approx(11.0, abs=0.5)
    assert presets["titan"].scale_height_km == pytest.approx(21.0, abs=1.5)
    assert presets["moon"].surface_gravity == pytest.approx(1.62, abs=0.01)

    # Venus: a runaway greenhouse of ~500 C.
    assert presets["venus"].greenhouse_warming_c > 450.0

    # Sunlight falls with the square of distance.
    assert presets["mars"].sunlight == pytest.approx(1.0 / 1.524 ** 2)
    assert presets["neptune"].sun_angular_radius_deg < 0.01


# =========================================================
# Components
# =========================================================

def test_earth_components_match_the_earth_defaults(presets):

    parts = components_for(presets["earth"])

    assert parts.planet.liquid == "water"
    assert parts.planet.palette == "biomes"
    assert parts.tectonics is not None
    assert parts.climate is not None

    # The derived atmosphere is Earth's (Hillaire's defaults).
    default = AtmosphereComponent()

    np.testing.assert_allclose(parts.atmosphere.rayleigh_scattering, default.rayleigh_scattering, rtol=0.05)
    assert parts.atmosphere.rayleigh_scale_height == pytest.approx(default.rayleigh_scale_height, rel=0.06)
    assert parts.atmosphere.ozone_absorption == default.ozone_absorption

    assert parts.body.name == "Earth"


def test_airless_moon(presets):

    parts = components_for(presets["moon"])

    assert parts.atmosphere is None
    assert parts.tectonics is None
    assert parts.planet.liquid == "none"
    assert parts.planet.palette == "mineral"
    assert parts.planet.radius == pytest.approx(1_737_400.0)
    assert parts.body.surface_pressure_bar == 0.0


def test_thick_and_hazy_atmospheres(presets):

    venus = components_for(presets["venus"]).atmosphere
    titan = components_for(presets["titan"]).atmosphere
    earth = components_for(presets["earth"]).atmosphere

    # Dense CO2 scatters far more than Earth's air.
    assert venus.rayleigh_scattering[2] > 50 * earth.rayleigh_scattering[2]
    assert titan.mie_scattering > earth.mie_scattering

    assert components_for(presets["titan"]).planet.liquid == "methane"


def test_gas_giant(presets):

    parts = components_for(presets["jupiter"])

    assert parts.planet.palette == "bands"
    assert parts.planet.bands > 0
    assert parts.climate is None
    assert parts.body.kind == "gas_giant"
    assert parts.body.oblateness > 0.06


def test_every_preset_builds_components(presets):

    for profile in presets.values():

        parts = components_for(profile)

        assert parts.planet.radius > 0
        assert parts.body.profile == profile.id

        if parts.atmosphere is not None:

            params = AtmosphereParameters.from_components(parts.planet.radius, parts.atmosphere)

            assert params.top_radius > params.ground_radius


# =========================================================
# Engine Support
# =========================================================

def test_dry_worlds_keep_their_basins(presets):

    planet = components_for(presets["moon"]).planet

    settings = terrain_settings_for(planet)

    assert not settings.has_liquid
    assert settings.min_elevation < 0.0

    terrain = Terrain(settings)

    data = build_chunk(ChunkKey(0, 1, 0, 0), terrain, 9)

    positions = data.mesh.positions[:81].astype(np.float64) + data.center

    radius = np.linalg.norm(positions, axis=1)

    elevation = data.mesh.vertices[:81, 6]

    # Below the reference radius where the ground is low
    # (a liquid world would sit at exactly the radius there).
    low = elevation < -100.0

    assert low.any()
    np.testing.assert_allclose(radius[low], settings.radius + elevation[low], atol=2.0)


def test_banded_giant_is_flat_with_bands():

    settings = TerrainSettings(radius=7.0e7, bands=12, has_liquid=False)

    terrain = Terrain(settings)

    directions = np.random.default_rng(0).normal(size=(500, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    np.testing.assert_array_equal(terrain.elevation(directions), 0.0)

    _, band = terrain.surface_climate(directions, np.zeros(500))

    assert band.min() >= 0.0 and band.max() <= 1.0
    assert band.std() > 0.2


def test_horizon_uses_the_lowest_ground():

    radius = 1_000_000.0

    camera = np.array([0.0, radius + 100.0, 0.0])

    # With basins 5 km deep, the horizon reaches farther.
    flat = LodSelector(radius, 1_000.0, 8, 1.5)
    deep = LodSelector(radius, 1_000.0, 8, 1.5, min_elevation=-5_000.0)

    assert deep.min_elevation == -5_000.0

    _, visible_flat = flat._measure([ChunkKey(2, 4, 8, 8)], camera)
    _, visible_deep = deep._measure([ChunkKey(2, 4, 8, 8)], camera)

    assert visible_flat[0] and visible_deep[0]


def test_vacuum_atmosphere_is_empty():

    vacuum = AtmosphereParameters.vacuum(1_737_400.0)

    assert vacuum.rayleigh_scattering == (0.0, 0.0, 0.0)
    assert vacuum.mie_scattering == 0.0
    assert vacuum.top_radius > vacuum.ground_radius

    assert vacuum.transmittance_to_sun(vacuum.ground_radius + 0.5, 0.5) == pytest.approx(np.ones(3))


def test_sun_size_from_distance(presets):

    # Apparent size scales with 1 / distance.
    ratio = presets["mars"].sun_angular_radius_deg / presets["earth"].sun_angular_radius_deg

    assert ratio == pytest.approx(1.0 / 1.524, rel=1e-3)
    assert math.isfinite(presets["pluto"].sun_angular_radius_deg)
