import math

import numpy as np
import pytest

from ecs.components import AtmosphereComponent, CameraControllerComponent
from graphics.atmosphere import AtmosphereParameters, pack_atmosphere_block, to_atmosphere_space
from planet.bodies import components_for, load_presets
from planet.chunk import build_chunk
from planet.cube_sphere import ChunkKey
from planet.terrain import Terrain, TerrainSettings, oblate_offset
from systems.camera_controller_system import clamp_altitude
from systems.planet_system import terrain_settings_for


JUPITER_RADIUS = 71_492_000.0


@pytest.fixture(scope="module")
def presets():

    return load_presets()


# =========================================================
# Shape
# =========================================================

def test_oblate_cloud_tops():

    f = 0.06487

    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])

    offset = oblate_offset(directions, JUPITER_RADIUS, f)

    # Equator at the equatorial radius, poles (1 - f) of it.
    assert offset[0] == pytest.approx(0.0)
    assert offset[1] == pytest.approx(-f * JUPITER_RADIUS, rel=1e-9)
    assert offset[2] == pytest.approx(offset[1])

    # Points on an ellipse: (x / a)^2 + (y / b)^2 = 1.
    angle = np.linspace(-1.5, 1.5, 50)
    d = np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=1)

    r = JUPITER_RADIUS + oblate_offset(d, JUPITER_RADIUS, f)

    a, b = JUPITER_RADIUS, JUPITER_RADIUS * (1.0 - f)

    np.testing.assert_allclose((r * d[:, 0] / a) ** 2 + (r * d[:, 1] / b) ** 2, 1.0, rtol=1e-12)


def test_giant_terrain_is_the_ellipsoid_within_bounds():

    settings = TerrainSettings(radius=JUPITER_RADIUS, bands=14, has_liquid=False, oblateness=0.065)

    terrain = Terrain(settings)

    assert settings.min_elevation == pytest.approx(-0.065 * JUPITER_RADIUS)

    for key in (ChunkKey(2, 0, 0, 0), ChunkKey(2, 3, 4, 7)):

        data = build_chunk(key, terrain, 9)

        assert settings.min_elevation - 1.0 <= data.min_elevation
        assert data.max_elevation <= terrain.max_elevation + 1.0


def test_atmosphere_space_makes_the_ellipsoid_a_sphere():

    f = 0.1

    for latitude in np.linspace(-1.5, 1.5, 7):

        d = np.array([math.cos(latitude), math.sin(latitude), 0.0])

        point = d * (JUPITER_RADIUS + oblate_offset(d[None, :], JUPITER_RADIUS, f)[0])

        stretched = to_atmosphere_space(point, flattening=f)

        assert np.linalg.norm(stretched) == pytest.approx(JUPITER_RADIUS, rel=1e-9)


def test_sun_is_packed_in_atmosphere_space():

    earth = AtmosphereParameters.from_components(6_371_000.0, AtmosphereComponent())

    sun = np.array([0.6, 0.8, 0.0])

    v = np.frombuffer(
        pack_atmosphere_block(earth, sun_direction=sun, sun_illuminance=(1.0, 1.0, 1.0), flattening=0.2, no_surface=True),
        dtype=np.float32
    ).reshape(14, 4)

    expected = np.array([0.6, 0.8 / 0.8, 0.0])
    expected /= np.linalg.norm(expected)

    np.testing.assert_allclose(v[7, :3], expected, atol=1e-6)

    # Flattening, no surface, opaque depth, where deep air begins.
    assert v[12, 0] == pytest.approx(0.2)
    assert v[12, 1] == 1.0
    assert v[12, 2] > v[12, 3] > 0.0


# =========================================================
# Bodies
# =========================================================

def test_giants_get_shape_and_weather(presets):

    jupiter = components_for(presets["jupiter"]).planet
    saturn = components_for(presets["saturn"]).planet
    neptune = components_for(presets["neptune"]).planet
    earth = components_for(presets["earth"]).planet

    assert jupiter.oblateness == pytest.approx(0.06487)
    assert saturn.oblateness > jupiter.oblateness

    # The Great Red Spot, south of the equator.
    assert jupiter.storm_latitude == pytest.approx(-22.0)
    assert jupiter.storm_size > 5_000_000.0

    assert saturn.polar_hexagon and not jupiter.polar_hexagon

    # Neptune's Great Dark Spot is dark.
    assert max(neptune.storm_color) < 0.5

    # Solid worlds are not flattened (their oceans would
    # flood the poles).
    assert earth.oblateness == 0.0
    assert terrain_settings_for(earth).oblateness == 0.0
    assert terrain_settings_for(jupiter).oblateness == pytest.approx(0.06487)


# =========================================================
# Descent
# =========================================================

def test_camera_may_descend_below_a_giants_cloud_tops():

    controller = CameraControllerComponent(
        planet_mode=True,
        planet_center=(0.0, 0.0, 0.0),
        planet_radius=1_000_000.0,
        min_altitude=10.0
    )

    deep = np.array([0.0, 990_000.0, 0.0])

    # Solid world: pushed back up to the ground.
    assert np.linalg.norm(clamp_altitude(deep, controller)) == pytest.approx(1_000_010.0)

    # Giant: free to go down to the descent limit.
    controller.descent = 20_000.0

    np.testing.assert_allclose(clamp_altitude(deep, controller), deep)

    assert np.linalg.norm(clamp_altitude(np.array([0.0, 900_000.0, 0.0]), controller)) == pytest.approx(980_010.0)
