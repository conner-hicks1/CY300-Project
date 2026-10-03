import math

import numpy as np
import pytest

from ecs.components import AtmosphereComponent
from graphics.atmosphere import AtmosphereParameters, pack_atmosphere_block
from graphics.clouds import MAP_SIZE, CloudParameters, cloud_cover_map
from math3d import quaternion
from planet.bodies import components_for, load_presets, parse_profile
from planet.climate import ClimateModel, ClimateSettings, ClimateField
from systems.render_system import _inverse_rotation


def area_mean(cover: np.ndarray) -> float:

    height = cover.shape[0]

    latitude = ((np.arange(height) + 0.5) / height - 0.5) * math.pi

    weight = np.cos(latitude)[:, None] * np.ones_like(cover)

    return float(np.sum(cover * weight) / np.sum(weight))


# =========================================================
# Cover Map
# =========================================================

@pytest.mark.parametrize("coverage", [0.05, 0.3, 0.62, 0.9])
def test_map_matches_the_mean_coverage(coverage):

    cover = cloud_cover_map(coverage)

    assert cover.shape == (MAP_SIZE[1], MAP_SIZE[0])
    assert 0.0 <= cover.min() and cover.max() <= 1.0
    assert area_mean(cover) == pytest.approx(coverage, abs=0.01)


@pytest.fixture(scope="module")
def ocean_climate():

    model = ClimateModel(ClimateSettings(resolution=24))

    state = model.compute(np.full(model.grid.cell_count, -1000.0))

    return ClimateField.from_state(model.grid, state, 1)


def test_clouds_follow_the_rain(ocean_climate):

    cover = cloud_cover_map(0.62, ocean_climate)

    assert area_mean(cover) == pytest.approx(0.62, abs=0.01)

    height = cover.shape[0]

    latitude = np.degrees(((np.arange(height) + 0.5) / height - 0.5) * math.pi)

    band = lambda low, high: cover[(np.abs(latitude) >= low) & (np.abs(latitude) < high)].mean()

    # The rainy equatorial belt is cloudier than the dry
    # subtropics (~20-30 degrees).
    assert band(0, 8) > band(20, 30) + 0.1


# =========================================================
# Shader Inputs
# =========================================================

def test_clouds_are_packed():

    earth = AtmosphereParameters.from_components(6_371_000.0, AtmosphereComponent())

    clouds = CloudParameters(coverage=0.6, altitude=4.0, optical_depth=14.0, scale=700.0, color=(1.0, 0.9, 0.8))

    frame = (0.0, 0.7071, 0.0, 0.7071)

    v = np.frombuffer(
        pack_atmosphere_block(earth, clouds=clouds, cloud_drift=0.25, planet_frame=frame),
        dtype=np.float32
    ).reshape(14, 4)

    np.testing.assert_allclose(v[9], (4.0, 14.0, 700.0, 1.0))
    np.testing.assert_allclose(v[10], (1.0, 0.9, 0.8, 0.25), rtol=1e-6)
    np.testing.assert_allclose(v[11], frame, rtol=1e-6)

    # No clouds: the flag is off.
    off = np.frombuffer(pack_atmosphere_block(earth), dtype=np.float32).reshape(14, 4)

    assert off[9, 3] == 0.0
    np.testing.assert_allclose(off[11], (0.0, 0.0, 0.0, 1.0))


def test_planet_frame_undoes_the_planet_rotation():

    q = quaternion.from_axis_angle(np.array([0.3, 1.0, 0.2]) / np.linalg.norm([0.3, 1.0, 0.2]), 0.8)

    world = np.eye(4)
    world[:3, :3] = quaternion.to_matrix3(q) * 2.0      # with scale

    inverse = np.array(_inverse_rotation(world))

    local = np.array([0.6, 0.0, 0.8])

    world_direction = quaternion.to_matrix3(q) @ local

    np.testing.assert_allclose(quaternion.to_matrix3(inverse) @ world_direction, local, atol=1e-9)


# =========================================================
# Bodies
# =========================================================

def test_bodies_get_their_clouds():

    presets = load_presets()

    earth = components_for(presets["earth"]).atmosphere
    mars = components_for(presets["mars"]).atmosphere
    venus = components_for(presets["venus"]).atmosphere

    assert earth.cloud_coverage == pytest.approx(0.55)
    assert earth.cloud_altitude == pytest.approx(4_000.0)

    # Mars's thin water-ice clouds, high and rare.
    assert mars.cloud_coverage < 0.2 and mars.cloud_optical_depth < 1.0
    assert mars.cloud_altitude > 10_000.0

    # Venus: no weather clouds (its deck is an aerosol layer).
    assert venus.cloud_coverage == 0.0
    assert venus.mie_layer_altitude > 40_000.0


def test_cloud_profile_is_validated():

    import json

    data = json.loads(open("assets/bodies/earth.json", encoding="utf-8").read())

    data["atmosphere"]["clouds"]["coverage"] = 1.5

    from planet.bodies import ProfileError

    with pytest.raises(ProfileError, match="coverage"):
        parse_profile(data, "earth")
