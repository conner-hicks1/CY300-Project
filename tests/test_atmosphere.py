import math

import numpy as np
import pytest

from ecs.components import AtmosphereComponent
from graphics.atmosphere import (
    SUN_ANGULAR_RADIUS,
    AtmosphereParameters,
    AtmosphereSky,
    pack_atmosphere_block
)
from graphics.uniform_blocks import ATMOSPHERE_BLOCK, UNIFORM_BLOCKS


RADIUS = 6_371_000.0


@pytest.fixture
def earth():

    return AtmosphereParameters.from_components(RADIUS, AtmosphereComponent())


# =========================================================
# Parameters
# =========================================================

def test_units_are_kilometers(earth):

    assert earth.ground_radius == pytest.approx(6371.0)
    assert earth.top_radius == pytest.approx(6471.0)

    # 1/Mm -> 1/km.
    assert earth.rayleigh_scattering == pytest.approx((5.802e-3, 13.558e-3, 33.1e-3))
    assert earth.rayleigh_scale_height == pytest.approx(8.0)
    assert earth.ozone_half_width == pytest.approx(15.0)


def test_bad_values_are_clamped():

    component = AtmosphereComponent(
        height=-5.0,
        mie_anisotropy=1.5,
        ground_albedo=3.0,
        rayleigh_scattering=(-1.0, 1.0, 1.0)
    )

    p = AtmosphereParameters.from_components(RADIUS, component)

    assert p.top_radius > p.ground_radius
    assert p.mie_anisotropy < 1.0
    assert p.ground_albedo == 1.0
    assert p.rayleigh_scattering[0] == 0.0


def test_parameters_are_hashable_for_change_detection(earth):

    same = AtmosphereParameters.from_components(RADIUS, AtmosphereComponent())

    assert earth == same and hash(earth) == hash(same)

    other = AtmosphereParameters.from_components(RADIUS, AtmosphereComponent(mie_scattering=10.0))

    assert other != earth


# =========================================================
# Transmittance (CPU reference of the LUT)
# =========================================================

def test_zenith_transmittance_matches_earth(earth):

    t = earth.transmittance_to_sun(earth.ground_radius, 1.0)

    # Clear-sky Earth: most light gets through, blue least.
    assert 0.9 < t[0] < 0.97
    assert 0.8 < t[1] < 0.92
    assert 0.7 < t[2] < 0.85
    assert t[0] > t[1] > t[2]


def test_low_sun_reddens(earth):

    high = earth.transmittance_to_sun(earth.ground_radius + 0.1, 0.5)
    low = earth.transmittance_to_sun(earth.ground_radius + 0.1, 0.03)

    assert (low < high).all()
    assert low[0] / low[2] > high[0] / high[2]


def test_planet_blocks_sun_below_horizon(earth):

    r = earth.ground_radius + 10.0

    horizon = -math.sqrt(1.0 - (earth.ground_radius / r) ** 2)

    assert earth.transmittance_to_sun(r, horizon - 0.01).sum() == 0.0
    assert earth.transmittance_to_sun(r, horizon + 0.01).sum() > 0.0


def test_transmittance_rises_with_altitude(earth):

    values = [
        earth.transmittance_to_sun(earth.ground_radius + h, 0.2)[2]
        for h in (0.0, 2.0, 10.0, 40.0, 99.0)
    ]

    assert values == sorted(values)
    assert values[-1] > 0.99


def test_extinction_profile(earth):

    ext = earth.extinction(np.array([0.0, 8.0, 25.0, 200.0]))

    # Rayleigh falls by e per scale height.
    assert ext[1, 0] < ext[0, 0]

    # Ozone peaks at its altitude: green absorption there
    # exceeds what scattering alone would give.
    assert ext[2, 1] > earth.rayleigh_scattering[1] * math.exp(-25.0 / 8.0)

    # Space is (almost) empty.
    assert ext[3].max() < 1e-6


# =========================================================
# Uniform Block
# =========================================================

def test_block_is_registered():

    assert ATMOSPHERE_BLOCK in UNIFORM_BLOCKS
    assert len({block.binding for block in UNIFORM_BLOCKS}) == len(UNIFORM_BLOCKS)


def test_disabled_block_is_zero():

    data = pack_atmosphere_block(None)

    assert len(data) == ATMOSPHERE_BLOCK.size
    assert not any(data)


def test_block_layout(earth):

    raw = pack_atmosphere_block(
        earth,
        planet_center_relative=(0.0, -6_372_000.0, 0.0),
        sun_direction=(0.0, 1.0, 0.0),
        sun_illuminance=(5.0, 4.0, 3.0),
        steps=32
    )

    assert len(raw) == ATMOSPHERE_BLOCK.size

    v = np.frombuffer(raw, dtype=np.float32).reshape(14, 4)

    np.testing.assert_allclose(v[0], (0.0, -6372.0, 0.0, 1.0))
    np.testing.assert_allclose(v[1], (6371.0, 6471.0, 0.8, 0.3), rtol=1e-6)
    np.testing.assert_allclose(v[2, :3], earth.rayleigh_scattering, rtol=1e-6)
    np.testing.assert_allclose(v[3], (*earth.mie_scattering, 1.2), rtol=1e-6)
    np.testing.assert_allclose(v[4, :3], earth.mie_absorption, rtol=1e-6)
    assert v[4, 3] == pytest.approx(math.radians(SUN_ANGULAR_RADIUS))
    assert v[6, 1] == 32.0
    assert v[6, 3] == 0.0                   # Earth's air is thin
    np.testing.assert_allclose(v[7], (0.0, 1.0, 0.0, 1.0))
    np.testing.assert_allclose(v[8], (5.0, 4.0, 3.0, 0.0))


def test_block_without_sun(earth):

    v = np.frombuffer(pack_atmosphere_block(earth), dtype=np.float32).reshape(14, 4)

    assert v[0, 3] == 1.0
    assert v[7, 3] == 0.0


def test_colored_aerosols():

    dust = AtmosphereComponent(
        mie_scattering=30.0,
        mie_absorption=10.0,
        mie_absorption_tint=(0.3, 0.8, 1.6)
    )

    p = AtmosphereParameters.from_components(RADIUS, dust)

    assert p.mie_scattering == pytest.approx((0.03, 0.03, 0.03))
    assert p.mie_absorption == pytest.approx((0.003, 0.008, 0.016))

    # Blue dims fastest through the dust: reddened sunlight.
    sun = p.transmittance_to_sun(RADIUS / 1000.0 + 0.01, 1.0)

    assert sun[0] > sun[1] > sun[2]


def test_thick_weight():

    earth = AtmosphereParameters.from_components(RADIUS, AtmosphereComponent())

    assert earth.thick_weight == 0.0

    cloudy = AtmosphereParameters.from_components(
        RADIUS,
        AtmosphereComponent(mie_scattering=400.0, mie_scale_height=15_000.0)
    )

    assert cloudy.vertical_scattering_depth > 6.0
    assert cloudy.thick_weight == 1.0


# =========================================================
# Environment Bake Key
# =========================================================

def sky(earth, **overrides):

    values = dict(
        parameters=earth,
        sun_direction=(0.0, 0.8, 0.6),
        sun_illuminance=(5.0, 5.0, 5.0),
        camera_up=(0.0, 1.0, 0.0),
        altitude_km=1.0,
        textures=(("uTransmittanceLut", 7),)
    )

    values.update(overrides)

    return AtmosphereSky(**values)


def test_bake_key_ignores_noise_and_textures(earth):

    base = sky(earth).bake_key()

    assert sky(earth, sun_direction=(0.0, 0.80001, 0.6)).bake_key() == base
    assert sky(earth, altitude_km=1.02).bake_key() == base
    assert sky(earth, textures=()).bake_key() == base


def test_bake_key_tracks_real_changes(earth):

    base = sky(earth).bake_key()

    assert sky(earth, sun_direction=(0.0, 0.6, 0.8)).bake_key() != base
    assert sky(earth, altitude_km=20.0).bake_key() != base
    assert sky(earth, camera_up=(1.0, 0.0, 0.0)).bake_key() != base

    other = AtmosphereParameters.from_components(RADIUS, AtmosphereComponent(height=50_000.0))

    assert sky(earth, parameters=other).bake_key() != base
