import json

import numpy as np
import pytest

from ecs.components import AtmosphereComponent, RingsComponent
from graphics.atmosphere import AtmosphereParameters, pack_atmosphere_block
from graphics.rings import MAX_BANDS, PROFILE_SIZE, flat_bands, ring_profile, unflat_bands
from planet.bodies import ProfileError, components_for, load_presets, parse_profile


SATURN_BANDS = [
    (74_500e3, 92_000e3, 0.08),        # C ring
    (92_000e3, 117_580e3, 1.8),        # B ring
    (117_580e3, 122_170e3, 0.1),       # Cassini Division
    (122_170e3, 133_400e3, 0.65),      # A ring
    (133_700e3, 136_775e3, 0.5),       # A ring past the Encke Gap
    (140_100e3, 140_350e3, 0.3),       # F ring
]


@pytest.fixture(scope="module")
def saturn_profile():

    return ring_profile(74_500e3, 140_500e3, SATURN_BANDS, seed=3)


def tau_at(profile, radius):

    u = (radius - 74_500e3) / (140_500e3 - 74_500e3)

    return float(profile[int(u * (len(profile) - 1))])


# =========================================================
# Profile
# =========================================================

def test_profile_follows_the_bands(saturn_profile):

    assert saturn_profile.shape == (PROFILE_SIZE,)
    assert saturn_profile.min() >= 0.0

    # The B ring is the densest; the C ring and the Cassini
    # Division sparse; the Encke Gap and the space before the
    # F ring empty.
    b = np.mean([tau_at(saturn_profile, r) for r in np.linspace(95e6, 115e6, 40)])
    c = np.mean([tau_at(saturn_profile, r) for r in np.linspace(76e6, 90e6, 40)])
    a = np.mean([tau_at(saturn_profile, r) for r in np.linspace(124e6, 132e6, 40)])

    assert b > a > c > 0.0

    assert tau_at(saturn_profile, 133_550e3) == 0.0
    assert tau_at(saturn_profile, 138_500e3) == 0.0
    assert tau_at(saturn_profile, 140_200e3) > 0.0


def test_ringlets_vary_within_a_band(saturn_profile):

    radii = np.linspace(95e6, 115e6, 400)

    values = np.array([tau_at(saturn_profile, r) for r in radii])

    # Structure, but around the band's depth.
    assert values.std() > 0.1 * values.mean()
    assert 0.6 * 1.8 < values.mean() < 1.4 * 1.8


def test_profile_is_deterministic():

    a = ring_profile(1e6, 2e6, [(1.2e6, 1.8e6, 1.0)], seed=5)
    b = ring_profile(1e6, 2e6, [(1.2e6, 1.8e6, 1.0)], seed=5)
    c = ring_profile(1e6, 2e6, [(1.2e6, 1.8e6, 1.0)], seed=6)

    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_band_flattening_round_trips():

    flat = flat_bands(SATURN_BANDS)

    assert len(flat) == 3 * MAX_BANDS

    assert unflat_bands(flat) == [tuple(map(float, band)) for band in SATURN_BANDS]

    # Defaults hold no bands.
    assert unflat_bands(RingsComponent().bands) == []


# =========================================================
# Bodies
# =========================================================

def test_ringed_bodies():

    presets = load_presets()

    saturn = components_for(presets["saturn"]).rings
    uranus = components_for(presets["uranus"]).rings

    assert saturn.inner_radius == pytest.approx(74_500e3)
    assert saturn.outer_radius == pytest.approx(140_500e3)
    assert len(unflat_bands(saturn.bands)) == 6

    # Uranus's rings are narrow and dark.
    assert max(uranus.color) < 0.1
    assert all(outer - inner < 1_000e3 for inner, outer, _ in unflat_bands(uranus.bands))

    assert components_for(presets["earth"]).rings is None


def test_ring_bands_are_validated():

    data = json.loads(open("assets/bodies/saturn.json", encoding="utf-8").read())

    data["rings"]["bands"][0] = [92000, 74500, 0.1]

    with pytest.raises(ProfileError, match="ring band"):
        parse_profile(data, "saturn")


# =========================================================
# Shader Inputs
# =========================================================

def test_rings_are_packed():

    p = AtmosphereParameters.from_components(58_232_000.0, AtmosphereComponent())

    v = np.frombuffer(pack_atmosphere_block(p, rings=(74_500.0, 140_500.0, 1.2)), dtype=np.float32).reshape(14, 4)

    np.testing.assert_allclose(v[13], (74_500.0, 140_500.0, 1.0, 1.2))

    off = np.frombuffer(pack_atmosphere_block(p), dtype=np.float32).reshape(14, 4)

    assert off[13, 2] == 0.0
