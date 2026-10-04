import numpy as np
import pytest

from graphics.atmosphere import (
    FLOOR_TEMPERATURE,
    AtmosphereParameters,
    deck_density,
    pack_atmosphere_block
)
from planet.air import AirColumn
from planet.bodies import air_column, components_for, load_presets


# =========================================================
# The Air Column
# =========================================================

@pytest.fixture(scope="module")
def presets():

    return load_presets()


def test_earth_column_matches_the_standard_atmosphere(presets):

    air = air_column(presets["earth"])

    # US standard atmosphere: 0.265 bar at 10 km, the
    # tropopause near 216 K.
    assert air.pressure(10.0) == pytest.approx(0.265, rel=0.05)
    assert air.temperature(20.0) == pytest.approx(216.0, abs=12.0)


def test_venus_column_matches_the_probes(presets):

    air = air_column(presets["venus"])

    # Venera / Pioneer Venus: ~1 bar near 50 km at ~350 K.
    assert air.pressure(50.0) == pytest.approx(1.0, rel=0.35)
    assert air.temperature(50.0) == pytest.approx(350.0, abs=25.0)


def test_jupiter_heats_along_the_adiabat_below_the_tops(presets):

    air = air_column(presets["jupiter"])

    # Galileo probe: ~22 bar and ~425 K some 150 km down.
    assert air.adiabatic_lapse == pytest.approx(2.0, rel=0.1)
    assert air.temperature(-150.0) == pytest.approx(450.0, abs=40.0)
    assert air.pressure(-150.0) == pytest.approx(20.0, rel=0.3)

    # Above the tops: near-isothermal.
    assert air.temperature(30.0) == air.temperature(0.0)


def test_altitude_of_pressure_inverts_pressure(presets):

    air = air_column(presets["jupiter"])

    for bar in (0.3, 1.0, 5.0, 40.0):
        assert air.pressure(air.altitude_of_pressure(bar)) == pytest.approx(bar, rel=1e-4)


def test_column_from_components_matches_the_profile(presets):

    for name in ("earth", "venus", "jupiter", "titan"):

        profile = presets[name]

        components = components_for(profile)

        expected = air_column(profile)

        air = AirColumn.from_scale_height(
            surface_pressure=components.body.surface_pressure_bar,
            surface_temperature=components.body.mean_temperature + 273.15,
            scale_height_km=components.atmosphere.rayleigh_scale_height / 1000.0,
            gravity=components.body.surface_gravity,
            adiabatic_exponent=components.atmosphere.adiabatic_exponent,
            lapse_rate=components.body.lapse_rate,
            giant=expected.giant,
            skin_temperature=expected.skin_temperature
        )

        assert air.molar_mass == pytest.approx(expected.molar_mass, rel=1e-3)
        assert air.pressure(-20.0 if air.giant else 8.0) == pytest.approx(
            expected.pressure(-20.0 if air.giant else 8.0), rel=1e-3
        )


# =========================================================
# Cloud Decks
# =========================================================

def _decks(atmosphere):

    return [atmosphere.decks[i:i + 9] for i in range(0, len(atmosphere.decks), 9) if atmosphere.decks[i + 2] > 0.0]


def test_jupiter_decks_condense_in_order_below_the_tops(presets):

    decks = _decks(components_for(presets["jupiter"]).atmosphere)

    # Ammonia, ammonium hydrosulfide, water: each deeper.
    assert len(decks) == 3

    tops = [d[1] for d in decks]

    assert tops == sorted(tops, reverse=True)
    assert all(d[0] < d[1] <= 1.0 for d in decks)

    # The water deck: thickest, with lightning, ~5 bar.
    water = decks[2]

    assert water[2] == max(d[2] for d in decks)
    assert water[8] > 0.0
    assert -110_000.0 < water[0] < water[1] < -60_000.0


def test_deck_density_is_soft_edged():

    assert deck_density(50.0, 40.0, 60.0) == pytest.approx(1.0)
    assert deck_density(80.0, 40.0, 60.0) == pytest.approx(0.0)
    assert 0.0 < deck_density(40.0, 40.0, 60.0) < 1.0


def test_decks_are_part_of_the_optical_depth(presets):

    venus = components_for(presets["venus"])

    p = AtmosphereParameters.from_components(venus.planet.radius, venus.atmosphere)

    # The clouds: an optical depth of ~30 between 31 and 90 km.
    clouds = sum(d[2] * (d[1] - d[0]) for d in p.decks)

    assert clouds == pytest.approx(27.8, rel=0.01)

    inside, above = p.extinction(np.array([55.0, 95.0]))

    assert inside[1] > 10.0 * above[1]


# =========================================================
# Packing and the Deep Air
# =========================================================

def test_decks_and_heat_are_packed(presets):

    jupiter = components_for(presets["jupiter"])

    p = AtmosphereParameters.from_components(jupiter.planet.radius, jupiter.atmosphere)

    v = np.frombuffer(
        pack_atmosphere_block(p, sun_direction=np.array([0.0, 1.0, 0.0]), no_surface=True, lightning_time=12.5),
        dtype=np.float32
    ).reshape(27, 4)

    # Temperature at the tops, the adiabat, cp/R - 1.
    assert v[13, 0] == pytest.approx(p.temperature)
    assert v[13, 1] == pytest.approx(p.lapse_rate, rel=1e-6)
    assert v[13, 2] == pytest.approx(p.adiabatic_exponent - 1.0, rel=1e-6)
    assert v[13, 3] > 0.0

    # Deck count, the lightning clock.
    assert v[14, 0] == 3.0
    assert v[14, 1] == pytest.approx(12.5)

    # The water deck: shape, scattering, absorption,
    # lightning.
    base, top, extinction, albedo, *_ , lightning = p.decks[2]

    assert v[15 + 6, 0] == pytest.approx(base, rel=1e-6)
    assert v[15 + 6, 1] == pytest.approx(top, rel=1e-6)
    np.testing.assert_allclose(v[15 + 7, :3], extinction * np.asarray(albedo), rtol=1e-5)
    np.testing.assert_allclose(v[15 + 8, :3], extinction * (1.0 - np.asarray(albedo)), rtol=1e-4)
    assert v[15 + 8, 3] == pytest.approx(lightning)

    # The model's bottom.
    assert v[12, 2] == pytest.approx(p.floor_depth)


def test_giants_are_modeled_down_to_glowing_heat(presets):

    for name in ("jupiter", "saturn", "uranus", "neptune"):

        components = components_for(presets[name])

        p = AtmosphereParameters.from_components(components.planet.radius, components.atmosphere)

        air = air_column(presets[name])

        # Where the adiabat reaches the floor's temperature:
        # far below the water clouds.
        assert air.temperature(-p.floor_depth) == pytest.approx(FLOOR_TEMPERATURE, rel=0.02)

        assert p.floor_depth > 5.0 * -min(d[0] for d in p.decks)


def test_rocky_worlds_have_no_deep_air(presets):

    earth = components_for(presets["earth"])

    p = AtmosphereParameters.from_components(earth.planet.radius, earth.atmosphere)

    assert p.deep_absorption == 0.0
    assert p.decks == ()
