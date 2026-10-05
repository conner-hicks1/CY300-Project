import math

from datetime import datetime, timezone

import numpy as np
import pytest

from graphics.bodies_block import BodySphere, pack_bodies_block
from graphics.comets import (
    ACTIVITY_LIMIT,
    MAX_COMETS,
    activity,
    coma_radiance,
    comet_view,
    dust_tail_length,
    pack_comet_uniforms
)
from graphics.star_field import MILKY_WAY_BRIGHTNESS, reflected_illuminance
from graphics.uniform_blocks import BODIES_BLOCK, MAX_BODIES
from planet.bodies import components_for, load_presets
from planet.orbits import date_of, seconds_since_j2000
from planet.stars import (
    catalog_available,
    galactic_basis,
    illuminance_to_magnitude,
    load_catalog,
    magnitude_to_illuminance,
    parse_catalog,
    radec_to_engine,
    random_catalog,
    temperature_of_color_index
)


AU = 1.495978707e11


@pytest.fixture(scope="module")
def presets():

    return load_presets()


# =========================================================
# The Stars
# =========================================================

def test_magnitudes_are_physical():

    # The Sun at Earth is the engine's sunlight, 5.
    assert magnitude_to_illuminance(-26.74) == pytest.approx(5.0)

    # Five magnitudes: a factor of 100.
    assert magnitude_to_illuminance(1.0) / magnitude_to_illuminance(6.0) == pytest.approx(100.0)

    # Sirius: ~8e-11 of sunlight.
    assert magnitude_to_illuminance(-1.46) / 5.0 == pytest.approx(7.7e-11, rel=0.02)

    assert illuminance_to_magnitude(magnitude_to_illuminance(3.3)) == pytest.approx(3.3)


def test_star_colors_from_color_index():

    # The Sun (B - V 0.65) ~5,800 K; Vega (0.0) ~10,000 K;
    # Betelgeuse (1.85) ~3,300 K.
    assert temperature_of_color_index(0.65) == pytest.approx(5800.0, rel=0.04)
    assert temperature_of_color_index(0.0) == pytest.approx(10_000.0, rel=0.1)
    assert temperature_of_color_index(1.85) == pytest.approx(3_300.0, rel=0.12)


def test_catalog_records_parse():

    # BSC5 fixed-width records: Sirius, and one without a
    # J2000 position (skipped).
    sirius = (
        " " * 75 + "064508.9-164258" + " " * 12 + "-1.46  +0.00"
    )
    blank = " " * 120

    ra, dec, magnitude, bv = parse_catalog([sirius, blank])

    assert len(ra) == 1

    assert ra[0] == pytest.approx(101.287, abs=0.01)
    assert dec[0] == pytest.approx(-16.716, abs=0.01)
    assert magnitude[0] == pytest.approx(-1.46)
    assert bv[0] == pytest.approx(0.0)


def test_galactic_axes():

    basis = galactic_basis()

    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-9)

    # The galactic center, and Deneb at l = 84.3, b = 2.0 deg.
    np.testing.assert_allclose(basis @ radec_to_engine(266.40510, -28.93617)[0], (1.0, 0.0, 0.0), atol=1e-5)

    g = basis @ radec_to_engine(310.358, 45.280)[0]

    assert math.degrees(math.atan2(g[1], g[0])) == pytest.approx(84.28, abs=0.05)
    assert math.degrees(math.asin(g[2])) == pytest.approx(2.0, abs=0.05)


def test_the_catalog_or_its_stand_in():

    catalog = load_catalog()

    if catalog_available():

        assert catalog.real
        assert len(catalog) > 9_000

        # Sirius is the brightest, in its place.
        brightest = int(np.argmin(catalog.magnitudes))

        assert catalog.magnitudes[brightest] == pytest.approx(-1.46)

        np.testing.assert_allclose(catalog.directions[brightest], radec_to_engine(101.287, -16.716)[0], atol=1e-4)

    else:

        assert not catalog.real

    # Colors have luminance 1.
    luminance = catalog.colors @ np.array([0.2126, 0.7152, 0.0722])

    np.testing.assert_allclose(luminance, 1.0, atol=1e-4)


def test_stand_in_has_the_real_counts():

    directions, magnitudes, _ = random_catalog()

    # ~9,000 to 6.5; a few hundred brighter than 3.5.
    assert 8_000 < len(magnitudes) < 10_000
    assert 100 < np.sum(magnitudes < 3.5) < 600

    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-9)


def test_milky_way_is_faint():

    # ~20 mag / arcsec^2: some 1e-8 of a sunlit surface's
    # radiance (5 / pi).
    assert 1e-9 < MILKY_WAY_BRIGHTNESS / (5.0 / math.pi) < 1e-7


# =========================================================
# Planets as Points, Planetshine
# =========================================================

def apparent_magnitude(sunlight, albedo, radius_km, distance_km, phase_deg):

    return float(illuminance_to_magnitude(
        reflected_illuminance(sunlight, albedo, radius_km, distance_km, math.radians(phase_deg))
    ))


def test_planets_shine_as_bright_as_they_do():

    # The full Moon: -12.7 (geometric albedo 0.12; the real
    # Moon is brighter at full by its opposition surge).
    assert apparent_magnitude(5.0, 0.12, 1737.4, 384_400.0, 0.0) == pytest.approx(-12.4, abs=0.5)

    # Jupiter at opposition: ~-2.7.
    assert apparent_magnitude(5.0 / 5.2 ** 2, 0.538, 69_911.0, 4.2 * AU / 1000.0, 0.0) == pytest.approx(-2.7, abs=0.3)

    # Venus near greatest brilliancy (~40 deg from the Sun,
    # crescent ~120 deg of phase): ~-4.5.
    assert apparent_magnitude(5.0 / 0.723 ** 2, 0.689, 6_051.8, 0.38 * AU / 1000.0, 120.0) == pytest.approx(-4.5, abs=0.5)


def test_phase_law():

    full = reflected_illuminance(1.0, 0.3, 1.0, 100.0, 0.0)

    # A Lambert sphere at quadrature: 1 / pi of full; nothing
    # at new.
    assert reflected_illuminance(1.0, 0.3, 1.0, 100.0, math.pi / 2) == pytest.approx(full / math.pi)
    assert reflected_illuminance(1.0, 0.3, 1.0, 100.0, math.pi) == pytest.approx(0.0, abs=1e-12)

    # Inside the body: none.
    assert reflected_illuminance(1.0, 0.3, 1.0, 0.5, 0.0) == 0.0


def test_earthshine_is_a_ten_thousandth_of_sunlight():

    # Full Earth over the Moon's night side.
    ratio = reflected_illuminance(1.0, 0.434, 6371.0, 384_400.0, 0.0)

    assert 5e-5 < ratio < 2e-4


def test_bodies_block_carries_their_light():

    body = BodySphere(center=(1e7, 0.0, 0.0), radius=6.371e6, light=(0.4, 0.45, 0.5))

    data = np.frombuffer(
        pack_bodies_block([body], np.zeros(3), 0.0047),
        dtype=np.float32
    ).reshape(-1, 4)

    assert len(data) * 16 == BODIES_BLOCK.size

    # After the header, spheres, glows and the ring's 4.
    np.testing.assert_allclose(data[1 + 2 * MAX_BODIES + 4, :3], (0.4, 0.45, 0.5), rtol=1e-6)


def test_bodies_know_how_they_look(presets):

    earth = components_for(presets["earth"]).body
    moon = components_for(presets["moon"]).body
    mars = components_for(presets["mars"]).body

    assert earth.geometric_albedo == pytest.approx(0.434)
    assert moon.geometric_albedo == pytest.approx(0.12)

    # Earth blue, Mars red.
    assert earth.disc_color[2] > earth.disc_color[0]
    assert mars.disc_color[0] > mars.disc_color[2]


# =========================================================
# Comets
# =========================================================

def test_activity_follows_the_sun():

    near = activity(30.0, 1.0)

    assert near == pytest.approx(30.0)

    # Steeply brighter inside 1 AU; nothing past ~4.5 AU.
    assert activity(30.0, 0.5) == pytest.approx(30.0 * 0.5 ** -3.5)
    assert activity(30.0, ACTIVITY_LIMIT[1] + 0.1) == 0.0
    assert activity(0.0, 1.0) == 0.0


def test_coma_brightness_from_afrho():

    # L(b) = E (A f rho) / (8 pi b): twice as far out, half
    # as bright.
    inner = coma_radiance(5.0, 10.0, 100.0)

    assert inner == pytest.approx(5.0 * 0.01 / (8.0 * math.pi * 100.0))
    assert coma_radiance(5.0, 10.0, 200.0) == pytest.approx(inner / 2.0)


def test_tails_grow_with_activity():

    assert dust_tail_length(200.0) == pytest.approx(3.0e6)
    assert dust_tail_length(800.0) == pytest.approx(6.0e6)
    assert dust_tail_length(1e-6) == pytest.approx(5.0e4)


def test_comet_view_geometry():

    sun = np.zeros(3)
    center = np.array([AU, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 0.0])

    view = comet_view(center, sun, normal, center + np.array([0.0, 0.0, 1e9]), 30.0, 1.0)

    # Tails away from the Sun; the dust lags the motion
    # (moving along normal x away = -z here, so it lags +z).
    np.testing.assert_allclose(view.away, (1.0, 0.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(view.lag, (0.0, 0.0, 1.0), atol=1e-9)

    assert view.center == pytest.approx((0.0, 0.0, -1e6))
    assert view.sunlight == pytest.approx(5.0)
    assert view.afrho == pytest.approx(0.03)

    # Too far out: inactive.
    assert comet_view(center * 6.0, sun, normal, center, 30.0, 1.0) is None


def test_comet_uniforms():

    sun = np.zeros(3)

    views = [
        comet_view(np.array([AU * (1.0 + 0.1 * i), 0.0, 0.0]), sun, (0.0, 1.0, 0.0), np.zeros(3), 30.0, 1.0)
        for i in range(MAX_COMETS + 2)
    ]

    packed = pack_comet_uniforms(views)

    assert packed["uCometCenter"].shape == (MAX_COMETS, 4)
    assert packed["uCometAxis"][0, 3] == pytest.approx(views[0].afrho)


def test_halley_returns_on_schedule(presets):

    halley = components_for(presets["halley"])

    assert halley.body.comet_afrho > components_for(presets["churyumov_gerasimenko"]).body.comet_afrho > 0.0

    orbit = halley.orbit

    # The last perihelion: 1986 February 9 (JPL: 9.46).
    mean = math.radians(orbit.mean_anomaly)

    perihelion = date_of(-mean / (2.0 * math.pi) * orbit.period)

    assert abs((perihelion - datetime(1986, 2, 9, 11, tzinfo=timezone.utc)).total_seconds()) < 2 * 86_400

    # Retrograde.
    assert orbit.inclination > 90.0
