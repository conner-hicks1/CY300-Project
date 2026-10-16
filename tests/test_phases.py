import math

import pytest

from planet.bodies import components_for, derived_frost_point, load_presets
from planet.phases import (
    SUBSTANCES,
    boiling_point,
    frost_point,
    liquid_range,
    vapor_pressure
)


WATER = SUBSTANCES["water"]
METHANE = SUBSTANCES["methane"]
NITROGEN = SUBSTANCES["nitrogen"]
CARBON_DIOXIDE = SUBSTANCES["carbon_dioxide"]


# =========================================================
# Phase Physics
# =========================================================

def test_boiling_points():

    assert boiling_point(WATER, 1.013) == pytest.approx(100.0, abs=1.0)
    assert boiling_point(NITROGEN, 1.013) == pytest.approx(-195.8, abs=2.0)
    assert boiling_point(METHANE, 1.013) == pytest.approx(-161.5, abs=2.5)

    # Higher pressure, higher boiling point; capped at the
    # critical point.
    assert boiling_point(WATER, 10.0) > boiling_point(WATER, 1.0)
    assert boiling_point(WATER, 500.0) == pytest.approx(WATER.critical_k - 273.15)


def test_no_liquid_below_the_triple_point():

    # Mars's 6 mbar is just under water's triple point.
    assert boiling_point(WATER, 0.006) is None

    phase = liquid_range("water", 0.006)

    assert not phase.can_be_liquid
    assert math.isinf(phase.freezing)


def test_vapor_pressure_is_continuous_at_the_triple_point():

    t = WATER.triple_k - 273.15

    assert vapor_pressure(WATER, t - 1e-6) == pytest.approx(WATER.triple_bar, rel=1e-4)
    assert vapor_pressure(WATER, t + 1e-6) == pytest.approx(WATER.triple_bar, rel=1e-4)


def test_frost_points():

    # Mars's carbon dioxide caps: ~-125 C.
    assert frost_point(CARBON_DIOXIDE, 0.006 * 0.95) == pytest.approx(-126.0, abs=4.0)

    # Pluto's nitrogen ice: ~-236 C.
    assert frost_point(NITROGEN, 1e-5) == pytest.approx(-236.0, abs=3.0)

    # Round trip: the vapor pressure at the frost point is
    # the partial pressure.
    point = frost_point(WATER, 1e-6)

    assert vapor_pressure(WATER, point) == pytest.approx(1e-6, rel=1e-6)

    # A gas above its triple point pressure condenses as a
    # liquid and freezes at the freezing point.
    assert frost_point(WATER, 0.05) == WATER.freezing_c


def test_lava_and_none_have_no_phase_range():

    assert liquid_range("lava", 1.0) is None
    assert liquid_range("none", 1.0) is None


# =========================================================
# Bodies
# =========================================================

@pytest.fixture(scope="module")
def presets():

    return load_presets()


def test_body_ices(presets):

    # Earth's snow: its own seas' freezing point.
    assert presets["earth"].ice == "water"
    assert presets["earth"].frost_point_c == pytest.approx(-1.9)

    # Mars's residual caps: water ice at a trace of vapor.
    assert presets["mars"].frost_point_c == pytest.approx(-76.0, abs=3.0)

    assert presets["pluto"].ice == "nitrogen"
    assert presets["pluto"].frost_point_c == pytest.approx(-236.0, abs=3.0)

    assert presets["moon"].ice == "none"

    assert presets["earth"].life and not presets["mars"].life

    parts = components_for(presets["mars"])

    assert parts.planet.ice == "water"
    assert not parts.planet.life


def test_frost_from_composition(presets):

    # Carbon dioxide frost on Mars from its 95% CO2 air.
    frost = derived_frost_point("carbon_dioxide", "none", presets["mars"].atmosphere)

    assert frost == pytest.approx(-127.0, abs=4.0)

    assert derived_frost_point("none", "none", None) < -500.0
