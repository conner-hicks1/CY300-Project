import math

import numpy as np
import pytest

from ecs.components import ClimateComponent, PlanetComponent
from planet.bodies import components_for, derived_frost_point, load_presets
from planet.climate import ClimateField, ClimateModel, ClimateSettings
from planet.phases import (
    SUBSTANCES,
    boiling_point,
    frost_point,
    liquid_range,
    vapor_pressure
)
from systems.climate_system import climate_settings_for
from systems.planet_system import terrain_settings_for


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


# =========================================================
# Climate
# =========================================================

def ocean_world(**settings):

    model = ClimateModel(ClimateSettings(resolution=16, **settings))

    state = model.compute(np.full(model.grid.cell_count, -1000.0))

    return model, state


def test_earth_seas_are_liquid_with_polar_ice():

    _, state = ocean_world()

    assert state.liquid_state == "partly frozen"
    assert 0.0 < state.frozen_fraction < 0.3


def test_seas_boil_away_close_to_the_star():

    _, state = ocean_world(stellar_flux=1361.0 * 4.0)

    assert state.liquid_state == "boiled away"


def test_seas_freeze_far_from_the_star():

    _, state = ocean_world(stellar_flux=1361.0 / 9.0)

    assert state.liquid_state == "frozen"
    assert state.frozen_fraction > 0.98


def test_boiled_seas_leave_dry_basins():

    model, state = ocean_world(stellar_flux=1361.0 * 4.0)

    field = ClimateField.from_state(model.grid, state, 1)

    assert field.liquid_boiled

    planet = PlanetComponent()

    assert terrain_settings_for(planet).has_liquid
    assert not terrain_settings_for(planet, field).has_liquid


def test_climate_settings_carry_the_liquid_range(presets):

    titan = components_for(presets["titan"])

    settings = climate_settings_for(titan.climate, titan.planet, titan.body)

    assert settings.liquid_freezing == pytest.approx(-200.0)
    assert settings.liquid_boiling == pytest.approx(-155.0, abs=3.0)

    # Water on a dry body at Mars's pressure: ice only.
    planet = PlanetComponent(liquid="water")

    mars = components_for(presets["mars"])

    settings = climate_settings_for(ClimateComponent(), planet, mars.body)

    assert math.isinf(settings.liquid_freezing)
    assert not settings.ocean

    # No liquid, no phase range.
    settings = climate_settings_for(ClimateComponent(), PlanetComponent(liquid="none"))

    assert settings.liquid_boiling is None
