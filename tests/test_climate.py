import math
import time

import numpy as np
import pytest

from core.jobs import JobSystem
from ecs.components import ClimateComponent, PlanetComponent
from planet.climate import (
    BIOMES,
    LAPSE_RATE,
    ClimateField,
    ClimateModel,
    ClimateSettings,
    annual_sunlight,
    classify_biomes,
    fallback_surface,
    prevailing_wind,
    sea_level_temperature
)
from planet.sphere_grid import SphereGrid
from scene.scene import Scene
from systems.climate_system import ClimateSystem


SETTINGS = ClimateSettings(resolution=24)


def latitude_of(grid: SphereGrid) -> np.ndarray:

    return np.degrees(np.arcsin(np.clip(grid.directions[:, 1], -1.0, 1.0)))


# =========================================================
# Sunlight and Temperature
# =========================================================

def test_annual_sunlight_matches_earth():

    # Earth: equator ~1.24x the mean, poles ~0.52x.
    assert annual_sunlight(0.0, 23.44) == pytest.approx(1.24, abs=0.01)
    assert annual_sunlight(1.0, 23.44) == pytest.approx(0.52, abs=0.01)

    # Averaged over the sphere (uniform in sin(latitude)) it is 1.
    x = np.linspace(-1.0, 1.0, 20_001)

    assert annual_sunlight(x, 23.44).mean() == pytest.approx(1.0, abs=1e-3)


def test_more_tilt_evens_out_sunlight():

    low = annual_sunlight(1.0, 5.0) / annual_sunlight(0.0, 5.0)
    high = annual_sunlight(1.0, 45.0) / annual_sunlight(0.0, 45.0)

    assert high > low


def test_temperature_offset_shifts_everything():

    base = sea_level_temperature(np.array([0.0, 0.7]), ClimateSettings())
    warm = sea_level_temperature(np.array([0.0, 0.7]), ClimateSettings(temperature_offset=5.0))

    np.testing.assert_allclose(warm - base, 5.0)


# =========================================================
# Winds
# =========================================================

def test_wind_belts():

    def wind_at(latitude_degrees):

        phi = math.radians(latitude_degrees)

        direction = np.array([[0.0, math.sin(phi), math.cos(phi)]])

        return prevailing_wind(direction)[0], direction[0]

    east = np.array([1.0, 0.0, 0.0])        # at longitude 0, east is +X

    trades, _ = wind_at(15.0)
    westerlies, _ = wind_at(45.0)

    assert trades @ east < 0.0           # from the east
    assert westerlies @ east > 0.0       # from the west

    # Trades blow toward the equator in both hemispheres.
    north_trades, _ = wind_at(15.0)
    south_trades, _ = wind_at(-15.0)

    assert north_trades[1] < 0.0 < south_trades[1]

    # Always tangent to the sphere.
    for latitude in (-70.0, -20.0, 0.0, 35.0, 80.0):

        wind, direction = wind_at(latitude)

        assert abs(wind @ direction) < 1e-9


# =========================================================
# Model
# =========================================================

@pytest.fixture(scope="module")
def ocean_world():

    model = ClimateModel(SETTINGS)

    return model, model.compute(np.full(model.grid.cell_count, -3_000.0))


def test_ocean_world_is_warm_at_the_equator_and_cold_at_the_poles(ocean_world):

    model, state = ocean_world

    latitude = np.abs(latitude_of(model.grid))

    equator = state.temperature[latitude < 10].mean()
    poles = state.temperature[latitude > 75].mean()

    assert 20.0 < equator < 35.0
    assert poles < 0.0


def test_rain_belts(ocean_world):

    model, state = ocean_world

    latitude = np.abs(latitude_of(model.grid))

    tropics = state.precipitation[latitude < 8].mean()
    subtropics = state.precipitation[(latitude > 25) & (latitude < 35)].mean()

    # Wet equator (rising air), dry horse latitudes (sinking air).
    assert tropics > 2.0 * subtropics

    # Normalized to an Earth-like global mean.
    assert state.precipitation.mean() == pytest.approx(1_000.0, rel=1e-3)


def test_mountains_are_colder_and_cast_rain_shadows():

    model = ClimateModel(SETTINGS)

    grid = model.grid

    elevation = np.full(grid.cell_count, -3_000.0)

    latitude = latitude_of(grid)

    longitude = np.degrees(np.arctan2(grid.directions[:, 0], grid.directions[:, 2]))

    # A continent in the westerlies (35-55 N), with a
    # north-south mountain range down its middle.
    land = (latitude > 35) & (latitude < 55) & (np.abs(longitude) < 40)

    elevation[land] = 300.0

    ridge = land & (np.abs(longitude) < 6)

    elevation[ridge] = 4_000.0

    state = model.compute(elevation)

    # Lapse rate: the range is colder than the lowland next to it.
    lowland = land & (np.abs(longitude) > 10) & (np.abs(longitude) < 20)

    assert state.temperature[ridge].mean() < state.temperature[lowland].mean() - 15.0

    # Westerlies blow from -longitude to +longitude: the
    # windward (west) side is wetter than the lee (east).
    west = land & (longitude < -8) & (longitude > -20)
    east = land & (longitude > 8) & (longitude < 20)

    assert state.precipitation[west].mean() > 1.3 * state.precipitation[east].mean()


def test_field_surface_applies_lapse_rate(ocean_world):

    model, state = ocean_world

    field = ClimateField.from_state(model.grid, state, version=1)

    directions = model.grid.directions[:10]

    low, _ = field.surface(directions, np.zeros(10))
    high, rain = field.surface(directions, np.full(10, 2_000.0))

    np.testing.assert_allclose(low - high, 2_000.0 * LAPSE_RATE, atol=1e-4)

    assert (rain >= 0.0).all()


# =========================================================
# Biomes
# =========================================================

@pytest.mark.parametrize("temperature, precipitation, biome", [
    (-20.0, 200.0, "Ice"),
    (-2.0, 300.0, "Tundra"),
    (2.0, 600.0, "Taiga"),
    (12.0, 1_500.0, "Temperate forest"),
    (12.0, 400.0, "Grassland"),
    (25.0, 100.0, "Desert"),
    (25.0, 900.0, "Savanna"),
    (26.0, 3_000.0, "Tropical rainforest"),
])
def test_biome_classes(temperature, precipitation, biome):

    assert BIOMES[int(classify_biomes(temperature, precipitation))] == biome


def test_fallback_without_a_model():

    directions = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    temperature, precipitation = fallback_surface(directions, np.array([0.0, 0.0]), np.array([0.5, 0.5]))

    assert temperature[0] > temperature[1] + 30.0
    assert (precipitation > 0).all()


# =========================================================
# Climate System
# =========================================================

def test_system_computes_and_follows_settings():

    jobs = JobSystem(workers=2)

    system = ClimateSystem(jobs)

    try:

        scene = Scene("climate test")

        entity = scene.create_entity()

        scene.add_component(entity, PlanetComponent())

        component = ClimateComponent(resolution=16)

        scene.add_component(entity, component)

        def pump(until):

            deadline = time.monotonic() + 30.0

            while time.monotonic() < deadline:

                system.update(scene)

                jobs.process_completions(budget_seconds=1.0)

                if until():
                    return

                time.sleep(0.002)

            pytest.fail("climate never computed")

        pump(lambda: system.field(entity) is not None)

        first = system.field(entity)

        component.temperature_offset = 10.0

        pump(lambda: system.field(entity).version != first.version)

        assert system.field(entity).mean_temperature > first.mean_temperature + 5.0

        scene.remove_component(entity, ClimateComponent)

        system.update(scene)

        assert system.field(entity) is None

    finally:

        system.shutdown()
        jobs.shutdown(wait=True)
