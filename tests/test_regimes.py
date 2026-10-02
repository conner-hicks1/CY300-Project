import numpy as np
import pytest

from planet.bodies import components_for, load_presets, relief_scale
from planet.chunk import build_chunk
from planet.cube_sphere import ChunkKey
from planet.regimes import (
    DEFAULT_TIME_STEPS,
    EpisodicResurfacing,
    HeatPipe,
    IceShell,
    StagnantLid
)
from planet.tectonics import (
    TectonicField,
    TectonicSettings,
    TectonicSimulation,
    simulation_for,
    surface_elevation
)
from planet.terrain import Terrain, TerrainSettings


REGIMES = ("stagnant_lid", "episodic_resurfacing", "heat_pipe", "ice_shell")


def settings(regime: str, **overrides) -> TectonicSettings:

    values = dict(
        regime=regime,
        resolution=32,
        radius=2_000_000.0,
        time_step=DEFAULT_TIME_STEPS[regime]
    )

    values.update(overrides)

    return TectonicSettings(**values)


@pytest.fixture(scope="module", params=REGIMES)
def regime_run(request):

    simulation = simulation_for(settings(request.param))

    return request.param, simulation, simulation.initial_state()


# =========================================================
# Common Behavior
# =========================================================

def test_factory():

    assert isinstance(simulation_for(TectonicSettings()), TectonicSimulation)
    assert isinstance(simulation_for(settings("stagnant_lid")), StagnantLid)
    assert isinstance(simulation_for(settings("episodic_resurfacing")), EpisodicResurfacing)
    assert isinstance(simulation_for(settings("heat_pipe")), HeatPipe)
    assert isinstance(simulation_for(settings("ice_shell")), IceShell)


def test_initial_state_is_one_still_plate(regime_run):

    regime, simulation, state = regime_run

    n = simulation.grid.cell_count

    assert state.regime == regime
    assert state.height.shape == (n,)
    assert np.all(np.isfinite(state.height))
    assert np.all(state.plate == 0)
    assert state.plate_total == 1 and state.speeds[0] == 0.0

    assert 0.0 <= state.continental.min() and state.continental.max() <= 1.0

    # The terrain reads the regime's height directly.
    np.testing.assert_array_equal(surface_elevation(state), state.height)

    [plate] = simulation.plates(state)

    assert plate.cell_fraction == 1.0


def test_deterministic(regime_run):

    regime, simulation, state = regime_run

    again = simulation_for(settings(regime)).initial_state()

    np.testing.assert_array_equal(state.height, again.height)

    np.testing.assert_array_equal(
        simulation.step(state).height,
        simulation_for(settings(regime)).step(again).height
    )


def test_step_advances_time_without_touching_the_input(regime_run):

    regime, simulation, state = regime_run

    before = state.height.copy()

    after = simulation.step(state)

    np.testing.assert_array_equal(state.height, before)

    assert after.time == pytest.approx(state.time + simulation.settings.time_step)
    assert after.step_index == state.step_index + 1
    assert after.regime == regime
    assert np.all(np.isfinite(after.height))


def test_field_carries_regime_and_age_scale(regime_run):

    regime, simulation, state = regime_run

    field = TectonicField.from_state(simulation.grid, state, 1)

    assert field.regime == regime
    assert field.age.max() <= field.age_scale


def test_relief_scales(regime_run):

    regime, _, state = regime_run

    taller = simulation_for(settings(regime, relief_scale=2.0)).initial_state()

    assert np.ptp(taller.height) > 1.5 * np.ptp(state.height)


# =========================================================
# Each Regime's Character
# =========================================================

def test_stagnant_lid_floods_its_lowlands_with_dark_lava():

    state = simulation_for(settings("stagnant_lid", resolution=48)).initial_state()

    low = state.height < np.percentile(state.height, 10)
    high = state.height > np.percentile(state.height, 70)

    assert state.continental[low].mean() < 0.2 < state.continental[high].mean()

    # Ancient: most crust is billions of years old.
    assert np.median(state.age) > 3_000.0


def test_venus_resurfacing_resets_the_plains():

    simulation = simulation_for(settings("episodic_resurfacing", resolution=48))

    state = simulation.initial_state()

    state.memory["next_resurfacing"] = 1.0

    after = simulation.step(state)

    survived = after.continental > 0.5

    # Flooded plains are brand new; surviving tesserae old.
    assert np.median(after.age[~survived]) < 1.0
    assert survived.mean() < 0.3
    assert after.memory["next_resurfacing"] > 300.0


def test_io_is_young_with_lava_lakes_and_tall_mountains():

    state = simulation_for(settings("heat_pipe", resolution=64)).initial_state()

    assert np.median(state.age) < 5.0
    assert (state.height < 0.0).mean() > 0.002          # caldera floors below the lava level
    assert state.height.max() > 2_000.0
    assert state.activity.max() > 0.0


def test_ice_shell_has_low_relief_and_dark_lineae():

    state = simulation_for(settings("ice_shell", resolution=64)).initial_state()

    assert np.abs(state.height).max() < 2_000.0
    assert state.continental.min() < 0.5
    assert np.median(state.continental) > 0.6
    assert state.age.max() < 100.0


# =========================================================
# Terrain and Bodies
# =========================================================

@pytest.mark.parametrize("regime", REGIMES)
def test_terrain_stays_within_its_bounds(regime):

    simulation = simulation_for(settings(regime))

    state = simulation.initial_state()

    field = TectonicField.from_state(simulation.grid, state, 1)

    terrain = Terrain(
        TerrainSettings(radius=2_000_000.0, has_liquid=False, mountain_height=3_000.0),
        field
    )

    for key in (ChunkKey(0, 1, 0, 0), ChunkKey(3, 2, 1, 1)):

        data = build_chunk(key, terrain, 9)

        assert terrain.min_elevation <= data.min_elevation
        assert data.max_elevation <= terrain.max_elevation


def test_bodies_get_their_regimes():

    presets = load_presets()

    expected = {
        "earth": "plate_tectonics",
        "mars": "stagnant_lid",
        "moon": "stagnant_lid",
        "venus": "episodic_resurfacing",
        "io": "heat_pipe",
        "europa": "ice_shell",
    }

    for body, regime in expected.items():

        tectonics = components_for(presets[body]).tectonics

        assert tectonics.regime == regime
        assert tectonics.time_step == DEFAULT_TIME_STEPS[regime]

    assert components_for(presets["jupiter"]).tectonics is None

    # Weaker gravity, taller relief.
    assert relief_scale(3.71) > relief_scale(9.81) == pytest.approx(1.0)


# =========================================================
# System
# =========================================================

@pytest.mark.parametrize("regime", REGIMES)
def test_system_runs_every_regime(regime):

    from core.jobs import JobSystem
    from ecs.components import PlanetComponent, TectonicsComponent
    from scene.scene import Scene
    from systems.tectonics_system import TectonicsSystem
    from tests.test_tectonics import pump

    jobs = JobSystem(workers=2)

    system = TectonicsSystem(jobs)

    scene = Scene("regime test")

    entity = scene.create_entity()

    scene.add_component(entity, PlanetComponent(radius=2_000_000.0, liquid="none"))

    component = TectonicsComponent(
        regime=regime,
        resolution=24,
        time_step=DEFAULT_TIME_STEPS[regime]
    )

    scene.add_component(entity, component)

    try:

        pump(scene, system, jobs, lambda: system.field(entity) is not None)

        first = system.field(entity)

        assert first.regime == regime

        system.step(entity, 2)

        pump(scene, system, jobs, lambda: system.field(entity).time >= 2 * component.time_step - 1e-9)

        assert component.simulated_time == pytest.approx(2 * component.time_step)
        assert len(system.status(entity, component).plates) == 1

    finally:

        system.shutdown()
        jobs.shutdown(wait=True)
