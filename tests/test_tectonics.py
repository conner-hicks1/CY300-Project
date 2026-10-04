import time

import numpy as np
import pytest

from core.jobs import JobSystem
from ecs.components import PlanetComponent, TectonicsComponent
from planet.sphere_grid import SphereGrid
from planet.tectonics import (
    ABYSSAL_DEPTH,
    RIDGE_DEPTH,
    TectonicField,
    TectonicSettings,
    TectonicSimulation,
    _merge_plates,
    _rotation_matrix,
    ocean_depth,
    plate_velocities,
    surface_elevation
)
from planet.terrain import Terrain, TerrainSettings
from scene.scene import Scene
from systems.tectonics_system import TectonicsSystem


SMALL = TectonicSettings(resolution=32, plate_count=8)


@pytest.fixture(scope="module")
def simulation():

    return TectonicSimulation(SMALL)


@pytest.fixture(scope="module")
def initial(simulation):

    return simulation.initial_state()


# =========================================================
# Sphere Grid
# =========================================================

def test_grid_cells_map_to_themselves():

    grid = SphereGrid(24)

    assert grid.cell_count == 6 * 24 * 24
    np.testing.assert_array_equal(grid.cell_of(grid.directions), np.arange(grid.cell_count))


def test_grid_neighbors_are_mutual_and_adjacent():

    grid = SphereGrid(24)

    for cell in range(0, grid.cell_count, 7):

        for other in grid.neighbors[cell]:

            assert cell in grid.neighbors[other]

            angle = np.arccos(np.clip(grid.directions[cell] @ grid.directions[other], -1, 1))

            assert angle < 1.5 * grid.cell_angle


def test_grid_sampling_is_continuous_across_face_edges():

    grid = SphereGrid(32)

    field = grid.pad(grid.directions[:, 0] + 0.5 * grid.directions[:, 2])

    # Walk across the +X / +Z face edge in tiny steps.
    angles = np.linspace(np.pi / 4 - 0.05, np.pi / 4 + 0.05, 400)

    points = np.stack((np.sin(angles), np.zeros_like(angles), np.cos(angles)), axis=1)

    values = grid.sample(field, points)

    assert np.abs(np.diff(values)).max() < 0.01

    expected = points[:, 0] + 0.5 * points[:, 2]

    assert np.abs(values - expected).max() < 0.02


# =========================================================
# Simulation
# =========================================================

def test_initial_state(simulation, initial):

    cells = simulation.grid.cell_count

    assert initial.plate.shape == (cells,)
    assert len(np.unique(initial.plate)) == SMALL.plate_count

    land = (initial.continental > 0.5).mean()

    assert abs(land - SMALL.land_fraction) < 0.05

    # Plate speeds near the requested cm/yr.
    surface = initial.speeds * SMALL.radius / 10_000.0

    assert 0.3 * SMALL.plate_speed < np.median(surface) < 2.0 * SMALL.plate_speed


def test_deterministic(simulation, initial):

    again = TectonicSimulation(SMALL).initial_state()

    np.testing.assert_array_equal(initial.plate, again.plate)
    np.testing.assert_array_equal(initial.continental, again.continental)

    a = simulation.step(initial)
    b = simulation.step(again)

    np.testing.assert_array_equal(a.plate, b.plate)
    np.testing.assert_allclose(a.land_height, b.land_height)


def test_step_does_not_modify_input(simulation, initial):

    before = initial.continental.copy()

    simulation.step(initial)

    np.testing.assert_array_equal(initial.continental, before)


def test_time_advances_and_crust_ages(simulation, initial):

    state = simulation.step(initial)

    assert state.time == pytest.approx(SMALL.time_step)
    assert state.step_index == 1

    # Moved crust is older; brand-new crust is age 0.
    assert (state.age >= 0).all()
    assert np.median(state.age) > np.median(initial.age)


def test_long_run_keeps_earth_like_statistics(simulation, initial):

    state = initial

    for _ in range(40):
        state = simulation.step(state)

    elevation = surface_elevation(state)

    assert 0.15 < (state.continental > 0.5).mean() < 0.45
    assert ABYSSAL_DEPTH - 1.0 <= elevation.min()
    assert elevation.max() > 1_500.0            # mountains formed
    assert (state.activity > 0.5).any()         # converging boundaries
    assert (state.activity < -0.5).any()        # spreading boundaries
    assert (state.age == 0).any() or (state.age < SMALL.time_step).any()   # new crust

    plates = len(np.unique(state.plate))

    assert 2 <= plates <= 2 * SMALL.plate_count


def test_ocean_depth_deepens_with_age_and_levels_off():

    ages = np.array([0.0, 10.0, 50.0, 150.0, 1000.0])

    depths = ocean_depth(ages)

    assert depths[0] == pytest.approx(RIDGE_DEPTH)
    assert (np.diff(depths) < 0).all()
    assert depths[-1] == pytest.approx(ABYSSAL_DEPTH, abs=1.0)


def test_rotation_matrix_moves_points_along_the_velocity(initial):

    axis = np.array([0.0, 1.0, 0.0])

    rotation = _rotation_matrix(axis, 0.01)

    point = np.array([1.0, 0.0, 0.0])

    moved = rotation @ point

    velocity = np.cross(axis, point)

    np.testing.assert_allclose((moved - point) / 0.01, velocity, atol=0.01)


def test_plate_velocity_is_tangent(initial):

    directions = np.array([[0.0, 0.0, 1.0], [0.6, 0.8, 0.0]])

    velocity = plate_velocities(initial, directions, np.array([0, 1]))

    np.testing.assert_allclose(np.einsum("ij,ij->i", velocity, directions), 0.0, atol=1e-12)


def test_merging_plates_conserves_cells(simulation, initial):

    state = simulation.step(initial)

    keep, absorbed = sorted(np.unique(state.plate))[:2]

    sizes = np.bincount(state.plate)

    _merge_plates(state, keep, absorbed)

    assert not (state.plate == absorbed).any()
    assert np.bincount(state.plate)[keep] == sizes[keep] + sizes[absorbed]


# =========================================================
# Field and Terrain
# =========================================================

def test_field_matches_state(simulation, initial):

    field = TectonicField.from_state(simulation.grid, initial, version=3)

    directions = simulation.grid.directions

    np.testing.assert_allclose(
        field.sample("elevation", directions),
        surface_elevation(initial),
        atol=1.0
    )

    np.testing.assert_array_equal(field.plate_at(directions), initial.plate)

    assert field.version == 3


def test_terrain_follows_the_field(simulation, initial):

    field = TectonicField.from_state(simulation.grid, initial, version=0)

    settings = TerrainSettings(mountain_height=0.0, detail_height=0.0)

    terrain = Terrain(settings, field)

    directions = simulation.grid.directions

    base = surface_elevation(initial)

    elevation = terrain.elevation(directions, 50_000.0)

    # Same land / sea pattern as the simulation (coastline
    # noise only moves the coast a little).
    agreement = ((elevation > 0) == (base > 0)).mean()

    assert agreement > 0.9

    data = terrain.tectonic_data(directions)

    np.testing.assert_array_equal(data[:, 0], initial.plate)

    # The noise-only terrain reports no plates.
    assert (Terrain(settings).tectonic_data(directions[:5])[:, 0] == -1).all()


# =========================================================
# Tectonics System
# =========================================================

@pytest.fixture
def world():

    jobs = JobSystem(workers=2)

    system = TectonicsSystem(jobs)

    scene = Scene("tectonics test")

    entity = scene.create_entity()

    scene.add_component(entity, PlanetComponent())
    scene.add_component(entity, TectonicsComponent(resolution=24, plate_count=6))

    yield scene, system, jobs, entity

    system.shutdown()
    jobs.shutdown(wait=True)


def pump(scene, system, jobs, until, timeout=30.0):

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:

        system.update(scene)

        jobs.process_completions(budget_seconds=1.0)

        if until():
            return

        time.sleep(0.002)

    pytest.fail("condition never met")


def test_system_initializes_and_steps(world):

    scene, system, jobs, entity = world

    component = scene.get_component(entity, TectonicsComponent)

    pump(scene, system, jobs, lambda: system.field(entity) is not None)

    first = system.field(entity)

    assert first.time == 0.0

    system.step(entity)

    pump(scene, system, jobs, lambda: system.field(entity).version > first.version)

    assert system.field(entity).time == pytest.approx(component.time_step)
    assert component.simulated_time == pytest.approx(component.time_step)


def test_system_catches_up_to_saved_time(world):

    scene, system, jobs, entity = world

    component = scene.get_component(entity, TectonicsComponent)

    # As if loaded from a file at 20 Myr.
    component.simulated_time = 20.0

    pump(
        scene, system, jobs,
        lambda: system.field(entity) is not None and system.field(entity).time >= 20.0 - 1e-6
    )

    assert system.field(entity).time == pytest.approx(20.0)


def test_system_play_and_reset(world):

    scene, system, jobs, entity = world

    component = scene.get_component(entity, TectonicsComponent)

    pump(scene, system, jobs, lambda: system.field(entity) is not None)

    system.rate = 10_000.0

    system.play(entity)

    pump(scene, system, jobs, lambda: system.field(entity).time >= 15.0)

    system.play(entity, False)

    assert component.simulated_time >= 15.0

    system.reset(entity, component)

    assert component.simulated_time == 0.0

    pump(scene, system, jobs, lambda: system.field(entity).time == 0.0)


def test_system_drops_removed_planets(world):

    scene, system, jobs, entity = world

    pump(scene, system, jobs, lambda: system.field(entity) is not None)

    scene.remove_component(entity, TectonicsComponent)

    system.update(scene)

    assert system.field(entity) is None


def test_collisions_raise_young_mountain_belts():

    # Converging plates lift fresh mountain belts (recent
    # uplift, "orogeny"), not just the ground.
    simulation = TectonicSimulation(TectonicSettings(seed=3, resolution=32, plate_count=8))

    state = simulation.initial_state()

    for _ in range(4):
        state = simulation.step(state)

    converging = state.activity > 1.0

    assert converging.any()

    assert state.orogeny[converging].mean() > 2.0 * state.orogeny[~converging].mean()
