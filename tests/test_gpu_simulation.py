import math

import numpy as np
import pytest

from planet import acceleration
from planet.climate import solve_temperature
from planet.hydrology import drainage
from planet.sphere_grid import sphere_grid
from planet.tectonics import _rotation_matrix, rotated_cells


# =========================================================
# GPU kernels against their NumPy references
# =========================================================
#
# Needs an OpenGL 4.3 context (a hidden window); skipped
# without one.


@pytest.fixture(scope="module")
def gpu():

    glfw = pytest.importorskip("glfw")

    if not glfw.init():
        pytest.skip("no windowing system")

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(64, 64, "compute tests", None, None)

    if not window:

        glfw.terminate()

        pytest.skip("no OpenGL 4.5 context")

    glfw.make_context_current(window)

    from graphics.compute import GpuQueue, compute_supported
    from graphics.gpu_simulation import GpuSimulation

    if not compute_supported():

        glfw.destroy_window(window)
        glfw.terminate()

        pytest.skip("no compute shaders")

    simulation = GpuSimulation(GpuQueue())

    yield simulation

    simulation.delete()

    glfw.destroy_window(window)
    glfw.terminate()


def test_rotated_cells(gpu):

    grid = sphere_grid(48)

    rotations = np.stack([
        _rotation_matrix(np.array(axis) / np.linalg.norm(axis), angle)
        for axis, angle in (((0.2, 1.0, 0.1), 0.03), ((1.0, -0.3, 0.5), -0.05), ((0.0, 0.0, 1.0), 0.2))
    ])

    expected = rotated_cells(grid.directions, rotations, grid.n)
    found = gpu.rotated_cells(grid.directions, rotations, grid.n)

    assert found.shape == expected.shape

    # Single precision differs only for directions on a cell
    # border.
    assert np.mean(found == expected) > 0.999

    assert gpu.enabled


def climate_inputs(n=32):

    grid = sphere_grid(n)

    latitude = grid.directions[:, 1]

    absorbed = 340.0 * (1.0 - 0.3) * (1.2 - 0.6 * latitude ** 2)

    emissivity = np.full(grid.cell_count, 0.6)

    transport = 0.6 * 4.0 / grid.cell_angle ** 2 * (1.0 + 3.0 * (grid.directions[:, 0] > 0.3))

    conductance = (transport[:, None] + transport[grid.neighbors]) * 0.125

    return absorbed, emissivity, conductance, grid.neighbors


def test_temperature_solve(gpu):

    absorbed, emissivity, conductance, neighbors = climate_inputs()

    expected = solve_temperature(absorbed, emissivity, conductance, neighbors, 255.0)
    found = gpu.solve_temperature(absorbed, emissivity, conductance, neighbors, 255.0)

    np.testing.assert_allclose(found, expected, atol=0.05)

    assert gpu.enabled


def test_drainage(gpu):

    grid = sphere_grid(48)

    d = grid.directions

    # Continents with basins and valleys.
    elevation = (
        3_000.0 * np.sin(3.0 * d[:, 0]) * np.cos(2.0 * d[:, 2])
        + 800.0 * np.sin(11.0 * d[:, 1] + 4.0 * d[:, 0])
        - 600.0
    )

    ocean = elevation < 0.0

    runoff = np.where(ocean, 0.0, 1.0 + 0.5 * np.sin(7.0 * d[:, 2]))

    filled_cpu, receiver_cpu, discharge_cpu = drainage(elevation, ocean, grid.neighbors, runoff)
    filled_gpu, receiver_gpu, discharge_gpu = gpu.drainage(elevation, ocean, grid.neighbors, runoff)

    # The same filled surface (lakes to their spill points).
    np.testing.assert_allclose(filled_gpu, filled_cpu, atol=0.05)

    # Every land cell drains (where the CPU's does), all
    # runoff reaches the sea, and big rivers agree (ties
    # between equally low neighbors may route differently).
    land = ~ocean

    assert np.array_equal(receiver_gpu[land] >= 0, receiver_cpu[land] >= 0)

    def into_sea(discharge, receiver):

        mouths = land & (receiver >= 0) & ocean[np.maximum(receiver, 0)]

        return discharge[mouths].sum()

    assert into_sea(discharge_gpu, receiver_gpu) == pytest.approx(into_sea(discharge_cpu, receiver_cpu), rel=1e-4)

    big = discharge_cpu > np.percentile(discharge_cpu[land], 99)

    assert np.corrcoef(discharge_gpu[big], discharge_cpu[big])[0, 1] > 0.9

    assert gpu.enabled


def test_accelerator_switch():

    assert acceleration.accelerator() is None

    class Fake:
        pass

    fake = Fake()

    acceleration.set_accelerator(fake)

    try:
        assert acceleration.accelerator() is fake
    finally:
        acceleration.set_accelerator(None)
