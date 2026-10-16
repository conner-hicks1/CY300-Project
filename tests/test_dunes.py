import dataclasses

import numpy as np
import pytest

from planet.bodies import components_for, erosion_settings, load_presets
from planet.dunes import DuneSettings, Dunes
from planet.terrain import Terrain
from systems.planet_system import terrain_settings_for


@pytest.fixture(scope="module")
def presets():

    return load_presets()


class _Grid:
    """Directions spread over the sphere (a cube's n x n faces, normalized)."""

    def __init__(self, n: int):

        u = (np.arange(n) + 0.5) / n * 2.0 - 1.0
        a, b = np.meshgrid(u, u)
        a, b, one = a.ravel(), b.ravel(), np.ones(n * n)

        faces = []
        for axis in range(3):
            for sign in (-1.0, 1.0):
                face = np.empty((n * n, 3))
                face[:, axis] = sign * one
                face[:, (axis + 1) % 3] = a
                face[:, (axis + 2) % 3] = b
                faces.append(face)

        directions = np.concatenate(faces)
        self.directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)


def sphere_grid(n: int) -> _Grid:

    return _Grid(n)


def mars_terrain(presets) -> Terrain:
    """Generated Mars (no measured map, so its craters are the generator's)."""

    parts = components_for(presets["mars"])
    parts.planet.elevation_map = ""
    parts.planet.color_map = ""

    return Terrain(terrain_settings_for(parts.planet))


def test_transverse_dunes_are_asymmetric():

    dunes = Dunes(DuneSettings(density=1.0, amplitude=50.0, wavelength=1_000.0), 3_390_000.0)

    grid = sphere_grid(48)

    cover = dunes.apply(grid.directions, 1e12)[1]

    center = grid.directions[np.argmax(cover)]

    # Across the crests (east-west at most latitudes).
    east = np.cross([0.0, 1.0, 0.0], center)
    east /= np.linalg.norm(east)

    s = np.linspace(-4_000.0, 4_000.0, 801)

    line = center[None, :] + s[:, None] * east[None, :] / 3_390_000.0
    line /= np.linalg.norm(line, axis=1, keepdims=True)

    height, _ = dunes.apply(line, 10.0)

    assert height.max() > 20.0
    assert height.min() >= 0.0

    # Steep slip faces: the steepest descent is much steeper
    # than the steepest climb, or vice versa.
    slope = np.diff(height)

    assert max(slope.max(), -slope.min()) > 2.0 * min(slope.max(), -slope.min())


def test_linear_dunes_stay_in_their_band():

    dunes = Dunes(DuneSettings(density=1.2, linear=True, max_latitude=30.0, wavelength=3_000.0), 2_574_700.0)

    grid = sphere_grid(32)

    cover = dunes.apply(grid.directions, 1e12)[1]

    latitude = np.degrees(np.arcsin(np.abs(grid.directions[:, 1])))

    assert cover[latitude > 31.0].max() == 0.0
    assert cover[latitude < 20.0].mean() > 0.5


def test_dunes_skip_when_too_small_to_see():

    dunes = Dunes(DuneSettings(density=1.0, wavelength=1_000.0), 3_390_000.0)

    directions = sphere_grid(16).directions

    height, cover = dunes.apply(directions, 5_000.0)

    assert not height.any()
    assert cover.any()


def test_bodies_get_their_dunes(presets):

    titan = erosion_settings(presets["titan"])
    mars = erosion_settings(presets["mars"])

    assert titan["dune_linear"] and titan["dune_max_latitude"] == 30.0
    assert not mars["dune_linear"] and mars["dune_darkening"] > 0.0

    # No air, no dunes.
    assert erosion_settings(presets["moon"])["dune_density"] == 0.0
    assert erosion_settings(presets["pluto"])["dune_density"] == 0.0


def test_dunes_bury_craters(presets):

    terrain = mars_terrain(presets)

    grid = sphere_grid(48)

    cover = terrain.dunes.apply(grid.directions, 1e12)[1]

    center = grid.directions[np.argmax(cover)]

    rng = np.random.default_rng(0)

    points = center + rng.normal(size=(2_000, 3)) * 0.002
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    with_dunes = terrain.elevation(points, 20.0)

    no_dunes = Terrain(dataclasses.replace(terrain.settings, dune_density=0.0))

    # Under the dune field the craters are mostly gone: the
    # surface is smoother at scales beyond the dunes.
    assert np.std(with_dunes) < np.std(no_dunes.elevation(points, 20.0))
