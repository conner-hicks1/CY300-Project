import time

import numpy as np
import pytest

import systems.planet_system as planet_system_module

from core.jobs import JobSystem
from ecs.components import (
    CameraControllerComponent,
    PlanetComponent,
    TransformComponent
)
from math3d.transform import Transform
from scene.scene import Scene
from systems.planet_system import PlanetSystem
from systems.transform_system import TransformSystem


RADIUS = 100_000.0


class FakeMesh:

    # Stands in for the GL mesh: records frees.

    live = 0

    def __init__(self, data):

        self.data = data
        self.bounding_center = np.zeros(3)
        self.bounding_radius = 1.0

        FakeMesh.live += 1

    @classmethod
    def from_data(cls, data):

        return cls(data)

    def delete(self):

        FakeMesh.live -= 1


class FakeMaterials:

    def get(self, handle):

        return "planet-material"


class FakeResources:

    materials = FakeMaterials()


@pytest.fixture
def world(monkeypatch):

    monkeypatch.setattr(planet_system_module, "Mesh", FakeMesh)

    FakeMesh.live = 0

    jobs = JobSystem(workers=2)

    system = PlanetSystem(FakeResources(), jobs, material=None)

    scene = Scene("planet test")

    planet = scene.create_entity()

    scene.add_component(planet, TransformComponent(transform=Transform(position=(0.0, -RADIUS, 0.0))))
    scene.add_component(planet, PlanetComponent(radius=RADIUS, resolution=9, max_depth=4, mountain_height=500.0))

    TransformSystem().update(scene)

    yield scene, system, jobs, planet

    system.shutdown()
    jobs.shutdown(wait=True)


def stream(scene, system, jobs, camera, frames=400):

    for _ in range(frames):

        system.update(scene, camera)

        jobs.process_completions(budget_seconds=1.0)

        if system.stats.chunks_building == 0 and system.stats.chunks_queued == 0:

            system.update(scene, camera)

            if system.stats.chunks_building == 0 and system.stats.chunks_queued == 0:
                return

        time.sleep(0.002)

    pytest.fail("planet never finished streaming")


def test_streams_until_complete(world):

    scene, system, jobs, _ = world

    stream(scene, system, jobs, np.array([0.0, 500.0, 0.0]))

    stats = system.stats

    assert stats.chunks_drawn > 0
    assert stats.chunks_loaded == FakeMesh.live

    items = system.draw_items

    assert len(items) == stats.chunks_drawn
    assert all(item.material == "planet-material" for item in items)

    # Chunk world matrices include the planet's position.
    centers = np.array([item.world_matrix[:3, 3] for item in items])

    distances = np.linalg.norm(centers - (0.0, -RADIUS, 0.0), axis=1)

    np.testing.assert_allclose(distances, RADIUS, rtol=1e-6)


def test_settings_change_rebuilds(world):

    scene, system, jobs, planet = world

    camera = np.array([0.0, 500.0, 0.0])

    stream(scene, system, jobs, camera)

    scene.get_component(planet, PlanetComponent).seed = 2

    system.update(scene, camera)

    # Old chunks freed immediately.
    assert FakeMesh.live == 0

    stream(scene, system, jobs, camera)

    assert system.stats.chunks_drawn > 0


def test_removed_planet_is_released(world):

    scene, system, jobs, planet = world

    camera = np.array([0.0, 500.0, 0.0])

    stream(scene, system, jobs, camera)

    scene.destroy_entity(planet)

    system.update(scene, camera)

    assert FakeMesh.live == 0
    assert system.draw_items == []


def test_camera_follows_ground(world):

    scene, system, jobs, _ = world

    camera = scene.create_entity()

    scene.add_component(camera, TransformComponent(transform=Transform(position=(0.0, 100.0, 0.0))))
    scene.add_component(camera, CameraControllerComponent(planet_mode=True))

    stream(scene, system, jobs, np.array([0.0, 100.0, 0.0]))

    system.update_camera_ground(scene)

    controller = scene.get_component(camera, CameraControllerComponent)

    assert controller.planet_center == pytest.approx((0.0, -RADIUS, 0.0))
    assert controller.planet_radius >= RADIUS
