import numpy as np
import pytest

from ecs.components import (
    HierarchyComponent,
    RotatorComponent,
    TransformComponent
)
from math3d.transform import Transform
from scene.scene import Scene
from systems.rotator_system import RotatorSystem
from systems.transform_system import TransformSystem


def add(scene, transform, parent=None):

    entity = scene.create_entity()

    scene.add_component(entity, TransformComponent(transform=transform))

    if parent is not None:
        scene.add_component(entity, HierarchyComponent(parent))

    return entity


def world_position(scene, entity):

    return scene.get_component(entity, TransformComponent).world_position


def test_child_inherits_parent_translation_and_rotation():

    scene = Scene("test")

    parent = add(scene, Transform(position=(1.0, 0.0, 0.0), rotation=(0.0, 90.0, 0.0)))
    child = add(scene, Transform(position=(2.0, 0.0, 0.0)), parent)

    TransformSystem().update(scene)

    # Yaw +90 turns local +X into world -Z.
    assert np.allclose(world_position(scene, child), (1.0, 0.0, -2.0), atol=1e-5)


def test_order_of_creation_does_not_matter():

    scene = Scene("test")

    # Child created before its parent is attached.
    child = add(scene, Transform(position=(0.0, 1.0, 0.0)))
    parent = add(scene, Transform(position=(0.0, 0.0, 5.0)))
    grandchild = add(scene, Transform(position=(0.0, 1.0, 0.0)), child)

    scene.add_component(child, HierarchyComponent(parent))

    TransformSystem().update(scene)

    assert np.allclose(world_position(scene, grandchild), (0.0, 2.0, 5.0))


def test_cycle_is_reported():

    scene = Scene("test")

    a = add(scene, Transform())
    b = add(scene, Transform(), a)

    scene.add_component(a, HierarchyComponent(b))

    with pytest.raises(Exception, match="cycle"):
        TransformSystem().update(scene)


def test_destroyed_parent_is_reported():

    scene = Scene("test")

    parent = add(scene, Transform())
    add(scene, Transform(), parent)

    scene.destroy_entity(parent)

    with pytest.raises(Exception, match="parent"):
        TransformSystem().update(scene)


def test_rotator_spins_on_fixed_steps():

    scene = Scene("test")

    entity = add(scene, Transform())

    scene.add_component(entity, RotatorComponent(degrees_per_second=(0.0, 30.0, 0.0)))

    system = RotatorSystem()

    for _ in range(60):
        system.fixed_update(scene, 1.0 / 60.0)

    rotation = scene.get_component(entity, TransformComponent).transform.rotation

    assert rotation[1] == pytest.approx(30.0, abs=1e-3)
