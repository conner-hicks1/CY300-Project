import numpy as np
import pytest

from ecs.components import (
    DirectionalLightComponent,
    HierarchyComponent,
    PointLightComponent,
    SpotLightComponent,
    TransformComponent
)
from math3d.transform import Transform
from scene.scene import Scene
from systems.render_system import RenderSystem
from systems.transform_system import TransformSystem


def add(scene, component, transform, parent=None):

    entity = scene.create_entity()

    scene.add_component(entity, TransformComponent(transform=transform))
    scene.add_component(entity, component)

    if parent is not None:
        scene.add_component(entity, HierarchyComponent(parent))

    return entity


def gl_free_render_system() -> RenderSystem:

    # Light gathering needs no GL; skip __init__, which
    # loads shaders.
    system = RenderSystem.__new__(RenderSystem)
    system._warnings = set()

    return system


def build(scene):

    TransformSystem().update(scene)

    return gl_free_render_system()._build_light_environment(scene.registry)


def test_gathers_all_light_types():

    scene = Scene("test")

    add(scene, DirectionalLightComponent(ambient=0.1), Transform(rotation=(-45.0, 30.0, 0.0)))
    add(scene, PointLightComponent(intensity=4.0, range=6.0), Transform(position=(1.5, 0.5, 1.0)))
    add(scene, SpotLightComponent(), Transform(position=(-2.0, 2.0, -1.0), rotation=(-90.0, 0.0, 0.0)))

    lighting = build(scene)

    assert lighting.ambient == pytest.approx(0.1)
    assert np.allclose(lighting.directional.direction, (-0.3535534, -0.7071068, -0.6123725), atol=1e-6)
    assert np.allclose(lighting.point_lights[0].position, (1.5, 0.5, 1.0))
    assert np.allclose(lighting.spot_lights[0].direction, (0.0, -1.0, 0.0), atol=1e-6)
    assert lighting.spot_lights[0].inner_cutoff == pytest.approx(np.cos(np.radians(15.0)))


def test_no_directional_means_no_ambient():

    scene = Scene("test")

    add(scene, PointLightComponent(), Transform())

    lighting = build(scene)

    assert lighting.directional is None
    assert lighting.ambient == 0.0


def test_light_positions_use_hierarchy():

    scene = Scene("test")

    pivot = scene.create_entity()
    scene.add_component(pivot, TransformComponent(transform=Transform(position=(0.0, 1.0, 0.0), rotation=(0.0, 90.0, 0.0))))

    add(scene, PointLightComponent(), Transform(position=(2.0, 0.0, 0.0)), parent=pivot)

    lighting = build(scene)

    assert np.allclose(lighting.point_lights[0].position, (0.0, 1.0, -2.0), atol=1e-5)


def test_second_directional_light_is_ignored():

    scene = Scene("test")

    add(scene, DirectionalLightComponent(intensity=1.0), Transform())
    add(scene, DirectionalLightComponent(intensity=9.0), Transform())

    lighting = build(scene)

    assert lighting.directional.intensity == 1.0


def test_inverted_spot_angles_are_clamped():

    scene = Scene("test")

    add(scene, SpotLightComponent(inner_angle=30.0, outer_angle=20.0), Transform())

    spot = build(scene).spot_lights[0]

    # inner clamped down to outer.
    assert spot.inner_cutoff == pytest.approx(spot.outer_cutoff)
    assert spot.outer_cutoff == pytest.approx(np.cos(np.radians(20.0)))


def test_excess_point_lights_and_bad_range_are_tolerated():

    from graphics.lighting import MAX_POINT_LIGHTS

    scene = Scene("test")

    for _ in range(MAX_POINT_LIGHTS + 3):
        add(scene, PointLightComponent(range=0.0), Transform())

    lighting = build(scene)

    assert len(lighting.point_lights) == MAX_POINT_LIGHTS
    assert all(light.range > 0.0 for light in lighting.point_lights)


def test_light_space_matrix_keeps_extent_in_clip_space():

    direction = np.array([-0.35, -0.7, -0.61])
    direction /= np.linalg.norm(direction)

    matrix = RenderSystem.light_space_matrix(direction, (0.0, 0.0, 0.0), 8.0)

    # Points inside the shadow area land inside NDC.
    for point in [(0, 0, 0), (7, 0, 7), (-7, 0, -7), (0, 2, 0)]:

        clip = matrix @ np.array([*point, 1.0])

        assert np.all(np.abs(clip[:3] / clip[3]) <= 1.0)


def test_light_space_matrix_handles_straight_down_light():

    matrix = RenderSystem.light_space_matrix((0.0, -1.0, 0.0), (0.0, 0.0, 0.0), 5.0)

    assert np.all(np.isfinite(matrix))
