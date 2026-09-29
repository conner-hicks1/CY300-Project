import json

import numpy as np
import pytest

from ecs.components import (
    CameraComponent,
    DirectionalLightComponent,
    HierarchyComponent,
    MeshRendererComponent,
    NameComponent,
    PointLightComponent,
    RotatorComponent,
    TransformComponent
)
from graphics.render_settings import RenderSettings, Tonemapper
from math3d.transform import Transform
from resources.resources import Resources
from scene.scene import Scene
from scene.scene_serializer import SceneFormatError, SceneSerializer


class Placeholder:
    """Stands in for GPU resources; no GL needed."""


@pytest.fixture
def resources():

    resources = Resources()

    resources.meshes.add("cube", Placeholder())
    resources.meshes.add("sphere", Placeholder())
    resources.materials.add("red", Placeholder())

    return resources


@pytest.fixture
def serializer(resources):

    loaded = []

    def load_model(path):
        loaded.append(path)
        return Placeholder()

    serializer = SceneSerializer(resources, load_model=load_model)
    serializer.loaded_models = loaded

    return serializer


def build_scene(resources):

    scene = Scene("Demo")

    root = scene.create_entity()
    scene.add_component(root, NameComponent("Root"))
    scene.add_component(root, TransformComponent(transform=Transform(position=(1, 2, 3), rotation=(10, 20, 30), scale=(2, 2, 2))))
    scene.add_component(root, MeshRendererComponent(
        mesh=resources.meshes.get_handle("cube"),
        material=resources.materials.get_handle("red"),
        casts_shadows=False
    ))
    scene.add_component(root, RotatorComponent(degrees_per_second=(0.0, 45.0, 0.0)))

    child = scene.create_entity()
    scene.add_component(child, NameComponent("Child"))
    scene.add_component(child, TransformComponent(transform=Transform(position=(0, 1, 0))))
    scene.add_component(child, HierarchyComponent(root))
    scene.add_component(child, PointLightComponent(color=(1.0, 0.5, 0.25), intensity=3.0, range=4.0))

    camera = scene.create_entity()
    scene.add_component(camera, NameComponent("Camera"))
    scene.add_component(camera, TransformComponent())
    scene.add_component(camera, CameraComponent(fov=60.0, primary=True))

    return scene


def find(scene, name):

    for entity in scene.entities():
        component = scene.try_get_component(entity, NameComponent)
        if component is not None and component.name == name:
            return entity

    raise AssertionError(f"no entity named {name}")


def test_round_trip_preserves_components(resources, serializer):

    scene = build_scene(resources)

    data = serializer.serialize(scene)

    # Must be plain JSON.
    data = json.loads(json.dumps(data))

    loaded = Scene("Other")
    serializer.deserialize(data, loaded)

    assert loaded.name == "Demo"
    assert loaded.entity_count == 3

    root = find(loaded, "Root")
    child = find(loaded, "Child")

    transform = loaded.get_component(root, TransformComponent).transform
    assert np.allclose(transform.position, (1, 2, 3))
    assert np.allclose(transform.rotation, (10, 20, 30))
    assert np.allclose(transform.scale, (2, 2, 2))

    renderer = loaded.get_component(root, MeshRendererComponent)
    assert resources.meshes.key_of(renderer.mesh) == "cube"
    assert resources.materials.key_of(renderer.material) == "red"
    assert renderer.casts_shadows is False

    assert loaded.get_component(root, RotatorComponent).degrees_per_second == (0.0, 45.0, 0.0)

    assert loaded.get_component(child, HierarchyComponent).parent == root
    assert loaded.get_component(child, PointLightComponent).color == (1.0, 0.5, 0.25)

    assert loaded.get_component(find(loaded, "Camera"), CameraComponent).primary is True


def test_serialized_output_is_stable(resources, serializer):

    scene = build_scene(resources)

    first = serializer.serialize(scene)

    loaded = Scene("x")
    serializer.deserialize(first, loaded)

    assert serializer.serialize(loaded) == first


def test_render_settings_round_trip(resources, serializer):

    settings = RenderSettings(exposure=2.5, tonemapper=Tonemapper.REINHARD, shadows_enabled=False)

    data = json.loads(json.dumps(serializer.serialize(Scene("s"), settings)))

    target = RenderSettings()
    serializer.deserialize(data, Scene("t"), target)

    assert target.exposure == 2.5
    assert target.tonemapper is Tonemapper.REINHARD
    assert target.shadows_enabled is False


def test_bad_file_leaves_scene_untouched(resources, serializer):

    scene = build_scene(resources)
    data = serializer.serialize(scene)

    data["entities"][0]["components"]["MeshRenderer"]["material"] = "missing"

    with pytest.raises(SceneFormatError, match="unknown material 'missing'"):
        serializer.deserialize(data, scene)

    assert scene.entity_count == 3
    assert scene.name == "Demo"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(format="other"), "Not a scene file"),
        (lambda d: d.update(version=99), "Unsupported scene version"),
        (lambda d: d["entities"][0]["components"].update(Bogus={}), "unknown component 'Bogus'"),
        (lambda d: d["entities"][0]["components"]["Transform"].update(position=[1, 2]), "list of 3 numbers"),
        (lambda d: d["entities"][0]["components"]["Rotator"].update(degrees_per_second="fast"), "list of 3 numbers"),
        (lambda d: d["entities"][2]["components"]["Camera"].update(fov="wide"), "must be a number"),
        (lambda d: d["entities"][2]["components"]["Camera"].update(zoom=2), "unknown Camera field"),
        (lambda d: d["entities"][1]["components"]["Hierarchy"].update(parent=42), "parent 42 does not exist"),
        (lambda d: d["entities"][1].update(id=0), "duplicate entity id"),
        (lambda d: d["entities"][0]["components"]["Name"].pop("name"), "Name.name is required"),
    ],
)
def test_validation_errors(resources, serializer, mutate, message):

    data = serializer.serialize(build_scene(resources))

    mutate(data)

    with pytest.raises(SceneFormatError, match=message):
        serializer.deserialize(data, Scene("x"))


def test_hierarchy_cycle_is_rejected(resources, serializer):

    data = serializer.serialize(build_scene(resources))

    data["entities"][0]["components"]["Hierarchy"] = {"parent": 1}

    with pytest.raises(SceneFormatError, match="cycle"):
        serializer.deserialize(data, Scene("x"))


def test_light_limits_are_enforced(resources, serializer):

    scene = Scene("lights")

    for _ in range(2):
        entity = scene.create_entity()
        scene.add_component(entity, TransformComponent())
        scene.add_component(entity, DirectionalLightComponent())

    data = serializer.serialize(scene)

    with pytest.raises(SceneFormatError, match="at most 1"):
        serializer.deserialize(data, Scene("x"))


def test_missing_transform_gets_default(resources, serializer):

    data = serializer.serialize(Scene("s"))

    data["entities"] = [{"id": 5, "components": {"Name": {"name": "Bare"}}}]

    scene = Scene("x")
    serializer.deserialize(data, scene)

    assert scene.has_component(find(scene, "Bare"), TransformComponent)


def test_model_meshes_load_on_demand(resources, serializer, tmp_path):

    model = tmp_path / "thing.obj"
    model.write_text("v 0 0 0\n")

    data = serializer.serialize(build_scene(resources))
    data["entities"][0]["components"]["MeshRenderer"]["mesh"] = str(model)

    scene = Scene("x")
    serializer.deserialize(data, scene)

    assert serializer.loaded_models == [str(model)]
    assert resources.meshes.contains(str(model))


def test_unknown_non_model_mesh_is_an_error(resources, serializer):

    data = serializer.serialize(build_scene(resources))
    data["entities"][0]["components"]["MeshRenderer"]["mesh"] = "teapot"

    with pytest.raises(SceneFormatError, match="unknown mesh 'teapot'"):
        serializer.deserialize(data, Scene("x"))


def test_duplicate_subtree(resources, serializer):

    scene = build_scene(resources)

    root = find(scene, "Root")

    copies = serializer.instantiate(
        serializer.encode_subtrees(scene, [root]),
        scene
    )

    assert scene.entity_count == 5

    new_root, new_child = copies[0], copies[1]

    assert new_root != root
    assert not scene.has_component(new_root, HierarchyComponent)
    assert scene.get_component(new_child, HierarchyComponent).parent == new_root


def test_duplicate_child_keeps_external_parent(resources, serializer):

    scene = build_scene(resources)

    root = find(scene, "Root")
    child = find(scene, "Child")

    copies = serializer.instantiate(
        serializer.encode_subtrees(scene, [child]),
        scene,
        root_parent=root
    )

    assert scene.get_component(copies[0], HierarchyComponent).parent == root


def test_save_and_load_file(resources, serializer, tmp_path):

    path = tmp_path / "scenes" / "demo.scene.json"

    serializer.save(path, build_scene(resources))

    assert not path.with_name(path.name + ".tmp").exists()

    scene = Scene("x")
    serializer.load(path, scene)

    assert scene.entity_count == 3


def test_load_reports_invalid_json(resources, serializer, tmp_path):

    path = tmp_path / "broken.json"
    path.write_text("{ nope")

    with pytest.raises(SceneFormatError, match="not valid JSON"):
        serializer.load(path, Scene("x"))


def test_file_ids_match_serialized_ids(resources, serializer):

    scene = build_scene(resources)

    ids = serializer.file_ids(scene)
    data = serializer.serialize(scene)

    camera = find(scene, "Camera")

    entry = next(e for e in data["entities"] if e["id"] == ids[camera.index])

    assert entry["components"]["Name"]["name"] == "Camera"


# =========================================================
# Compatibility / Model Materials
# =========================================================

def test_obsolete_component_field_is_ignored(resources, serializer):

    scene = Scene("s")
    sun = scene.create_entity()
    scene.add_component(sun, TransformComponent())
    scene.add_component(sun, DirectionalLightComponent(intensity=4.0))

    data = serializer.serialize(scene)
    data["entities"][0]["components"]["DirectionalLight"]["ambient"] = 0.1   # pre-IBL files

    loaded = Scene("x")
    serializer.deserialize(data, loaded)

    light = loaded.get_component(loaded.entities()[0], DirectionalLightComponent)

    assert light.intensity == 4.0


def test_unknown_render_settings_are_skipped(resources, serializer):

    data = serializer.serialize(Scene("s"), RenderSettings())
    data["render_settings"]["shadow_extent"] = 8.0          # removed setting
    data["render_settings"]["exposure"] = 0.5

    settings = RenderSettings()
    serializer.deserialize(data, Scene("x"), settings)

    assert settings.exposure == 0.5


def test_model_material_loads_on_demand(resources, tmp_path):

    from scene.scene_serializer import model_material_key

    model = tmp_path / "thing.gltf"
    model.write_text("{}")

    requested = []

    def load_material(key):
        requested.append(key)
        return Placeholder()

    serializer = SceneSerializer(resources, load_model=lambda p: Placeholder(), load_material=load_material)

    data = serializer.serialize(build_scene(resources))
    key = model_material_key(str(model))
    data["entities"][0]["components"]["MeshRenderer"]["material"] = key

    scene = Scene("x")
    serializer.deserialize(data, scene)

    assert requested == [key]
    assert resources.materials.contains(key)


def test_model_material_without_loader_is_an_error(resources, serializer, tmp_path):

    from scene.scene_serializer import model_material_key

    model = tmp_path / "thing.gltf"
    model.write_text("{}")

    data = serializer.serialize(build_scene(resources))
    data["entities"][0]["components"]["MeshRenderer"]["material"] = model_material_key(str(model))

    with pytest.raises(SceneFormatError, match="unknown material"):
        serializer.deserialize(data, Scene("x"))
