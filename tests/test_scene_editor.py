import numpy as np
import pytest

from imgui_bundle import imgui

from ecs.components import (
    CameraComponent,
    CameraControllerComponent,
    DirectionalLightComponent,
    HierarchyComponent,
    MeshRendererComponent,
    NameComponent,
    PointLightComponent,
    RotatorComponent,
    TransformComponent
)
from graphics.lighting import MAX_POINT_LIGHTS
from graphics.render_settings import RenderSettings
from math3d.camera import Camera
from math3d.transform import Transform
from resources.resources import Resources
from scene.scene import Scene
from scene.scene_serializer import SceneSerializer
from systems.rotator_system import RotatorSystem
from systems.transform_system import TransformSystem

from editor.scene_editor import SceneEditor


# =========================================================
# Fixtures
# =========================================================
#
# SceneEditor's commands need no GL: an ImGui context (for
# the gizmo module) and a render system that provides the
# last camera and settings are enough.

class FakeMesh:

    def __init__(self):
        self.bounds = (np.full(3, -0.5), np.full(3, 0.5))


class FakeRenderSystem:

    def __init__(self):

        self.settings = RenderSettings()

        self.last_camera = Camera(
            position=(0.0, 0.0, 10.0),
            target=(0.0, 0.0, 0.0),
            aspect_ratio=1.0
        )


class FakeWindow:

    def __init__(self):
        self.title = ""
        self.should_close = False

    def set_title(self, title):
        self.title = title

    def set_should_close(self, value):
        self.should_close = value


@pytest.fixture(scope="module", autouse=True)
def imgui_context():

    context = imgui.create_context()

    yield

    imgui.destroy_context(context)


@pytest.fixture
def editor():

    resources = Resources()
    resources.meshes.add("cube", FakeMesh())
    resources.materials.add("red", object())

    scene = Scene("Test")

    def populate(target):

        entity = target.create_entity()
        target.add_component(entity, NameComponent("Demo cube"))
        target.add_component(entity, TransformComponent())

    editor = SceneEditor(
        scene,
        resources,
        FakeRenderSystem(),
        FakeWindow(),
        SceneSerializer(resources, load_model=lambda path: FakeMesh()),
        populate_demo_scene=populate,
        load_model=lambda path: FakeMesh()
    )

    return editor


def update_world(editor):
    TransformSystem().update(editor.scene)


def end_frame(editor):
    # What draw() does at the end of every frame.
    editor._commit_record()


def names(editor):
    return sorted(editor.entity_name(e) for e in editor.scene.entities())


def make(editor, name, position=(0, 0, 0), parent=None, components=()):

    entity = editor.create_entity(name, list(components), parent=parent)

    editor.scene.get_component(entity, TransformComponent).transform.position = position

    end_frame(editor)
    update_world(editor)

    return entity


# =========================================================
# Create / Delete / Duplicate
# =========================================================

def test_create_places_entity_in_front_of_camera_and_selects_it(editor):

    entity = editor.create_entity("Thing")

    position = editor.scene.get_component(entity, TransformComponent).transform.position

    assert np.allclose(position, (0.0, 0.0, 6.0))
    assert editor.selected == entity


def test_delete_removes_whole_subtree(editor):

    parent = make(editor, "Parent")
    child = make(editor, "Child", parent=parent)
    other = make(editor, "Other")

    # Selection inside the deleted subtree is cleared...
    editor.select(child)
    editor.delete(parent)

    assert names(editor) == ["Other"]
    assert editor.selected is None

    # ...and selection elsewhere is kept.
    extra = make(editor, "Extra")
    editor.select(other)
    editor.delete(extra)

    assert editor.selected == other


def test_duplicate_copies_subtree_and_keeps_parent(editor):

    root = make(editor, "Root")
    branch = make(editor, "Branch", parent=root)
    make(editor, "Leaf", parent=branch)

    copy = editor.duplicate(branch)

    assert names(editor) == ["Branch", "Branch (copy)", "Leaf", "Leaf", "Root"]
    assert editor.scene.get_component(copy, HierarchyComponent).parent == root
    assert editor.selected == copy


def test_duplicate_refuses_to_exceed_light_limit(editor):

    for index in range(MAX_POINT_LIGHTS):
        make(editor, f"Light {index}", components=[PointLightComponent()])

    light = editor.scene.entities()[0]

    assert editor.duplicate(light) is None
    assert editor.count_components(PointLightComponent) == MAX_POINT_LIGHTS


def test_light_limits_for_adding(editor):

    assert editor.can_add_component(DirectionalLightComponent)[0]

    make(editor, "Sun", components=[DirectionalLightComponent()])

    allowed, reason = editor.can_add_component(DirectionalLightComponent)

    assert not allowed
    assert "one directional light" in reason


# =========================================================
# Reparent
# =========================================================

def test_reparent_keeps_world_transform(editor):

    parent = make(editor, "Parent", position=(5, 0, 0))

    parent_transform = editor.scene.get_component(parent, TransformComponent).transform
    parent_transform.rotation = (0, 90, 0)
    parent_transform.scale = (2, 2, 2)

    child = make(editor, "Child", position=(1, 2, 3))

    update_world(editor)

    before = editor.scene.get_component(child, TransformComponent).world_matrix.copy()

    assert editor.reparent(child, parent)

    update_world(editor)

    after = editor.scene.get_component(child, TransformComponent).world_matrix

    assert np.allclose(before, after, atol=1e-4)

    # And back out again.
    assert editor.reparent(child, None)

    update_world(editor)

    assert np.allclose(editor.scene.get_component(child, TransformComponent).world_matrix, before, atol=1e-4)
    assert not editor.scene.has_component(child, HierarchyComponent)


def test_reparent_rejects_cycles(editor):

    parent = make(editor, "Parent")
    child = make(editor, "Child", parent=parent)
    grandchild = make(editor, "Grandchild", parent=child)

    assert not editor.reparent(parent, grandchild)
    assert not editor.reparent(parent, parent)

    assert not editor.scene.has_component(parent, HierarchyComponent)


# =========================================================
# Undo / Redo / Dirty
# =========================================================

def test_undo_redo_restores_scene_and_selection(editor):

    cube = make(editor, "Cube", position=(1, 0, 0))

    editor.select(cube)

    transform = editor.scene.get_component(cube, TransformComponent).transform
    transform.position = (4, 0, 0)

    editor.record("Move Cube")
    end_frame(editor)

    editor.undo()

    restored = editor.selected

    assert editor.entity_name(restored) == "Cube"
    assert np.allclose(editor.scene.get_component(restored, TransformComponent).transform.position, (1, 0, 0))

    editor.redo()

    assert np.allclose(editor.scene.get_component(editor.selected, TransformComponent).transform.position, (4, 0, 0))


def test_multiple_records_in_one_frame_are_one_step(editor):

    make(editor, "A")

    editor.create_entity("B")
    editor.create_entity("C")

    end_frame(editor)

    editor.undo()

    assert names(editor) == ["A"]


def test_dirty_tracks_saved_state(editor, tmp_path):

    assert not editor.dirty

    make(editor, "Thing")

    assert editor.dirty

    editor._write(tmp_path / "scene.scene.json")

    assert not editor.dirty

    editor.undo()

    assert editor.dirty

    editor.redo()

    assert not editor.dirty


def test_open_restores_file_and_clears_history(editor, tmp_path):

    make(editor, "Saved")

    path = tmp_path / "a.scene.json"

    editor._write(path)

    make(editor, "Unsaved")

    assert editor.load_initial(path)

    assert names(editor) == ["Saved"]
    assert not editor.history.can_undo
    assert not editor.dirty


def test_bad_file_keeps_current_scene(editor, tmp_path):

    make(editor, "Keep me")

    path = tmp_path / "bad.scene.json"
    path.write_text('{"format": "nope"}')

    assert not editor.load_initial(path)

    assert names(editor) == ["Keep me"]


def test_new_demo_scene(editor):

    editor.new_scene(demo=True)

    assert names(editor) == ["Demo cube"]
    assert editor.scene_path is None
    assert not editor.dirty


def test_request_close_runs_immediately_when_clean(editor):

    closed = []

    editor.request_close(lambda: closed.append(True))

    assert closed == [True]


# =========================================================
# Play Mode
# =========================================================

def test_stop_restores_pre_play_scene(editor):

    spinner = make(editor, "Spinner", components=[RotatorComponent(degrees_per_second=(0, 90, 0))])

    editor.select(spinner)
    editor.play()

    for _ in range(30):
        RotatorSystem().fixed_update(editor.scene, 1 / 60)

    make(editor, "Made while playing")

    editor.stop()

    assert names(editor) == ["Spinner"]

    rotation = editor.scene.get_component(editor.selected, TransformComponent).transform.rotation

    assert np.allclose(rotation, 0.0)


def test_edits_while_playing_are_not_recorded(editor):

    editor.play()

    make(editor, "Temporary")

    editor.stop()

    assert not editor.history.can_undo
    assert not editor.dirty


# =========================================================
# Picking / Focus
# =========================================================

def test_pick_candidates_cover_meshes_and_markers(editor):

    resources = editor.resources

    mesh_entity = make(editor, "Mesh", components=[
        MeshRendererComponent(mesh=resources.meshes.get_handle("cube"), material=resources.materials.get_handle("red"))
    ])

    light = make(editor, "Light", components=[PointLightComponent()])

    camera = make(editor, "Camera", components=[CameraComponent(primary=True)])

    keys = [candidate.key for candidate in editor._pick_candidates()]

    assert mesh_entity in keys
    assert light in keys
    assert camera not in keys


def test_focus_moves_controllable_camera_toward_selection(editor):

    camera = make(editor, "Camera", position=(0, 0, 50), components=[CameraComponent(primary=True), CameraControllerComponent()])
    target = make(editor, "Target", position=(3, 0, 0))

    update_world(editor)

    editor.select(target)
    editor.focus_selected()

    position = editor.scene.get_component(camera, TransformComponent).transform.position

    # Looking down -Z at a small target: camera ends up just behind it on +Z.
    assert np.allclose(position[:2], (3, 0), atol=1e-4)
    assert 0.0 < position[2] < 5.0
