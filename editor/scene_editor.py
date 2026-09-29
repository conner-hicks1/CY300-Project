import json
import time

from collections.abc import Callable
from pathlib import Path

import numpy as np

from imgui_bundle import imgui, imguizmo
from imgui_bundle import portable_file_dialogs as pfd

from core.logger import Logger
from core.window import Window

from ecs.components import (
    CameraComponent,
    CameraControllerComponent,
    DirectionalLightComponent,
    HierarchyComponent,
    MeshRendererComponent,
    NameComponent,
    PointLightComponent,
    SpotLightComponent,
    TransformComponent
)
from ecs.entity import Entity

from graphics.lighting import MAX_POINT_LIGHTS, MAX_SPOT_LIGHTS

from math3d.matrices import decompose_trs
from math3d.transform import Transform

from resources.resources import Resources

from scene.scene import Scene
from scene.scene_serializer import (
    SceneFormatError,
    SceneSerializer
)

from systems.render_system import RenderSystem

from editor.hierarchy_panel import draw_hierarchy_panel
from editor.history import Snapshot, UndoHistory
from editor.inspector_panel import draw_inspector_panel
from editor.picking import PickCandidate, pick, screen_ray


gizmo = imguizmo.im_guizmo

SCENE_DIRECTORY = Path("assets/scenes")

SCENE_FILE_FILTERS = [
    "Scene files (*.scene.json)", "*.scene.json",
    "JSON files (*.json)", "*.json",
    "All files", "*",
]

MODEL_FILE_FILTERS = [
    "Models (*.obj *.gltf *.glb)", "*.obj *.gltf *.glb",
    "All files", "*",
]


class SceneEditor:

    # =====================================================
    # Scene Editor
    # =====================================================
    #
    # Owns editor state (selection, undo, file, play mode)
    # and draws the menu bar, toolbar, hierarchy, inspector
    # and viewport gizmo. Panels live in
    # editor/hierarchy_panel.py and
    # editor/inspector_panel.py and call the commands here.
    #
    # Undo: panels call record(label) when an edit
    # finishes; one snapshot of the scene is pushed at the
    # end of that frame (see editor/history.py).
    #
    # Play mode: Play snapshots the scene and lets the
    # simulation (fixed update) run; Stop restores the
    # snapshot. Edits made while playing are discarded, as
    # in Unity. Undo and file operations are disabled
    # while playing.

    GIZMO_OPERATIONS = {
        "Move": gizmo.OPERATION.translate,
        "Rotate": gizmo.OPERATION.rotate,
        "Scale": gizmo.OPERATION.scale,
    }

    STATUS_SECONDS = 4.0

    def __init__(
        self,
        scene: Scene,
        resources: Resources,
        render_system: RenderSystem,
        window: Window,
        serializer: SceneSerializer,
        populate_demo_scene: Callable[[Scene], None],
        load_model: Callable[[str], object],
        load_model_material: Callable[[str], object] | None = None
    ):
        """
        load_model_material: model path -> Material handle
            for the model's own material (glTF), used when
            importing models. None = always use a default
            material.
        """

        self.scene = scene
        self.resources = resources
        self.render_system = render_system
        self.window = window
        self.serializer = serializer

        self._populate_demo_scene = populate_demo_scene
        self._load_model = load_model
        self._load_model_material = load_model_material

        self.visible = True

        self.selected: Entity | None = None

        self.scene_path: Path | None = None

        # Gizmo
        self.gizmo_operation: str | None = "Move"
        self.gizmo_world_space = False
        self.snap_enabled = False
        self.snap_translate = 0.25
        self.snap_rotate = 15.0
        self.snap_scale = 0.1

        # Play mode
        self.playing = False
        self._play_snapshot: Snapshot | None = None

        # Undo
        self.history = UndoHistory()
        self._pending_record: str | None = None
        self._saved_scene_text = ""

        self._gizmo_was_using = False

        # Native dialog in flight: (dialog, on_ready).
        self._dialog = None

        # (text, is_error, time shown)
        self._status: tuple[str, bool, float] | None = None

        self._title = ""

        gizmo.set_im_gui_context(
            imgui.get_current_context()
        )

        self._reset_history()

    # =====================================================
    # Frame
    # =====================================================

    def draw(
        self,
        looking: bool
    ):
        """
        Draw the editor UI. Call between ImGui
        begin_frame() and end_frame(), after the scene has
        been rendered this frame.

        looking: True while the user is flying the camera
            (RMB held); shortcuts and picking are off.
        """

        self._poll_dialog()

        if not self.scene.is_alive(self.selected):
            self.selected = None

        if self.visible:

            self._draw_menu_bar()
            self._draw_toolbar()

            draw_hierarchy_panel(self)
            draw_inspector_panel(self)

            self._draw_gizmo()

            if not looking:

                self._handle_shortcuts()
                self._handle_picking()

        self._draw_status()

        self._commit_record()

        self._update_title()

    # =====================================================
    # Selection
    # =====================================================

    def select(
        self,
        entity: Entity | None
    ):

        self.selected = entity

    def entity_name(
        self,
        entity: Entity
    ) -> str:

        name = self.scene.try_get_component(
            entity,
            NameComponent
        )

        return (
            name.name
            if name is not None
            else f"Entity {entity.index}"
        )

    # =====================================================
    # Undo / Redo
    # =====================================================

    def record(
        self,
        label: str
    ):
        """
        Mark that an edit finished this frame. One undo step
        is recorded at the end of the frame, however many
        record() calls there were.
        """

        if self._pending_record is None:
            self._pending_record = label

    def _commit_record(self):

        label = self._pending_record

        self._pending_record = None

        if label is None or self.playing:
            return

        self.history.push(
            self._snapshot(label)
        )

    def undo(self):

        if self.playing:
            return self.set_status("Undo is disabled while playing.", True)

        snapshot = self.history.undo()

        if snapshot is not None:

            self._restore(snapshot)

            self.set_status(f"Undo: {self._undone_label(snapshot)}")

    def redo(self):

        if self.playing:
            return self.set_status("Redo is disabled while playing.", True)

        snapshot = self.history.redo()

        if snapshot is not None:

            self._restore(snapshot)

            self.set_status(f"Redo: {snapshot.label}")

    def _undone_label(
        self,
        snapshot: Snapshot
    ) -> str:

        # After undo, the redo slot holds the edit that was
        # just undone.
        return self.history.redo_label or snapshot.label

    def _snapshot(
        self,
        label: str = ""
    ) -> Snapshot:

        # Render settings are not part of undo: they are
        # tweaked live in the Engine panel, and undoing a
        # transform should not also revert exposure.

        text = json.dumps(
            self.serializer.serialize(self.scene),
            sort_keys=True
        )

        selected_id = (
            self.serializer.file_ids(self.scene).get(self.selected.index)
            if self.selected is not None
            and self.scene.is_alive(self.selected)
            else None
        )

        return Snapshot(
            scene=text,
            selected_id=selected_id,
            label=label
        )

    def _restore(
        self,
        snapshot: Snapshot
    ):

        mapping = self.serializer.deserialize(
            json.loads(snapshot.scene),
            self.scene
        )

        self.selected = mapping.get(
            snapshot.selected_id
        )

    def _reset_history(self):

        snapshot = self._snapshot(
            "Open"
        )

        self.history.reset(
            snapshot
        )

        self._saved_scene_text = snapshot.scene

    @property
    def dirty(
        self
    ) -> bool:

        current = self.history.current

        return (
            current is not None
            and current.scene != self._saved_scene_text
        )

    # =====================================================
    # Play Mode
    # =====================================================

    def play(self):

        if self.playing:
            return

        self._play_snapshot = self._snapshot()

        self.playing = True

        self.set_status("Playing. Stop restores the scene.")

    def stop(self):

        if not self.playing:
            return

        self.playing = False

        if self._play_snapshot is not None:
            self._restore(self._play_snapshot)

        self._play_snapshot = None

        self.set_status("Stopped; scene restored.")

    def toggle_play(self):

        if self.playing:
            self.stop()
        else:
            self.play()

    # =====================================================
    # Entity Commands
    # =====================================================

    def create_entity(
        self,
        name: str,
        components: list = (),
        parent: Entity | None = None,
        label: str | None = None
    ) -> Entity:

        scene = self.scene

        entity = scene.create_entity()

        scene.add_component(
            entity,
            NameComponent(name)
        )

        position = (
            (0.0, 0.0, 0.0)
            if parent is not None
            else self._spawn_position()
        )

        scene.add_component(
            entity,
            TransformComponent(
                transform=Transform(position=position)
            )
        )

        if parent is not None:

            scene.add_component(
                entity,
                HierarchyComponent(parent)
            )

        for component in components:

            scene.add_component(
                entity,
                component
            )

        self.select(entity)

        self.record(label or f"Create {name}")

        return entity

    def _spawn_position(
        self
    ) -> tuple[float, float, float]:

        # A few units in front of the camera.

        camera = self.render_system.last_camera

        if camera is None:
            return (0.0, 0.0, 0.0)

        forward = camera.target - camera.position

        forward = forward / max(float(np.linalg.norm(forward)), 1e-6)

        return tuple(
            float(v)
            for v in camera.position + forward * 4.0
        )

    def delete(
        self,
        entity: Entity
    ):

        if not self.scene.is_alive(entity):
            return

        name = self.entity_name(entity)

        for victim in reversed(self.subtree(entity)):
            self.scene.destroy_entity(victim)

        if not self.scene.is_alive(self.selected):
            self.selected = None

        self.record(f"Delete {name}")

    def duplicate(
        self,
        entity: Entity
    ) -> Entity | None:

        if not self.scene.is_alive(entity):
            return None

        if not self._light_budget_allows(self.subtree(entity)):

            self.set_status(
                "Cannot duplicate: the copy would exceed the "
                "light limit.",
                True
            )

            return None

        hierarchy = self.scene.try_get_component(
            entity,
            HierarchyComponent
        )

        copies = self.serializer.instantiate(
            self.serializer.encode_subtrees(self.scene, [entity]),
            self.scene,
            root_parent=(
                hierarchy.parent
                if hierarchy is not None
                else None
            )
        )

        copy = copies[0]

        name = self.scene.try_get_component(
            copy,
            NameComponent
        )

        if name is not None:
            name.name = f"{name.name} (copy)"

        self.select(copy)

        self.record(f"Duplicate {self.entity_name(entity)}")

        return copy

    def reparent(
        self,
        entity: Entity,
        new_parent: Entity | None
    ) -> bool:
        """
        Move `entity` under `new_parent` (None = root),
        keeping its world position/rotation/scale.
        """

        scene = self.scene

        if new_parent is not None:

            if (
                new_parent == entity
                or self.is_descendant(new_parent, entity)
            ):

                self.set_status(
                    "Cannot parent an entity to itself or one of "
                    "its children.",
                    True
                )

                return False

        current = scene.try_get_component(
            entity,
            HierarchyComponent
        )

        if (
            (current.parent if current is not None else None)
            == new_parent
        ):
            return False

        transform_component = scene.get_component(
            entity,
            TransformComponent
        )

        world = transform_component.world_matrix

        parent_world = (
            scene.get_component(new_parent, TransformComponent).world_matrix
            if new_parent is not None
            else np.identity(4)
        )

        try:

            local = np.linalg.inv(parent_world) @ world

        except np.linalg.LinAlgError:

            # Zero-scaled parent: keep the local transform.
            local = transform_component.transform.matrix

        position, rotation, scale = decompose_trs(local)

        transform = transform_component.transform

        transform.position = position
        transform.rotation = rotation
        transform.scale = scale

        if current is not None:
            scene.remove_component(entity, HierarchyComponent)

        if new_parent is not None:
            scene.add_component(entity, HierarchyComponent(new_parent))

        self.record(
            f"Parent {self.entity_name(entity)}"
            if new_parent is not None
            else f"Unparent {self.entity_name(entity)}"
        )

        return True

    # =====================================================
    # Hierarchy Queries
    # =====================================================

    def children_map(
        self
    ) -> dict[int, list[Entity]]:

        children: dict[int, list[Entity]] = {}

        for entity in self.scene.entities():

            hierarchy = self.scene.try_get_component(
                entity,
                HierarchyComponent
            )

            if hierarchy is not None:

                children.setdefault(
                    hierarchy.parent.index,
                    []
                ).append(entity)

        return children

    def subtree(
        self,
        root: Entity
    ) -> list[Entity]:
        """root followed by all descendants (parents first)."""

        children = self.children_map()

        result = []
        stack = [root]

        while stack:

            entity = stack.pop()

            result.append(entity)

            stack.extend(
                reversed(children.get(entity.index, []))
            )

        return result

    def is_descendant(
        self,
        entity: Entity,
        ancestor: Entity
    ) -> bool:

        seen = set()

        current = entity

        while current is not None and current.index not in seen:

            seen.add(current.index)

            hierarchy = self.scene.try_get_component(
                current,
                HierarchyComponent
            )

            if hierarchy is None:
                return False

            if hierarchy.parent == ancestor:
                return True

            current = hierarchy.parent

        return False

    # =====================================================
    # Light Limits
    # =====================================================

    def count_components(
        self,
        component_type: type
    ) -> int:

        return len(
            self.scene.view(component_type)
        )

    def can_add_component(
        self,
        component_type: type
    ) -> tuple[bool, str]:
        """(allowed, reason if not)."""

        limits = {
            DirectionalLightComponent: (1, "Only one directional light is supported."),
            PointLightComponent: (MAX_POINT_LIGHTS, f"At most {MAX_POINT_LIGHTS} point lights."),
            SpotLightComponent: (MAX_SPOT_LIGHTS, f"At most {MAX_SPOT_LIGHTS} spot lights."),
        }

        if component_type in limits:

            limit, reason = limits[component_type]

            if self.count_components(component_type) >= limit:
                return False, reason

        return True, ""

    def _light_budget_allows(
        self,
        entities: list[Entity]
    ) -> bool:

        for component_type in (
            DirectionalLightComponent,
            PointLightComponent,
            SpotLightComponent,
        ):

            extra = sum(
                1
                for entity in entities
                if self.scene.has_component(entity, component_type)
            )

            if extra == 0:
                continue

            limit = {
                DirectionalLightComponent: 1,
                PointLightComponent: MAX_POINT_LIGHTS,
                SpotLightComponent: MAX_SPOT_LIGHTS,
            }[component_type]

            if self.count_components(component_type) + extra > limit:
                return False

        return True

    # =====================================================
    # Camera Focus
    # =====================================================

    def focus_selected(self):
        """Move the editor camera to frame the selection."""

        if self.selected is None:
            return

        scene = self.scene

        camera_entity = next(
            (
                entity
                for entity, camera in scene.registry.view_with(CameraComponent)
                if camera.primary
                and scene.has_component(entity, CameraControllerComponent)
            ),
            None
        )

        if camera_entity is None:

            self.set_status(
                "No controllable primary camera to move.",
                True
            )

            return

        target_transform = scene.get_component(
            self.selected,
            TransformComponent
        )

        target = target_transform.world_position

        # Distance from the selection's size, if it has a mesh.

        radius = 0.5

        renderer = scene.try_get_component(
            self.selected,
            MeshRendererComponent
        )

        if renderer is not None:

            mesh = self.resources.meshes.try_get(renderer.mesh)

            if mesh is not None and mesh.bounds is not None:

                low, high = mesh.bounds

                corners = np.array(
                    [
                        [x, y, z, 1.0]
                        for x in (low[0], high[0])
                        for y in (low[1], high[1])
                        for z in (low[2], high[2])
                    ]
                ) @ target_transform.world_matrix.T

                radius = max(
                    float(np.max(np.linalg.norm(corners[:, :3] - target, axis=1))),
                    0.1
                )

        camera_transform = scene.get_component(
            camera_entity,
            TransformComponent
        )

        forward = camera_transform.world_forward

        camera_transform.transform.position = (
            target
            - forward * (radius * 3.0)
        )

    # =====================================================
    # Files
    # =====================================================

    def new_scene(
        self,
        demo: bool
    ):

        def action():

            if self.playing:
                self.stop()

            self.scene.clear()

            self.scene.name = "Demo Scene" if demo else "Untitled"

            if demo:
                self._populate_demo_scene(self.scene)
            else:
                self._populate_empty_scene()

            self.new_scene_loaded(
                "Demo Scene" if demo else "Untitled"
            )

            self.set_status("New scene.")

        self._confirm_discard_then(action)

    def new_scene_loaded(
        self,
        name: str = "Demo Scene"
    ):
        """
        The scene was just filled by code (not from a file):
        no path, nothing selected, fresh undo baseline. It
        counts as unsaved only once edited.
        """

        self.scene.name = name
        self.scene_path = None
        self.selected = None

        self._reset_history()

    def _populate_empty_scene(self):

        # Every scene needs a camera to look through and
        # benefits from a light; an empty one gets both.

        scene = self.scene

        camera = scene.create_entity()

        scene.add_component(camera, NameComponent("Camera"))
        scene.add_component(camera, TransformComponent(transform=Transform(position=(0.0, 1.5, 5.0), rotation=(-15.0, 0.0, 0.0))))
        scene.add_component(camera, CameraComponent(primary=True))
        scene.add_component(camera, CameraControllerComponent())

        sun = scene.create_entity()

        scene.add_component(sun, NameComponent("Sun"))
        scene.add_component(sun, TransformComponent(transform=Transform(rotation=(-50.0, 30.0, 0.0))))
        scene.add_component(sun, DirectionalLightComponent())

    def open_scene(
        self,
        path=None
    ):

        if path is not None:

            self._confirm_discard_then(
                lambda: self._load(Path(path))
            )

            return

        def chosen(result):

            if result:
                self._load(Path(result[0]))

        self._confirm_discard_then(
            lambda: self._start_dialog(
                pfd.open_file(
                    "Open scene",
                    str(SCENE_DIRECTORY.resolve()),
                    SCENE_FILE_FILTERS
                ),
                chosen
            )
        )

    def load_initial(
        self,
        path
    ) -> bool:
        """Load at startup (no confirmation). False on error."""

        return self._load(
            Path(path)
        )

    def _load(
        self,
        path: Path
    ) -> bool:

        if self.playing:
            self.stop()

        try:

            self.serializer.load(
                path,
                self.scene,
                self.render_system.settings
            )

        except SceneFormatError as error:

            Logger.error(
                "[Editor] Failed to open %s: %s",
                path,
                error
            )

            self.set_status(
                f"Open failed: {error}",
                True
            )

            return False

        self.scene_path = _display_path(path)
        self.selected = None

        self._reset_history()

        self.set_status(f"Opened {self.scene_path}")

        return True

    def save(
        self,
        then: Callable[[], None] | None = None
    ):

        if self.playing:
            return self.set_status("Stop playing before saving.", True)

        if self.scene_path is None:
            return self.save_as(then)

        self._write(self.scene_path)

        if then is not None:
            then()

    def save_as(
        self,
        then: Callable[[], None] | None = None
    ):

        if self.playing:
            return self.set_status("Stop playing before saving.", True)

        default = (
            self.scene_path
            if self.scene_path is not None
            else SCENE_DIRECTORY / f"{_slug(self.scene.name)}.scene.json"
        )

        def chosen(result: str):

            if not result:
                return

            path = Path(result)

            if path.suffix.lower() != ".json":
                path = path.with_name(path.name + ".scene.json")

            self._write(path)

            if then is not None:
                then()

        SCENE_DIRECTORY.mkdir(parents=True, exist_ok=True)

        self._start_dialog(
            pfd.save_file(
                "Save scene as",
                str(Path(default).resolve()),
                SCENE_FILE_FILTERS
            ),
            chosen
        )

    def _write(
        self,
        path: Path
    ):

        try:

            self.serializer.save(
                path,
                self.scene,
                self.render_system.settings
            )

        except (OSError, SceneFormatError) as error:

            Logger.error("[Editor] Failed to save %s: %s", path, error)

            self.set_status(f"Save failed: {error}", True)

            return

        self.scene_path = _display_path(path)

        # Saved state = current undo state.
        self._commit_record()

        self._saved_scene_text = self.history.current.scene

        self.set_status(f"Saved {self.scene_path}")

    def import_model(self):

        def chosen(result):

            if not result:
                return

            key = str(_display_path(Path(result[0]))).replace("\\", "/")

            try:

                mesh = self.resources.meshes.load(
                    key,
                    lambda: self._load_model(key)
                )

            except Exception as error:

                Logger.exception("[Editor] Model import failed: %s", key)

                self.set_status(f"Import failed: {error}", True)

                return

            # The model's own material if it has one (glTF),
            # else the default.

            material = None

            if self._load_model_material is not None:

                try:
                    material = self._load_model_material(key)

                except Exception:

                    Logger.exception(
                        "[Editor] Could not import the material of %s; "
                        "using the default.",
                        key
                    )

            renderer = MeshRendererComponent(
                mesh=mesh,
                material=material or _default_material(self.resources)
            )

            self.create_entity(
                Path(key).stem,
                [renderer],
                label=f"Import {Path(key).name}"
            )

        self._start_dialog(
            pfd.open_file(
                "Import model",
                str(Path("assets/models").resolve()),
                MODEL_FILE_FILTERS
            ),
            chosen
        )

    # -----------------------------------------------------
    # Unsaved-changes prompt
    # -----------------------------------------------------

    def request_close(
        self,
        close: Callable[[], None]
    ):
        """
        Called when the window is asked to close. Runs
        `close` now if nothing is unsaved, otherwise after
        the user chooses Save / Discard (never on Cancel).
        """

        self._confirm_discard_then(close)

    def _confirm_discard_then(
        self,
        action: Callable[[], None]
    ):

        if self._dialog is not None:

            self.set_status("Finish the open dialog first.", True)

            return

        if not self.dirty:

            action()

            return

        def answered(button):

            if button == pfd.button.yes:
                self.save(then=action)

            elif button == pfd.button.no:
                action()

        self._start_dialog(
            pfd.message(
                "Unsaved changes",
                f"Save changes to '{self.scene.name}' first?",
                pfd.choice.yes_no_cancel,
                pfd.icon.warning
            ),
            answered
        )

    def _start_dialog(
        self,
        dialog,
        on_ready: Callable[[object], None]
    ):

        self._dialog = (
            dialog,
            on_ready
        )

    def _poll_dialog(self):

        # Native dialogs run asynchronously; the engine keeps
        # rendering while one is open.

        if self._dialog is None:
            return

        dialog, on_ready = self._dialog

        if not dialog.ready(0):
            return

        self._dialog = None

        on_ready(
            dialog.result()
        )

    @property
    def dialog_open(
        self
    ) -> bool:

        return self._dialog is not None

    # =====================================================
    # Menu Bar
    # =====================================================

    def _draw_menu_bar(self):

        if not imgui.begin_main_menu_bar():
            return

        editing = not self.playing and self._dialog is None

        if imgui.begin_menu("File"):

            if imgui.menu_item("New Empty Scene", "", False, editing)[0]:
                self.new_scene(demo=False)

            if imgui.menu_item("New Demo Scene", "", False, editing)[0]:
                self.new_scene(demo=True)

            imgui.separator()

            if imgui.menu_item("Open...", "Ctrl+O", False, editing)[0]:
                self.open_scene()

            if imgui.menu_item("Save", "Ctrl+S", False, editing)[0]:
                self.save()

            if imgui.menu_item("Save As...", "Ctrl+Shift+S", False, editing)[0]:
                self.save_as()

            imgui.separator()

            if imgui.menu_item("Import Model...", "", False, editing)[0]:
                self.import_model()

            imgui.separator()

            if imgui.menu_item("Exit", "", False, self._dialog is None)[0]:
                self.request_close(lambda: self.window.set_should_close(True))

            imgui.end_menu()

        if imgui.begin_menu("Edit"):

            undo_label = self.history.undo_label
            redo_label = self.history.redo_label

            if imgui.menu_item(
                f"Undo {undo_label}".strip(),
                "Ctrl+Z",
                False,
                editing and self.history.can_undo
            )[0]:
                self.undo()

            if imgui.menu_item(
                f"Redo {redo_label}".strip(),
                "Ctrl+Y",
                False,
                editing and self.history.can_redo
            )[0]:
                self.redo()

            imgui.separator()

            has_selection = self.selected is not None

            if imgui.menu_item("Duplicate", "Ctrl+D", False, has_selection)[0]:
                self.duplicate(self.selected)

            if imgui.menu_item("Delete", "Delete", False, has_selection)[0]:
                self.delete(self.selected)

            if imgui.menu_item("Focus Selection", "F", False, has_selection)[0]:
                self.focus_selected()

            imgui.end_menu()

        if imgui.begin_menu("Create"):

            self.draw_create_menu_items(parent=None)

            imgui.end_menu()

        imgui.end_main_menu_bar()

    def draw_create_menu_items(
        self,
        parent: Entity | None
    ):
        """Shared by the menu bar and hierarchy context menus."""

        if imgui.menu_item("Empty", "", False)[0]:
            self.create_entity("Empty", parent=parent)

        if imgui.begin_menu("Mesh"):

            for key, handle in self.resources.meshes.handle_items():

                if key.startswith("engine/"):
                    continue

                if imgui.menu_item(key, "", False)[0]:

                    self.create_entity(
                        Path(key).stem.capitalize(),
                        [
                            MeshRendererComponent(
                                mesh=handle,
                                material=_default_material(self.resources)
                            )
                        ],
                        parent=parent
                    )

            imgui.end_menu()

        imgui.separator()

        for label, component_type, factory in (
            ("Point Light", PointLightComponent, lambda: PointLightComponent(intensity=4.0, range=6.0)),
            ("Spot Light", SpotLightComponent, lambda: SpotLightComponent(intensity=8.0)),
            ("Directional Light", DirectionalLightComponent, DirectionalLightComponent),
        ):

            allowed, reason = self.can_add_component(component_type)

            if imgui.menu_item(label, "", False, allowed)[0]:
                self.create_entity(label, [factory()], parent=parent)

            if not allowed:
                imgui.set_item_tooltip(reason)

        if imgui.menu_item("Camera", "", False)[0]:
            self.create_entity("Camera", [CameraComponent()], parent=parent)

    # =====================================================
    # Toolbar
    # =====================================================

    def _draw_toolbar(self):

        io = imgui.get_io()

        imgui.set_next_window_pos(
            (io.display_size.x * 0.5, imgui.get_frame_height() + 6.0),
            imgui.Cond_.always,
            (0.5, 0.0)
        )

        flags = (
            imgui.WindowFlags_.no_title_bar.value
            | imgui.WindowFlags_.no_resize.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.always_auto_resize.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.no_focus_on_appearing.value
        )

        imgui.begin("##toolbar", None, flags)

        for label, shortcut, operation in (
            ("Select", "Q", None),
            ("Move", "W", "Move"),
            ("Rotate", "E", "Rotate"),
            ("Scale", "R", "Scale"),
        ):

            if imgui.radio_button(label, self.gizmo_operation == operation):
                self.gizmo_operation = operation

            imgui.set_item_tooltip(shortcut)

            imgui.same_line()

        imgui.text("|")
        imgui.same_line()

        if imgui.button("World" if self.gizmo_world_space else "Local"):
            self.gizmo_world_space = not self.gizmo_world_space

        imgui.set_item_tooltip("Gizmo space (click to toggle)")

        imgui.same_line()

        _, self.snap_enabled = imgui.checkbox("Snap", self.snap_enabled)

        imgui.set_item_tooltip(
            f"Move {self.snap_translate} / rotate {self.snap_rotate} deg / "
            f"scale {self.snap_scale} (hold Ctrl to invert)"
        )

        imgui.same_line()
        imgui.text("|")
        imgui.same_line()

        if self.playing:

            imgui.push_style_color(imgui.Col_.button, (0.75, 0.2, 0.15, 1.0))

            if imgui.button("Stop"):
                self.stop()

            imgui.pop_style_color()

        elif imgui.button("Play"):

            self.play()

        imgui.set_item_tooltip("Ctrl+P. Stop restores the scene.")

        imgui.end()

    # =====================================================
    # Viewport Gizmo
    # =====================================================

    def _draw_gizmo(self):

        gizmo.begin_frame()

        io = imgui.get_io()

        gizmo.set_drawlist(
            imgui.get_background_draw_list()
        )

        gizmo.set_rect(
            0.0,
            0.0,
            io.display_size.x,
            io.display_size.y
        )

        camera = self.render_system.last_camera

        entity = self.selected

        using = False

        if (
            entity is not None
            and camera is not None
            and self.gizmo_operation is not None
        ):

            using = self._manipulate(
                entity,
                camera
            )

        # Record one undo step when a drag ends.

        if self._gizmo_was_using and not using:

            self.record(
                f"{self.gizmo_operation or 'Transform'} "
                f"{self.entity_name(entity) if entity is not None else ''}".strip()
            )

        self._gizmo_was_using = using

    def _manipulate(
        self,
        entity: Entity,
        camera
    ) -> bool:

        transform_component = self.scene.try_get_component(
            entity,
            TransformComponent
        )

        if transform_component is None:
            return False

        view = gizmo.Matrix16(_column_major(camera.view_matrix))
        projection = gizmo.Matrix16(_column_major(camera.projection_matrix))

        world = transform_component.world_matrix

        matrix = gizmo.Matrix16(_column_major(world))

        operation_name = self.gizmo_operation

        snap = None

        # Ctrl temporarily inverts the snap setting.
        snapping = self.snap_enabled != imgui.get_io().key_ctrl

        if snapping:

            step = {
                "Move": self.snap_translate,
                "Rotate": self.snap_rotate,
                "Scale": self.snap_scale,
            }[operation_name]

            snap = gizmo.Matrix3([step, step, step])

        changed = gizmo.manipulate(
            view,
            projection,
            self.GIZMO_OPERATIONS[operation_name],
            gizmo.MODE.world if self.gizmo_world_space else gizmo.MODE.local,
            matrix,
            None,
            snap
        )

        if changed:

            new_world = np.array(
                matrix.values,
                dtype=np.float64
            ).reshape(4, 4).T

            self._apply_world_matrix(
                entity,
                new_world,
                operation_name
            )

        return gizmo.is_using()

    def _apply_world_matrix(
        self,
        entity: Entity,
        new_world: np.ndarray,
        operation_name: str
    ):

        scene = self.scene

        hierarchy = scene.try_get_component(
            entity,
            HierarchyComponent
        )

        parent_world = (
            scene.get_component(hierarchy.parent, TransformComponent).world_matrix
            if hierarchy is not None
            else np.identity(4)
        )

        try:
            local = np.linalg.inv(parent_world) @ new_world

        except np.linalg.LinAlgError:
            return

        position, rotation, scale = decompose_trs(local)

        transform = scene.get_component(
            entity,
            TransformComponent
        ).transform

        # Only write the channel being edited, so e.g.
        # moving never perturbs rotation through Euler
        # round-off.

        if operation_name == "Move":
            transform.position = position

        elif operation_name == "Rotate":
            transform.rotation = rotation

        elif operation_name == "Scale":
            transform.scale = scale

        # Keep this frame's world matrix in sync so the
        # gizmo does not lag a frame behind the mouse.

        scene.get_component(
            entity,
            TransformComponent
        ).world_matrix = (
            parent_world @ transform.matrix
        ).astype(np.float32)

    # =====================================================
    # Picking
    # =====================================================

    def _handle_picking(self):

        io = imgui.get_io()

        if not imgui.is_mouse_clicked(imgui.MouseButton_.left):
            return

        if (
            io.want_capture_mouse
            or gizmo.is_over()
            or gizmo.is_using()
        ):
            return

        camera = self.render_system.last_camera

        if camera is None:
            return

        ray = screen_ray(
            io.mouse_pos.x,
            io.mouse_pos.y,
            io.display_size.x,
            io.display_size.y,
            camera.view_matrix,
            camera.projection_matrix
        )

        self.select(
            pick(
                ray,
                self._pick_candidates()
            )
        )

    def _pick_candidates(
        self
    ) -> list[PickCandidate]:

        scene = self.scene
        registry = scene.registry

        candidates = []

        marker = self.render_system.settings.gizmo_scale * 0.8

        for entity, transform in registry.view_with(TransformComponent):

            renderer = registry.try_get(
                entity,
                MeshRendererComponent
            )

            if renderer is not None:

                mesh = self.resources.meshes.try_get(renderer.mesh)

                if mesh is None or mesh.bounds is None:
                    continue

                candidates.append(
                    PickCandidate(
                        entity,
                        transform.world_matrix,
                        *mesh.bounds
                    )
                )

                continue

            # The camera being looked through cannot be
            # clicked (the ray starts inside it).

            camera = registry.try_get(entity, CameraComponent)

            if camera is not None and camera.primary:
                continue

            # Everything else (lights, cameras, empties) is
            # picked by a small marker box at its position.

            model = np.identity(4)
            model[:3, 3] = transform.world_position

            candidates.append(
                PickCandidate(
                    entity,
                    model,
                    np.full(3, -marker),
                    np.full(3, marker)
                )
            )

        return candidates

    # =====================================================
    # Shortcuts
    # =====================================================

    def _handle_shortcuts(self):

        io = imgui.get_io()

        # Typing into a text field must not trigger
        # shortcuts (e.g. Delete).

        if io.want_text_input or self._dialog is not None:
            return

        def pressed(key):
            return imgui.is_key_pressed(key, False)

        ctrl = io.key_ctrl
        shift = io.key_shift

        if ctrl:

            if pressed(imgui.Key.z):

                if shift:
                    self.redo()
                else:
                    self.undo()

            elif pressed(imgui.Key.y):
                self.redo()

            elif pressed(imgui.Key.s):

                if shift:
                    self.save_as()
                else:
                    self.save()

            elif pressed(imgui.Key.o):
                self.open_scene()

            elif pressed(imgui.Key.d) and self.selected is not None:
                self.duplicate(self.selected)

            elif pressed(imgui.Key.p):
                self.toggle_play()

            return

        if pressed(imgui.Key.q):
            self.gizmo_operation = None

        elif pressed(imgui.Key.w):
            self.gizmo_operation = "Move"

        elif pressed(imgui.Key.e):
            self.gizmo_operation = "Rotate"

        elif pressed(imgui.Key.r):
            self.gizmo_operation = "Scale"

        elif pressed(imgui.Key.f):
            self.focus_selected()

        elif pressed(imgui.Key.escape):
            self.select(None)

        elif (
            pressed(imgui.Key.delete)
            and self.selected is not None
        ):
            self.delete(self.selected)

    # =====================================================
    # Status / Title
    # =====================================================

    def set_status(
        self,
        text: str,
        error: bool = False
    ):

        self._status = (
            text,
            error,
            time.monotonic()
        )

    def _draw_status(self):

        if self._status is None:
            return

        text, error, shown = self._status

        if time.monotonic() - shown > self.STATUS_SECONDS:

            self._status = None

            return

        io = imgui.get_io()

        size = imgui.calc_text_size(text)

        draw_list = imgui.get_foreground_draw_list()

        x = io.display_size.x - size.x - 16.0
        y = io.display_size.y - size.y - 12.0

        draw_list.add_rect_filled(
            (x - 8.0, y - 4.0),
            (x + size.x + 8.0, y + size.y + 4.0),
            imgui.get_color_u32((0.0, 0.0, 0.0, 0.7))
        )

        draw_list.add_text(
            (x, y),
            imgui.get_color_u32(
                (1.0, 0.45, 0.4, 1.0)
                if error
                else (0.85, 0.95, 0.85, 1.0)
            ),
            text
        )

    def _update_title(self):

        location = (
            str(self.scene_path)
            if self.scene_path is not None
            else "unsaved"
        )

        title = (
            f"{'* ' if self.dirty else ''}{self.scene.name} "
            f"({location}){'  [PLAYING]' if self.playing else ''}"
            " - OpenGL Engine"
        )

        if title != self._title:

            self.window.set_title(title)

            self._title = title


# =========================================================
# Helpers
# =========================================================

def _column_major(
    matrix
) -> list[float]:

    return np.asarray(
        matrix,
        dtype=np.float32
    ).T.ravel().tolist()


def _display_path(
    path: Path
) -> Path:
    """Path relative to the project if inside it."""

    path = Path(path).resolve()

    try:
        return path.relative_to(Path.cwd().resolve())

    except ValueError:
        return path


def _slug(
    name: str
) -> str:

    cleaned = "".join(
        c.lower() if c.isalnum() else "_"
        for c in name
    ).strip("_")

    return cleaned or "scene"


def _default_material(
    resources: Resources
):

    for key, handle in resources.materials.handle_items():

        if not key.startswith("engine/"):
            return handle

    raise SceneFormatError(
        "No material is loaded to assign to a new mesh."
    )
