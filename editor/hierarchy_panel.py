from typing import TYPE_CHECKING

from imgui_bundle import imgui

from ecs.components import (
    CameraComponent,
    DirectionalLightComponent,
    HierarchyComponent,
    MeshRendererComponent,
    PointLightComponent,
    SpotLightComponent
)
from ecs.entity import Entity

if TYPE_CHECKING:
    from editor.scene_editor import SceneEditor


# =========================================================
# Hierarchy Panel
# =========================================================
#
# Tree of entities by parent. Click selects, drag an
# entity onto another to parent it (onto the empty area
# below the tree to unparent), right-click for a context
# menu.

DRAG_PAYLOAD = "ENTITY"

_TAGS = (
    (CameraComponent, "camera"),
    (DirectionalLightComponent, "sun"),
    (PointLightComponent, "point"),
    (SpotLightComponent, "spot"),
    (MeshRendererComponent, "mesh"),
)


def draw_hierarchy_panel(
    editor: "SceneEditor"
):

    io = imgui.get_io()

    top = imgui.get_frame_height() + 4.0

    imgui.set_next_window_pos(
        (io.display_size.x - 340.0, top),
        imgui.Cond_.first_use_ever
    )

    imgui.set_next_window_size(
        (330.0, 300.0),
        imgui.Cond_.first_use_ever
    )

    imgui.begin("Hierarchy")

    scene = editor.scene

    imgui.text_disabled(
        f"{scene.name}  ({scene.entity_count} entities)"
    )

    imgui.separator()

    children = editor.children_map()

    roots = [
        entity
        for entity in scene.entities()
        if not scene.has_component(entity, HierarchyComponent)
    ]

    # Deferred: modifying the scene mid-tree would change
    # what later rows iterate over.
    actions: list = []

    for root in roots:

        _draw_node(
            editor,
            root,
            children,
            actions
        )

    # -----------------------------------------------------
    # Empty area: drop target to unparent, click to
    # deselect, right-click to create.
    # -----------------------------------------------------

    remaining = imgui.get_content_region_avail()

    imgui.invisible_button(
        "##hierarchy_background",
        (max(remaining.x, 1.0), max(remaining.y, 40.0))
    )

    if imgui.is_item_clicked():
        editor.select(None)

    if imgui.begin_drag_drop_target():

        payload = imgui.accept_drag_drop_payload_py_id(
            DRAG_PAYLOAD
        )

        if payload is not None:

            dragged = _entity_from_payload(editor, payload.data_id)

            if dragged is not None:
                actions.append(lambda: editor.reparent(dragged, None))

        imgui.end_drag_drop_target()

    if imgui.begin_popup_context_item("##hierarchy_background_menu"):

        imgui.text_disabled("Create")
        imgui.separator()

        editor.draw_create_menu_items(parent=None)

        imgui.end_popup()

    imgui.end()

    for action in actions:
        action()


def _draw_node(
    editor: "SceneEditor",
    entity: Entity,
    children: dict[int, list[Entity]],
    actions: list
):

    scene = editor.scene

    child_list = children.get(
        entity.index,
        []
    )

    flags = (
        imgui.TreeNodeFlags_.open_on_arrow.value
        | imgui.TreeNodeFlags_.open_on_double_click.value
        | imgui.TreeNodeFlags_.span_avail_width.value
        | imgui.TreeNodeFlags_.default_open.value
    )

    if not child_list:
        flags |= imgui.TreeNodeFlags_.leaf.value

    if entity == editor.selected:
        flags |= imgui.TreeNodeFlags_.selected.value

    tags = [
        tag
        for component_type, tag in _TAGS
        if scene.has_component(entity, component_type)
    ]

    label = editor.entity_name(entity)

    if tags:
        label = f"{label}  [{', '.join(tags)}]"

    imgui.push_id(
        f"entity{entity.index}_{entity.generation}"
    )

    opened = imgui.tree_node_ex(
        "##node",
        flags,
        label
    )

    # Select on click, but not when the click only
    # expanded/collapsed the node.

    if (
        imgui.is_item_clicked()
        and not imgui.is_item_toggled_open()
    ):
        editor.select(entity)

    # -----------------------------------------------------
    # Drag source / drop target
    # -----------------------------------------------------

    if imgui.begin_drag_drop_source():

        imgui.set_drag_drop_payload_py_id(
            DRAG_PAYLOAD,
            _payload_id(entity)
        )

        imgui.text(
            f"Move {editor.entity_name(entity)}"
        )

        imgui.end_drag_drop_source()

    if imgui.begin_drag_drop_target():

        payload = imgui.accept_drag_drop_payload_py_id(
            DRAG_PAYLOAD
        )

        if payload is not None:

            dragged = _entity_from_payload(editor, payload.data_id)

            if dragged is not None and dragged != entity:
                actions.append(
                    lambda d=dragged: editor.reparent(d, entity)
                )

        imgui.end_drag_drop_target()

    # -----------------------------------------------------
    # Context menu
    # -----------------------------------------------------

    if imgui.begin_popup_context_item():

        editor.select(entity)

        if imgui.begin_menu("Create Child"):

            editor.draw_create_menu_items(parent=entity)

            imgui.end_menu()

        if imgui.menu_item("Duplicate", "Ctrl+D", False)[0]:
            actions.append(lambda: editor.duplicate(entity))

        is_child = scene.has_component(entity, HierarchyComponent)

        if imgui.menu_item("Unparent", "", False, is_child)[0]:
            actions.append(lambda: editor.reparent(entity, None))

        if imgui.menu_item("Focus", "F", False)[0]:
            actions.append(editor.focus_selected)

        imgui.separator()

        if imgui.menu_item("Delete", "Delete", False)[0]:
            actions.append(lambda: editor.delete(entity))

        imgui.end_popup()

    if opened:

        for child in child_list:

            _draw_node(
                editor,
                child,
                children,
                actions
            )

        imgui.tree_pop()

    imgui.pop_id()


# =========================================================
# Drag Payload
# =========================================================
#
# ImGui payloads here carry a single integer, so pack the
# entity's index and generation into one.

def _payload_id(
    entity: Entity
) -> int:

    return (entity.generation << 32) | entity.index


def _entity_from_payload(
    editor: "SceneEditor",
    data_id: int
) -> Entity | None:

    entity = Entity(
        index=data_id & 0xFFFFFFFF,
        generation=data_id >> 32
    )

    return (
        entity
        if editor.scene.is_alive(entity)
        else None
    )
