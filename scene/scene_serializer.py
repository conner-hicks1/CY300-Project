import json

from collections.abc import Callable
from dataclasses import MISSING, dataclass, fields
from enum import Enum
from pathlib import Path

from core.exceptions import ResourceError
from core.logger import Logger

from ecs.components import (
    CameraComponent,
    CameraControllerComponent,
    DirectionalLightComponent,
    HierarchyComponent,
    MeshRendererComponent,
    NameComponent,
    PlanetComponent,
    PointLightComponent,
    RotatorComponent,
    SpotLightComponent,
    TransformComponent
)
from ecs.entity import Entity

from graphics.lighting import (
    MAX_POINT_LIGHTS,
    MAX_SPOT_LIGHTS
)
from graphics.render_settings import (
    RenderSettings,
    Tonemapper
)

from math3d.transform import Transform

from resources.resources import Resources

from scene.scene import Scene


# =========================================================
# Scene Files
# =========================================================
#
# JSON layout:
#
#     {
#       "format": "cy300-scene",
#       "version": 1,
#       "name": "Test Scene",
#       "render_settings": { "exposure": 1.0, ... },
#       "entities": [
#         {
#           "id": 0,
#           "components": {
#             "Name":      { "name": "Cube" },
#             "Transform": { "position": [...], "rotation": [...], "scale": [...] },
#             "Hierarchy": { "parent": 3 },
#             "MeshRenderer": { "mesh": "cube", "material": "floor", ... },
#             ...
#           }
#         }
#       ]
#     }
#
# Entity ids are file-local; entities get new handles
# when loaded and Hierarchy parents are remapped.
#
# Resources are referenced by key. Meshes whose key is a
# model file path (.obj / .gltf / .glb) are loaded on
# demand; everything else (materials, built-in meshes)
# must already be loaded.
#
# Loading is two-phase: the whole file is decoded and
# validated first, and only then is the scene replaced,
# so a bad file never destroys the current scene.

SCENE_FORMAT = "cy300-scene"
SCENE_FORMAT_VERSION = 1

MODEL_EXTENSIONS = (".obj", ".gltf", ".glb")

# Materials imported from a model file are keyed
# "<model path>#material".
MODEL_MATERIAL_SUFFIX = "material"


def model_material_key(
    model_path: str
) -> str:

    return f"{model_path}#{MODEL_MATERIAL_SUFFIX}"


class SceneFormatError(ResourceError):
    """A scene file is malformed or references missing resources."""


# =========================================================
# Codec Contexts
# =========================================================

@dataclass(slots=True)
class _EncodeContext:

    resources: Resources

    # entity.index -> file id
    ids: dict[int, int]


@dataclass(slots=True)
class _DecodeContext:

    resources: Resources

    load_model: Callable[[str], object]

    # material key -> Material for model materials
    # ("<model path>#material"); None if unsupported.
    load_material: Callable[[str], object] | None = None

    # Where in the file we are, for error messages.
    location: str = ""

    def error(
        self,
        message: str
    ) -> SceneFormatError:

        return SceneFormatError(
            f"{self.location}: {message}"
            if self.location
            else message
        )


# Hierarchy parents are file ids until every entity
# exists; this placeholder carries the id through.

@dataclass(slots=True)
class _PendingParent:

    parent_id: int


# =========================================================
# Component Codecs
# =========================================================

@dataclass(frozen=True, slots=True)
class ComponentCodec:

    # Key used in scene files and the editor.
    name: str

    component_type: type

    encode: Callable[[object, _EncodeContext], dict]
    decode: Callable[[dict, _DecodeContext], object]

    # Editor: builds a new component for "Add component";
    # None = not addable from the editor.
    create_default: Callable[[Resources], object] | None = None

    # Editor: whether "Remove component" is offered.
    removable: bool = True


def _dataclass_codec(
    name: str,
    component_type: type,
    create_default: Callable[[Resources], object] | None = None,
    removable: bool = True,
    removed_fields: frozenset[str] = frozenset()
) -> ComponentCodec:
    """
    Codec for plain dataclass components whose fields are
    floats, ints, bools, strings or tuples of numbers.
    Values are validated/coerced against each field's
    default, so a file cannot put a string where a float
    belongs.

    removed_fields: fields older versions wrote; they are
    ignored (with a warning) so old scene files still load.
    """

    component_fields = {
        field.name: field
        for field in fields(component_type)
    }

    def encode(
        component,
        _context
    ) -> dict:

        return {
            field_name: _to_json(getattr(component, field_name))
            for field_name in component_fields
        }

    def decode(
        data: dict,
        context: _DecodeContext
    ):

        obsolete = set(data) & removed_fields

        if obsolete:

            Logger.warning(
                "[SceneSerializer] %s: ignoring obsolete %s field(s): %s",
                context.location,
                name,
                ", ".join(sorted(obsolete))
            )

        unknown = set(data) - set(component_fields) - removed_fields

        if unknown:

            raise context.error(
                f"unknown {name} field(s): {', '.join(sorted(unknown))}"
            )

        values = {}

        for field_name, field in component_fields.items():

            label = f"{name}.{field_name}"

            if field_name not in data:

                if (
                    field.default is MISSING
                    and field.default_factory is MISSING
                ):
                    raise context.error(f"{label} is required.")

                continue

            values[field_name] = _coerce_like(
                data[field_name],
                _example_value(field),
                label,
                context
            )

        return component_type(
            **values
        )

    return ComponentCodec(
        name=name,
        component_type=component_type,
        encode=encode,
        decode=decode,
        create_default=create_default,
        removable=removable
    )


def _example_value(
    field
):
    """
    A value of the field's type, used to validate and
    coerce file values: the field's default if it has one,
    otherwise a zero value of its annotated type.
    """

    if field.default is not MISSING:
        return field.default

    if field.default_factory is not MISSING:
        return field.default_factory()

    zero_values = {
        str: "",
        float: 0.0,
        int: 0,
        bool: False,
    }

    if field.type not in zero_values:

        raise SceneFormatError(
            f"Field '{field.name}' has no default and an "
            f"unsupported type {field.type!r}."
        )

    return zero_values[field.type]


# ---------------------------------------------------------
# Transform
# ---------------------------------------------------------

def _encode_transform(
    component: TransformComponent,
    _context
) -> dict:

    transform = component.transform

    return {
        "position": _to_json(transform.position),
        "rotation": _to_json(transform.rotation),
        "scale": _to_json(transform.scale),
    }


def _decode_transform(
    data: dict,
    context: _DecodeContext
) -> TransformComponent:

    unknown = set(data) - {"position", "rotation", "scale"}

    if unknown:

        raise context.error(
            f"unknown Transform field(s): {', '.join(sorted(unknown))}"
        )

    values = {
        key: _coerce_like(
            data[key],
            default,
            f"Transform.{key}",
            context
        )
        for key, default in (
            ("position", (0.0, 0.0, 0.0)),
            ("rotation", (0.0, 0.0, 0.0)),
            ("scale", (1.0, 1.0, 1.0)),
        )
        if key in data
    }

    return TransformComponent(
        transform=Transform(**values)
    )


# ---------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------

def _encode_hierarchy(
    component: HierarchyComponent,
    context: _EncodeContext
) -> dict:

    return {
        "parent": context.ids[component.parent.index]
    }


def _decode_hierarchy(
    data: dict,
    context: _DecodeContext
) -> _PendingParent:

    parent = data.get(
        "parent"
    )

    if (
        set(data) != {"parent"}
        or not isinstance(parent, int)
        or isinstance(parent, bool)
    ):

        raise context.error(
            "Hierarchy must be {\"parent\": <entity id>}."
        )

    return _PendingParent(
        parent
    )


# ---------------------------------------------------------
# Mesh Renderer
# ---------------------------------------------------------

def _encode_mesh_renderer(
    component: MeshRendererComponent,
    context: _EncodeContext
) -> dict:

    resources = context.resources

    mesh_key = resources.meshes.key_of(component.mesh)
    material_key = resources.materials.key_of(component.material)

    if mesh_key is None or material_key is None:

        raise SceneFormatError(
            "MeshRenderer references a mesh or material that "
            "is not a named resource."
        )

    return {
        "mesh": mesh_key,
        "material": material_key,
        "casts_shadows": component.casts_shadows,
    }


def _decode_mesh_renderer(
    data: dict,
    context: _DecodeContext
) -> MeshRendererComponent:

    unknown = set(data) - {"mesh", "material", "casts_shadows"}

    if unknown:

        raise context.error(
            f"unknown MeshRenderer field(s): {', '.join(sorted(unknown))}"
        )

    mesh_key = data.get("mesh")
    material_key = data.get("material")

    if not isinstance(mesh_key, str) or not isinstance(material_key, str):

        raise context.error(
            "MeshRenderer needs string 'mesh' and 'material' keys."
        )

    resources = context.resources

    mesh = resources.meshes.try_get_handle(
        mesh_key
    )

    if mesh is None:

        mesh = _load_model_mesh(
            mesh_key,
            context
        )

    material = resources.materials.try_get_handle(
        material_key
    )

    if material is None:
        material = _load_model_material(material_key, context)

    if material is None:

        available = ", ".join(
            key
            for key, _ in resources.materials.handle_items()
        )

        raise context.error(
            f"unknown material '{material_key}' "
            f"(loaded: {available or 'none'})."
        )

    casts_shadows = _coerce_like(
        data.get("casts_shadows", True),
        True,
        "MeshRenderer.casts_shadows",
        context
    )

    return MeshRendererComponent(
        mesh=mesh,
        material=material,
        casts_shadows=casts_shadows
    )


def _load_model_material(
    key: str,
    context: _DecodeContext
):
    """
    Materials imported with a model are keyed
    "<model path>#material"; load one on demand.
    """

    model_path, separator, suffix = key.partition("#")

    if (
        not separator
        or suffix != MODEL_MATERIAL_SUFFIX
        or context.load_material is None
        or not Path(model_path).is_file()
    ):
        return None

    return context.resources.materials.load(
        key,
        lambda: context.load_material(key)
    )


def _load_model_mesh(
    key: str,
    context: _DecodeContext
):

    path = Path(
        key
    )

    if (
        path.suffix.lower() not in MODEL_EXTENSIONS
        or not path.is_file()
    ):

        available = ", ".join(
            key
            for key, _ in context.resources.meshes.handle_items()
        )

        raise context.error(
            f"unknown mesh '{key}' (not loaded and not a model "
            f"file; loaded: {available or 'none'})."
        )

    return context.resources.meshes.load(
        key,
        lambda: context.load_model(key)
    )


def _default_mesh_renderer(
    resources: Resources
) -> MeshRendererComponent:

    meshes = resources.meshes.handle_items()
    materials = resources.materials.handle_items()

    if not meshes or not materials:

        raise SceneFormatError(
            "Adding a MeshRenderer needs at least one loaded "
            "mesh and material."
        )

    # Prefer a user-facing resource over engine internals.

    def pick(items, preferred):

        for key, handle in items:

            if key == preferred:
                return handle

        for key, handle in items:

            if not key.startswith("engine/"):
                return handle

        return items[0][1]

    return MeshRendererComponent(
        mesh=pick(meshes, "cube"),
        material=pick(materials, "cube")
    )


# ---------------------------------------------------------
# Registry
# ---------------------------------------------------------
#
# Order matters: it is the order components appear in the
# inspector and in files.

COMPONENT_CODECS: tuple[ComponentCodec, ...] = (

    _dataclass_codec(
        "Name",
        NameComponent,
        removable=False
    ),

    ComponentCodec(
        name="Transform",
        component_type=TransformComponent,
        encode=_encode_transform,
        decode=_decode_transform,
        removable=False
    ),

    ComponentCodec(
        name="Hierarchy",
        component_type=HierarchyComponent,
        encode=_encode_hierarchy,
        decode=_decode_hierarchy,

        # Parenting is done by dragging in the hierarchy
        # panel, not by adding/removing a component (which
        # would not preserve the world transform).
        create_default=None,
        removable=False
    ),

    ComponentCodec(
        name="MeshRenderer",
        component_type=MeshRendererComponent,
        encode=_encode_mesh_renderer,
        decode=_decode_mesh_renderer,
        create_default=_default_mesh_renderer
    ),

    _dataclass_codec(
        "Camera",
        CameraComponent,
        create_default=lambda _: CameraComponent()
    ),

    _dataclass_codec(
        "CameraController",
        CameraControllerComponent,
        create_default=lambda _: CameraControllerComponent()
    ),

    _dataclass_codec(
        "Planet",
        PlanetComponent,
        create_default=lambda _: PlanetComponent()
    ),

    _dataclass_codec(
        "DirectionalLight",
        DirectionalLightComponent,
        create_default=lambda _: DirectionalLightComponent(),

        # Replaced by image-based lighting.
        removed_fields=frozenset({"ambient"})
    ),

    _dataclass_codec(
        "PointLight",
        PointLightComponent,
        create_default=lambda _: PointLightComponent()
    ),

    _dataclass_codec(
        "SpotLight",
        SpotLightComponent,
        create_default=lambda _: SpotLightComponent()
    ),

    _dataclass_codec(
        "Rotator",
        RotatorComponent,
        create_default=lambda _: RotatorComponent()
    ),
)

CODECS_BY_NAME: dict[str, ComponentCodec] = {
    codec.name: codec
    for codec in COMPONENT_CODECS
}

CODECS_BY_TYPE: dict[type, ComponentCodec] = {
    codec.component_type: codec
    for codec in COMPONENT_CODECS
}


# =========================================================
# Serializer
# =========================================================

class SceneSerializer:

    def __init__(
        self,
        resources: Resources,
        load_model: Callable[[str], object] | None = None,
        load_material: Callable[[str], object] | None = None
    ):
        """
        load_model: path -> Mesh, used for model-file mesh
            keys that are not loaded yet. Defaults to
            MeshFactory.load_model (needs a GL context).

        load_material: "<model path>#material" -> Material,
            for model materials not loaded yet. Without it,
            such keys must already be loaded.
        """

        if load_model is None:

            from graphics.mesh_factory import MeshFactory

            load_model = MeshFactory.load_model

        self._resources = resources
        self._load_model = load_model
        self._load_material = load_material

    # =====================================================
    # Encode
    # =====================================================

    def serialize(
        self,
        scene: Scene,
        render_settings: RenderSettings | None = None
    ) -> dict:

        entities = scene.entities()

        data = {
            "format": SCENE_FORMAT,
            "version": SCENE_FORMAT_VERSION,
            "name": scene.name,
            "entities": self._encode_entities(
                scene,
                entities
            ),
        }

        if render_settings is not None:

            data["render_settings"] = encode_render_settings(
                render_settings
            )

        return data

    def file_ids(
        self,
        scene: Scene
    ) -> dict[int, int]:
        """
        entity.index -> the id serialize() gives it. Lets
        callers (undo) remember a selection across a
        save/load round trip.
        """

        return {
            entity.index: file_id
            for file_id, entity in enumerate(scene.entities())
        }

    def encode_subtrees(
        self,
        scene: Scene,
        roots: list[Entity]
    ) -> list[dict]:
        """
        Encode `roots` and all their descendants (for
        duplicate / copy). Hierarchy links to entities
        outside the set are dropped from the roots.
        """

        children = self._children_by_parent(
            scene
        )

        ordered: list[Entity] = []
        seen: set[int] = set()

        def visit(entity: Entity):

            if entity.index in seen:
                return

            seen.add(entity.index)
            ordered.append(entity)

            for child in children.get(entity.index, []):
                visit(child)

        for root in roots:
            visit(root)

        return self._encode_entities(
            scene,
            ordered
        )

    def _encode_entities(
        self,
        scene: Scene,
        entities: list[Entity]
    ) -> list[dict]:

        ids = {
            entity.index: file_id
            for file_id, entity in enumerate(entities)
        }

        context = _EncodeContext(
            resources=self._resources,
            ids=ids
        )

        encoded = []

        for entity in entities:

            components = {}

            for codec in COMPONENT_CODECS:

                component = scene.try_get_component(
                    entity,
                    codec.component_type
                )

                if component is None:
                    continue

                # Links to entities outside this set (e.g.
                # the parent of a duplicated subtree root)
                # are not written.

                if (
                    isinstance(component, HierarchyComponent)
                    and component.parent.index not in ids
                ):
                    continue

                components[codec.name] = codec.encode(
                    component,
                    context
                )

            encoded.append(
                {
                    "id": ids[entity.index],
                    "components": components,
                }
            )

        return encoded

    @staticmethod
    def _children_by_parent(
        scene: Scene
    ) -> dict[int, list[Entity]]:

        children: dict[int, list[Entity]] = {}

        for entity in scene.entities():

            hierarchy = scene.try_get_component(
                entity,
                HierarchyComponent
            )

            if hierarchy is not None:

                children.setdefault(
                    hierarchy.parent.index,
                    []
                ).append(
                    entity
                )

        return children

    # =====================================================
    # Decode
    # =====================================================

    def deserialize(
        self,
        data: dict,
        scene: Scene,
        render_settings: RenderSettings | None = None
    ) -> dict[int, Entity]:
        """
        Replace the scene's contents with `data`. Returns
        file id -> new Entity. On any error the scene is
        left untouched.
        """

        if not isinstance(data, dict):

            raise SceneFormatError(
                "Scene file must contain a JSON object."
            )

        if data.get("format") != SCENE_FORMAT:

            raise SceneFormatError(
                f"Not a scene file (format is {data.get('format')!r}, "
                f"expected {SCENE_FORMAT!r})."
            )

        version = data.get(
            "version"
        )

        if version != SCENE_FORMAT_VERSION:

            raise SceneFormatError(
                f"Unsupported scene version {version!r} "
                f"(this engine reads version {SCENE_FORMAT_VERSION})."
            )

        name = data.get(
            "name",
            scene.name
        )

        if not isinstance(name, str) or not name.strip():

            raise SceneFormatError(
                "Scene 'name' must be a non-empty string."
            )

        # Phase 1: decode everything (may raise).

        plan = self._decode_entities(
            data.get("entities", [])
        )

        settings = (
            decode_render_settings(
                data["render_settings"]
            )
            if render_settings is not None
            and "render_settings" in data
            else None
        )

        # Phase 2: apply (cannot fail on file content).

        scene.clear()

        scene.name = name

        mapping = self._apply(
            scene,
            plan,
            root_parent=None
        )

        if settings is not None:

            for field in fields(RenderSettings):

                setattr(
                    render_settings,
                    field.name,
                    getattr(settings, field.name)
                )

        Logger.info(
            "[SceneSerializer] Loaded scene '%s' (%d entities).",
            name,
            len(mapping)
        )

        return mapping

    def instantiate(
        self,
        entity_data: list[dict],
        scene: Scene,
        root_parent: Entity | None = None
    ) -> dict[int, Entity]:
        """
        Add entities (e.g. from encode_subtrees) to the
        scene without clearing it. Entities whose parent is
        not in the data are parented to `root_parent`.
        """

        plan = self._decode_entities(
            entity_data
        )

        return self._apply(
            scene,
            plan,
            root_parent=root_parent
        )

    def _decode_entities(
        self,
        entity_data
    ) -> list[tuple[int, list[object]]]:

        if not isinstance(entity_data, list):

            raise SceneFormatError(
                "'entities' must be a list."
            )

        plan: list[tuple[int, list[object]]] = []
        ids: set[int] = set()

        for position, item in enumerate(entity_data):

            context = _DecodeContext(
                resources=self._resources,
                load_model=self._load_model,
                load_material=self._load_material,
                location=f"entities[{position}]"
            )

            if not isinstance(item, dict):
                raise context.error("entity must be an object.")

            file_id = item.get(
                "id"
            )

            if (
                not isinstance(file_id, int)
                or isinstance(file_id, bool)
            ):
                raise context.error("entity needs an integer 'id'.")

            if file_id in ids:
                raise context.error(f"duplicate entity id {file_id}.")

            ids.add(file_id)

            context.location = f"entity {file_id}"

            components_data = item.get(
                "components",
                {}
            )

            if not isinstance(components_data, dict):
                raise context.error("'components' must be an object.")

            components = []

            for name, component_data in components_data.items():

                codec = CODECS_BY_NAME.get(
                    name
                )

                if codec is None:

                    raise context.error(
                        f"unknown component '{name}' "
                        f"(known: {', '.join(CODECS_BY_NAME)})."
                    )

                if not isinstance(component_data, dict):

                    raise context.error(
                        f"component '{name}' must be an object."
                    )

                components.append(
                    codec.decode(
                        component_data,
                        context
                    )
                )

            # Every entity needs a transform to take part
            # in the scene; add a default rather than fail.

            if not any(
                isinstance(component, TransformComponent)
                for component in components
            ):

                components.append(
                    TransformComponent()
                )

            plan.append(
                (file_id, components)
            )

        self._validate_plan(
            plan,
            ids
        )

        return plan

    @staticmethod
    def _validate_plan(
        plan: list[tuple[int, list[object]]],
        ids: set[int]
    ):

        parents: dict[int, int] = {}

        counts = {
            DirectionalLightComponent: 0,
            PointLightComponent: 0,
            SpotLightComponent: 0,
        }

        for file_id, components in plan:

            for component in components:

                if isinstance(component, _PendingParent):

                    if component.parent_id not in ids:

                        raise SceneFormatError(
                            f"entity {file_id}: parent {component.parent_id} "
                            "does not exist."
                        )

                    parents[file_id] = component.parent_id

                if type(component) in counts:
                    counts[type(component)] += 1

        # Cycle check: walk up from every entity.

        for start in parents:

            seen = {start}
            current = parents.get(start)

            while current is not None:

                if current in seen:

                    raise SceneFormatError(
                        f"entity {start}: hierarchy contains a cycle."
                    )

                seen.add(current)
                current = parents.get(current)

        for component_type, limit, label in (
            (DirectionalLightComponent, 1, "directional light"),
            (PointLightComponent, MAX_POINT_LIGHTS, "point lights"),
            (SpotLightComponent, MAX_SPOT_LIGHTS, "spot lights"),
        ):

            if counts[component_type] > limit:

                raise SceneFormatError(
                    f"scene has {counts[component_type]} {label}; "
                    f"the engine supports at most {limit}."
                )

    @staticmethod
    def _apply(
        scene: Scene,
        plan: list[tuple[int, list[object]]],
        root_parent: Entity | None
    ) -> dict[int, Entity]:

        mapping: dict[int, Entity] = {
            file_id: scene.create_entity()
            for file_id, _ in plan
        }

        for file_id, components in plan:

            entity = mapping[file_id]

            has_parent = False

            for component in components:

                if isinstance(component, _PendingParent):

                    parent = mapping.get(
                        component.parent_id
                    )

                    if parent is None:
                        continue

                    component = HierarchyComponent(
                        parent
                    )

                    has_parent = True

                scene.add_component(
                    entity,
                    component
                )

            if (
                not has_parent
                and root_parent is not None
            ):

                scene.add_component(
                    entity,
                    HierarchyComponent(root_parent)
                )

        return mapping

    # =====================================================
    # Files
    # =====================================================

    def save(
        self,
        path,
        scene: Scene,
        render_settings: RenderSettings | None = None
    ):

        path = Path(
            path
        )

        data = self.serialize(
            scene,
            render_settings
        )

        text = json.dumps(
            data,
            indent=2
        ) + "\n"

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Write-then-rename so a crash mid-save cannot
        # leave a truncated scene file.

        temporary = path.with_name(
            path.name + ".tmp"
        )

        temporary.write_text(
            text,
            encoding="utf-8"
        )

        temporary.replace(
            path
        )

        Logger.info(
            "[SceneSerializer] Saved '%s' to %s.",
            scene.name,
            path
        )

    def load(
        self,
        path,
        scene: Scene,
        render_settings: RenderSettings | None = None
    ) -> dict[int, Entity]:

        path = Path(
            path
        )

        try:

            data = json.loads(
                path.read_text(encoding="utf-8")
            )

        except OSError as error:

            raise SceneFormatError(
                f"Cannot read scene file {path}: {error}"
            ) from None

        except json.JSONDecodeError as error:

            raise SceneFormatError(
                f"{path} is not valid JSON: {error}"
            ) from None

        try:

            return self.deserialize(
                data,
                scene,
                render_settings
            )

        except SceneFormatError as error:

            raise SceneFormatError(
                f"{path.name}: {error}"
            ) from None


# =========================================================
# Render Settings
# =========================================================

def encode_render_settings(
    settings: RenderSettings
) -> dict:

    return {
        field.name: _to_json(getattr(settings, field.name))
        for field in fields(RenderSettings)
    }


def decode_render_settings(
    data: dict
) -> RenderSettings:

    context = _DecodeContext(
        resources=None,
        load_model=None,
        location="render_settings"
    )

    if not isinstance(data, dict):
        raise context.error("must be an object.")

    defaults = RenderSettings()

    known = {
        field.name
        for field in fields(RenderSettings)
    }

    # Settings come and go between engine versions and are
    # never essential to a scene: unknown ones (e.g. from
    # an older engine) are skipped, not fatal.

    unknown = set(data) - known

    if unknown:

        Logger.warning(
            "[SceneSerializer] Ignoring unknown render setting(s): %s",
            ", ".join(sorted(unknown))
        )

    values = {}

    for key, value in data.items():

        if key in unknown:
            continue

        if key == "tonemapper":

            try:
                values[key] = Tonemapper[str(value)]

            except KeyError:

                raise context.error(
                    f"unknown tonemapper {value!r} "
                    f"(known: {', '.join(t.name for t in Tonemapper)})."
                ) from None

            continue

        values[key] = _coerce_like(
            value,
            getattr(defaults, key),
            key,
            context
        )

    return RenderSettings(
        **values
    )


# =========================================================
# Value Helpers
# =========================================================

def _to_json(
    value
):

    if isinstance(value, Enum):
        return value.name

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, (tuple, list)):
        return [_to_json(v) for v in value]

    if isinstance(value, float):
        # Keep files readable: 0.30000001192092896 -> 0.3
        return round(value, 6)

    return value


def _coerce_like(
    value,
    default,
    label: str,
    context: _DecodeContext
):
    """
    Convert a JSON value to the type of `default`,
    rejecting values that do not fit.
    """

    if isinstance(default, bool):

        if not isinstance(value, bool):
            raise context.error(f"{label} must be true or false.")

        return value

    if isinstance(default, (int, float)):

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise context.error(f"{label} must be a number.")

        return type(default)(value)

    if isinstance(default, str):

        if not isinstance(value, str):
            raise context.error(f"{label} must be a string.")

        return value

    if isinstance(default, tuple):

        if (
            not isinstance(value, list)
            or len(value) != len(default)
        ):
            raise context.error(
                f"{label} must be a list of {len(default)} numbers."
            )

        return tuple(
            _coerce_like(v, d, label, context)
            for v, d in zip(value, default)
        )

    raise context.error(
        f"{label} has an unsupported type."
    )
