import base64
import json
import struct

from pathlib import Path

import numpy as np

from core.exceptions import ResourceError
from core.logger import Logger

from graphics.mesh_data import MeshData
from math3d.matrices import normal_matrix


# =========================================================
# Model Loading
# =========================================================
#
# Loads geometry into a single MeshData. Materials are not
# imported; assign an engine Material to the resulting
# mesh.
#
#     .obj           Wavefront OBJ (v / vt / vn / f)
#     .gltf / .glb   glTF 2.0 (triangle primitives; node
#                    transforms of the default scene are
#                    baked in)

def load_model_data(
    path
) -> MeshData:

    path = Path(
        path
    )

    if not path.is_file():

        raise ResourceError(
            f"Model file does not exist: {path}"
        )

    suffix = path.suffix.lower()

    if suffix == ".obj":
        data = load_obj(path)

    elif suffix in (".gltf", ".glb"):
        data = load_gltf(path)

    else:

        raise ResourceError(
            f"Unsupported model format '{suffix}': {path}"
        )

    Logger.info(
        "[ModelLoader] Loaded '%s' (%d vertices, %d triangles).",
        path.name,
        data.vertex_count,
        data.triangle_count
    )

    return data


# =========================================================
# OBJ
# =========================================================

def load_obj(
    path
) -> MeshData:

    path = Path(
        path
    )

    positions: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    normals: list[tuple[float, float, float]] = []

    # (position, texcoord, normal) index triple -> vertex

    vertex_lookup: dict[tuple[int, int, int], int] = {}

    out_positions = []
    out_uvs = []
    out_normals = []

    indices: list[int] = []

    missing_normals = False

    def resolve(
        token: str,
        count: int,
        line_number: int
    ) -> int:

        # OBJ indices are 1-based; negatives count back
        # from the most recent element. Returns -1 for an
        # omitted index.

        if token == "":
            return -1

        try:
            value = int(token)

        except ValueError:

            raise ResourceError(
                f"{path.name}:{line_number}: bad index '{token}'."
            ) from None

        index = (
            value - 1
            if value > 0
            else count + value
        )

        if not 0 <= index < count:

            raise ResourceError(
                f"{path.name}:{line_number}: index {value} "
                f"out of range (have {count})."
            )

        return index

    with path.open(
        encoding="utf-8",
        errors="replace"
    ) as file:

        for line_number, raw_line in enumerate(
            file,
            start=1
        ):

            line = raw_line.split(
                "#",
                1
            )[0].strip()

            if not line:
                continue

            parts = line.split()
            keyword = parts[0]

            try:

                if keyword == "v":

                    positions.append(
                        tuple(float(v) for v in parts[1:4])
                    )

                elif keyword == "vt":

                    u = float(parts[1])

                    v = (
                        float(parts[2])
                        if len(parts) > 2
                        else 0.0
                    )

                    texcoords.append(
                        (u, v)
                    )

                elif keyword == "vn":

                    normals.append(
                        tuple(float(v) for v in parts[1:4])
                    )

                elif keyword == "f":

                    corners = []

                    for token in parts[1:]:

                        fields = token.split("/")

                        fields += [""] * (3 - len(fields))

                        key = (
                            resolve(fields[0], len(positions), line_number),
                            resolve(fields[1], len(texcoords), line_number),
                            resolve(fields[2], len(normals), line_number),
                        )

                        if key[0] < 0:

                            raise ResourceError(
                                f"{path.name}:{line_number}: "
                                "face vertex has no position."
                            )

                        if key[2] < 0:
                            missing_normals = True

                        vertex = vertex_lookup.get(
                            key
                        )

                        if vertex is None:

                            vertex = len(out_positions)

                            vertex_lookup[key] = vertex

                            out_positions.append(
                                positions[key[0]]
                            )

                            out_uvs.append(
                                texcoords[key[1]]
                                if key[1] >= 0
                                else (0.0, 0.0)
                            )

                            out_normals.append(
                                normals[key[2]]
                                if key[2] >= 0
                                else (0.0, 0.0, 0.0)
                            )

                        corners.append(
                            vertex
                        )

                    if len(corners) < 3:

                        raise ResourceError(
                            f"{path.name}:{line_number}: "
                            "face needs at least 3 vertices."
                        )

                    # Fan-triangulate polygons.

                    for k in range(1, len(corners) - 1):

                        indices.extend(
                            (
                                corners[0],
                                corners[k],
                                corners[k + 1]
                            )
                        )

                # o, g, s, usemtl, mtllib, l, p: ignored.

            except (ValueError, IndexError) as error:

                raise ResourceError(
                    f"{path.name}:{line_number}: "
                    f"malformed '{keyword}' line: {error}"
                ) from None

    if not indices:

        raise ResourceError(
            f"OBJ contains no faces: {path}"
        )

    return MeshData.from_attributes(
        positions=out_positions,
        indices=indices,
        normals=(
            None
            if missing_normals
            else out_normals
        ),
        uvs=out_uvs
    )


# =========================================================
# glTF 2.0
# =========================================================

_GLB_MAGIC = 0x46546C67      # "glTF"
_GLB_CHUNK_JSON = 0x4E4F534A # "JSON"
_GLB_CHUNK_BIN = 0x004E4942  # "BIN\0"

_COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

_TYPE_SIZES = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}

_MODE_TRIANGLES = 4


def load_gltf(
    path
) -> MeshData:

    path = Path(
        path
    )

    document, glb_binary = _read_gltf_document(
        path
    )

    buffers = [
        _load_buffer(path, buffer, glb_binary)
        for buffer in document.get("buffers", [])
    ]

    all_positions = []
    all_normals = []
    all_uvs = []
    all_indices = []

    have_normals = True
    vertex_offset = 0

    for mesh_index, world in _mesh_instances(
        document
    ):

        mesh = document["meshes"][mesh_index]

        normal_transform = normal_matrix(
            world
        )

        mirrored = np.linalg.det(
            world[:3, :3]
        ) < 0.0

        for primitive in mesh.get("primitives", []):

            mode = primitive.get(
                "mode",
                _MODE_TRIANGLES
            )

            if mode != _MODE_TRIANGLES:

                Logger.warning(
                    "[ModelLoader] %s: skipping primitive "
                    "with mode %d (only triangles supported).",
                    path.name,
                    mode
                )

                continue

            attributes = primitive["attributes"]

            if "POSITION" not in attributes:

                raise ResourceError(
                    f"{path.name}: primitive has no POSITION."
                )

            positions = _read_accessor(
                document,
                buffers,
                attributes["POSITION"]
            ).astype(np.float64)

            count = len(
                positions
            )

            # Bake the node transform into the vertices.

            positions = (
                np.hstack(
                    (positions, np.ones((count, 1)))
                )
                @ world.T
            )[:, :3]

            if "NORMAL" in attributes:

                normals = _read_accessor(
                    document,
                    buffers,
                    attributes["NORMAL"]
                ).astype(np.float64) @ normal_transform.T.astype(np.float64)

                lengths = np.linalg.norm(
                    normals,
                    axis=1,
                    keepdims=True
                )

                normals = normals / np.maximum(
                    lengths,
                    1e-12
                )

            else:

                have_normals = False

                normals = np.zeros(
                    (count, 3)
                )

            if "TEXCOORD_0" in attributes:

                uvs = _read_accessor(
                    document,
                    buffers,
                    attributes["TEXCOORD_0"]
                ).astype(np.float64)

                # glTF UV origin is top-left; engine
                # textures are flipped to bottom-left.

                uvs[:, 1] = 1.0 - uvs[:, 1]

            else:

                uvs = np.zeros(
                    (count, 2)
                )

            if "indices" in primitive:

                indices = _read_accessor(
                    document,
                    buffers,
                    primitive["indices"]
                ).reshape(-1).astype(np.int64)

            else:

                indices = np.arange(
                    count,
                    dtype=np.int64
                )

            triangles = indices.reshape(
                -1,
                3
            )

            # A mirroring transform flips winding.

            if mirrored:
                triangles = triangles[:, [0, 2, 1]]

            all_positions.append(positions)
            all_normals.append(normals)
            all_uvs.append(uvs)
            all_indices.append(triangles.reshape(-1) + vertex_offset)

            vertex_offset += count

    if not all_indices:

        raise ResourceError(
            f"glTF contains no triangle geometry: {path}"
        )

    return MeshData.from_attributes(
        positions=np.vstack(all_positions),
        indices=np.concatenate(all_indices),
        normals=(
            np.vstack(all_normals)
            if have_normals
            else None
        ),
        uvs=np.vstack(all_uvs)
    )


# ---------------------------------------------------------
# Document / Buffers
# ---------------------------------------------------------

def _read_gltf_document(
    path: Path
) -> tuple[dict, bytes | None]:

    raw = path.read_bytes()

    if path.suffix.lower() == ".gltf":

        try:
            return json.loads(raw.decode("utf-8")), None

        except (UnicodeDecodeError, json.JSONDecodeError) as error:

            raise ResourceError(
                f"Invalid glTF JSON in {path}: {error}"
            ) from None

    # GLB: 12-byte header, then chunks.

    if len(raw) < 12:

        raise ResourceError(
            f"GLB file too small: {path}"
        )

    magic, version, length = struct.unpack_from(
        "<III",
        raw,
        0
    )

    if magic != _GLB_MAGIC or version != 2:

        raise ResourceError(
            f"Not a glTF 2.0 binary file: {path}"
        )

    document = None
    binary = None
    offset = 12

    while offset + 8 <= min(length, len(raw)):

        chunk_length, chunk_type = struct.unpack_from(
            "<II",
            raw,
            offset
        )

        chunk = raw[
            offset + 8:offset + 8 + chunk_length
        ]

        if chunk_type == _GLB_CHUNK_JSON:
            document = json.loads(chunk.decode("utf-8"))

        elif chunk_type == _GLB_CHUNK_BIN and binary is None:
            binary = bytes(chunk)

        offset += 8 + chunk_length

    if document is None:

        raise ResourceError(
            f"GLB has no JSON chunk: {path}"
        )

    return document, binary


def _load_buffer(
    path: Path,
    buffer: dict,
    glb_binary: bytes | None
) -> bytes:

    uri = buffer.get(
        "uri"
    )

    if uri is None:

        # GLB-embedded buffer.

        if glb_binary is None:

            raise ResourceError(
                f"{path.name}: buffer has no uri and no GLB BIN chunk."
            )

        return glb_binary

    if uri.startswith("data:"):

        try:

            _, encoded = uri.split(
                ",",
                1
            )

            return base64.b64decode(
                encoded
            )

        except ValueError as error:

            raise ResourceError(
                f"{path.name}: invalid data URI: {error}"
            ) from None

    buffer_path = (
        path.parent
        / uri
    )

    if not buffer_path.is_file():

        raise ResourceError(
            f"{path.name}: buffer file not found: {buffer_path}"
        )

    return buffer_path.read_bytes()


def _read_accessor(
    document: dict,
    buffers: list[bytes],
    accessor_index: int
) -> np.ndarray:

    accessor = document["accessors"][accessor_index]

    if "sparse" in accessor:

        raise ResourceError(
            "Sparse glTF accessors are not supported."
        )

    dtype = np.dtype(
        _COMPONENT_DTYPES[accessor["componentType"]]
    ).newbyteorder("<")

    components = _TYPE_SIZES[
        accessor["type"]
    ]

    count = accessor["count"]

    if "bufferView" not in accessor:

        # Accessor with no data is all zeros by spec.

        return np.zeros(
            (count, components),
            dtype=dtype
        )

    view = document["bufferViews"][
        accessor["bufferView"]
    ]

    data = buffers[
        view["buffer"]
    ]

    base = (
        view.get("byteOffset", 0)
        + accessor.get("byteOffset", 0)
    )

    element_size = (
        dtype.itemsize
        * components
    )

    stride = view.get(
        "byteStride",
        element_size
    )

    if stride == element_size:

        array = np.frombuffer(
            data,
            dtype=dtype,
            count=count * components,
            offset=base
        )

    else:

        # Interleaved: gather each element.

        raw = np.frombuffer(
            data,
            dtype=np.uint8,
            count=(count - 1) * stride + element_size,
            offset=base
        )

        rows = np.lib.stride_tricks.as_strided(
            raw,
            shape=(count, element_size),
            strides=(stride, 1)
        )

        array = np.ascontiguousarray(
            rows
        ).view(dtype)

    return array.reshape(
        count,
        components
    ).copy()


# ---------------------------------------------------------
# Scene Graph
# ---------------------------------------------------------

def _mesh_instances(
    document: dict
):
    """
    Yield (mesh index, world matrix) for every node with a
    mesh in the default scene. Documents without scenes
    use every mesh once with an identity transform.
    """

    nodes = document.get(
        "nodes",
        []
    )

    scenes = document.get(
        "scenes",
        []
    )

    if not scenes:

        for mesh_index in range(
            len(document.get("meshes", []))
        ):

            yield mesh_index, np.identity(4)

        return

    scene = scenes[
        document.get("scene", 0)
    ]

    stack = [
        (node_index, np.identity(4))
        for node_index in scene.get("nodes", [])
    ]

    while stack:

        node_index, parent = stack.pop()

        node = nodes[node_index]

        world = parent @ _node_matrix(node)

        if "mesh" in node:
            yield node["mesh"], world

        for child in node.get("children", []):
            stack.append((child, world))


def _node_matrix(
    node: dict
) -> np.ndarray:

    if "matrix" in node:

        # glTF matrices are column-major.

        return np.array(
            node["matrix"],
            dtype=np.float64
        ).reshape(4, 4).T

    tx, ty, tz = node.get(
        "translation",
        (0.0, 0.0, 0.0)
    )

    x, y, z, w = node.get(
        "rotation",
        (0.0, 0.0, 0.0, 1.0)
    )

    sx, sy, sz = node.get(
        "scale",
        (1.0, 1.0, 1.0)
    )

    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )

    matrix = np.identity(4)

    matrix[:3, :3] = rotation * np.array([sx, sy, sz])
    matrix[:3, 3] = (tx, ty, tz)

    return matrix
