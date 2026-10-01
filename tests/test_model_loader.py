import json
import struct

from pathlib import Path

import numpy as np
import pytest

from core.exceptions import ResourceError
from graphics.model_loader import load_model_data

from tests.test_mesh_data import assert_counter_clockwise, assert_valid_tangents


MODELS = Path(__file__).resolve().parent / "fixtures" / "models"


# =========================================================
# OBJ
# =========================================================

def test_torus_obj():

    data = load_model_data(MODELS / "torus.obj")

    assert data.triangle_count == 48 * 24 * 2

    assert_counter_clockwise(data)
    assert_valid_tangents(data)


def test_obj_polygon_triangulation_and_negative_indices(tmp_path):

    path = tmp_path / "quad.obj"

    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "vn 0 0 1\n"
        "f -4//1 -3//1 -2//1 -1//1\n"
    )

    data = load_model_data(path)

    assert data.triangle_count == 2
    assert np.allclose(data.normals, (0.0, 0.0, 1.0))


def test_obj_without_normals_gets_computed_normals(tmp_path):

    path = tmp_path / "tri.obj"

    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

    data = load_model_data(path)

    assert np.allclose(data.normals, (0.0, 0.0, 1.0))


def test_obj_bad_index_reports_line(tmp_path):

    path = tmp_path / "bad.obj"

    path.write_text("v 0 0 0\nf 1 2 3\n")

    with pytest.raises(ResourceError, match="bad.obj:2"):
        load_model_data(path)


def test_unsupported_extension(tmp_path):

    path = tmp_path / "model.fbx"
    path.write_bytes(b"")

    with pytest.raises(ResourceError, match="Unsupported"):
        load_model_data(path)


# =========================================================
# glTF / GLB
# =========================================================

def test_pyramid_gltf_applies_node_transform():

    data = load_model_data(MODELS / "pyramid.gltf")

    assert data.triangle_count == 6

    assert_counter_clockwise(data)
    assert_valid_tangents(data)

    # The root node yaws 45 degrees, turning the square
    # base's corners onto the X/Z axes.
    base = data.positions[np.isclose(data.positions[:, 1], 0.0)]
    corner_distance = np.sqrt(0.5)

    assert np.allclose(np.abs(base[:, [0, 2]]).max(axis=1), corner_distance, atol=1e-5)


def test_glb_matches_gltf(tmp_path):

    source = MODELS / "pyramid.gltf"
    document = json.loads(source.read_text())

    # Convert the embedded data URI into a GLB BIN chunk.
    import base64

    uri = document["buffers"][0].pop("uri")
    binary = base64.b64decode(uri.split(",", 1)[1])

    json_chunk = json.dumps(document).encode()
    json_chunk += b" " * (-len(json_chunk) % 4)
    binary += b"\0" * (-len(binary) % 4)

    glb = (
        struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(json_chunk) + 8 + len(binary))
        + struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
        + struct.pack("<II", len(binary), 0x004E4942) + binary
    )

    path = tmp_path / "pyramid.glb"
    path.write_bytes(glb)

    expected = load_model_data(source)
    actual = load_model_data(path)

    assert np.allclose(expected.vertices, actual.vertices)
    assert np.array_equal(expected.indices, actual.indices)


def test_gltf_uv_flip():

    data = load_model_data(MODELS / "pyramid.gltf")

    # Apex UV is (0.5, 0.0) in glTF (top) -> (0.5, 1.0) in engine space.
    apex = np.isclose(data.positions[:, 1], 0.8)

    assert np.allclose(data.uvs[apex], (0.5, 1.0))


# =========================================================
# glTF Materials
# =========================================================

def test_pyramid_gltf_material_description():

    from graphics.model_loader import load_gltf_material

    material = load_gltf_material(MODELS / "pyramid.gltf")

    assert material.name == "Terracotta"
    assert material.base_color == pytest.approx((0.8, 0.36, 0.18))
    assert material.metallic == 0.0
    assert material.roughness == pytest.approx(0.55)
    assert material.textures == {}


def test_obj_has_no_gltf_material():

    from graphics.model_loader import load_gltf_material

    assert load_gltf_material(MODELS / "torus.obj") is None


def test_gltf_material_textures_file_and_embedded(tmp_path):

    import base64

    from graphics.model_loader import load_gltf_material

    source = json.loads((MODELS / "pyramid.gltf").read_text())

    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )

    (tmp_path / "albedo.png").write_bytes(png)

    source["images"] = [
        {"uri": "albedo.png"},
        {"uri": "data:image/png;base64," + base64.b64encode(png).decode()},
    ]
    source["textures"] = [{"source": 0}, {"source": 1}]
    source["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
    source["materials"][0]["normalTexture"] = {"index": 1, "scale": 0.5}
    source["materials"][0]["emissiveFactor"] = [1.0, 0.5, 0.0]

    path = tmp_path / "textured.gltf"
    path.write_text(json.dumps(source))

    material = load_gltf_material(path)

    assert material.textures["base_color"].path == tmp_path / "albedo.png"
    assert material.textures["normal"].data == png
    assert material.normal_scale == 0.5
    assert material.emissive == pytest.approx((1.0, 0.5, 0.0))
