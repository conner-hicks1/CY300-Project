import numpy as np
import pytest

from graphics.mesh_data import FLOATS_PER_VERTEX, MeshData
from graphics.mesh_factory import MeshFactory


def assert_counter_clockwise(data: MeshData):
    """
    Every triangle must be counter-clockwise when viewed
    from the side its vertex normals face. Face culling
    depends on this.
    """

    triangles = data.indices.reshape(-1, 3)

    p = data.positions
    n = data.normals

    face = np.cross(
        p[triangles[:, 1]] - p[triangles[:, 0]],
        p[triangles[:, 2]] - p[triangles[:, 0]]
    )

    vertex_normal = n[triangles].sum(axis=1)

    alignment = np.einsum("ij,ij->i", face, vertex_normal)

    assert np.all(alignment > 0.0), (
        f"{np.count_nonzero(alignment <= 0.0)} triangle(s) wound clockwise"
    )


def assert_valid_tangents(data: MeshData):

    tangents = data.tangents

    assert np.allclose(np.linalg.norm(tangents[:, :3], axis=1), 1.0, atol=1e-4)
    assert np.allclose(np.einsum("ij,ij->i", tangents[:, :3], data.normals), 0.0, atol=1e-4)
    assert set(np.unique(tangents[:, 3])) <= {-1.0, 1.0}


SHAPES = {
    "triangle": MeshFactory.triangle_data,
    "quad": MeshFactory.quad_data,
    "plane": MeshFactory.plane_data,
    "cube": MeshFactory.cube_data,
    "sphere": MeshFactory.sphere_data,
}


@pytest.mark.parametrize("name", SHAPES)
def test_factory_shapes_are_counter_clockwise(name):

    assert_counter_clockwise(SHAPES[name]())


@pytest.mark.parametrize("name", SHAPES)
def test_factory_shapes_have_valid_tangents(name):

    data = SHAPES[name]()

    assert data.vertices.shape[1] == FLOATS_PER_VERTEX

    assert_valid_tangents(data)


def test_quad_tangent_follows_u_axis():

    data = MeshFactory.quad_data()

    assert np.allclose(data.tangents, (1.0, 0.0, 0.0, 1.0))


def test_cube_counts():

    data = MeshFactory.cube_data()

    assert data.vertex_count == 24
    assert data.triangle_count == 12


def test_sphere_is_on_radius_with_outward_normals():

    data = MeshFactory.sphere_data(segments=16, rings=8, radius=2.0)

    assert np.allclose(np.linalg.norm(data.positions, axis=1), 2.0, atol=1e-5)
    assert np.allclose(data.positions / 2.0, data.normals, atol=1e-5)


def test_missing_normals_are_computed():

    data = MeshData.from_attributes(
        positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        indices=[0, 1, 2]
    )

    assert np.allclose(data.normals, (0.0, 0.0, 1.0))


def test_rejects_out_of_range_index():

    with pytest.raises(Exception):
        MeshData.from_attributes(positions=[(0, 0, 0)] * 3, indices=[0, 1, 3])
