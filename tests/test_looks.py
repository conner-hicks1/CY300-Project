import numpy as np
import pytest

from graphics.mesh_data import MeshData
from graphics.mesh_factory import MeshFactory
from vessel import looks
from vessel.parts import load_parts


def _attributes(data: MeshData):

    vertices = data.vertices

    return vertices[:, 0:3], vertices[:, 3:6], vertices[:, 6:9], vertices[:, 9:11]


def test_a_lathed_cylinder_faces_out():

    data = MeshFactory.lathe_data([(0.0, 0.0), (1.0, 0.0), (1.0, 2.0), (0.0, 2.0)], 16)

    positions, normals, _, _ = _attributes(data)

    # Side vertices: normals straight out; caps: down, up.
    side = np.isclose(np.hypot(positions[:, 0], positions[:, 2]), 1.0) & (np.abs(normals[:, 1]) < 0.5)

    radial = positions[side][:, [0, 2]]

    np.testing.assert_allclose(normals[side][:, [0, 2]], radial / np.linalg.norm(radial, axis=1, keepdims=True), atol=1e-5)

    assert (normals[np.isclose(positions[:, 1], 0.0) & (np.abs(normals[:, 1]) > 0.5)][:, 1] < 0).all()
    assert (normals[np.isclose(positions[:, 1], 2.0) & (np.abs(normals[:, 1]) > 0.5)][:, 1] > 0).all()

    # Triangles wind counter-clockwise seen from outside.
    triangles = data.indices.reshape(-1, 3)

    a, b, c = positions[triangles[:, 0]], positions[triangles[:, 1]], positions[triangles[:, 2]]

    face = np.cross(b - a, c - a)

    centers = (a + b + c) / 3.0

    big = np.linalg.norm(face, axis=1) > 1e-9

    outward = np.where(np.abs(centers[:, 1] - 1.0)[:, None] < 0.99, centers * [1.0, 0.0, 1.0], centers * [0.0, 1.0, 0.0] - [0.0, 1.0, 0.0])

    assert (np.einsum("ij,ij->i", face[big], outward[big]) > 0.0).mean() > 0.95


def test_creases_and_tiling():

    # A smooth bend shares its normal; a square corner keeps
    # one each side.
    data = MeshFactory.lathe_data([(1.0, 0.0), (1.0, 1.0), (0.5, 1.0)], 8)

    positions, normals, _, _ = _attributes(data)

    corner = np.isclose(positions[:, 1], 1.0) & np.isclose(np.hypot(positions[:, 0], positions[:, 2]), 1.0)

    assert {round(float(n), 3) for n in normals[corner][:, 1]} == {0.0, 1.0}

    # Tiled: u in whole repeats round (2 pi r / tile), v in m / tile.
    tiled = MeshFactory.lathe_data([(1.0, 0.0), (1.0, 3.0)], 8, tile=1.0)

    _, _, _, uvs = _attributes(tiled)

    assert uvs[:, 0].max() == pytest.approx(round(2.0 * np.pi))
    assert uvs[:, 1].max() == pytest.approx(3.0)

    # Vertex colors per segment.
    colored = MeshFactory.lathe_data([(1.0, 0.0), (1.0, 1.0), (1.0, 2.0)], 8, colors=[(1, 0, 0), (0, 1, 0)])

    _, _, colors, _ = _attributes(colored)

    assert {tuple(row) for row in np.round(colors, 3)} == {(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)}


def test_merge():

    one = MeshFactory.lathe_data([(1.0, 0.0), (1.0, 1.0)], 8)
    two = MeshFactory.lathe_data([(0.5, 1.0), (0.5, 2.0)], 8)

    both = MeshData.merge([one, two])

    assert len(both.vertices) == len(one.vertices) + len(two.vertices)
    assert both.indices.max() == len(both.vertices) - 1
    assert len(both.indices) == len(one.indices) + len(two.indices)


@pytest.mark.parametrize("part_id", sorted(load_parts()))
def test_every_part_has_a_look(part_id):

    part = load_parts()[part_id]

    pieces = looks.pieces(part)

    assert pieces

    for piece in pieces:

        assert piece.finish in looks.FINISHES

        profile = np.asarray(piece.profile)

        # Within the part (a little proud for rings and lips).
        assert profile[:, 1].min() >= -1e-6 and profile[:, 1].max() <= part.height + 1e-6
        assert profile[:, 0].min() >= 0.0 and profile[:, 0].max() <= part.radius * 1.12 + 0.05

        MeshFactory.lathe_data(piece.profile, piece.segments, piece.tile)


def test_an_interstage_covers_the_engine_above():

    parts = load_parts()

    separator = parts["decoupler_2m5"]

    plain = looks.pieces(separator)
    skirted = looks.pieces(separator, skirt=2.5)

    assert len(skirted) == len(plain) + 1

    top = max(max(height for _, height in piece.profile) for piece in skirted)

    assert top == pytest.approx(separator.height + 2.5)

    # By finish, one mesh each.
    groups = looks.merged(parts["engine_booster"])

    assert set(groups) >= {"bell", "steel", "metal"}
