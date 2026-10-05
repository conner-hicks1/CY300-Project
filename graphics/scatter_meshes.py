import math

import numpy as np

from graphics.mesh_data import MeshData


# =========================================================
# Scatter Meshes: Trees, Rocks, Grass
# =========================================================
#
# Procedural, low-poly, in unit size (trees and grass 1 m
# tall, rocks 1 m across; the instance scales them), with
# vertex colors the instance tints. Foliage normals point
# out from the crown's middle rather than off each facet, so
# a crown shades as a soft mass, the way leaves do.
#
# Trees have two levels of detail: near (crowns of several
# clumps, layered conifer whorls) and far (one shape each,
# ~20 triangles).

BARK = (0.09, 0.07, 0.05)


class _Builder:

    def __init__(self):

        self.positions = []
        self.normals = []
        self.colors = []
        self.indices = []
        self.count = 0

    def add(
        self,
        positions,
        normals,
        colors,
        indices
    ):

        positions = np.asarray(positions, dtype=np.float64)

        self.positions.append(positions)
        self.normals.append(np.asarray(normals, dtype=np.float64))
        self.colors.append(np.broadcast_to(np.asarray(colors, dtype=np.float64), positions.shape).copy())
        self.indices.append(np.asarray(indices, dtype=np.int64) + self.count)

        self.count += len(positions)

    def mesh(
        self
    ) -> MeshData:

        positions = np.vstack(self.positions)
        normals = np.vstack(self.normals)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)

        tangents = np.zeros((len(positions), 4))
        tangents[:, 0] = 1.0
        tangents[:, 3] = 1.0

        return MeshData.from_attributes(
            positions=positions,
            indices=np.concatenate(self.indices),
            normals=normals,
            colors=np.vstack(self.colors),
            tangents=tangents
        )


def _icosphere(
    subdivisions: int
) -> tuple[np.ndarray, np.ndarray]:

    t = (1.0 + math.sqrt(5.0)) / 2.0

    vertices = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]

    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]

    vertices = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in vertices]

    for _ in range(subdivisions):

        cache = {}
        refined = []

        def middle(i, j):

            key = (min(i, j), max(i, j))

            if key not in cache:

                m = vertices[i] + vertices[j]

                vertices.append(m / np.linalg.norm(m))

                cache[key] = len(vertices) - 1

            return cache[key]

        for a, b, c in faces:

            ab, bc, ca = middle(a, b), middle(b, c), middle(c, a)

            refined += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]

        faces = refined

    return np.array(vertices), np.array(faces)


def _noise(
    points: np.ndarray,
    seed: int,
    frequency: float
) -> np.ndarray:
    """Smooth pseudo-noise on points (sums of random sines)."""

    rng = np.random.default_rng(seed)

    total = np.zeros(len(points))

    for _ in range(6):

        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)

        total += np.sin(points @ direction * frequency * (0.6 + 0.8 * rng.random()) + rng.random() * 6.28)

    return total / 6.0


def _cylinder(
    builder: _Builder,
    bottom: float,
    top: float,
    radius_bottom: float,
    radius_top: float,
    sides: int,
    color,
    offset=(0.0, 0.0, 0.0)
):

    angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)

    ring = np.stack((np.cos(angles), np.zeros(sides), np.sin(angles)), axis=1)

    offset = np.asarray(offset)

    positions = np.vstack((ring * radius_bottom + (0.0, bottom, 0.0), ring * radius_top + (0.0, top, 0.0))) + offset

    normals = np.vstack((ring, ring))

    indices = []

    for i in range(sides):

        j = (i + 1) % sides

        indices += [i, sides + j, j, i, sides + i, sides + j]

    builder.add(positions, normals, color, indices)


def _blob(
    builder: _Builder,
    center,
    radii,
    color,
    seed: int,
    subdivisions: int = 1,
    roughness: float = 0.18,
    crown_center=None
):
    """A lumpy ellipsoid (a clump of leaves, or a rock)."""

    vertices, faces = _icosphere(subdivisions)

    bumps = 1.0 + roughness * _noise(vertices, seed, 5.0)

    positions = vertices * bumps[:, None] * np.asarray(radii) + np.asarray(center)

    if crown_center is None:
        normals = vertices
    else:
        normals = positions - np.asarray(crown_center)

    # Darker inside the crown and underneath.
    shade = 0.75 + 0.25 * (vertices[:, 1] * 0.5 + 0.5)

    colors = np.asarray(color)[None, :] * shade[:, None]

    builder.add(positions, normals, colors, faces.ravel())


# ---------------------------------------------------------
# Trees
# ---------------------------------------------------------

def conifer(
    near: bool = True
) -> MeshData:

    b = _Builder()

    leaves = (0.018, 0.042, 0.022)

    _cylinder(b, 0.0, 0.35, 0.025, 0.015, 6 if near else 4, BARK)

    if not near:

        # One cone.
        sides = 6

        angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)

        ring = np.stack((np.cos(angles) * 0.22, np.full(sides, 0.15), np.sin(angles) * 0.22), axis=1)

        positions = np.vstack((ring, [(0.0, 1.0, 0.0)]))

        normals = np.vstack((ring * (1, 0, 1) + (0, 0.5, 0), [(0.0, 1.0, 0.0)]))

        indices = []

        for i in range(sides):
            indices += [i, sides, (i + 1) % sides]

        b.add(positions, normals, leaves, indices)

        return b.mesh()

    rng = np.random.default_rng(11)

    # Whorls of branches: drooping, ragged skirts, narrower up
    # the tree, each with a dozen branch tips sticking out
    # (alternate rim vertices pushed out and down), so the
    # outline is spiky and light comes through the layers.
    layers = 10

    for layer in range(layers):

        f = layer / (layers - 1)

        bottom = 0.14 + 0.76 * f
        top = bottom + 0.16 - 0.04 * f

        radius = 0.25 * (1.0 - f) ** 0.85 + 0.025

        sides = 14

        angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False) + rng.random() * 0.8

        tips = np.arange(sides) % 2 == 0

        jag = np.where(tips, 1.0 + 0.35 * rng.random(sides), 0.55 + 0.15 * rng.random(sides))

        droop = np.where(tips, 0.07, 0.0) * (1.0 - 0.5 * f)

        ring = np.stack((np.cos(angles) * radius * jag, bottom - droop, np.sin(angles) * radius * jag), axis=1)

        apex = np.array([[0.0, top, 0.0]])

        positions = np.vstack((ring, apex))

        normals = np.vstack((ring * (1.0, 0.0, 1.0) + (0.0, radius * 1.2, 0.0), [(0.0, 1.0, 0.0)]))

        shade = 0.8 + 0.4 * rng.random()

        indices = []

        for i in range(sides):
            indices += [i, sides, (i + 1) % sides]

        # Tips lighter (new growth), the inside darker.
        colors = np.vstack((
            np.where(tips[:, None], 1.15, 0.8) * np.asarray(leaves) * shade,
            np.asarray(leaves)[None, :] * shade * 0.5
        ))

        b.add(positions, normals, colors, indices)

        # Underside, darker.
        under = np.array([[0.0, bottom + 0.02, 0.0]])

        b.add(
            np.vstack((ring, under)),
            np.vstack((-normals[:sides] * (1, -1, 1), [(0.0, -1.0, 0.0)])),
            np.asarray(leaves) * shade * 0.5,
            [v for i in range(sides) for v in (i, (i + 1) % sides, sides)]
        )

    return b.mesh()


def _broadleaf_crown(
    b: _Builder,
    center,
    spread,
    height,
    clumps: int,
    leaves,
    seed: int,
    near: bool,
    flat: float = 1.0
):

    rng = np.random.default_rng(seed)

    center = np.asarray(center, dtype=np.float64)

    if not near:

        _blob(b, center, (spread, height * 0.5, spread), leaves, seed, subdivisions=0, roughness=0.1, crown_center=center)

        return

    for i in range(clumps):

        angle = rng.random() * 2.0 * math.pi

        reach = spread * 0.55 * math.sqrt(rng.random())

        offset = np.array([math.cos(angle) * reach, (rng.random() - 0.4) * height * 0.45 * flat, math.sin(angle) * reach])

        size = spread * (0.45 + 0.25 * rng.random())

        shade = 0.8 + 0.4 * rng.random()

        _blob(
            b,
            center + offset,
            (size, size * 0.8 * flat, size),
            np.asarray(leaves) * shade,
            seed * 31 + i,
            subdivisions=1,
            roughness=0.22,
            crown_center=center
        )


def broadleaf(
    near: bool = True
) -> MeshData:

    b = _Builder()

    _cylinder(b, 0.0, 0.45, 0.035, 0.022, 7 if near else 4, BARK)

    _broadleaf_crown(b, (0.0, 0.62, 0.0), 0.3, 0.6, 9, (0.03, 0.068, 0.018), 5, near)

    return b.mesh()


def rainforest(
    near: bool = True
) -> MeshData:

    b = _Builder()

    _cylinder(b, 0.0, 0.75, 0.03, 0.02, 7 if near else 4, (0.12, 0.10, 0.07))

    _broadleaf_crown(b, (0.0, 0.84, 0.0), 0.32, 0.3, 7, (0.014, 0.045, 0.012), 9, near, flat=0.6)

    return b.mesh()


def acacia(
    near: bool = True
) -> MeshData:

    b = _Builder()

    _cylinder(b, 0.0, 0.55, 0.04, 0.03, 6 if near else 4, BARK)

    if near:

        # Forked: two leaning limbs.
        for side in (-1.0, 1.0):
            _cylinder(b, 0.0, 0.3, 0.025, 0.015, 5, BARK, offset=(side * 0.08, 0.5, 0.0))

    _broadleaf_crown(b, (0.0, 0.85, 0.0), 0.55, 0.18, 6, (0.045, 0.065, 0.02), 13, near, flat=0.35)

    return b.mesh()


# ---------------------------------------------------------
# Rocks
# ---------------------------------------------------------

def rock(
    variant: int
) -> MeshData:
    """A boulder 1 m across, flattened, lumpy; its color is the instance's."""

    b = _Builder()

    rng = np.random.default_rng(100 + variant)

    vertices, faces = _icosphere(1)

    lumps = 1.0 + 0.28 * _noise(vertices, 200 + variant, 2.5) + 0.08 * _noise(vertices, 300 + variant, 7.0)

    stretch = np.array([0.5 + 0.15 * rng.random(), 0.28 + 0.12 * rng.random(), 0.45 + 0.15 * rng.random()])

    positions = vertices * lumps[:, None] * stretch

    # Faceted: each face its own vertices and flat normal
    # (broken rock), shades of grey to tint.
    tris = positions[faces]

    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])

    shade = 0.8 + 0.4 * rng.random(len(faces))

    b.add(
        tris.reshape(-1, 3),
        np.repeat(normals, 3, axis=0),
        np.repeat(np.stack((shade, shade, shade), axis=1), 3, axis=0),
        np.arange(len(faces) * 3)
    )

    return b.mesh()


# ---------------------------------------------------------
# Grass
# ---------------------------------------------------------

def grass_tuft(
    variant: int
) -> MeshData:
    """Blades 1 m tall (scaled to ~0.3-0.7 m), both sides."""

    b = _Builder()

    rng = np.random.default_rng(500 + variant)

    blades = 9

    for i in range(blades):

        angle = rng.random() * 2.0 * math.pi

        lean = 0.15 + 0.3 * rng.random()

        foot = np.array([math.cos(angle), 0.0, math.sin(angle)]) * 0.06 * rng.random()

        out = np.array([math.cos(angle), 0.0, math.sin(angle)])
        side = np.array([-math.sin(angle), 0.0, math.cos(angle)]) * 0.012

        height = 0.6 + 0.4 * rng.random()

        tip = foot + out * lean * height + np.array([0.0, height, 0.0])
        mid = foot + out * lean * 0.35 * height + np.array([0.0, 0.55 * height, 0.0])

        positions = np.array([foot - side, foot + side, mid - side * 0.7, mid + side * 0.7, tip])

        normal = np.array([0.0, 1.0, 0.0]) + 0.4 * out

        normals = np.repeat(normal[None, :], 5, axis=0)

        # Darker at the base.
        shade = np.array([0.5, 0.5, 0.85, 0.85, 1.1])[:, None]

        front = [0, 1, 3, 0, 3, 2, 2, 3, 4]
        back = [0, 3, 1, 0, 2, 3, 2, 4, 3]

        b.add(positions, normals, shade * np.ones(3), front + back)

    return b.mesh()


# Kind -> (near, far) builders (planet/scatter.py kinds).
def build_meshes() -> dict[tuple[int, str], MeshData]:

    meshes = {}

    trees = (conifer, broadleaf, rainforest, acacia)

    for kind, builder in enumerate(trees):

        meshes[(kind, "near")] = builder(True)
        meshes[(kind, "far")] = builder(False)

    for variant in range(4):
        meshes[(4 + variant, "near")] = rock(variant)

    for variant in range(2):
        meshes[(8 + variant, "near")] = grass_tuft(variant)

    return meshes
