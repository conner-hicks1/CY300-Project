"""
Generate the procedural demo assets.

    py -3.14 tools/generate_assets.py

Outputs (committed, so running this is only needed after
changing it):

    assets/textures/tiles_albedo.png     sRGB color
    assets/textures/tiles_normal.png     tangent-space normal map (OpenGL, +Y up)
    assets/textures/tiles_orm.png        R occlusion, G roughness, B metallic (glTF packing)
    assets/textures/uv_checker.png       sRGB color test pattern (cube)
    tests/fixtures/models/torus.obj      OBJ with v/vt/vn
    tests/fixtures/models/pyramid.gltf   glTF 2.0, embedded buffer, node transform

Deterministic: the same script always produces the same
files.
"""

import base64
import json
import sys

from pathlib import Path

import numpy as np

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent

TEXTURES = ROOT / "assets" / "textures"
MODELS = ROOT / "tests" / "fixtures" / "models"


# =========================================================
# Tiles Texture Set
# =========================================================

def generate_tiles(
    size: int = 512,
    tiles: int = 4,
    grout: float = 0.04,
    bevel: float = 0.06
):
    rng = np.random.default_rng(1234)

    # Coordinates within a tile, 0..1.

    coords = (np.arange(size) + 0.5) / size * tiles
    u = coords[None, :] % 1.0
    v = coords[:, None] % 1.0

    tile_x = np.floor(coords[None, :]).astype(int).repeat(size, 0)
    tile_y = np.floor(coords[:, None]).astype(int).repeat(size, 1)

    # Distance to the nearest tile edge.

    edge = np.minimum(
        np.minimum(u, 1.0 - u),
        np.minimum(v, 1.0 - v)
    )

    # Height: 0 in the grout, ramps up across the bevel,
    # 1 on the tile face, plus faint surface noise.

    height = np.clip(
        (edge - grout) / bevel,
        0.0,
        1.0
    )

    height = height * height * (3.0 - 2.0 * height)  # smoothstep

    noise = rng.normal(0.0, 1.0, (size, size))
    noise = _blur(noise, 3) * 0.02

    height = height + noise * (height > 0.5)

    in_grout = edge < grout

    # -----------------------------------------------------
    # Albedo
    # -----------------------------------------------------

    tile_tint = rng.uniform(
        0.85,
        1.0,
        (tiles, tiles)
    )[tile_y, tile_x]

    base = np.array([0.78, 0.74, 0.68])
    grout_color = np.array([0.22, 0.21, 0.20])

    albedo = (
        base[None, None, :]
        * tile_tint[..., None]
        * (1.0 + noise[..., None] * 2.0)
    )

    albedo[in_grout] = grout_color

    _save_rgb(
        TEXTURES / "tiles_albedo.png",
        albedo
    )

    # -----------------------------------------------------
    # Normal Map
    # -----------------------------------------------------
    #
    # Image rows run top-to-bottom but textures are flipped
    # on load so +V points up; dh/dv = -dh/drow.

    strength = 6.0

    dh_du = np.gradient(height, axis=1) * size / tiles
    dh_dv = -np.gradient(height, axis=0) * size / tiles

    normal = np.stack(
        (
            -dh_du / strength,
            -dh_dv / strength,
            np.ones_like(height)
        ),
        axis=-1
    )

    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)

    _save_rgb(
        TEXTURES / "tiles_normal.png",
        normal * 0.5 + 0.5
    )

    # -----------------------------------------------------
    # Occlusion / Roughness / Metallic
    # -----------------------------------------------------
    #
    # glTF packs these into one texture: R occlusion,
    # G roughness, B metallic. Glazed tiles are smooth, the
    # grout is rough and sits in a groove (occluded), and
    # nothing is metal.

    roughness = np.where(
        in_grout,
        0.9,
        0.4 + noise * 4.0
    )

    # Darken toward the groove: the bevel ramps back up.
    occlusion = 0.55 + 0.45 * np.clip(height, 0.0, 1.0)

    metallic = np.zeros_like(height)

    _save_rgb(
        TEXTURES / "tiles_orm.png",
        np.stack(
            (occlusion, np.clip(roughness, 0.05, 1.0), metallic),
            axis=-1
        )
    )


# =========================================================
# Vessel Surfaces
# =========================================================
#
# Tileable surfaces for rocket parts (vessel/looks.py
# finishes), each an albedo (a multiplier near white: the
# part's color comes from its vertices), a normal map and an
# ORM (R occlusion, G a roughness factor, B metallic 1):
#
#   hull     skin panels with seams and rivet rows (two by
#            two in a tile)
#   foil     crinkled insulation foil
#   shield   an ablative heat shield: charred, pitted
#   cells    solar cells: deep blue squares in a silver grid
#            (eight by eight in a tile, a metre on a wing)
#   tubes    an engine bell: the regenerative cooling tubes
#            brazed side by side (round it, u), burnt bronze
#            and blue near the throat (along it, v), dark at
#            the exit
#   foam     sprayed-on insulation foam: lumpy, matte

def _height_to_normal(
    height: np.ndarray,
    strength: float
) -> np.ndarray:

    size = height.shape[0]

    dh_du = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5 * size
    dh_dv = -(np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5 * size

    normal = np.stack((-dh_du / strength, -dh_dv / strength, np.ones_like(height)), axis=-1)

    return normal / np.linalg.norm(normal, axis=-1, keepdims=True)


def _save_surface(
    name: str,
    albedo: np.ndarray,
    height: np.ndarray,
    occlusion: np.ndarray,
    roughness: np.ndarray,
    strength: float
):

    _save_rgb(TEXTURES / f"{name}_albedo.png", albedo)
    _save_rgb(TEXTURES / f"{name}_normal.png", _height_to_normal(height, strength) * 0.5 + 0.5)
    _save_rgb(
        TEXTURES / f"{name}_orm.png",
        np.stack((np.clip(occlusion, 0.0, 1.0), np.clip(roughness, 0.05, 1.0), np.ones_like(height)), axis=-1)
    )


def generate_hull(
    size: int = 512
):

    rng = np.random.default_rng(301)

    coords = (np.arange(size) + 0.5) / size

    u = np.broadcast_to(coords[None, :], (size, size))
    v = np.broadcast_to(coords[:, None], (size, size))

    # Two panels each way: seams at 0 and 0.5.
    seam_u = np.minimum(np.abs((u * 2.0) % 1.0), 1.0 - (u * 2.0) % 1.0)
    seam_v = np.minimum(np.abs((v * 2.0) % 1.0), 1.0 - (v * 2.0) % 1.0)

    seam = np.minimum(seam_u, seam_v)

    groove = 1.0 - np.clip(seam / 0.012, 0.0, 1.0)

    # Rivets along each seam, a row each side.
    rivet = np.zeros((size, size))

    for offset in (0.03, 0.97):

        along_v = np.abs(((v * 2.0) % 1.0) - offset) < 0.006
        spots_u = np.abs(((u * 32.0) % 1.0) - 0.5) < 0.18

        rivet = np.maximum(rivet, (along_v & spots_u).astype(float))

        along_u = np.abs(((u * 2.0) % 1.0) - offset) < 0.006
        spots_v = np.abs(((v * 32.0) % 1.0) - 0.5) < 0.18

        rivet = np.maximum(rivet, (along_u & spots_v).astype(float))

    rivet = _blur(rivet, 1)

    # Each panel a shade apart; faint streaks of wear.
    panel = rng.uniform(0.93, 1.0, (2, 2))[(v * 2.0).astype(int) % 2, (u * 2.0).astype(int) % 2]

    streaks = _blur(rng.normal(0.0, 1.0, (size, size)), 6)
    streaks = _blur(streaks, 2) * 0.05

    height = -groove * 0.5 + rivet * 0.35 + streaks * 0.2

    albedo = (panel * (1.0 + streaks) * (1.0 - 0.12 * groove))[..., None] * np.ones(3)

    occlusion = 1.0 - 0.2 * groove

    roughness = 0.9 + streaks * 2.0 + 0.1 * groove

    _save_surface("hull", albedo, height, occlusion, roughness, 6.0)


def generate_foil(
    size: int = 512
):

    rng = np.random.default_rng(302)

    # Crinkles: sharp ridges of folded noise at two scales.
    coarse = _blur(rng.normal(0.0, 1.0, (size, size)), 10)
    fine = _blur(rng.normal(0.0, 1.0, (size, size)), 3)

    coarse /= np.abs(coarse).max()
    fine /= np.abs(fine).max()

    height = np.abs(coarse) * 0.7 + np.abs(fine) * 0.3

    albedo = (0.85 + 0.15 * (1.0 - height))[..., None] * np.ones(3)

    occlusion = 0.8 + 0.2 * height

    roughness = 0.7 + 0.5 * np.abs(fine)

    _save_surface("foil", albedo, height, occlusion, roughness, 1.2)


def generate_shield(
    size: int = 512
):

    rng = np.random.default_rng(303)

    char = _blur(rng.normal(0.0, 1.0, (size, size)), 8)
    char /= np.abs(char).max()

    # Pits: small dark dimples scattered over it.
    pits = (rng.uniform(0.0, 1.0, (size, size)) > 0.997).astype(float)
    pits = _blur(pits, 2)
    pits /= max(pits.max(), 1e-9)

    height = char * 0.3 - pits * 0.8

    albedo = (0.75 + 0.25 * char - 0.4 * pits)[..., None] * np.array([1.0, 0.92, 0.85])

    occlusion = 1.0 - 0.5 * pits

    roughness = np.full((size, size), 1.0) - 0.1 * char

    _save_surface("shield", np.clip(albedo, 0.0, 1.0), height, occlusion, roughness, 2.0)


def generate_cells(
    size: int = 512,
    cells: int = 8
):

    rng = np.random.default_rng(304)

    coords = (np.arange(size) + 0.5) / size

    u = np.broadcast_to(coords[None, :], (size, size))
    v = np.broadcast_to(coords[:, None], (size, size))

    # Distance to the nearest cell edge (in cells).
    fu = (u * cells) % 1.0
    fv = (v * cells) % 1.0

    edge = np.minimum(np.minimum(fu, 1.0 - fu), np.minimum(fv, 1.0 - fv))

    gap = 1.0 - np.clip((edge - 0.02) / 0.015, 0.0, 1.0)

    # Each cell a shade apart; thin fingers across it.
    shade = rng.uniform(0.85, 1.0, (cells, cells))[(v * cells).astype(int) % cells, (u * cells).astype(int) % cells]

    fingers = (np.abs(((v * cells * 12.0) % 1.0) - 0.5) < 0.04).astype(float) * 0.25

    blue = np.array([0.13, 0.2, 0.52])
    silver = np.array([0.75, 0.77, 0.8])

    cell = blue * (shade * (1.0 + 0.6 * fingers))[..., None]

    albedo = cell * (1.0 - gap[..., None]) + silver * gap[..., None]

    height = -0.6 * gap + 0.05 * fingers

    occlusion = 1.0 - 0.15 * gap

    roughness = 0.35 + 0.5 * gap

    _save_surface("cells", np.clip(albedo, 0.0, 1.0), height, occlusion, roughness, 3.0)


def generate_tubes(
    size: int = 512,
    tubes: int = 96
):

    rng = np.random.default_rng(305)

    coords = (np.arange(size) + 0.5) / size

    u = np.broadcast_to(coords[None, :], (size, size))
    v = np.broadcast_to(coords[:, None], (size, size))

    # Each tube a rounded ridge; the seams between dark.
    phase = (u * tubes) % 1.0

    ridge = np.sqrt(np.clip(1.0 - (2.0 * phase - 1.0) ** 2, 0.0, 1.0))

    seam = 1.0 - np.clip(ridge / 0.35, 0.0, 1.0)

    # Along the bell (v: its profile from the exit, 0, up to
    # the throat and back down inside): the heat's colors.
    heat = np.exp(-((v - 0.42) / 0.16) ** 2)

    exit_dark = np.clip(1.0 - v / 0.2, 0.0, 1.0)

    bronze = np.array([0.62, 0.45, 0.3])
    blue = np.array([0.32, 0.34, 0.5])
    steel = np.array([0.48, 0.47, 0.46])

    tint = steel * (1.0 - heat[..., None]) + (bronze * 0.6 + blue * 0.4) * heat[..., None]

    tint = tint * (1.0 - 0.45 * exit_dark[..., None])

    streaks = _blur(rng.normal(0.0, 1.0, (size, size)), 3) * 0.04

    albedo = tint * (0.85 + 0.15 * ridge[..., None] + streaks[..., None]) * (1.0 - 0.35 * seam[..., None])

    height = ridge * 0.6 - seam * 0.2

    occlusion = 1.0 - 0.3 * seam

    roughness = 0.75 + 0.2 * seam - 0.15 * heat

    _save_surface("tubes", np.clip(albedo, 0.0, 1.0), height, occlusion, roughness, 4.0)


def generate_foam(
    size: int = 512
):

    rng = np.random.default_rng(306)

    lumps = _blur(rng.normal(0.0, 1.0, (size, size)), 5)
    fine = _blur(rng.normal(0.0, 1.0, (size, size)), 1)

    lumps /= np.abs(lumps).max()
    fine /= np.abs(fine).max()

    height = 0.7 * lumps + 0.3 * fine

    albedo = (0.88 + 0.08 * lumps + 0.04 * fine)[..., None] * np.ones(3)

    occlusion = 0.85 + 0.15 * (lumps * 0.5 + 0.5)

    roughness = np.full((size, size), 0.95)

    _save_surface("foam", np.clip(albedo, 0.0, 1.0), height, occlusion, roughness, 2.5)


# =========================================================
# UV Checker
# =========================================================

def generate_uv_checker(
    size: int = 512,
    cells: int = 8
):
    """
    Colorful checkerboard: hue sweeps along U, brightness
    rises along V, so texture orientation is obvious on any
    face. A light grid separates the cells, and one corner
    cell is marked to show where UV (0, 0) is.
    """

    coords = (np.arange(size) + 0.5) / size

    u = coords[None, :].repeat(size, 0)

    # Image rows run top-down; textures are flipped on load,
    # so v = 1 - row.
    v = 1.0 - coords[:, None].repeat(size, 1)

    cell_u = np.floor(u * cells)
    cell_v = np.floor(v * cells)

    checker = (cell_u + cell_v) % 2 == 0

    hue = (cell_u + 0.5) / cells

    # HSV -> RGB (saturation 0.6)
    k = (np.stack((5.0, 3.0, 1.0)) [:, None, None] + hue[None] * 6.0) % 6.0

    rgb = 1.0 - 0.6 * np.clip(np.minimum(k, 4.0 - k), 0.0, 1.0)

    value = np.where(checker, 0.9, 0.55) * (0.55 + 0.45 * (cell_v + 0.5) / cells)

    image = np.moveaxis(rgb, 0, -1) * value[..., None]

    # Grid lines.
    line = 0.03

    fu = (u * cells) % 1.0
    fv = (v * cells) % 1.0

    on_line = (fu < line) | (fu > 1 - line) | (fv < line) | (fv > 1 - line)

    image[on_line] = 0.93

    # Origin marker: dark triangle in the (0, 0) cell.
    origin = (cell_u == 0) & (cell_v == 0) & (fu + fv < 0.6)

    image[origin] = 0.08

    _save_rgb(
        TEXTURES / "uv_checker.png",
        image
    )


def _blur(
    image: np.ndarray,
    radius: int
) -> np.ndarray:

    kernel = np.ones(2 * radius + 1) / (2 * radius + 1)

    for axis in (0, 1):

        image = np.apply_along_axis(
            lambda row: np.convolve(
                np.pad(row, radius, mode="wrap"),
                kernel,
                mode="valid"
            ),
            axis,
            image
        )

    return image


def _save_rgb(
    path: Path,
    rgb: np.ndarray
):

    path.parent.mkdir(parents=True, exist_ok=True)

    pixels = np.clip(
        np.round(rgb * 255.0),
        0,
        255
    ).astype(np.uint8)

    Image.fromarray(pixels, "RGB").save(path)

    print(f"wrote {path.relative_to(ROOT)}")


# =========================================================
# Torus (OBJ)
# =========================================================

def generate_torus(
    major_radius: float = 0.35,
    minor_radius: float = 0.15,
    major_segments: int = 48,
    minor_segments: int = 24
):
    lines = [
        "# Procedural torus - generated by tools/generate_assets.py",
        "o Torus",
    ]

    for i in range(major_segments + 1):

        theta = 2.0 * np.pi * i / major_segments

        for j in range(minor_segments + 1):

            phi = 2.0 * np.pi * j / minor_segments

            ring = major_radius + minor_radius * np.cos(phi)

            x = ring * np.cos(theta)
            y = minor_radius * np.sin(phi)
            z = ring * np.sin(theta)

            nx = np.cos(phi) * np.cos(theta)
            ny = np.sin(phi)
            nz = np.cos(phi) * np.sin(theta)

            lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
            lines.append(f"vt {i / major_segments:.6f} {j / minor_segments:.6f}")
            lines.append(f"vn {nx:.6f} {ny:.6f} {nz:.6f}")

    columns = minor_segments + 1

    def vertex(i, j):

        index = i * columns + j + 1   # OBJ is 1-based

        return f"{index}/{index}/{index}"

    for i in range(major_segments):

        for j in range(minor_segments):

            # Quad corners; order chosen so the face is
            # counter-clockwise seen from outside (checked
            # by tests/test_model_loader.py).

            lines.append(
                "f "
                + " ".join(
                    (
                        vertex(i, j),
                        vertex(i, j + 1),
                        vertex(i + 1, j + 1),
                        vertex(i + 1, j),
                    )
                )
            )

    path = MODELS / "torus.obj"

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {path.relative_to(ROOT)}")


# =========================================================
# Pyramid (glTF)
# =========================================================

def generate_pyramid():

    apex = np.array([0.0, 0.8, 0.0])

    base = [
        np.array([-0.5, 0.0, 0.5]),
        np.array([0.5, 0.0, 0.5]),
        np.array([0.5, 0.0, -0.5]),
        np.array([-0.5, 0.0, -0.5]),
    ]

    positions = []
    normals = []
    uvs = []
    indices = []

    # Four sides, flat shaded (3 unique vertices each).

    for k in range(4):

        a = base[k]
        b = base[(k + 1) % 4]

        normal = np.cross(b - a, apex - a)
        normal /= np.linalg.norm(normal)

        start = len(positions)

        positions += [a, b, apex]
        normals += [normal] * 3

        # glTF UV origin is top-left.
        uvs += [(0.0, 1.0), (1.0, 1.0), (0.5, 0.0)]

        indices += [start, start + 1, start + 2]

    # Base, facing down (counter-clockwise from below).

    start = len(positions)

    positions += [base[0], base[3], base[2], base[1]]
    normals += [np.array([0.0, -1.0, 0.0])] * 4
    uvs += [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    indices += [start, start + 1, start + 2, start + 2, start + 3, start]

    positions = np.array(positions, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    uvs = np.array(uvs, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint16)

    # Pack: positions | normals | uvs | indices

    blobs = [
        positions.tobytes(),
        normals.tobytes(),
        uvs.tobytes(),
        indices.tobytes(),
    ]

    offsets = []
    buffer = b""

    for blob in blobs:

        # 4-byte alignment per the spec.
        buffer += b"\0" * (-len(buffer) % 4)

        offsets.append(len(buffer))

        buffer += blob

    ARRAY_BUFFER = 34962
    ELEMENT_ARRAY_BUFFER = 34963
    FLOAT = 5126
    UNSIGNED_SHORT = 5123

    document = {
        "asset": {
            "version": "2.0",
            "generator": "tools/generate_assets.py",
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {
                # Parent node rotates 45 degrees around Y
                # to exercise node transforms.
                "name": "PyramidRoot",
                "rotation": [0.0, 0.38268343, 0.0, 0.92387953],
                "children": [1],
            },
            {
                "name": "Pyramid",
                "mesh": 0,
                "scale": [1.0, 1.0, 1.0],
            },
        ],
        "materials": [
            {
                "name": "Terracotta",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.8, 0.36, 0.18, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.55,
                },
            }
        ],
        "meshes": [
            {
                "name": "Pyramid",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 1,
                            "TEXCOORD_0": 2,
                        },
                        "indices": 3,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "buffers": [
            {
                "byteLength": len(buffer),
                "uri": "data:application/octet-stream;base64,"
                + base64.b64encode(buffer).decode("ascii"),
            }
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[0], "byteLength": len(blobs[0]), "target": ARRAY_BUFFER},
            {"buffer": 0, "byteOffset": offsets[1], "byteLength": len(blobs[1]), "target": ARRAY_BUFFER},
            {"buffer": 0, "byteOffset": offsets[2], "byteLength": len(blobs[2]), "target": ARRAY_BUFFER},
            {"buffer": 0, "byteOffset": offsets[3], "byteLength": len(blobs[3]), "target": ELEMENT_ARRAY_BUFFER},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": FLOAT,
                "count": len(positions),
                "type": "VEC3",
                "min": positions.min(axis=0).tolist(),
                "max": positions.max(axis=0).tolist(),
            },
            {"bufferView": 1, "componentType": FLOAT, "count": len(normals), "type": "VEC3"},
            {"bufferView": 2, "componentType": FLOAT, "count": len(uvs), "type": "VEC2"},
            {"bufferView": 3, "componentType": UNSIGNED_SHORT, "count": len(indices), "type": "SCALAR"},
        ],
    }

    path = MODELS / "pyramid.gltf"

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {path.relative_to(ROOT)}")


# =========================================================
# Main
# =========================================================

def main() -> int:

    generate_tiles()
    generate_hull()
    generate_foil()
    generate_shield()
    generate_cells()
    generate_tubes()
    generate_foam()
    generate_uv_checker()
    generate_torus()
    generate_pyramid()

    return 0


if __name__ == "__main__":

    sys.exit(main())
