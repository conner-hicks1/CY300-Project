import math

from dataclasses import dataclass

import numpy as np

from graphics.mesh_data import MeshData
from graphics.mesh_factory import MeshFactory

from vessel.parts import PartDefinition


# =========================================================
# How Parts Look
# =========================================================
#
# Each part as a few lathed pieces (graphics/mesh_factory.py
# lathe_data: a profile turned about the stack's axis), each
# with a finish: a tank's painted skin between bare metal
# rings, an engine's mount, chamber and bell (dark inside),
# an ogive nose, a parachute's packed dome, a heat shield's
# ablative face, a probe core's gold foil, a dish's
# parabola. Measured from the part's bottom (0) to its top
# (its height), within its radii.
#
# Finishes, shared by every part (the material: how metal,
# how rough, which surface texture; vertex colors carry the
# piece's own color, or the part's):
#
#   paint   the part's color, a satin finish, hull panels
#   metal   bare aluminium, rings and fittings
#   steel   dark metal: engine mounts, separators' bands
#   bell    an engine bell: dark, burnt metal
#   foil    gold insulation foil, crinkled
#   shield  ablative heat shield, charred and pitted
#   glass   a lens or a window: dark, glossy
#   fabric  a parachute's pack (the part's color)
#   cells   solar cells (a wing's, drawn by
#           systems/vessel_system.py: not lathed)
#   foam    a hydrogen tank's sprayed-on orange insulation
#
# Besides the lathed pieces, details off the axis (details()):
# boxes and small cylinders round the part, a few copies
# evenly spaced: RCS thruster quads, windows and a hatch, a
# tank's cable raceway, an engine's turbopump and gimbal
# actuators, a separator's bolts, a dish's feed horn, a
# probe's instrument boxes, an RTG's fins.

FINISHES = {
    #          metallic  roughness  texture
    "paint":  (0.05,     0.5,       "hull"),
    "metal":  (0.9,      0.32,      "hull"),
    "steel":  (0.85,     0.45,      None),
    "bell":   (0.8,      0.55,      "tubes"),
    "foil":   (1.0,      0.28,      "foil"),
    "shield": (0.0,      0.92,      "shield"),
    "glass":  (0.1,      0.06,      None),
    "fabric": (0.0,      0.85,      None),
    "cells":  (0.1,      0.3,       "cells"),
    "foam":   (0.0,      0.85,      "foam"),
}

ALUMINIUM = (0.78, 0.79, 0.8)
STEEL = (0.36, 0.36, 0.38)
BELL = (0.3, 0.27, 0.25)
BELL_INSIDE = (0.12, 0.1, 0.09)
GOLD = (1.0, 0.76, 0.33)
CHAR = (0.2, 0.15, 0.11)
GLASS = (0.04, 0.05, 0.07)
DARK_PAINT = (0.12, 0.12, 0.13)


@dataclass(frozen=True, slots=True)
class Piece:

    profile: tuple                  # ((radius, height), ...) bottom up (outside), back down (inside)
    finish: str
    color: tuple | None = None      # None: the part's own
    segments: int = 40
    tile: float | None = 1.0        # m a surface texture's repeat (None: stretched)


def _band(
    radius: float,
    bottom: float,
    top: float,
    proud: float
) -> tuple:
    """A ring standing `proud` m off a skin of `radius`, from bottom to top (its edges squared)."""

    return ((radius, bottom), (radius + proud, bottom), (radius + proud, top), (radius, top))


def _cylinder(
    radius: float,
    bottom: float,
    top: float
) -> tuple:

    return ((0.0, bottom), (radius, bottom), (radius, top), (0.0, top))


def _frustum(
    bottom_radius: float,
    top_radius: float,
    bottom: float,
    top: float
) -> tuple:

    return ((0.0, bottom), (bottom_radius, bottom), (top_radius, top), (0.0, top))


def _curve(
    function,
    bottom: float,
    top: float,
    steps: int = 12
) -> list:
    """Profile points (radius(t), height) for t from 0 (bottom) to 1 (top)."""

    return [(max(function(k / steps), 0.0), bottom + (top - bottom) * k / steps) for k in range(steps + 1)]


def pieces(
    part: PartDefinition,
    skirt: float = 0.0
) -> list[Piece]:
    """
    The part's look (its shape within its own size). skirt:
    a separator under an engine, how high an interstage
    skirt goes up round it (m).
    """

    kind = part.kind

    h = part.height
    r_top = part.radius_top
    r_bottom = part.radius_bottom

    if part.docking:
        return _port(part)

    if part.isru is not None:
        return _plant(part)

    if part.antenna_range > 0.0 and kind == "antenna":
        return _dish(part)

    if kind == "tank":
        return _tank(part)

    if kind == "engine":
        return _engine(part)

    if kind == "decoupler":
        return _separator(part, skirt)

    if kind == "parachute":
        return _parachute(part)

    if kind == "command":

        if part.crew >= 4:
            return _habitat(part)

        if part.crew > 0:
            return _capsule(part)

        return _probe(part)

    if kind == "science":
        return _instrument(part)

    if kind == "power":
        return _power(part)

    if kind == "legs":
        return _leg(part)

    if part.max_temperature >= 2_500.0:
        return _heat_shield(part)

    # Structure: a nose cone (a point on top), else an
    # adapter.
    if r_top < 0.25 * r_bottom:
        return _nose(part)

    ring = min(0.06, 0.15 * h)

    return [
        Piece(_frustum(r_bottom, r_top, ring, h - ring), "paint"),
        Piece(_band(r_bottom, 0.0, ring, 0.012), "metal", ALUMINIUM),
        Piece(_band(r_top, h - ring, h, 0.012), "metal", ALUMINIUM),
    ]


def _tank(part) -> list[Piece]:

    h, r = part.height, part.radius

    ring = min(0.08, 0.04 * h)

    # Hydrogen: sprayed foam; methane: bare stainless steel.
    skin = {"hydrolox": "foam", "methalox": "metal"}.get(part.propellant_type, "paint")

    looks = [
        Piece(_cylinder(r, ring, h - ring), skin, tile=3.0),
        Piece(_band(r * 0.995, 0.0, ring, 0.02), "metal", ALUMINIUM),
        Piece(_band(r * 0.995, h - ring, h, 0.02), "metal", ALUMINIUM),
    ]

    # Long tanks: a dark band near the top, as the old moon
    # rockets wore (a roll pattern to read from the ground).
    if h >= 4.0 * r:
        looks.append(Piece(_band(r, h * 0.78, h * 0.86, 0.004), "paint", DARK_PAINT, tile=3.0))

    return looks


def _engine(part) -> list[Piece]:

    h = part.height
    mount = part.radius_top
    exit_radius = part.radius_bottom * 0.94

    throat = 0.24 * exit_radius
    throat_height = 0.5 * h

    # The bell: flaring from the throat to the exit, its
    # wall a few cm thick (its inside dark).
    def flare(t):
        return exit_radius - (exit_radius - throat) * (t ** 0.6)

    outside = _curve(flare, 0.0, throat_height)
    inside = [(max(radius - 0.025, 0.0), height + 0.01) for radius, height in reversed(outside)]

    bell = tuple([(exit_radius - 0.025, 0.0)] + outside + inside[:-1] + [(0.0, inside[-1][1])])

    chamber = 0.42 * exit_radius

    return [
        Piece(bell, "bell", (0.85, 0.8, 0.75), segments=48, tile=None),
        Piece(_cylinder(chamber, throat_height, h * 0.68), "steel", STEEL, tile=None),
        Piece(_frustum(chamber * 1.2, mount * 0.9, h * 0.68, h * 0.9), "steel", STEEL, tile=None),
        Piece(_cylinder(mount, h * 0.9, h), "metal", ALUMINIUM, tile=None),
    ]


def _separator(part, skirt: float = 0.0) -> list[Piece]:

    h, r = part.height, part.radius

    looks = [
        Piece(_cylinder(r * 0.99, 0.0, h), "steel", STEEL, tile=None),
        Piece(_band(r * 0.99, h * 0.42, h * 0.58, 0.012), "metal", (0.86, 0.72, 0.4)),
    ]

    # An interstage: a skirt up round the engine above,
    # dropped with the separator (the engine bare after).
    if skirt > 0.0:
        looks.append(Piece(((r * 0.985, h), (r, h), (r, h + skirt), (r * 0.985, h + skirt), (r * 0.985, h)), "paint", tile=3.0))

    return looks


def _parachute(part) -> list[Piece]:

    h, r_top, r_bottom = part.height, part.radius_top, part.radius_bottom

    base = 0.18 * h

    def dome(t):
        return r_bottom * math.cos(t * 0.5 * math.pi) ** 0.55 * (1.0 - t) + r_top * 0.5 * t

    return [
        Piece(_cylinder(r_bottom, 0.0, base), "metal", ALUMINIUM, tile=None),
        Piece(tuple([(0.0, base)] + _curve(dome, base, h) + [(0.0, h)]), "fabric", tile=None),
    ]


def _capsule(part) -> list[Piece]:

    h, r_top, r_bottom = part.height, part.radius_top, part.radius_bottom

    rim = 0.06 * h
    hatch = 0.12 * h

    # The cone's skin bows out a little (a blunt body).
    def side(t):
        return r_bottom + (r_top - r_bottom) * t + 0.04 * r_bottom * math.sin(math.pi * t)

    looks = [
        Piece(tuple([(0.0, rim)] + _curve(side, rim, h - hatch) + [(0.0, h - hatch)]), "paint", tile=0.8),
        Piece(_band(r_bottom * 0.99, 0.0, rim, 0.02), "metal", ALUMINIUM),
        Piece(_cylinder(r_top * 0.9, h - hatch, h), "metal", ALUMINIUM),
    ]

    # Windows: a dark band where the crew looks out.
    looks.append(Piece(_band(side(0.55) - 0.005, rim + (h - rim - hatch) * 0.52, rim + (h - rim - hatch) * 0.6, 0.01), "glass", GLASS, tile=None))

    return looks


def _probe(part) -> list[Piece]:

    h, r = part.height, part.radius

    return [
        Piece(_cylinder(r * 0.97, 0.05 * h, 0.95 * h), "foil", GOLD, segments=8, tile=0.5),
        Piece(_band(r * 0.97, 0.0, 0.05 * h, 0.015), "metal", ALUMINIUM),
        Piece(_band(r * 0.97, 0.95 * h, h, 0.015), "metal", ALUMINIUM),
    ]


def _habitat(part) -> list[Piece]:

    h, r = part.height, part.radius

    end = 0.08 * h

    # An expandable shell: soft bulges between straps.
    def shell(t):
        return r * (0.95 + 0.05 * abs(math.sin(3.0 * math.pi * t)))

    return [
        Piece(tuple([(0.0, end)] + _curve(shell, end, h - end, 36) + [(0.0, h - end)]), "fabric", (0.92, 0.91, 0.88), segments=48, tile=None),
        Piece(_cylinder(r * 0.8, 0.0, end), "metal", ALUMINIUM),
        Piece(_cylinder(r * 0.8, h - end, h), "metal", ALUMINIUM),
        Piece(_band(r * 0.99, h * 0.62, h * 0.66, 0.01), "glass", GLASS, tile=None),
    ]


def _heat_shield(part) -> list[Piece]:

    h, r = part.height, part.radius

    # Its face bows down a little; a metal ring round it.
    return [
        Piece(((0.0, 0.0), (r * 0.6, 0.12 * h), (r, 0.55 * h), (r, h), (0.0, h)), "shield", CHAR, tile=0.6),
        Piece(_band(r * 0.995, 0.7 * h, h, 0.012), "metal", ALUMINIUM),
    ]


def _nose(part) -> list[Piece]:

    h, r_bottom, r_top = part.height, part.radius_bottom, part.radius_top

    # An ogive: full at the base, curving in to the tip.
    def ogive(t):
        return r_top + (r_bottom - r_top) * math.sqrt(max(1.0 - t * t, 0.0))

    return [
        Piece(tuple([(0.0, 0.0)] + _curve(ogive, 0.0, h, 16) + [(0.0, h)]), "paint", tile=0.8),
    ]


def _port(part) -> list[Piece]:

    h, r_top, r_bottom = part.height, part.radius_top, part.radius_bottom

    collar = 0.55 * h

    return [
        Piece(_cylinder(r_bottom, 0.0, collar), "steel", STEEL, tile=None),
        Piece(((0.0, collar), (r_top, collar), (r_top, h), (r_top * 0.72, h), (r_top * 0.72, h - 0.04), (0.0, h - 0.04)), "metal", ALUMINIUM, tile=None),
    ]


def _plant(part) -> list[Piece]:

    h, r = part.height, part.radius

    return [
        Piece(_cylinder(r, 0.0, 0.8 * h), "paint", tile=1.0),
        Piece(tuple([(0.0, 0.8 * h)] + _curve(lambda t: r * math.cos(t * 0.5 * math.pi), 0.8 * h, h) + [(0.0, h)]), "metal", ALUMINIUM, tile=None),
        Piece(_band(r, 0.25 * h, 0.3 * h, 0.03), "steel", STEEL, tile=None),
        Piece(_band(r, 0.55 * h, 0.6 * h, 0.03), "steel", STEEL, tile=None),
    ]


def _dish(part) -> list[Piece]:

    h, rim = part.height, part.radius_top

    stalk = max(part.radius_bottom * 0.6, 0.03)

    base = 0.35 * h

    # A parabola, front and back (a few mm thick).
    def face(t):
        return rim * math.sqrt(t)

    front = _curve(face, base, h, 10)
    back = [(radius, height - 0.02) for radius, height in reversed(front)]

    return [
        Piece(_cylinder(stalk, 0.0, base + 0.02), "steel", STEEL, tile=None),
        Piece(tuple(back + front[1:]), "paint", (0.93, 0.93, 0.92), segments=48, tile=None),
    ]


def _instrument(part) -> list[Piece]:

    h, r_top, r_bottom = part.height, part.radius_top, part.radius_bottom

    experiment = part.experiment

    looks = [Piece(_frustum(r_bottom, r_top, 0.0, 0.85 * h), "paint", tile=None)]

    if experiment is not None and experiment.maps:
        looks.append(Piece(_cylinder(r_top * 0.7, 0.85 * h, h), "glass", GLASS, tile=None))
    else:
        looks.append(Piece(_cylinder(r_top * 0.9, 0.85 * h, h), "metal", ALUMINIUM, tile=None))

    return looks


def _leg(part) -> list[Piece]:

    h, r = part.height, part.radius

    pad = 0.08 * h

    # A strut with a shock absorber, a hinge at the top, a
    # wide pad at the foot (drawn swung out from the hinge,
    # systems/vessel_system.py).
    return [
        Piece(_frustum(r, r * 0.5, 0.0, pad), "metal", ALUMINIUM, segments=16, tile=None),
        Piece(_cylinder(r * 0.22, pad, 0.55 * h), "steel", STEEL, segments=12, tile=None),
        Piece(_cylinder(r * 0.32, 0.55 * h, 0.92 * h), "paint", segments=12, tile=None),
        Piece(_cylinder(r * 0.4, 0.92 * h, h), "metal", ALUMINIUM, segments=12, tile=None),
    ]


def _power(part) -> list[Piece]:

    h, r = part.height, part.radius

    ring = min(0.05, 0.12 * h)

    rings = [
        Piece(_band(r * 0.99, 0.0, ring, 0.012), "metal", ALUMINIUM, tile=None),
        Piece(_band(r * 0.99, h - ring, h, 0.012), "metal", ALUMINIUM, tile=None),
    ]

    # Solar wings' housing: a drum, its wings' roots round
    # the middle (the wings drawn apart).
    if part.solar is not None:
        return [
            Piece(_cylinder(r * 0.97, ring, h - ring), "paint", tile=1.0),
            Piece(_band(r * 0.97, 0.35 * h, 0.65 * h, 0.02), "cells", (1.0, 1.0, 1.0), tile=0.6),
        ] + rings

    # A reactor: a shielded core under a cone of radiator.
    if part.generator >= 2_000.0:
        return [
            Piece(_cylinder(r * 0.45, 0.0, 0.35 * h), "steel", STEEL, tile=None),
            Piece(_frustum(r, r * 0.4, 0.35 * h, h), "paint", (0.93, 0.93, 0.92), segments=48, tile=None),
            Piece(_band(r * 0.45, 0.3 * h, 0.36 * h, 0.03), "metal", ALUMINIUM, tile=None),
        ]

    # An RTG: a finned, dark cylinder (fins: a hexagon's
    # edges, standing out).
    if part.generator > 0.0:
        return [
            Piece(_cylinder(r * 0.55, 0.0, h), "steel", STEEL, segments=24, tile=None),
            Piece(_cylinder(r, 0.1 * h, 0.9 * h), "bell", BELL, segments=6, tile=None),
            Piece(_band(r * 0.55, 0.92 * h, h, 0.02), "metal", ALUMINIUM, tile=None),
        ]

    # A battery pack: dark cells banded.
    return [
        Piece(_cylinder(r * 0.98, ring, h - ring), "paint", DARK_PAINT, tile=1.0),
        Piece(_band(r * 0.98, 0.45 * h, 0.55 * h, 0.006), "paint", (0.85, 0.55, 0.15), tile=None),
    ] + rings


# ---------------------------------------------------------
# Details off the axis
# ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Detail:
    """
    A box (size: across, up, out m) or a cylinder (size:
    radius, length m) on the part: its center `out` m from
    the axis, `up` m above the part's bottom; `count` copies
    evenly round from `angle` deg. A cylinder stands along the
    stack, or (tilt 90) points straight out.
    """

    shape: str
    size: tuple
    out: float
    up: float
    finish: str
    color: tuple | None = None
    count: int = 1
    angle: float = 0.0
    tilt: float = 0.0


DARK = (0.08, 0.08, 0.09)


def _surface(
    part,
    height: float
) -> float:
    """The part's radius at a height (a frustum)."""

    share = min(max(height / max(part.height, 1e-6), 0.0), 1.0)

    return part.radius_bottom + (part.radius_top - part.radius_bottom) * share


def details(
    part: PartDefinition
) -> list[Detail]:
    """The part's details off its axis (its shape within its own size, a little proud)."""

    kind = part.kind
    h = part.height

    found = []

    if part.docking or part.legs is not None or part.solar is not None:
        return found

    if kind == "command":

        if part.crew >= 4:

            r = part.radius

            found += [
                Detail("cylinder", (0.16, 0.05), r * 0.99, 0.64 * h, "glass", GLASS, count=4, angle=45.0, tilt=90.0),
                Detail("box", (0.04, 0.8 * h, 0.04), r * 1.02, 0.5 * h, "metal", ALUMINIUM, count=8, angle=22.5),
            ]

        elif part.crew > 0:

            window = 0.6 * h

            found += [
                Detail("box", (0.34, 0.3, 0.03), _surface(part, window) * 0.97, window, "glass", GLASS, angle=-22.0),
                Detail("box", (0.34, 0.3, 0.03), _surface(part, window) * 0.97, window, "glass", GLASS, angle=22.0),
                Detail("box", (0.34, 0.3, 0.03), _surface(part, window) * 0.97, window, "glass", GLASS, angle=180.0),
                Detail("box", (0.62, 0.72, 0.025), _surface(part, 0.45 * h) * 0.98, 0.45 * h, "metal", ALUMINIUM, angle=90.0),
            ]

        else:

            r = part.radius

            found.append(Detail("box", (0.26, 0.26, 0.14), r * 0.97 + 0.07, 0.5 * h, "foil", GOLD, count=4, angle=45.0))

        if part.rcs_thrust > 0.0:

            at = 0.78 * h if part.crew else 0.5 * h

            r = _surface(part, at)

            found += [
                Detail("box", (0.22, 0.26, 0.12), r + 0.04, at, "paint", DARK_PAINT, count=4, angle=0.0),
                Detail("cylinder", (0.035, 0.12), r + 0.14, at, "bell", BELL, count=4, angle=0.0, tilt=90.0),
            ]

        return found

    if kind == "tank":

        r = part.radius

        # A cable raceway down its side.
        found.append(Detail("box", (0.14 * r / 1.25 + 0.06, 0.9 * h, 0.07), r + 0.03, 0.5 * h, "metal", ALUMINIUM, angle=90.0))

        if h >= 4.0:
            found.append(Detail("box", (0.08, 0.9 * h, 0.05), r + 0.02, 0.5 * h, "metal", ALUMINIUM, angle=270.0))

        return found

    if kind == "engine":

        mount = part.radius_top

        chamber = 0.42 * part.radius_bottom * 0.94

        found += [
            # A turbopump beside the chamber, its exhaust duct.
            Detail("cylinder", (0.32 * chamber + 0.04, 0.3 * h), chamber * 1.9, 0.72 * h, "steel", STEEL),
            Detail("cylinder", (0.12 * chamber + 0.02, 0.25 * h), chamber * 1.9, 0.48 * h, "bell", BELL),
            # Gimbal actuators from the mount to the chamber.
            Detail("box", (0.05 + 0.03 * mount, 0.28 * h, 0.05 + 0.03 * mount), chamber * 1.35, 0.76 * h, "metal", ALUMINIUM, count=2, angle=135.0),
        ]

        return found

    if kind == "decoupler":

        r = part.radius

        found.append(Detail("box", (0.07, 0.6 * h, 0.05), r * 0.99 + 0.02, 0.5 * h, "steel", STEEL, count=12 if r > 1.0 else 8, angle=15.0))

        return found

    if kind == "antenna":

        top = h + 0.1 * part.radius_top

        found += [
            # The feed horn at the dish's focus, on three struts.
            Detail("cylinder", (0.06 * part.radius_top + 0.02, 0.12 * h), 0.0, top, "steel", STEEL),
            Detail("box", (0.02, 0.6 * h, 0.02), 0.45 * part.radius_top, 0.72 * h, "metal", ALUMINIUM, count=3),
        ]

        return found

    if kind == "power" and part.generator > 0.0 and part.generator < 2_000.0:

        r = part.radius

        # Radiator fins.
        depth = 0.45 * r

        found.append(Detail("box", (0.025, 0.8 * h, depth), 0.55 * r + 0.5 * depth, 0.5 * h, "bell", BELL, count=8, angle=22.5))

        return found

    if kind == "power" and part.battery > 0.0:

        found.append(Detail("box", (0.2, 0.08, 0.03), part.radius * 0.98 + 0.01, 0.5 * part.height, "metal", ALUMINIUM, count=4, angle=45.0))

        return found

    return found


def _place(
    data: MeshData,
    matrix: np.ndarray,
    offset: np.ndarray,
    color
) -> MeshData:
    """A mesh moved: positions by matrix and offset, normals and tangents turned, all one color."""

    vertices = data.vertices.copy()

    vertices[:, 0:3] = vertices[:, 0:3] @ matrix.T + offset

    normals = vertices[:, 3:6] @ np.linalg.inv(matrix)

    vertices[:, 3:6] = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    tangents = vertices[:, 11:14] @ matrix.T

    vertices[:, 11:14] = tangents / np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)

    vertices[:, 6:9] = color

    return MeshData(vertices.astype(np.float32), data.indices.copy())


_UNIT_CYLINDER = None


def detail_data(
    detail: Detail,
    color
) -> list[MeshData]:
    """A detail's copies as meshes in the part's frame."""

    global _UNIT_CYLINDER

    if detail.shape == "box":
        unit = MeshFactory.cube_data()
        scale = np.diag(detail.size)
    else:

        if _UNIT_CYLINDER is None:
            _UNIT_CYLINDER = MeshFactory.lathe_data(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5)), 12, None)

        unit = _UNIT_CYLINDER

        radius, length = detail.size

        scale = np.diag((radius, length, radius))

    meshes = []

    up = np.array([0.0, 1.0, 0.0])

    for k in range(max(detail.count, 1)):

        angle = math.radians(detail.angle) + 2.0 * math.pi * k / max(detail.count, 1)

        out = np.array([math.cos(angle), 0.0, math.sin(angle)])
        across = np.array([-math.sin(angle), 0.0, math.cos(angle)])

        # Local x across, y up, z out.
        turn = np.column_stack((across, up, out))

        if detail.tilt:

            tilt = math.radians(detail.tilt)

            # Leaning out about the across axis: y toward out.
            lean = np.array([
                [1.0, 0.0, 0.0],
                [0.0, math.cos(tilt), -math.sin(tilt)],
                [0.0, math.sin(tilt), math.cos(tilt)],
            ])

            turn = turn @ lean

        meshes.append(_place(unit, turn @ scale, out * detail.out + up * detail.up, color))

    return meshes


def merged_details(
    part: PartDefinition
) -> dict[str, list[MeshData]]:
    """The details by finish: finish -> meshes."""

    groups: dict[str, list[MeshData]] = {}

    for detail in details(part):
        groups.setdefault(detail.finish, []).extend(detail_data(detail, detail.color or part.color))

    return groups


def merged(
    part: PartDefinition,
    skirt: float = 0.0
) -> dict[str, tuple[list, list]]:
    """The look by finish: finish -> (pieces' profiles, their colors), for one mesh each."""

    groups: dict[str, tuple[list, list]] = {}

    for piece in pieces(part, skirt):

        profiles, colors = groups.setdefault(piece.finish, ([], []))

        profiles.append(piece)
        colors.append(piece.color or part.color)

    return groups
