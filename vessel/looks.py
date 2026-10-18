import math

from dataclasses import dataclass

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

FINISHES = {
    #          metallic  roughness  texture
    "paint":  (0.05,     0.5,       "hull"),
    "metal":  (0.9,      0.32,      "hull"),
    "steel":  (0.85,     0.45,      None),
    "bell":   (0.8,      0.55,      None),
    "foil":   (1.0,      0.28,      "foil"),
    "shield": (0.0,      0.92,      "shield"),
    "glass":  (0.1,      0.06,      None),
    "fabric": (0.0,      0.85,      None),
    "cells":  (0.1,      0.3,       "cells"),
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

    looks = [
        Piece(_cylinder(r, ring, h - ring), "paint", tile=3.0),
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
        Piece(bell, "bell", BELL, segments=48, tile=None),
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
