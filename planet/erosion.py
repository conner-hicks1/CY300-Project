import math

import numpy as np


# =========================================================
# Carved Slopes (an erosion filter)
# =========================================================
#
# Running water cuts gullies straight down a slope; they
# join into valleys, the valleys into bigger ones, and the
# ridges between them are left sharp. A simulation would
# need the whole landscape on a grid; the terrain here is a
# function sampled chunk by chunk at any resolution, so the
# effect is built per point instead (after the "erosion
# filter" idea for procedural terrain, Clay John 2021 and
# Rune Johansen 2023):
#
#   * the slope's direction comes from the relief at a
#     coarser scale (a cheap gradient of the larger forms)
#   * each octave lays stripes down that slope: a cosine
#     across it, its phase set by a few jittered cells
#     nearby, so the stripes break into gullies of finite
#     length that start and end at random
#   * the next, finer octave follows the slope as the
#     coarser gullies have bent it, so small gullies run
#     into the bigger ones (branching, as in real drainage)
#   * only cut down (valleys), more on steep ground, none on
#     flats, nothing finer than the samples can show
#
# Wavelengths from ~2.4 km (valleys between spurs) down to
# ~130 m (gullies); depth ~15% of the wavelength on steep
# slopes. Over a measured map, only what it is too coarse to
# show (it has the larger valleys already).

WAVELENGTHS = (2_400.0, 900.0, 340.0, 130.0)     # m

DEPTH = 0.15                # of the wavelength

# Gullies run this many times longer down the slope than
# they are wide.
ELONGATION = 2.5

# Slopes (rise / run) where the cutting starts and is full.
SLOPE_RANGE = (0.03, 0.25)


def _hash(
    cells: np.ndarray,
    salt: int
) -> np.ndarray:
    """(n, 3) int64 cells -> (n, 3) floats in [0, 1)."""

    x = cells[:, 0] * 73_856_093
    y = cells[:, 1] * 19_349_663
    z = cells[:, 2] * 83_492_791

    h = (x ^ y ^ z ^ (salt * 2_654_435_761)) & 0xFFFFFFFF

    out = np.empty((len(cells), 3))

    for axis in range(3):

        h = (h * 1_103_515_245 + 12_345) & 0x7FFFFFFF

        out[:, axis] = (h & 0xFFFF) / 65_536.0

    return out


def _smoothstep(
    edge0: float,
    edge1: float,
    x
):

    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)

    return t * t * (3.0 - 2.0 * t)


def tangent_frame(
    directions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Two unit tangents at each unit direction."""

    reference = np.where(
        np.abs(directions[:, 1:2]) < 0.9,
        np.array([[0.0, 1.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0]])
    )

    first = np.cross(directions, reference)
    first /= np.linalg.norm(first, axis=1, keepdims=True)

    second = np.cross(directions, first)

    return first, second


def relief_gradient(
    relief,
    directions: np.ndarray,
    radius: float,
    scale: float
) -> np.ndarray:
    """
    (n, 3) gradient (rise / run, tangent) of `relief`
    (directions, spacing) -> m, at features of `scale` m and
    up: finite differences half a scale apart.
    """

    first, second = tangent_frame(directions)

    step = 0.5 * scale / radius

    points = np.vstack((
        directions,
        directions + step * first,
        directions + step * second
    ))

    points /= np.linalg.norm(points, axis=1, keepdims=True)

    heights = relief(points, scale).reshape(3, -1)

    run = step * radius

    return (
        ((heights[1] - heights[0]) / run)[:, None] * first
        + ((heights[2] - heights[0]) / run)[:, None] * second
    )


def gullies(
    directions: np.ndarray,
    radius: float,
    gradient: np.ndarray,
    spacing: float,
    strength: float,
    seed: int,
    largest: float = float("inf")
) -> np.ndarray:
    """
    Height change (m, <= 0) at unit `directions` on a body of
    `radius`, given the larger forms' gradient there; no
    octave finer than ~4 samples (`spacing`), none longer
    than `largest` (m).
    """

    n = len(directions)

    cut = np.zeros(n)

    if strength <= 0.0 or n == 0:
        return cut

    points = directions * radius

    gradient = np.array(gradient, dtype=np.float64)

    for octave, wavelength in enumerate(WAVELENGTHS):

        if spacing > 0.0 and wavelength < 4.0 * spacing:
            break

        if wavelength > largest:
            continue

        # Faded in as the samples get fine enough (from 4 down
        # to 8 per wavelength): a level of detail that starts
        # showing an octave shows it gently, not with a jump.
        resolve = 1.0 if spacing <= 0.0 else 1.0 - float(_smoothstep(wavelength / 8.0, wavelength / 4.0, spacing))

        steepness = np.linalg.norm(gradient, axis=1)

        amount = _smoothstep(*SLOPE_RANGE, steepness) * strength * resolve

        active = amount > 1e-3

        if not np.any(active):
            continue

        # Across the slope (the gullies run along it).
        across = np.cross(directions, gradient)

        norm = np.linalg.norm(across, axis=1, keepdims=True)

        across = np.divide(across, norm, out=np.zeros_like(across), where=norm > 1e-12)

        downhill = np.cross(across, directions)

        q = points / wavelength

        base = np.floor(q - 0.5).astype(np.int64)

        wave = np.zeros(n)
        slope_wave = np.zeros(n)
        total = np.zeros(n)

        for corner in range(8):

            offset = np.array([(corner >> 0) & 1, (corner >> 1) & 1, (corner >> 2) & 1])

            cells = base + offset

            jitter = _hash(cells, seed * 31 + octave)

            center = cells + 0.5 + 0.7 * (jitter - 0.5)

            delta = q - center

            # Cells stretched down the slope: long gullies.
            along = np.einsum("ij,ij->i", delta, downhill)

            distance2 = np.einsum("ij,ij->i", delta, delta) - (1.0 - 1.0 / ELONGATION ** 2) * along * along

            weight = np.maximum(1.0 - distance2 / 1.6, 0.0) ** 2

            phase = 2.0 * math.pi * np.einsum("ij,ij->i", delta, across)

            wave += weight * np.cos(phase)
            slope_wave += weight * np.sin(phase)

            total += weight

        total = np.maximum(total, 1e-9)

        wave /= total
        slope_wave /= total

        depth = DEPTH * wavelength * amount

        # Valleys only: 0 on the crests, -depth in the
        # gullies.
        cut += depth * (wave - 1.0) * 0.5

        # The slope as this octave left it (its derivative
        # across the slope, the weights taken as constant):
        # finer gullies follow it into the coarser ones.
        gradient += (-0.5 * depth * slope_wave * (2.0 * math.pi / wavelength))[:, None] * across

    return cut
