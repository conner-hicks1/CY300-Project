import numpy as np


# =========================================================
# Gradient Noise
# =========================================================
#
# Vectorized 3D Perlin noise ("improved noise", Perlin
# 2002) and fractal sums built from it. Everything works
# on (n, 3) float64 arrays of points so one call evaluates
# a whole chunk's grid.
#
# Terrain samples noise at points on the unit sphere
# (times a frequency), so there are no seams between cube
# faces: neighbouring chunks query the same 3D field.

# The 12 cube-edge gradient directions.
_GRADIENTS = np.array(
    [
        (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
        (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
        (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
    ],
    dtype=np.float64
)

# Added to each octave's coordinates so octaves do not all
# share a lattice point at the origin (which would line up
# their features).
_OCTAVE_OFFSET = np.array([19.1, 7.3, 13.7])


class Perlin:

    __slots__ = ("_perm", "_gradient_of")

    def __init__(
        self,
        seed: int
    ):

        rng = np.random.default_rng(seed)

        permutation = rng.permutation(256)

        # Doubled so lookups of (perm[x] + y + 1) never wrap.
        self._perm = np.concatenate(
            (permutation, permutation)
        ).astype(np.int64)

        # Hash -> gradient, folded into one table lookup.
        self._gradient_of = _GRADIENTS[self._perm % 12]

    def __call__(
        self,
        points: np.ndarray
    ) -> np.ndarray:
        """Noise in roughly [-1, 1] at (n, 3) points."""

        points = np.asarray(points, dtype=np.float64)

        floor = np.floor(points)

        f = points - floor

        cell = floor.astype(np.int64) & 255

        x, y, z = cell[:, 0], cell[:, 1], cell[:, 2]

        perm = self._perm

        a = perm[x] + y
        b = perm[x + 1] + y

        aa = perm[a] + z
        ab = perm[a + 1] + z
        ba = perm[b] + z
        bb = perm[b + 1] + z

        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]

        gx, gy, gz = fx - 1.0, fy - 1.0, fz - 1.0

        grad = self._gradient_of

        def corner(h, px, py, pz):

            g = grad[h]

            return g[:, 0] * px + g[:, 1] * py + g[:, 2] * pz

        # Quintic fade: C2-continuous, so normals derived
        # from the height field have no creases.
        u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)

        ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]

        x00 = corner(aa, fx, fy, fz)
        x00 += ux * (corner(ba, gx, fy, fz) - x00)

        x10 = corner(ab, fx, gy, fz)
        x10 += ux * (corner(bb, gx, gy, fz) - x10)

        x01 = corner(aa + 1, fx, fy, gz)
        x01 += ux * (corner(ba + 1, gx, fy, gz) - x01)

        x11 = corner(ab + 1, fx, gy, gz)
        x11 += ux * (corner(bb + 1, gx, gy, gz) - x11)

        y0 = x00 + uy * (x10 - x00)
        y1 = x01 + uy * (x11 - x01)

        return y0 + uz * (y1 - y0)


# =========================================================
# Fractal Sums
# =========================================================

def fbm(
    noise: Perlin,
    points: np.ndarray,
    octaves: int,
    lacunarity: float = 2.0,
    gain: float = 0.5
) -> np.ndarray:
    """
    Fractional Brownian motion: octaves of noise at rising
    frequency and falling amplitude. Normalized so the
    result stays in about [-1, 1] for any octave count.
    """

    total = np.zeros(len(points))

    amplitude = 1.0
    frequency = 1.0
    norm = 0.0

    for octave in range(max(1, octaves)):

        total += amplitude * noise(points * frequency + octave * _OCTAVE_OFFSET)

        norm += amplitude

        amplitude *= gain
        frequency *= lacunarity

    return total / norm


def ridged(
    noise: Perlin,
    points: np.ndarray,
    octaves: int,
    lacunarity: float = 2.0,
    gain: float = 0.5
) -> np.ndarray:
    """
    Ridged multifractal in [0, 1]: sharp crests where the
    noise crosses zero (mountain ranges). Each octave is
    weighted by the previous one, so detail gathers on the
    ridges and valleys stay smooth.
    """

    total = np.zeros(len(points))

    weight = np.ones(len(points))

    amplitude = 1.0
    frequency = 1.0
    norm = 0.0

    for octave in range(max(1, octaves)):

        signal = 1.0 - np.abs(noise(points * frequency + octave * _OCTAVE_OFFSET))

        signal *= signal

        signal *= weight

        weight = np.clip(signal * 2.0, 0.0, 1.0)

        total += amplitude * signal

        norm += amplitude

        amplitude *= gain
        frequency *= lacunarity

    return total / norm


def octaves_for_spacing(
    base_wavelength: float,
    spacing: float,
    maximum: int
) -> int:
    """
    Octaves worth evaluating for a grid with `spacing`
    between samples: stop once an octave's wavelength drops
    below two samples (it would only alias).
    """

    if spacing <= 0.0:
        return maximum

    count = 1

    wavelength = base_wavelength

    while count < maximum and wavelength * 0.5 >= 2.0 * spacing:

        wavelength *= 0.5
        count += 1

    return count
