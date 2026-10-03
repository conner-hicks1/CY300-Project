import math

import numpy as np

from OpenGL.GL import (
    GL_CLAMP_TO_EDGE,
    GL_FLOAT,
    GL_LINEAR,
    GL_R32F,
    GL_RED,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    glBindTexture,
    glDeleteTextures,
    glGenTextures,
    glTexImage2D,
    glTexParameteri
)


# =========================================================
# Planetary Rings
# =========================================================
#
# A disk of ice and rock particles in the planet's
# equatorial plane (assets/shaders/rings.frag.glsl draws
# it). How opaque it is at each distance from the planet
# (its normal optical depth) is a radial profile baked here
# into a 1D texture: the bands of the profile (Saturn's C,
# B and A rings, the Cassini Division, the Encke Gap, the F
# ring; Uranus's narrow dark rings) with fine ringlets
# within them. The same texture lets lit surfaces look up
# the rings' shadow.

# Samples across the rings.
PROFILE_SIZE = 2048

# Bands per ring system (RingsComponent.bands holds this
# many (inner m, outer m, optical depth) triples).
MAX_BANDS = 8


def ring_profile(
    inner: float,
    outer: float,
    bands,
    seed: int = 1,
    size: int = PROFILE_SIZE
) -> np.ndarray:
    """
    Normal optical depth at `size` radii evenly spaced from
    `inner` to `outer` (m). bands: (inner m, outer m, tau)
    triples (tau 0 = unused slot).
    """

    radius = inner + (np.arange(size) + 0.5) / size * (outer - inner)

    tau = np.zeros(size)

    span = max(outer - inner, 1.0)

    for band_inner, band_outer, band_tau in bands:

        if band_tau <= 0.0 or band_outer <= band_inner:
            continue

        # Edges soft over a texel or so (narrow gaps such as
        # the Encke Gap must stay open).
        edge = 0.75 * span / size

        inside = _smoothstep(band_inner - edge, band_inner + edge, radius) * (
            1.0 - _smoothstep(band_outer - edge, band_outer + edge, radius)
        )

        tau = np.maximum(tau, band_tau * inside)

    # Ringlets: the bands are made of thousands of narrow
    # rings of varying density (1D fractal noise in radius).
    rng = np.random.default_rng(seed)

    ringlets = np.zeros(size)
    amplitude = 0.5
    total = 0.0

    for octave in range(6):

        count = 16 * 2 ** octave

        knots = rng.random(count + 1)

        x = np.linspace(0.0, count, size)

        i = np.minimum(x.astype(int), count - 1)
        f = x - i
        f = f * f * (3.0 - 2.0 * f)

        ringlets += amplitude * (knots[i] * (1.0 - f) + knots[i + 1] * f)

        total += amplitude
        amplitude *= 0.6

    ringlets /= total

    return (tau * (0.45 + 1.1 * ringlets)).astype(np.float32)


def flat_bands(
    bands
) -> tuple[float, ...]:
    """(inner, outer, tau) triples -> RingsComponent.bands (MAX_BANDS slots)."""

    values = []

    for band in list(bands)[:MAX_BANDS]:
        values.extend(float(v) for v in band)

    values.extend([0.0] * (3 * MAX_BANDS - len(values)))

    return tuple(values)


def unflat_bands(
    values
) -> list[tuple[float, float, float]]:

    values = list(values)

    return [
        (values[i], values[i + 1], values[i + 2])
        for i in range(0, len(values) - 2, 3)
        if values[i + 2] > 0.0
    ]


class RingProfileTexture:

    # The profile as a texture (R32F, PROFILE_SIZE x 1). A
    # cleared 1x1 stands in without rings, so the sampler
    # always has a 2D texture bound.

    def __init__(self):

        self.texture_id = int(glGenTextures(1))

        self._key = None

        self._upload(np.zeros(1, dtype=np.float32))

    def update(
        self,
        key,
        make_profile
    ) -> bool:

        if key == self._key:
            return False

        self._key = key

        self._upload(make_profile())

        return True

    def clear(self):

        if self._key is not None:

            self._key = None

            self._upload(np.zeros(1, dtype=np.float32))

    def _upload(
        self,
        profile: np.ndarray
    ):

        data = np.ascontiguousarray(profile, dtype=np.float32)

        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, len(data), 1, 0, GL_RED, GL_FLOAT, data)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        glBindTexture(GL_TEXTURE_2D, 0)

    def delete(self):

        if self.texture_id:

            glDeleteTextures(1, [self.texture_id])

            self.texture_id = 0


def disc_mesh_data(
    segments: int = 1024,
    rings: int = 6
):
    """
    A unit disc in the XZ plane (rings of quads around a
    small center fan), finely divided around so its outer
    edge stays round up close. The fragment shader cuts the
    annulus out of it.
    """

    from graphics.mesh_data import MeshData

    angles = np.arange(segments) * (2.0 * math.pi / segments)

    radii = np.linspace(0.0, 1.0, rings + 1)[1:]

    positions = [(0.0, 0.0, 0.0)]

    for r in radii:
        for a in angles:
            positions.append((r * math.cos(a), 0.0, r * math.sin(a)))

    indices = []

    # Center fan.
    for s in range(segments):
        indices += [0, 1 + s, 1 + (s + 1) % segments]

    for ring in range(rings - 1):

        a0 = 1 + ring * segments
        b0 = 1 + (ring + 1) * segments

        for s in range(segments):

            s1 = (s + 1) % segments

            indices += [a0 + s, b0 + s, b0 + s1, a0 + s, b0 + s1, a0 + s1]

    positions = np.asarray(positions, dtype=np.float32)

    normals = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (len(positions), 1))

    return MeshData.from_attributes(
        positions=positions,
        indices=np.asarray(indices, dtype=np.uint32),
        normals=normals,
        uvs=positions[:, [0, 2]]
    )


def _smoothstep(
    edge0: float,
    edge1: float,
    x
) -> np.ndarray:

    t = np.clip((np.asarray(x) - edge0) / (edge1 - edge0), 0.0, 1.0)

    return t * t * (3.0 - 2.0 * t)
