import math

from dataclasses import dataclass

import numpy as np

from OpenGL.GL import (
    GL_CLAMP_TO_EDGE,
    GL_LINEAR,
    GL_R8,
    GL_RED,
    GL_REPEAT,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNPACK_ALIGNMENT,
    GL_UNSIGNED_BYTE,
    glBindTexture,
    glDeleteTextures,
    glGenTextures,
    glPixelStorei,
    glTexImage2D,
    glTexParameteri
)


# =========================================================
# Weather Clouds
# =========================================================
#
# A cloud layer on a planet with air (assets/shaders/
# include/clouds.glsl draws it). Where it is cloudy comes
# from a cover map built here: from the climate (rain falls
# where air rises and its vapor condenses: the equatorial
# belt and the storm tracks are cloudy, the subtropical
# deserts clear), scaled so the planet averages its
# observed cloud cover. The shader adds the clouds' shapes.


@dataclass(frozen=True, slots=True)
class CloudParameters:

    # Mean fraction of the sky covered (Earth ~0.67).
    coverage: float

    # Height of the layer (km).
    altitude: float

    # Optical depth at the thickest (Earth's clouds ~5-30).
    optical_depth: float

    # Largest cloud features (km).
    scale: float

    color: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Drift around the planet (m / s; the winds aloft).
    speed: float = 0.0


# Cover map size (equirectangular).
MAP_SIZE = (256, 128)


def cloud_cover_map(
    coverage: float,
    climate=None,
    size: tuple[int, int] = MAP_SIZE
) -> np.ndarray:
    """
    (height, width) cloud cover fraction, 0..1, rows from
    the south pole up, columns from longitude -180 degrees
    (as include/clouds.glsl cloudCover() samples it).

    climate: planet.climate.ClimateField (rainfall), or None
    for a generic pattern (cloudier at the equator and in
    mid-latitudes).
    """

    width, height = size

    latitude = ((np.arange(height) + 0.5) / height - 0.5) * math.pi
    longitude = ((np.arange(width) + 0.5) / width - 0.5) * 2.0 * math.pi

    lon, lat = np.meshgrid(longitude, latitude)

    directions = np.stack(
        (np.cos(lat) * np.cos(lon), np.sin(lat), np.cos(lat) * np.sin(lon)),
        axis=-1
    ).reshape(-1, 3)

    if climate is not None:

        rain = np.maximum(climate.grid.sample(climate.precipitation, directions), 0.0)

        relative = rain / max(float(np.mean(rain)), 1e-6)

        shape = 0.15 + 0.85 * _smoothstep(0.1, 2.0, relative)

    else:

        shape = 0.55 + 0.25 * np.cos(2.0 * lat.reshape(-1)) + 0.2 * np.cos(4.0 * lat.reshape(-1))

    # Scale to the mean coverage (area-weighted).
    weight = np.cos(lat.reshape(-1))

    target = float(np.clip(coverage, 0.0, 1.0))

    low, high = 0.0, 20.0

    for _ in range(40):

        middle = 0.5 * (low + high)

        mean = float(np.sum(np.clip(shape * middle, 0.0, 1.0) * weight) / np.sum(weight))

        if mean < target:
            low = middle
        else:
            high = middle

    cover = np.clip(shape * 0.5 * (low + high), 0.0, 1.0)

    return cover.reshape(height, width).astype(np.float32)


class CloudMap:

    # The cover map as a texture (R8, equirectangular). A
    # cleared 1x1 map stands in when there are no clouds, so
    # the sampler always has a 2D texture bound.

    def __init__(self):

        self.texture_id = int(glGenTextures(1))

        self._key = None

        self._upload(np.zeros((1, 1), dtype=np.float32))

    def update(
        self,
        key,
        make_map
    ) -> bool:
        """Rebuild when `key` changed (make_map() -> the cover array)."""

        if key == self._key:
            return False

        self._key = key

        self._upload(make_map())

        return True

    def clear(self):

        if self._key is not None:

            self._key = None

            self._upload(np.zeros((1, 1), dtype=np.float32))

    def _upload(
        self,
        cover: np.ndarray
    ):

        data = np.ascontiguousarray(np.clip(cover * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8))

        height, width = data.shape

        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, data)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # Longitude wraps; latitude stops at the poles.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        glBindTexture(GL_TEXTURE_2D, 0)

    def delete(self):

        if self.texture_id:

            glDeleteTextures(1, [self.texture_id])

            self.texture_id = 0


def _smoothstep(
    edge0: float,
    edge1: float,
    x
) -> np.ndarray:

    t = np.clip((np.asarray(x) - edge0) / (edge1 - edge0), 0.0, 1.0)

    return t * t * (3.0 - 2.0 * t)
