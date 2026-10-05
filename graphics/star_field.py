import ctypes
import math

from dataclasses import dataclass

import numpy as np

from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_REPEAT,
    GL_TEXTURE_2D,
    GL_TEXTURE_WRAP_S,
    GL_DYNAMIC_DRAW,
    GL_FLOAT,
    GL_FALSE,
    GL_POINTS,
    GL_PROGRAM_POINT_SIZE,
    GL_STATIC_DRAW,
    glBindBuffer,
    glBindVertexArray,
    glBufferData,
    glBufferSubData,
    glDeleteBuffers,
    glDeleteVertexArrays,
    glDisable,
    glDrawArrays,
    glEnable,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glBindTexture,
    glTexParameteri,
    glVertexAttribPointer
)

from graphics.framebuffer import ColorFormat, DepthMode, Framebuffer, FramebufferSpec
from graphics.render_command import RenderCommand

from planet.stars import StarCatalog, galactic_basis, magnitude_to_illuminance


# =========================================================
# Star Field
# =========================================================
#
# The night sky behind everything (planet/stars.py for the
# catalog, assets/shaders/stars.*.glsl and milky_way.frag.glsl
# for the drawing): the bright stars as points at their real
# directions and brightness, the Milky Way's glow, and the
# planets and moons too far away to show a disc, as points of
# light that move against the stars.
#
# All at physical brightness: invisible beside anything
# sunlit, as in a photograph of the Moon or of Earth from
# orbit, and plain to the dark-adapted eye in the dark.

# Brightest part of the Milky Way (the Sagittarius star
# clouds, ~20 mag / arcsec^2) as radiance in engine units:
# the illuminance of magnitude 20 over a square arcsecond.
MILKY_WAY_BRIGHTNESS = float(magnitude_to_illuminance(20.0)) / (math.radians(1.0 / 3600.0) ** 2)

# Twinkling amplitude at the zenith through Earth's air at
# sea level (scales with the air's density over the camera).
TWINKLE = 0.25

_FLOATS = 8                         # per point: direction, illuminance, color, angular radius
_MAX_BODY_POINTS = 64

# The Milky Way's baked map (galactic longitude x latitude):
# ~0.18 deg per texel, finer than its dust lanes.
MILKY_WAY_MAP_SIZE = (2048, 1024)


@dataclass(frozen=True, slots=True)
class BodyPoint:
    """A planet or moon seen as a point of light."""

    direction: tuple[float, float, float]
    illuminance: float              # engine units
    color: tuple[float, float, float]   # luminance 1
    angular_radius: float           # rad


def reflected_illuminance(
    sunlight: float,
    geometric_albedo: float,
    radius: float,
    distance: float,
    phase_angle: float
) -> float:
    """
    Light from a sunlit sphere reaching an observer
    (illuminance, in the units of `sunlight`, the sunlight at
    the sphere): geometric albedo times the disc's solid angle
    over pi, times the Lambert sphere's phase law
    (sin a + (pi - a) cos a) / pi: full at opposition, half
    lit at quadrature (a third of full), nothing new.
    """

    if distance <= radius:
        return 0.0

    a = min(max(phase_angle, 0.0), math.pi)

    phase = (math.sin(a) + (math.pi - a) * math.cos(a)) / math.pi

    return sunlight * geometric_albedo * (radius / distance) ** 2 * phase


def pack_points(
    directions,
    illuminance,
    colors,
    angular_radii=None
) -> np.ndarray:

    directions = np.asarray(directions, dtype=np.float32).reshape(-1, 3)

    n = len(directions)

    data = np.zeros((n, _FLOATS), dtype=np.float32)

    data[:, 0:3] = directions
    data[:, 3] = np.asarray(illuminance, dtype=np.float32).reshape(n)
    data[:, 4:7] = np.asarray(colors, dtype=np.float32).reshape(n, 3)

    if angular_radii is not None:
        data[:, 7] = np.asarray(angular_radii, dtype=np.float32).reshape(n)

    return data


class _PointBuffer:

    def __init__(
        self,
        data: np.ndarray | None,
        capacity: int
    ):

        self.vao = int(glGenVertexArrays(1))
        self.vbo = int(glGenBuffers(1))

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        if data is not None:
            glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        else:
            glBufferData(GL_ARRAY_BUFFER, capacity * _FLOATS * 4, None, GL_DYNAMIC_DRAW)

        stride = _FLOATS * 4

        for location, offset in ((0, 0), (1, 16)):

            glEnableVertexAttribArray(location)
            glVertexAttribPointer(location, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))

        glBindVertexArray(0)

        self.count = 0 if data is None else len(data)
        self.capacity = capacity

    def update(
        self,
        data: np.ndarray
    ):

        data = data[:self.capacity]

        self.count = len(data)

        if self.count:

            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)

    def draw(self):

        if self.count:

            glBindVertexArray(self.vao)
            glDrawArrays(GL_POINTS, 0, self.count)
            glBindVertexArray(0)

    def delete(self):

        glDeleteBuffers(1, [self.vbo])
        glDeleteVertexArrays(1, [self.vao])


class StarField:

    def __init__(
        self,
        catalog: StarCatalog
    ):

        self.catalog = catalog

        self._stars = _PointBuffer(
            pack_points(catalog.directions, catalog.illuminance(), catalog.colors),
            len(catalog)
        )

        self._bodies = _PointBuffer(None, _MAX_BODY_POINTS)

        self.galactic = galactic_basis().astype(np.float32)

        self._milky_way_map: Framebuffer | None = None

    def set_bodies(
        self,
        points: list[BodyPoint]
    ):

        if not points:

            self._bodies.count = 0

            return

        self._bodies.update(
            pack_points(
                [p.direction for p in points],
                [p.illuminance for p in points],
                [p.color for p in points],
                [p.angular_radius for p in points]
            )
        )

    @property
    def baked(
        self
    ) -> bool:

        return self._milky_way_map is not None

    def bake_milky_way(
        self,
        shader,
        draw_fullscreen
    ):
        """
        Render the Milky Way's model into its map, once (it
        binds its own framebuffer: rebind the caller's after).
        """

        width, height = MILKY_WAY_MAP_SIZE

        target = Framebuffer(
            FramebufferSpec(
                width=width,
                height=height,
                color_format=ColorFormat.RGBA16F,
                depth_mode=DepthMode.NONE
            )
        )

        target.bind()

        shader.bind()

        shader.set_bool("uBake", True)

        draw_fullscreen(shader)

        # Longitude wraps around.
        glBindTexture(GL_TEXTURE_2D, target.color_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glBindTexture(GL_TEXTURE_2D, 0)

        self._milky_way_map = target

    def draw_milky_way(
        self,
        shader,
        draw_fullscreen
    ):

        shader.bind()

        shader.set_bool("uBake", False)

        for name, axis in zip(("uGalacticX", "uGalacticY", "uGalacticZ"), self.galactic):
            shader.set_vec3(name, tuple(float(v) for v in axis))

        shader.set_float("uBrightness", MILKY_WAY_BRIGHTNESS)

        RenderCommand.bind_texture(self._milky_way_map.color_texture_id, 0)

        shader.set_int("uMilkyWayMap", 0)

        draw_fullscreen(shader)

    def draw_points(
        self,
        shader,
        viewport: tuple[int, int],
        twinkle: float,
        camera_up,
        time_seconds: float,
        planet_sphere=(0.0, 0.0, 0.0, 0.0)
    ):

        shader.bind()

        shader.set_vec2("uViewportSize", (float(viewport[0]), float(viewport[1])))
        shader.set_float("uTwinkle", float(twinkle))
        shader.set_vec3("uCameraUp", tuple(float(v) for v in camera_up))
        shader.set_float("uTime", float(time_seconds % 10_000.0))
        shader.set_vec4("uPlanetSphere", tuple(float(v) for v in planet_sphere))

        glEnable(GL_PROGRAM_POINT_SIZE)

        self._stars.draw()
        self._bodies.draw()

        glDisable(GL_PROGRAM_POINT_SIZE)

    def delete(self):

        self._stars.delete()
        self._bodies.delete()

        if self._milky_way_map is not None:

            self._milky_way_map.delete()

            self._milky_way_map = None
