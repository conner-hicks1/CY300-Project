import math

from dataclasses import dataclass

import numpy as np

from OpenGL.GL import GL_TEXTURE_2D

from core.logger import Logger

from graphics.framebuffer import (
    ColorFormat,
    DepthMode,
    Framebuffer,
    FramebufferSpec
)
from graphics.render_command import RenderCommand
from graphics.uniform_blocks import ATMOSPHERE_BLOCK


# =========================================================
# Atmospheric Scattering
# =========================================================
#
# After Hillaire, "A Scalable and Production Ready Sky and
# Atmosphere Rendering Technique" (EGSR 2020):
#
#   transmittance LUT   256 x 64: how much sunlight
#                       survives from altitude r toward a
#                       sun at zenith cosine mu (0 if the
#                       planet is in the way)
#   multi-scattering    32 x 32: light scattered more than
#                       once, folded into one isotropic
#                       term per (altitude, sun angle)
#   diffuse             32 x 32: daylight that filters down
#                       through an optically thick
#                       atmosphere (Venus's clouds, Titan's
#                       haze), where the multiple-scattering
#                       approximation breaks down
#
# Both depend only on the atmosphere's parameters, so they
# are rebuilt only when those change. The sky and aerial
# perspective are then a short raymarch per pixel
# (assets/shaders/atmosphere.frag.glsl), and lit surfaces
# look up the sunlight reaching them in the transmittance
# LUT (sunsets, night side).
#
# Shaders work in kilometers: float32 then resolves
# altitude to well under a meter at Earth's radius.


# Apparent radius of the sun (degrees), for the disc.
SUN_ANGULAR_RADIUS = 0.2678

# Disc brightness relative to the sun's illuminance; far
# brighter than the sky, so it blooms.
SUN_DISC_BRIGHTNESS = 40.0


@dataclass(frozen=True, slots=True)
class AtmosphereParameters:

    # Everything in kilometers and 1/km.

    ground_radius: float
    top_radius: float

    rayleigh_scattering: tuple[float, float, float]
    rayleigh_scale_height: float

    mie_scattering: tuple[float, float, float]
    mie_absorption: tuple[float, float, float]
    mie_scale_height: float
    mie_anisotropy: float

    ozone_absorption: tuple[float, float, float]
    ozone_altitude: float
    ozone_half_width: float

    ground_albedo: float

    @classmethod
    def vacuum(
        cls,
        planet_radius: float
    ) -> "AtmosphereParameters":
        """
        No air (Mercury, the Moon): nothing scatters, so the
        sky is black and the sun a sharp disc.
        """

        ground = planet_radius / 1000.0

        return cls(
            ground_radius=ground,
            top_radius=ground + 1.0,
            rayleigh_scattering=(0.0, 0.0, 0.0),
            rayleigh_scale_height=1.0,
            mie_scattering=(0.0, 0.0, 0.0),
            mie_absorption=(0.0, 0.0, 0.0),
            mie_scale_height=1.0,
            mie_anisotropy=0.0,
            ozone_absorption=(0.0, 0.0, 0.0),
            ozone_altitude=25.0,
            ozone_half_width=15.0,
            ground_albedo=0.1
        )

    @classmethod
    def from_components(
        cls,
        planet_radius: float,
        component
    ) -> "AtmosphereParameters":
        """
        planet_radius: sea level (m).
        component: AtmosphereComponent (meters, 1/Mm).
        """

        per_mm_to_per_km = 1e-3

        return cls(
            ground_radius=planet_radius / 1000.0,
            top_radius=(planet_radius + max(component.height, 1.0)) / 1000.0,
            rayleigh_scattering=tuple(
                max(float(v), 0.0) * per_mm_to_per_km
                for v in component.rayleigh_scattering
            ),
            rayleigh_scale_height=max(component.rayleigh_scale_height, 1.0) / 1000.0,
            mie_scattering=_tinted(component.mie_scattering, component.mie_scattering_tint),
            mie_absorption=_tinted(component.mie_absorption, component.mie_absorption_tint),
            mie_scale_height=max(component.mie_scale_height, 1.0) / 1000.0,
            mie_anisotropy=float(np.clip(component.mie_anisotropy, -0.99, 0.99)),
            ozone_absorption=tuple(
                max(float(v), 0.0) * per_mm_to_per_km
                for v in component.ozone_absorption
            ),
            ozone_altitude=component.ozone_altitude / 1000.0,
            ozone_half_width=max(component.ozone_thickness, 1.0) / 2000.0,
            ground_albedo=float(np.clip(component.ground_albedo, 0.0, 1.0))
        )

    # -----------------------------------------------------
    # CPU reference (tests, and the sun's color for the
    # editor). Mirrors include/atmosphere.glsl.
    # -----------------------------------------------------

    @property
    def vertical_scattering_depth(
        self
    ) -> float:
        """
        Scattering optical depth (green) straight up from the
        ground: Earth ~0.1, Mars ~0.3, Titan and Venus many.
        """

        return (
            self.rayleigh_scattering[1] * self.rayleigh_scale_height
            + self.mie_scattering[1] * self.mie_scale_height
        )

    @property
    def thick_weight(
        self
    ) -> float:
        """
        0..1: how much of the sky's multiple scattering comes
        from the thick-atmosphere diffuse model instead of
        Hillaire's (exact for Earth-like air, too dark where
        sunlight scatters many times before reaching the
        ground).
        """

        return _smoothstep(1.5, 6.0, self.vertical_scattering_depth)

    def extinction(
        self,
        altitude_km
    ) -> np.ndarray:
        """(n, 3) extinction coefficients (1/km) at altitudes."""

        altitude = np.maximum(np.asarray(altitude_km, dtype=np.float64), 0.0)[:, None]

        rayleigh = np.exp(-altitude / self.rayleigh_scale_height) * self.rayleigh_scattering

        mie = np.exp(-altitude / self.mie_scale_height) * (
            np.asarray(self.mie_scattering) + np.asarray(self.mie_absorption)
        )

        ozone = (
            np.maximum(0.0, 1.0 - np.abs(altitude - self.ozone_altitude) / self.ozone_half_width)
            * self.ozone_absorption
        )

        return rayleigh + mie + ozone

    def transmittance_to_sun(
        self,
        radius_km: float,
        sun_cos_zenith: float,
        steps: int = 200
    ) -> np.ndarray:
        """Fraction of sunlight (rgb) reaching this altitude."""

        r = max(radius_km, self.ground_radius + 1e-3)
        mu = float(np.clip(sun_cos_zenith, -1.0, 1.0))

        origin = np.array([0.0, r])
        direction = np.array([math.sqrt(max(0.0, 1.0 - mu * mu)), mu])

        if _ray_sphere(origin, direction, self.ground_radius) > 0.0:
            return np.zeros(3)

        length = _ray_sphere(origin, direction, self.top_radius)

        t = (np.arange(steps) + 0.5) / steps * length

        points = origin + t[:, None] * direction

        altitude = np.linalg.norm(points, axis=1) - self.ground_radius

        optical_depth = self.extinction(altitude).sum(axis=0) * (length / steps)

        return np.exp(-optical_depth)


def _tinted(
    value: float,
    tint
) -> tuple[float, float, float]:
    """Coefficient (1/Mm) times a per-channel tint, in 1/km."""

    return tuple(
        max(float(value), 0.0) * max(float(t), 0.0) * 1e-3
        for t in tint
    )


def _smoothstep(
    edge0: float,
    edge1: float,
    x: float
) -> float:

    t = min(max((x - edge0) / (edge1 - edge0), 0.0), 1.0)

    return t * t * (3.0 - 2.0 * t)


def _ray_sphere(
    origin: np.ndarray,
    direction: np.ndarray,
    radius: float
) -> float:
    """Distance to the first hit ahead (-1 if none)."""

    b = float(origin @ direction)
    c = float(origin @ origin) - radius * radius

    if c > 0.0 and b > 0.0:
        return -1.0

    d = b * b - c

    if d < 0.0:
        return -1.0

    s = math.sqrt(d)

    return -b - s if c > 0.0 else -b + s


# =========================================================
# Uniform Block
# =========================================================
#
#     vec4 uPlanetCenter;            xyz center relative to the camera (km),
#                                    w 1 = atmosphere present
#     vec4 uAtmosphereRadii;         x ground radius, y top radius (km),
#                                    z Mie anisotropy g, w ground albedo
#     vec4 uRayleighScattering;      rgb 1/km, w scale height (km)
#     vec4 uMieScattering;           rgb 1/km, w scale height (km)
#     vec4 uMieAbsorption;           rgb 1/km, w sun angular radius (rad)
#     vec4 uOzoneAbsorption;         rgb 1/km, w center altitude (km)
#     vec4 uOzoneParams;             x half width (km), y raymarch steps,
#                                    z 1 = haze over geometry (aerial perspective),
#                                    w thick-atmosphere weight (diffuse LUT)
#     vec4 uAtmosphereSunDirection;  xyz toward the sun, w 1 = sun present
#     vec4 uSunIlluminance;          rgb sun color * intensity, w disc brightness

def pack_atmosphere_block(
    parameters: AtmosphereParameters | None,
    planet_center_relative=(0.0, 0.0, 0.0),
    sun_direction=None,
    sun_illuminance=(0.0, 0.0, 0.0),
    steps: int = 24,
    aerial_perspective: bool = True,
    sun_angular_radius: float = SUN_ANGULAR_RADIUS
) -> bytes:
    """
    parameters None packs a disabled atmosphere.
    planet_center_relative: planet center minus camera (m).
    sun_direction: unit vector toward the sun (None = none).
    sun_angular_radius: apparent radius of the sun (degrees;
        smaller farther from the star).
    """

    data = np.zeros(ATMOSPHERE_BLOCK.size // 4, dtype=np.float32)

    if parameters is None:
        return data.tobytes()

    p = parameters

    data[0:3] = np.asarray(planet_center_relative, dtype=np.float64) / 1000.0
    data[3] = 1.0

    data[4:8] = (p.ground_radius, p.top_radius, p.mie_anisotropy, p.ground_albedo)

    data[8:11] = p.rayleigh_scattering
    data[11] = p.rayleigh_scale_height

    data[12:15] = p.mie_scattering
    data[15] = p.mie_scale_height

    data[16:19] = p.mie_absorption
    data[19] = math.radians(sun_angular_radius)

    data[20:23] = p.ozone_absorption
    data[23] = p.ozone_altitude

    data[24] = p.ozone_half_width
    data[25] = float(steps)
    data[26] = 1.0 if aerial_perspective else 0.0
    data[27] = p.thick_weight

    if sun_direction is not None:

        data[28:31] = sun_direction
        data[31] = 1.0

        data[32:35] = sun_illuminance

    data[35] = SUN_DISC_BRIGHTNESS

    return data.tobytes()


# =========================================================
# Lookup Tables
# =========================================================

class AtmosphereLuts:

    TRANSMITTANCE_SIZE = (256, 64)
    MULTI_SCATTERING_SIZE = (32, 32)
    DIFFUSE_SIZE = (32, 32)

    def __init__(
        self,
        renderer,
        get_shader
    ):
        """
        get_shader(name) -> Shader for "atmosphere_transmittance",
        "atmosphere_multiscatter" and "atmosphere_diffuse".
        The atmosphere block must be uploaded before update().
        """

        self._renderer = renderer
        self._get_shader = get_shader

        self._transmittance = Framebuffer(
            FramebufferSpec(
                *self.TRANSMITTANCE_SIZE,
                color_format=ColorFormat.RGBA16F,
                depth_mode=DepthMode.NONE
            )
        )

        self._multi_scattering = Framebuffer(
            FramebufferSpec(
                *self.MULTI_SCATTERING_SIZE,
                color_format=ColorFormat.RGBA16F,
                depth_mode=DepthMode.NONE
            )
        )

        self._diffuse = Framebuffer(
            FramebufferSpec(
                *self.DIFFUSE_SIZE,
                color_format=ColorFormat.RGBA16F,
                depth_mode=DepthMode.NONE
            )
        )

        self._baked: AtmosphereParameters | None = None

        self.bake_count = 0

    def update(
        self,
        parameters: AtmosphereParameters
    ) -> bool:
        """
        Rebuild the LUTs if the parameters changed. The
        caller sets fullscreen state (no depth / culling).
        """

        if parameters == self._baked:
            return False

        shader = self._get_shader("atmosphere_transmittance")

        self._transmittance.bind()

        self._renderer.draw_fullscreen(shader)

        shader = self._get_shader("atmosphere_multiscatter")

        shader.bind()

        RenderCommand.bind_texture(self._transmittance.color_texture_id, 0)

        shader.set_int("uTransmittanceLut", 0)

        self._multi_scattering.bind()

        self._renderer.draw_fullscreen(shader)

        self._diffuse.bind()

        self._renderer.draw_fullscreen(self._get_shader("atmosphere_diffuse"))

        self._baked = parameters

        self.bake_count += 1

        Logger.debug("[Atmosphere] Built lookup tables (#%d).", self.bake_count)

        return True

    def textures(
        self
    ) -> dict[str, tuple[int, int]]:

        return {
            "uTransmittanceLut": (self._transmittance.color_texture_id, GL_TEXTURE_2D),
            "uMultiScatteringLut": (self._multi_scattering.color_texture_id, GL_TEXTURE_2D),
            "uDiffuseLut": (self._diffuse.color_texture_id, GL_TEXTURE_2D),
        }

    def delete(self):

        self._transmittance.delete()
        self._multi_scattering.delete()
        self._diffuse.delete()


# =========================================================
# Atmosphere as the IBL Source
# =========================================================

@dataclass(frozen=True, slots=True)
class AtmosphereSky:

    # What the environment bake needs to know to decide
    # whether the sky seen from the camera changed: the
    # atmosphere, the sun, and where the camera is (its
    # local "up" and altitude).

    parameters: AtmosphereParameters
    sun_direction: tuple[float, float, float] | None
    sun_illuminance: tuple[float, float, float]
    camera_up: tuple[float, float, float]
    altitude_km: float

    # (sampler name, texture id) of the LUTs; not part of
    # the bake key.
    textures: tuple[tuple[str, int], ...] = ()

    shader_name: str = "ibl_atmosphere"

    def bake_key(
        self
    ) -> tuple:

        def rounded(values, digits):

            if values is None:
                return None

            return tuple(round(float(v), digits) for v in values)

        # Altitude in ~19% steps: the sky changes slowly
        # with height except near the ground.
        altitude_bucket = round(math.log2(self.altitude_km + 0.5) * 4.0)

        return (
            self.parameters,
            rounded(self.sun_direction, 3),
            rounded(self.sun_illuminance, 3),
            rounded(self.camera_up, 2),
            altitude_bucket,
        )

    def apply(
        self,
        shader
    ):
        # The atmosphere block is frame-wide; the LUTs go on
        # units 0 and 1 (the bake binds nothing else).

        for unit, (name, texture_id) in enumerate(self.textures):

            RenderCommand.bind_texture(texture_id, unit)

            shader.set_int(name, unit)
