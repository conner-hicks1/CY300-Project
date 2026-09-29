from dataclasses import dataclass

import numpy as np

from OpenGL.GL import (
    GL_TEXTURE_2D,
    GL_TEXTURE_CUBE_MAP
)

from core.logger import Logger

from graphics.cubemap import (
    Cubemap,
    CubemapCaptureTarget
)
from graphics.framebuffer import (
    ColorFormat,
    DepthMode,
    Framebuffer,
    FramebufferSpec
)
from graphics.lighting import PREFILTER_MIP_LEVELS
from graphics.render_settings import RenderSettings
from graphics.shader import Shader


# =========================================================
# Sky Parameters
# =========================================================

@dataclass(frozen=True, slots=True)
class SkyParameters:

    # Unit vector toward the sun (None = no sun).
    sun_direction: tuple[float, float, float] | None

    # Sun color * intensity.
    sun_radiance: tuple[float, float, float]

    zenith: tuple[float, float, float]
    horizon: tuple[float, float, float]
    ground: tuple[float, float, float]

    intensity: float

    # Apparent sun radius in degrees.
    sun_size: float

    @classmethod
    def from_scene(
        cls,
        settings: RenderSettings,
        light_direction=None,
        light_color=(1.0, 1.0, 1.0),
        light_intensity: float = 0.0
    ) -> "SkyParameters":
        """
        light_direction: direction the directional light
            travels (None without one).
        """

        if light_direction is None:

            sun_direction = None
            sun_radiance = (0.0, 0.0, 0.0)

        else:

            toward = -np.asarray(light_direction, dtype=np.float64)
            toward = toward / np.linalg.norm(toward)

            sun_direction = tuple(float(v) for v in toward)

            sun_radiance = tuple(
                float(c) * float(light_intensity)
                for c in light_color
            )

        return cls(
            sun_direction=sun_direction,
            sun_radiance=sun_radiance,
            zenith=tuple(settings.sky_zenith_color),
            horizon=tuple(settings.sky_horizon_color),
            ground=tuple(settings.ground_color),
            intensity=float(settings.sky_intensity),
            sun_size=float(settings.sun_size)
        )

    def bake_key(
        self
    ) -> tuple:
        """
        Changes whenever the baked lighting would visibly
        change. Rounded so float noise (e.g. a re-derived
        light direction) does not trigger a re-bake.
        """

        def rounded(values, digits=3):

            if values is None:
                return None

            return tuple(round(float(v), digits) for v in values)

        # The disc is not baked (see include/sky.glsl), so
        # sun_size is not part of the key.

        return (
            rounded(self.sun_direction),
            rounded(self.sun_radiance),
            rounded(self.zenith),
            rounded(self.horizon),
            rounded(self.ground),
            round(self.intensity, 3),
        )

    def apply(
        self,
        shader: Shader
    ):
        """Set the include/sky.glsl uniforms."""

        has_sun = self.sun_direction is not None

        shader.set_vec3("uSunDirection", self.sun_direction if has_sun else (0.0, 1.0, 0.0))
        shader.set_vec3("uSunRadiance", self.sun_radiance)
        shader.set_float("uHasSun", 1.0 if has_sun else 0.0)

        shader.set_vec3("uSkyZenith", self.zenith)
        shader.set_vec3("uSkyHorizon", self.horizon)
        shader.set_vec3("uSkyGround", self.ground)
        shader.set_float("uSkyIntensity", self.intensity)

        radius = np.radians(max(self.sun_size, 0.01))

        shader.set_vec2(
            "uSunDisc",
            (float(np.cos(radius * 1.25)), float(np.cos(radius)))
        )


# =========================================================
# Environment (Image-Based Lighting)
# =========================================================

class Environment:

    # =====================================================
    # Baked Lighting from the Sky
    # =====================================================
    #
    # Split-sum IBL (Karis 2013). From the procedural sky:
    #
    #   environment   256^2 cubemap (+ mips), the sky itself
    #   irradiance     32^2 cubemap, diffuse convolution
    #   prefiltered   128^2 cubemap, one mip per roughness
    #   BRDF LUT      256^2, scene-independent, baked once
    #
    # Baking runs only when SkyParameters.bake_key()
    # changes, and at most every MIN_FRAMES_BETWEEN_BAKES
    # frames, so dragging the sun around stays interactive.

    ENVIRONMENT_SIZE = 256
    IRRADIANCE_SIZE = 32
    PREFILTER_SIZE = 128
    BRDF_LUT_SIZE = 256

    MIN_FRAMES_BETWEEN_BAKES = 4

    def __init__(
        self,
        renderer,
        get_shader
    ):
        """
        get_shader(name) -> Shader for "ibl_sky",
        "ibl_irradiance", "ibl_prefilter" or "ibl_brdf"
        (looked up per bake so hot reload works).
        """

        self._renderer = renderer
        self._get_shader = get_shader

        environment_mips = int(np.log2(self.ENVIRONMENT_SIZE)) + 1

        self.environment = Cubemap(
            self.ENVIRONMENT_SIZE,
            environment_mips,
            "environment"
        )

        self.irradiance = Cubemap(
            self.IRRADIANCE_SIZE,
            1,
            "irradiance"
        )

        self.prefiltered = Cubemap(
            self.PREFILTER_SIZE,
            PREFILTER_MIP_LEVELS,
            "prefiltered"
        )

        self._capture = CubemapCaptureTarget()

        self._brdf_lut = Framebuffer(
            FramebufferSpec(
                width=self.BRDF_LUT_SIZE,
                height=self.BRDF_LUT_SIZE,
                color_format=ColorFormat.RGBA16F,
                depth_mode=DepthMode.NONE
            )
        )

        self._baked_key = None
        self._frames_since_bake = self.MIN_FRAMES_BETWEEN_BAKES

        self.bake_count = 0

        self._bake_brdf_lut()

    @property
    def brdf_lut_texture_id(
        self
    ) -> int:

        return self._brdf_lut.color_texture_id

    # =====================================================
    # Update
    # =====================================================

    def update(
        self,
        sky: SkyParameters
    ) -> bool:
        """
        Re-bake if the sky changed. Returns True if it
        baked. The caller disables depth test / culling.
        """

        self._frames_since_bake += 1

        key = sky.bake_key()

        if key == self._baked_key:
            return False

        # First bake always happens immediately.
        if (
            self._baked_key is not None
            and self._frames_since_bake < self.MIN_FRAMES_BETWEEN_BAKES
        ):
            return False

        self.bake(sky)

        self._baked_key = key
        self._frames_since_bake = 0

        return True

    # =====================================================
    # Baking
    # =====================================================

    def bake(
        self,
        sky: SkyParameters
    ):

        renderer = self._renderer

        # -------------------------------------------------
        # 1. Sky -> environment cubemap
        # -------------------------------------------------

        shader = self._get_shader("ibl_sky")

        shader.bind()

        sky.apply(shader)

        for face in range(6):

            self._capture.bind_face(self.environment, face, 0)

            shader.set_int("uFace", face)

            renderer.draw_fullscreen(shader)

        # Mips feed the filtered lookups below.
        self.environment.generate_mipmaps()

        # -------------------------------------------------
        # 2. Diffuse irradiance
        # -------------------------------------------------

        shader = self._get_shader("ibl_irradiance")

        shader.bind()
        shader.set_int("uEnvironment", 0)

        self.environment.bind(0)

        for face in range(6):

            self._capture.bind_face(self.irradiance, face, 0)

            shader.set_int("uFace", face)

            renderer.draw_fullscreen(shader)

        # -------------------------------------------------
        # 3. Specular prefilter, one mip per roughness
        # -------------------------------------------------

        shader = self._get_shader("ibl_prefilter")

        shader.bind()
        shader.set_int("uEnvironment", 0)
        shader.set_float("uEnvironmentSize", float(self.ENVIRONMENT_SIZE))

        self.environment.bind(0)

        for level in range(PREFILTER_MIP_LEVELS):

            roughness = level / (PREFILTER_MIP_LEVELS - 1)

            shader.set_float("uRoughness", roughness)

            # Mip-filtered samples converge quickly; mip 0
            # (mirror) is a straight copy.
            shader.set_int("uSampleCount", 1 if level == 0 else 64)

            for face in range(6):

                self._capture.bind_face(self.prefiltered, face, level)

                shader.set_int("uFace", face)

                renderer.draw_fullscreen(shader)

        self.bake_count += 1

        Logger.debug(
            "[Environment] Baked IBL (#%d).",
            self.bake_count
        )

    def _bake_brdf_lut(self):

        shader = self._get_shader("ibl_brdf")

        self._brdf_lut.bind()

        self._renderer.draw_fullscreen(shader)

    # =====================================================
    # Binding
    # =====================================================

    def textures(
        self
    ) -> dict[str, tuple[int, int]]:
        """sampler name -> (texture id, GL target)."""

        return {
            "uIrradianceMap": (self.irradiance.id, GL_TEXTURE_CUBE_MAP),
            "uPrefilterMap": (self.prefiltered.id, GL_TEXTURE_CUBE_MAP),
            "uBrdfLut": (self.brdf_lut_texture_id, GL_TEXTURE_2D),
        }

    # =====================================================
    # Cleanup
    # =====================================================

    def delete(self):

        self.environment.delete()
        self.irradiance.delete()
        self.prefiltered.delete()

        self._capture.delete()
        self._brdf_lut.delete()
