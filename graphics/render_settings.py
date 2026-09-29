from dataclasses import dataclass
from enum import IntEnum


# =========================================================
# Tone Mapping
# =========================================================
#
# Values must match the uTonemapper switch in
# assets/shaders/post.frag.glsl.

class Tonemapper(IntEnum):

    # Clamp to [0, 1]. Shows clipping.
    NONE = 0

    # x / (1 + x). Gentle, desaturates highlights.
    REINHARD = 1

    # Narkowicz's ACES filmic fit. Punchier contrast.
    ACES = 2


# =========================================================
# Render Settings
# =========================================================
#
# Runtime-tweakable renderer options (edited live by the
# debug UI, saved with scenes). Colors are linear.

@dataclass(slots=True)
class RenderSettings:

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    # Tuned for a sun of intensity ~5 plus sky light.
    exposure: float = 0.75
    tonemapper: Tonemapper = Tonemapper.ACES
    gamma: float = 2.2

    # Fast approximate anti-aliasing on the final image.
    fxaa_enabled: bool = True

    # -----------------------------------------------------
    # Bloom
    # -----------------------------------------------------
    #
    # Energy-conserving: the image is blended toward its
    # blurred version by `bloom_intensity`, so there is no
    # brightness threshold; very bright (HDR) areas simply
    # dominate the blur.

    bloom_enabled: bool = True
    bloom_intensity: float = 0.04

    # Upsample filter radius in UV units (larger = wider).
    bloom_radius: float = 0.005

    # -----------------------------------------------------
    # Sky / Image-Based Lighting
    # -----------------------------------------------------

    # When off, the background is `clear_color` (the sky
    # still lights the scene through IBL).
    show_sky: bool = True
    clear_color: tuple[float, float, float] = (0.01, 0.01, 0.015)

    sky_zenith_color: tuple[float, float, float] = (0.05, 0.12, 0.36)
    sky_horizon_color: tuple[float, float, float] = (0.24, 0.32, 0.44)
    ground_color: tuple[float, float, float] = (0.09, 0.08, 0.07)
    sky_intensity: float = 1.0

    # Apparent radius of the sun disc, in degrees.
    sun_size: float = 0.8

    # Scales ambient light and reflections from the sky.
    ibl_intensity: float = 1.0

    # -----------------------------------------------------
    # Shadows
    # -----------------------------------------------------

    shadows_enabled: bool = True

    # Directional light: cascaded shadow maps covering the
    # view out to `shadow_distance`, split between
    # `cascade_count` maps. Lambda blends logarithmic (1.0,
    # more detail close up) and uniform (0.0) splits.

    shadow_distance: float = 30.0
    cascade_count: int = 4
    cascade_split_lambda: float = 0.75

    # Resolution of each cascade / each spot shadow map.
    shadow_map_size: int = 2048
    spot_shadow_map_size: int = 1024

    # Receiver offset along the surface normal, in shadow
    # texels. Too low = acne, too high = shadows detach
    # from their casters.
    shadow_normal_offset: float = 1.5

    # Small constant depth bias (in NDC depth).
    shadow_depth_bias: float = 0.0005

    # -----------------------------------------------------
    # Debug
    # -----------------------------------------------------

    show_light_gizmos: bool = True
    gizmo_scale: float = 0.12

    # Tints the scene by shadow cascade.
    visualize_cascades: bool = False
