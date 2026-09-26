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
# debug UI).

@dataclass(slots=True)
class RenderSettings:

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    # Linear-space background color.
    clear_color: tuple[float, float, float] = (0.01, 0.01, 0.015)

    exposure: float = 1.0
    tonemapper: Tonemapper = Tonemapper.ACES
    gamma: float = 2.2

    # -----------------------------------------------------
    # Shadows (directional light)
    # -----------------------------------------------------

    shadows_enabled: bool = True

    shadow_map_size: int = 2048

    # The shadow map covers a square of
    # 2 * shadow_extent world units around shadow_center.
    # Smaller = sharper shadows over a smaller area.

    shadow_extent: float = 8.0
    shadow_center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Slope-scaled depth bias: bias_max at grazing angles,
    # bias_min facing the light. Too low = shadow acne,
    # too high = shadows detach from casters
    # ("peter-panning").

    shadow_bias_min: float = 0.0005
    shadow_bias_max: float = 0.005

    # -----------------------------------------------------
    # Debug
    # -----------------------------------------------------

    show_light_gizmos: bool = True
    gizmo_scale: float = 0.12
