from dataclasses import dataclass, field

import numpy as np


# =========================================================
# Limits
# =========================================================
#
# Injected into every shader as #defines (see
# graphics/uniform_blocks.ENGINE_SHADER_DEFINES), so this
# is the single place to change them.

MAX_POINT_LIGHTS = 8
MAX_SPOT_LIGHTS = 4


# =========================================================
# Directional Light
# =========================================================
#
# Renderer-side light data. Kept free of ECS types, the
# same way the Renderer receives a math3d Camera rather
# than camera components.

@dataclass(slots=True)
class DirectionalLight:

    # Normalized world-space direction the light travels.
    direction: np.ndarray

    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    casts_shadows: bool = False


# =========================================================
# Point Light
# =========================================================

@dataclass(slots=True)
class PointLight:

    position: np.ndarray

    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    range: float = 10.0


# =========================================================
# Spot Light
# =========================================================

@dataclass(slots=True)
class SpotLight:

    position: np.ndarray

    # Normalized world-space direction the cone points.
    direction: np.ndarray

    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    range: float = 10.0

    # Cosines of the inner / outer half-angles.
    inner_cutoff: float = 0.966
    outer_cutoff: float = 0.906


# =========================================================
# Light Environment
# =========================================================
#
# Every light affecting a frame.

@dataclass(slots=True)
class LightEnvironment:

    ambient: float = 0.0

    directional: DirectionalLight | None = None

    point_lights: list[PointLight] = field(
        default_factory=list
    )

    spot_lights: list[SpotLight] = field(
        default_factory=list
    )
