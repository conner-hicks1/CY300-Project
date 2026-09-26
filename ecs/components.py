from dataclasses import dataclass, field

from core.handle import Handle
from math3d.transform import Transform


# =========================================================
# Transform
# =========================================================

@dataclass(slots=True)
class TransformComponent:
    transform: Transform = field(
        default_factory=Transform
    )


# =========================================================
# Mesh Renderer
# =========================================================

@dataclass(slots=True)
class MeshRendererComponent:
    mesh: Handle
    material: Handle


# =========================================================
# Camera
# =========================================================

@dataclass(slots=True)
class CameraComponent:
    fov: float = 45.0
    near: float = 0.1
    far: float = 100.0
    primary: bool = False


# =========================================================
# Directional Light
# =========================================================
#
# Direction comes from the owning entity's
# TransformComponent.transform.forward.

@dataclass(slots=True)
class DirectionalLightComponent:
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    ambient: float = 0.1


# =========================================================
# Point Light
# =========================================================
#
# Position comes from the owning entity's
# TransformComponent.transform.position.
#
# range: distance at which the light fades to zero.

@dataclass(slots=True)
class PointLightComponent:
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    range: float = 10.0


# =========================================================
# Spot Light
# =========================================================
#
# Position and direction come from the owning entity's
# TransformComponent (position / forward).
#
# Angles are half-angles in degrees measured from the
# spot direction. Full intensity inside inner_angle,
# fading to zero at outer_angle.

@dataclass(slots=True)
class SpotLightComponent:
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    range: float = 10.0
    inner_angle: float = 15.0
    outer_angle: float = 25.0


# =========================================================
# Camera Controller
# =========================================================

@dataclass(slots=True)
class CameraControllerComponent:

    movement_speed: float = 3.0
    mouse_sensitivity: float = 0.1

    # Prevent looking exactly straight up/down.
    min_pitch: float = -89.0
    max_pitch: float = 89.0