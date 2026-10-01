from dataclasses import dataclass, field

import numpy as np

from core.handle import Handle
from ecs.entity import Entity
from math3d.transform import Transform


# =========================================================
# Name
# =========================================================
#
# Human-readable label used by the debug UI and logs.

@dataclass(slots=True)
class NameComponent:
    name: str


# =========================================================
# Transform
# =========================================================
#
# `transform` is the local transform (relative to the
# parent, or to the world when there is no parent).
#
# `world_matrix` is computed each frame by TransformSystem.
# Anything that needs a world-space position/direction
# (rendering, lights, cameras) should use the world_*
# helpers rather than `transform` directly.

def _identity() -> np.ndarray:

    return np.identity(
        4,
        dtype=np.float64
    )


@dataclass(slots=True)
class TransformComponent:

    transform: Transform = field(
        default_factory=Transform
    )

    world_matrix: np.ndarray = field(
        default_factory=_identity
    )

    # -----------------------------------------------------
    # World-Space Helpers
    # -----------------------------------------------------

    @property
    def world_position(
        self
    ) -> np.ndarray:

        return np.array(
            self.world_matrix[:3, 3],
            dtype=np.float64
        )

    @property
    def world_forward(
        self
    ) -> np.ndarray:

        return self._world_direction(
            (0.0, 0.0, -1.0)
        )

    @property
    def world_up(
        self
    ) -> np.ndarray:

        return self._world_direction(
            (0.0, 1.0, 0.0)
        )

    def _world_direction(
        self,
        local
    ) -> np.ndarray:

        world = (
            self.world_matrix[:3, :3]
            @ np.asarray(local, dtype=np.float64)
        )

        length = float(
            np.linalg.norm(world)
        )

        # A zero-scaled parent can collapse directions;
        # fall back to the unrotated local axis.

        if length <= 1e-8:

            return np.asarray(
                local,
                dtype=np.float64
            )

        return (
            world
            / length
        ).astype(
            np.float64
        )


# =========================================================
# Hierarchy
# =========================================================
#
# Makes the owning entity's transform relative to
# `parent`. The parent must have a TransformComponent.

@dataclass(slots=True)
class HierarchyComponent:
    parent: Entity


# =========================================================
# Mesh Renderer
# =========================================================

@dataclass(slots=True)
class MeshRendererComponent:
    mesh: Handle
    material: Handle
    casts_shadows: bool = True


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
# Direction comes from the owning entity's world forward.
# It also places the sun in the sky. (Ambient light comes
# from the sky through image-based lighting; see
# RenderSettings.ibl_intensity.)

@dataclass(slots=True)
class DirectionalLightComponent:
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 5.0
    casts_shadows: bool = True


# =========================================================
# Point Light
# =========================================================
#
# Position comes from the owning entity's world position.
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
# world position / forward.
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
    casts_shadows: bool = True


# =========================================================
# Rotator
# =========================================================
#
# Spins the entity's local transform at a constant rate
# (degrees per second around X, Y, Z). Driven by
# RotatorSystem on the fixed timestep.

@dataclass(slots=True)
class RotatorComponent:
    degrees_per_second: tuple[float, float, float] = (
        0.0,
        30.0,
        0.0
    )


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

    # Planet mode: "up" points away from planet_center
    # instead of world +Y, so the horizon stays level
    # anywhere on a sphere and Q/E move radially.
    planet_mode: bool = False
    planet_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    planet_radius: float = 0.0

    # Speed grows with altitude above planet_radius:
    # speed = max(movement_speed, altitude * altitude_speed).
    # 0 disables scaling.
    altitude_speed: float = 0.0

    # Closest the camera may get to the surface.
    min_altitude: float = 1.0
