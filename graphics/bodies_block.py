import math

from dataclasses import dataclass

import numpy as np

from graphics.uniform_blocks import BODIES_BLOCK, MAX_BODIES


# =========================================================
# Bodies Block
# =========================================================
#
# Every planet and moon in the scene as a sphere, for
# eclipses: a point is lit by the part of the sun's disc no
# other body hides (assets/shaders/include/bodies.glsl).
# The Moon's shadow on Earth is a solar eclipse, Earth's on
# the Moon a lunar one; a body with air bends a little red
# sunset light into its umbra (the "blood moon").
#
# The ring system lives here too, apart from the atmosphere
# block: rings belong to their planet whichever body's air
# is being drawn (Saturn's rings in Titan's sky).


# Sunlight Earth's atmosphere refracts into its umbra at
# the Moon, per unit sun, red: a totally eclipsed Moon is
# really ~1e-4 as bright as full, but the dark-adapted eye
# (or a camera, exposing for it) sees it plainly, copper
# red; it is shown at that brightness, as in photographs.
# Scaled by the body's air.
UMBRA_GLOW = (8.0e-2, 2.0e-2, 4.0e-3)


@dataclass(frozen=True, slots=True)
class BodySphere:

    # World position (m) and radius (m).
    center: tuple[float, float, float]
    radius: float

    # Sunlight reaching its umbra through its air (fraction,
    # per channel; 0 = airless).
    glow: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Geometric albedo times its color: the sunlight it
    # reflects onto its neighbors (planetshine).
    light: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class RingSystem:

    # The ringed planet: world position (m), quaternion
    # taking world directions into its frame, equatorial
    # radius (m) and flattening.
    center: tuple[float, float, float]
    frame: tuple[float, float, float, float]
    planet_radius: float
    flattening: float

    # The rings: inner and outer radius (m), optical depth
    # scale.
    inner: float
    outer: float
    opacity: float = 1.0


def umbra_glow(
    surface_pressure_bar: float
) -> tuple[float, float, float]:
    """Light in a body's umbra from its air (Earth's at 1 bar)."""

    if surface_pressure_bar <= 0.01:
        return (0.0, 0.0, 0.0)

    scale = min(surface_pressure_bar, 3.0)

    return tuple(c * scale for c in UMBRA_GLOW)


def eclipsing(
    body: BodySphere,
    target_center,
    target_radius: float,
    toward_sun,
    sun_angular_radius: float
) -> bool:
    """
    Whether the body's shadow (umbra and penumbra) can
    reach a sphere: the body must be sunward of it, and the
    target within the widening penumbra cone.
    """

    toward_sun = np.asarray(toward_sun, dtype=np.float64)

    offset = np.asarray(target_center, dtype=np.float64) - np.asarray(body.center, dtype=np.float64)

    behind = -float(offset @ toward_sun)

    if behind <= 0.0:
        return False

    across = float(np.linalg.norm(offset + behind * toward_sun))

    penumbra = body.radius + behind * math.tan(sun_angular_radius)

    return across < penumbra + target_radius


def pack_bodies_block(
    bodies,
    camera_position,
    sun_angular_radius: float,
    eclipsing_count: int = 0,
    rings: RingSystem | None = None
) -> bytes:
    """
    bodies: BodySphere list (at most MAX_BODIES), the ones
        eclipsing the atmosphere's planet first.
    camera_position: world position (m); centers are stored
        relative to it (km), like everything rendered.
    sun_angular_radius: apparent radius of the sun (rad).
    eclipsing_count: how many of the first bodies shade the
        planet whose air is drawn (the sky dims under them).
    """

    data = np.zeros(BODIES_BLOCK.size // 4, dtype=np.float32)

    camera = np.asarray(camera_position, dtype=np.float64)

    bodies = list(bodies)[:MAX_BODIES]

    data[0:4] = (len(bodies), sun_angular_radius, min(eclipsing_count, len(bodies)), 0.0)

    spheres = 4
    glows = spheres + 4 * MAX_BODIES

    for i, body in enumerate(bodies):

        data[spheres + 4 * i:spheres + 4 * i + 3] = (np.asarray(body.center, dtype=np.float64) - camera) / 1000.0
        data[spheres + 4 * i + 3] = body.radius / 1000.0

        data[glows + 4 * i:glows + 4 * i + 3] = body.glow

    ring = glows + 4 * MAX_BODIES

    lights = ring + 16

    for i, body in enumerate(bodies):
        data[lights + 4 * i:lights + 4 * i + 3] = body.light

    if rings is not None:

        data[ring:ring + 3] = (np.asarray(rings.center, dtype=np.float64) - camera) / 1000.0
        data[ring + 3] = rings.planet_radius / 1000.0

        data[ring + 4:ring + 8] = rings.frame

        data[ring + 8:ring + 12] = (rings.inner / 1000.0, rings.outer / 1000.0, 1.0, max(rings.opacity, 0.0))

        data[ring + 12] = min(max(rings.flattening, 0.0), 0.5)

    else:

        # Identity frame, no rings.
        data[ring + 7] = 1.0

    return data.tobytes()
