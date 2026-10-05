import math

from dataclasses import dataclass

import numpy as np


# =========================================================
# Comets: Coma and Tails
# =========================================================
#
# Near the Sun a comet's ices sublimate and the gas drags
# dust off the nucleus (assets/shaders/comets.frag.glsl draws
# it):
#
#   coma        dust around the nucleus, its density falling
#               as 1 / r^2 (steady outflow), out to where the
#               Sun's light pressure sweeps it back
#   dust tail   swept away from the Sun, and curved: the
#               grains keep their orbital speed while they
#               drift outward, so the tail lags the comet's
#               motion; a broad fan, yellowish (sunlight
#               off the grains), brightest seen toward the
#               Sun (they scatter forward)
#   ion tail    gas ionized by sunlight, carried straight
#               out by the solar wind: narrow, blue (CO+
#               glows near 420 nm)
#
# How much dust: A f rho (A'Hearn 1984), the measure comet
# observers report: albedo x the dust's filling factor of
# an aperture x its radius. For a 1 / r^2 coma it fixes the
# surface brightness at a projected distance b from the
# nucleus:
#
#     L(b) = E (A f rho) / (8 pi b)
#
# with E the sunlight on the comet, so the coma here is as
# bright as the measured comets are (67P at perihelion:
# ~1,500 cm; Halley in 1986: ~1e5 cm). Activity grows
# steeply toward the Sun (~r^-3.5) and stops past ~4 AU,
# where water ice no longer sublimates.

MAX_COMETS = 4

# Activity's growth toward the Sun, and where it stops (AU).
ACTIVITY_SLOPE = 3.5
ACTIVITY_LIMIT = (3.0, 4.5)

# Dust tail: length for an A f rho of 2e4 cm (km), how it
# scales, its bounds; the curve (lag at its end, as a share
# of its length).
DUST_TAIL_LENGTH = 3.0e6
DUST_TAIL_BOUNDS = (5.0e4, 3.0e7)
DUST_TAIL_BEND = 0.12

# The coma's extent at 1 AU (km), growing with distance (the
# Sun's light pressure turns the dust back sooner closer in).
COMA_SIZE = 1.0e5


@dataclass(frozen=True, slots=True)
class CometView:
    """One comet as the shader needs it (camera-relative km)."""

    center: tuple[float, float, float]
    coma_size: float                # km
    away: tuple[float, float, float]     # unit: from the Sun
    afrho: float                    # km
    lag: tuple[float, float, float]      # unit: behind its motion, across `away`
    dust_length: float              # km
    sunlight: float                 # engine illuminance
    gas: float
    ion_length: float               # km


def activity(
    afrho_1au_m: float,
    distance_au: float
) -> float:
    """A f rho (m) at a distance from the Sun, from its value at 1 AU."""

    if afrho_1au_m <= 0.0 or distance_au <= 0.0:
        return 0.0

    low, high = ACTIVITY_LIMIT

    t = min(max((distance_au - low) / (high - low), 0.0), 1.0)

    fade = 1.0 - t * t * (3.0 - 2.0 * t)

    return afrho_1au_m * distance_au ** -ACTIVITY_SLOPE * fade


def coma_radiance(
    sunlight: float,
    afrho_m: float,
    projected_km: float
) -> float:
    """Surface brightness of a 1 / r^2 coma at a projected distance (engine units)."""

    return sunlight * (afrho_m / 1000.0) / (8.0 * math.pi * max(projected_km, 1e-3))


def dust_tail_length(
    afrho_m: float
) -> float:
    """km."""

    low, high = DUST_TAIL_BOUNDS

    return min(max(DUST_TAIL_LENGTH * math.sqrt(max(afrho_m, 0.0) / 200.0), low), high)


def comet_view(
    center,
    sun_position,
    orbit_normal,
    camera_position,
    afrho_1au_m: float,
    gas: float,
    star_luminosity: float = 1.0
) -> CometView | None:
    """
    center, sun_position, camera_position: world (m);
    orbit_normal: unit (the orbit's angular momentum).
    None when the comet is inactive.
    """

    center = np.asarray(center, dtype=np.float64)

    offset = center - np.asarray(sun_position, dtype=np.float64)

    distance = float(np.linalg.norm(offset))

    if distance <= 0.0:
        return None

    au = distance / 1.495978707e11

    afrho = activity(afrho_1au_m, au)

    if afrho <= 0.0:
        return None

    away = offset / distance

    # Behind its motion: the orbital velocity's part across
    # the Sun's direction is normal x away.
    lag = -np.cross(np.asarray(orbit_normal, dtype=np.float64), away)

    norm = float(np.linalg.norm(lag))

    lag = lag / norm if norm > 1e-9 else np.zeros(3)

    dust = dust_tail_length(afrho)

    relative = (center - np.asarray(camera_position, dtype=np.float64)) / 1000.0

    return CometView(
        center=tuple(float(v) for v in relative),
        coma_size=COMA_SIZE * au,
        away=tuple(float(v) for v in away),
        afrho=afrho / 1000.0,
        lag=tuple(float(v) for v in lag),
        dust_length=dust,
        sunlight=5.0 * star_luminosity / au ** 2,
        gas=gas,
        ion_length=1.5 * dust * max(gas, 0.1)
    )


def pack_comet_uniforms(
    comets: list[CometView]
) -> dict[str, np.ndarray]:
    """Uniform arrays for comets.frag.glsl (4 vec4 per comet)."""

    comets = comets[:MAX_COMETS]

    center = np.zeros((MAX_COMETS, 4), dtype=np.float32)
    axis = np.zeros((MAX_COMETS, 4), dtype=np.float32)
    bend = np.zeros((MAX_COMETS, 4), dtype=np.float32)
    params = np.zeros((MAX_COMETS, 4), dtype=np.float32)

    for i, c in enumerate(comets):

        center[i] = (*c.center, c.coma_size)
        axis[i] = (*c.away, c.afrho)
        bend[i] = (*c.lag, c.dust_length)
        params[i] = (c.sunlight, c.gas, c.ion_length, 0.0)

    return {
        "uCometCenter": center,
        "uCometAxis": axis,
        "uCometBend": bend,
        "uCometParams": params,
    }
