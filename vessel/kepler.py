import math

from dataclasses import dataclass

import numpy as np


# =========================================================
# Two-Body Orbits
# =========================================================
#
# A vessel coasting outside the air moves on a conic about
# the body it orbits: propagating it exactly (universal
# variables, Curtis, Orbital Mechanics for Engineering
# Students, algorithms 3.3-3.4) lets time warp run at any
# rate without integration error ("on rails"). The same
# state gives the orbit's shape for the HUD: apoapsis,
# periapsis, period.
#
# Positions and velocities are relative to the body's center
# in a non-rotating frame (m, m/s); mu = G M (m^3/s^2).


def _stumpff_c(
    z: float
) -> float:

    if z > 1e-8:
        return (1.0 - math.cos(math.sqrt(z))) / z

    if z < -1e-8:
        return (math.cosh(math.sqrt(-z)) - 1.0) / -z

    return 0.5 - z / 24.0


def _stumpff_s(
    z: float
) -> float:

    if z > 1e-8:
        root = math.sqrt(z)
        return (root - math.sin(root)) / root ** 3

    if z < -1e-8:
        root = math.sqrt(-z)
        return (math.sinh(root) - root) / root ** 3

    return 1.0 / 6.0 - z / 120.0


def propagate(
    position,
    velocity,
    mu: float,
    dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """The state `dt` seconds later on the same conic."""

    r0_vec = np.asarray(position, dtype=np.float64)
    v0_vec = np.asarray(velocity, dtype=np.float64)

    r0 = float(np.linalg.norm(r0_vec))
    v0 = float(np.linalg.norm(v0_vec))

    if dt == 0.0 or r0 == 0.0:
        return r0_vec.copy(), v0_vec.copy()

    vr0 = float(np.dot(r0_vec, v0_vec)) / r0

    # Reciprocal of the semi-major axis (< 0: hyperbola).
    alpha = 2.0 / r0 - v0 * v0 / mu

    # Whole revolutions change nothing on an ellipse (and
    # keep Newton's start close).
    if alpha > 1e-12:
        period = 2.0 * math.pi / math.sqrt(mu * alpha ** 3)
        dt = math.fmod(dt, period)

    sqrt_mu = math.sqrt(mu)

    x = sqrt_mu * abs(alpha) * dt if abs(alpha) > 1e-12 else sqrt_mu * dt / r0

    for _ in range(60):

        z = alpha * x * x

        c = _stumpff_c(z)
        s = _stumpff_s(z)

        f = (
            r0 * vr0 / sqrt_mu * x * x * c
            + (1.0 - alpha * r0) * x ** 3 * s
            + r0 * x
            - sqrt_mu * dt
        )

        df = (
            r0 * vr0 / sqrt_mu * x * (1.0 - alpha * x * x * s)
            + (1.0 - alpha * r0) * x * x * c
            + r0
        )

        step = f / df

        x -= step

        if abs(step) < 1e-10 * max(1.0, abs(x)):
            break

    z = alpha * x * x

    c = _stumpff_c(z)
    s = _stumpff_s(z)

    f = 1.0 - x * x / r0 * c
    g = dt - x ** 3 / sqrt_mu * s

    r_vec = f * r0_vec + g * v0_vec

    r = float(np.linalg.norm(r_vec))

    f_dot = sqrt_mu / (r * r0) * (alpha * x ** 3 * s - x)
    g_dot = 1.0 - x * x / r * c

    v_vec = f_dot * r0_vec + g_dot * v0_vec

    return r_vec, v_vec


@dataclass(frozen=True, slots=True)
class OrbitShape:

    semi_major_axis: float          # m (< 0: hyperbolic)
    eccentricity: float
    periapsis: float                # m from the center
    apoapsis: float | None          # m from the center; None: escaping
    period: float | None            # s; None: escaping
    inclination: float              # deg, to the body's equator (pole given)


def orbit_shape(
    position,
    velocity,
    mu: float,
    pole=(0.0, 1.0, 0.0)
) -> OrbitShape:
    """The conic through a state."""

    r_vec = np.asarray(position, dtype=np.float64)
    v_vec = np.asarray(velocity, dtype=np.float64)

    r = float(np.linalg.norm(r_vec))
    v2 = float(np.dot(v_vec, v_vec))

    h_vec = np.cross(r_vec, v_vec)
    h = float(np.linalg.norm(h_vec))

    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
    e = float(np.linalg.norm(e_vec))

    energy = v2 / 2.0 - mu / r

    pole = np.asarray(pole, dtype=np.float64)

    inclination = (
        math.degrees(math.acos(float(np.clip(np.dot(h_vec, pole) / (h * np.linalg.norm(pole)), -1.0, 1.0))))
        if h > 0.0
        else 0.0
    )

    # Semi-latus rectum: periapsis = p / (1 + e) holds for
    # every conic (and a radial fall, h = 0, has p = 0).
    p = h * h / mu

    periapsis = p / (1.0 + e)

    if energy < 0.0 and e < 1.0:

        a = -mu / (2.0 * energy)

        return OrbitShape(
            semi_major_axis=a,
            eccentricity=e,
            periapsis=periapsis,
            apoapsis=a * (1.0 + e),
            period=2.0 * math.pi * math.sqrt(a ** 3 / mu),
            inclination=inclination
        )

    return OrbitShape(
        semi_major_axis=-mu / (2.0 * energy) if energy != 0.0 else math.inf,
        eccentricity=e,
        periapsis=periapsis,
        apoapsis=None,
        period=None,
        inclination=inclination
    )


def circular_speed(
    mu: float,
    radius: float
) -> float:

    return math.sqrt(mu / radius)
