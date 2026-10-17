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


def _first_guess(
    r0_vec: np.ndarray,
    v0_vec: np.ndarray,
    r0: float,
    alpha: float,
    mu: float,
    dt
):
    """
    Newton's starting point for the universal anomaly
    (Vallado, algorithm 8): linear in time on an ellipse, the
    asymptotic log on a hyperbola (where the linear guess
    overshoots into overflow).
    """

    sqrt_mu = math.sqrt(mu)

    if alpha > 1e-12:
        return sqrt_mu * alpha * dt

    if alpha < -1e-12:

        a = 1.0 / alpha

        sign = np.sign(dt) if np.ndim(dt) else math.copysign(1.0, dt)

        rv = float(np.dot(r0_vec, v0_vec))

        numerator = -2.0 * mu * alpha * np.abs(dt)
        denominator = abs(rv * sign + math.sqrt(-mu * a) * (1.0 - r0 * alpha)) if not np.ndim(dt) else np.abs(
            rv * sign + math.sqrt(-mu * a) * (1.0 - r0 * alpha)
        )

        # (At dt = 0 the anomaly is 0.)
        ratio = np.where(np.asarray(dt) == 0.0, 1.0, numerator / np.maximum(denominator, 1e-30))

        asymptotic = sign * math.sqrt(-a) * np.log(np.maximum(ratio, 1.0))

        linear = sqrt_mu * np.asarray(dt) / r0

        # The smaller of the two: the linear guess overshoots
        # over long times, the asymptotic one is poor over
        # short ones; from below Newton converges.
        guess = np.where(np.abs(linear) < np.abs(asymptotic), linear, asymptotic)

        return guess if np.ndim(dt) else float(guess)

    return sqrt_mu * dt / r0


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

    x = _first_guess(r0_vec, v0_vec, r0, alpha, mu, dt)

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


def _stumpff_arrays(
    z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    c = np.empty_like(z)
    s = np.empty_like(z)

    positive = z > 1e-8
    negative = z < -1e-8
    small = ~(positive | negative)

    root = np.sqrt(z[positive])
    c[positive] = (1.0 - np.cos(root)) / z[positive]
    s[positive] = (root - np.sin(root)) / root ** 3

    root = np.sqrt(-z[negative])
    c[negative] = (np.cosh(root) - 1.0) / -z[negative]
    s[negative] = (np.sinh(root) - root) / root ** 3

    c[small] = 0.5 - z[small] / 24.0
    s[small] = 1.0 / 6.0 - z[small] / 120.0

    return c, s


def propagate_many(
    position,
    velocity,
    mu: float,
    times
) -> tuple[np.ndarray, np.ndarray]:
    """
    propagate() at many times at once (vectorized): (n, 3)
    positions and velocities. For drawing whole orbits.
    Samples the solver cannot reach (a path straight through
    the body's center) come back NaN.
    """

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):

        r_vec, v_vec = _propagate_many(position, velocity, mu, times)

    bad = ~(np.isfinite(r_vec).all(axis=1) & np.isfinite(v_vec).all(axis=1))

    # (Or diverged: farther than light goes in a month.)
    with np.errstate(invalid="ignore"):
        bad |= ~(np.abs(r_vec) < 1e15).all(axis=1)

    r_vec[bad] = np.nan
    v_vec[bad] = np.nan

    return r_vec, v_vec


def _propagate_many(
    position,
    velocity,
    mu: float,
    times
) -> tuple[np.ndarray, np.ndarray]:

    r0_vec = np.asarray(position, dtype=np.float64)
    v0_vec = np.asarray(velocity, dtype=np.float64)

    dt = np.asarray(times, dtype=np.float64).copy()

    r0 = float(np.linalg.norm(r0_vec))
    v0 = float(np.linalg.norm(v0_vec))

    if r0 == 0.0:
        return np.tile(r0_vec, (len(dt), 1)), np.tile(v0_vec, (len(dt), 1))

    vr0 = float(np.dot(r0_vec, v0_vec)) / r0

    alpha = 2.0 / r0 - v0 * v0 / mu

    if alpha > 1e-12:
        period = 2.0 * math.pi / math.sqrt(mu * alpha ** 3)
        dt = np.fmod(dt, period)

    sqrt_mu = math.sqrt(mu)

    x = _first_guess(r0_vec, v0_vec, r0, alpha, mu, dt)

    for _ in range(60):

        z = alpha * x * x

        c, s = _stumpff_arrays(z)

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

        x = x - step

        if np.all(~np.isfinite(step) | (np.abs(step) < 1e-10 * np.maximum(1.0, np.abs(x)))):
            break

    z = alpha * x * x

    c, s = _stumpff_arrays(z)

    f = 1.0 - x * x / r0 * c
    g = dt - x ** 3 / sqrt_mu * s

    r_vec = f[:, None] * r0_vec[None, :] + g[:, None] * v0_vec[None, :]

    r = np.linalg.norm(r_vec, axis=1)

    f_dot = sqrt_mu / (r * r0) * (alpha * x ** 3 * s - x)
    g_dot = 1.0 - x * x / r * c

    v_vec = f_dot[:, None] * r0_vec[None, :] + g_dot[:, None] * v0_vec[None, :]

    return r_vec, v_vec


def time_to_radius(
    position,
    velocity,
    mu: float,
    radius: float
) -> float | None:
    """
    Seconds until the vessel, outbound, first reaches
    `radius` from the center (0 if already beyond); None if
    its conic never gets that far.
    """

    r_vec = np.asarray(position, dtype=np.float64)
    v_vec = np.asarray(velocity, dtype=np.float64)

    r = float(np.linalg.norm(r_vec))

    if r >= radius:
        return 0.0

    h_vec = np.cross(r_vec, v_vec)

    h2 = float(np.dot(h_vec, h_vec))

    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r

    e = float(np.linalg.norm(e_vec))

    if e < 1e-9 or h2 <= 0.0:
        return None

    p = h2 / mu

    cos_target = (p / radius - 1.0) / e

    if cos_target < -1.0:
        return None

    def anomaly(r_now: float, outbound: bool) -> float:

        cos_nu = min(max((p / r_now - 1.0) / e, -1.0), 1.0)

        nu = math.acos(cos_nu)

        return nu if outbound else -nu

    def time_since_periapsis(nu: float) -> float:

        if e < 1.0:

            a = p / (1.0 - e * e)

            big_e = 2.0 * math.atan(math.sqrt((1.0 - e) / (1.0 + e)) * math.tan(nu / 2.0))

            return (big_e - e * math.sin(big_e)) / math.sqrt(mu / a ** 3)

        a = p / (e * e - 1.0)

        f = 2.0 * math.atanh(min(math.sqrt((e - 1.0) / (e + 1.0)) * math.tan(nu / 2.0), 1.0 - 1e-15))

        return (e * math.sinh(f) - f) / math.sqrt(mu / a ** 3)

    now = anomaly(r, float(np.dot(r_vec, v_vec)) >= 0.0)

    target = math.acos(min(max(cos_target, -1.0), 1.0))

    return max(time_since_periapsis(target) - time_since_periapsis(now), 0.0)


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
