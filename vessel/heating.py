import math

import numpy as np


# =========================================================
# Aerodynamic Heating
# =========================================================
#
# Fast through the air, the air ahead of the vessel is
# squeezed and heated; the part facing the flow takes it.
# Its heat flux at the stagnation point (Sutton and Graves):
#
#   q = k sqrt(rho / r_n) v^3     W/m^2
#
# (rho the air's density, r_n the nose's radius, v the speed
# through the air): ~1.6 MW/m^2 coming back from low Earth
# orbit, ~3x that at the Moon's return speed (v^3). Each
# part's skin has its own temperature: the
# part leading (the top of the stack, nose first; the bottom
# of it, tail first) takes the flux over its face, every
# part a little along its side when the flow comes at an
# angle; all of them radiate (sigma eps T^4) back out. A
# part hotter than it can stand burns up, and with it the
# vessel: a heat shield (rated ~3,300 K) under the capsule
# brings it home.
#
# The skin heats, not the whole part: an areal heat
# capacity per m^2 of skin (a few mm of metal), so heating
# and cooling take seconds to minutes, not hours.

# Sutton-Graves constant for air (kg^0.5 / m).
SUTTON_GRAVES = 1.7415e-4

STEFAN_BOLTZMANN = 5.670374419e-8

EMISSIVITY = 0.8

# J / (K m^2) of skin.
SKIN_HEAT_CAPACITY = 25_000.0

# Along the side of a part, at an angle to the flow, a
# fraction of the stagnation flux.
SIDE_SHARE = 0.08

# A nose no sharper than this (m): flux grows as the
# radius shrinks, and a point's flux is not infinite.
MIN_NOSE_RADIUS = 0.2

# What a skin settles to in space (K): sunlight on one side,
# the dark on the other.
SPACE_TEMPERATURE = 250.0


def stagnation_flux(
    density: float,
    speed: float,
    nose_radius: float
) -> float:
    """W/m^2 at the front of a body through air (Sutton-Graves)."""

    if density <= 0.0 or speed <= 0.0:
        return 0.0

    return SUTTON_GRAVES * math.sqrt(density / max(nose_radius, MIN_NOSE_RADIUS)) * speed ** 3


def equilibrium_temperature(
    flux: float,
    ambient: float = 3.0
) -> float:
    """Where a face taking `flux` W/m^2 radiates it all back out (K)."""

    return (flux / (STEFAN_BOLTZMANN * EMISSIVITY) + ambient ** 4) ** 0.25


def heat(
    vessel,
    up: np.ndarray,
    air_velocity: np.ndarray,
    density: float,
    ambient: float,
    dt: float
) -> object | None:
    """
    Heat and cool every part's skin over `dt` s. up: the
    stack's axis (inertial); air_velocity: the vessel's
    through the air. Returns the first part past its limit
    (it burns up), else None.
    """

    if not vessel.parts:
        return None

    speed = float(np.linalg.norm(air_velocity))

    core = vessel.core

    # Which end meets the air: the top moving nose first.
    cosine = float(np.dot(up, air_velocity)) / speed if speed > 1e-6 else 0.0

    leading = None
    face = 0.0
    nose = 1.0

    if core:

        if cosine >= 0.0:
            leading = core[0]
            nose = leading.part.radius_top
        else:
            leading = core[-1]
            nose = leading.part.radius_bottom

        face = math.pi * max(leading.part.radius, nose) ** 2

    # The flux on a blunt face: the part's own radius sets
    # the nose's when the end is flat; a cone's tip is
    # sharper.
    flux = stagnation_flux(density, speed, max(nose, 0.5 * (leading.part.radius if leading else 1.0)))

    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))

    # (For the glow of coming in.)
    vessel.heat_flux = flux * abs(cosine)
    vessel.heat_end = "top" if cosine >= 0.0 else "bottom"

    burned = None

    for state in vessel.parts:

        part = state.part

        side = 2.0 * math.pi * part.radius * part.height

        skin = side + math.pi * part.radius ** 2

        gained = flux * SIDE_SHARE * sine * 0.5 * side

        if state is leading:
            gained += flux * abs(cosine) * face

        lost = STEFAN_BOLTZMANN * EMISSIVITY * skin * (state.temperature ** 4 - ambient ** 4)

        capacity = SKIN_HEAT_CAPACITY * skin

        change = (gained - lost) / capacity * dt

        # (Stiff when hot: never past where it would settle.)
        if change < 0.0:
            state.temperature = max(state.temperature + change, ambient)
        else:
            settle = ((gained / (STEFAN_BOLTZMANN * EMISSIVITY * skin)) + ambient ** 4) ** 0.25
            state.temperature = min(state.temperature + change, max(settle, state.temperature))

        if burned is None and state.temperature > part.max_temperature:
            burned = state

    return burned


def hottest(
    vessel
) -> object | None:
    """The part nearest its limit (by share of it)."""

    return max(vessel.parts, key=lambda state: state.temperature / state.part.max_temperature, default=None)
