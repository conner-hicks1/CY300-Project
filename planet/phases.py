import math

from dataclasses import dataclass


# =========================================================
# Phases of Volatiles
# =========================================================
#
# When a substance on a body's surface is solid, liquid or
# gas. Each substance has a triple point (below its
# pressure there is no liquid at all, only ice and vapor)
# and a critical point (above its temperature no liquid
# surface either: a supercritical fluid). Between them the
# boiling point follows the vapor-pressure curve, from the
# Clausius-Clapeyron relation:
#
#     ln(P / P_triple) = -(L / R) (1 / T - 1 / T_triple)
#
# with L the latent heat of vaporization (above the triple
# point) or sublimation (below it, for frost).
#
# That decides:
#   - whether a body's seas are open, frozen or boiled away
#     (Earth moved to Venus's orbit loses its oceans;
#     water on Mars cannot stay liquid at 6 mbar)
#   - where ice caps form: a gas frosts out where the
#     ground is colder than the temperature at which its
#     vapor pressure equals its partial pressure in the air
#     (carbon dioxide on Mars at ~-127 C, nitrogen on Pluto
#     at ~-236 C)

GAS_CONSTANT = 8.314        # J / (mol K)
KELVIN = 273.15


@dataclass(frozen=True, slots=True)
class Substance:

    name: str

    # Gas formula in atmosphere compositions
    # (planet/bodies.py GASES).
    gas: str

    triple_k: float
    triple_bar: float
    critical_k: float
    critical_bar: float

    # J / mol.
    vaporization_heat: float
    sublimation_heat: float

    # Where the liquid freezes (C). Below the pure
    # substance's triple point for mixtures: sea water
    # (salt), Titan's methane-ethane-nitrogen seas.
    freezing_c: float


SUBSTANCES: dict[str, Substance] = {
    "water": Substance(
        name="water",
        gas="H2O",
        triple_k=273.16,
        triple_bar=0.00611657,
        critical_k=647.1,
        critical_bar=220.64,
        vaporization_heat=43_300.0,
        sublimation_heat=51_060.0,
        freezing_c=-1.9
    ),
    "methane": Substance(
        name="methane",
        gas="CH4",
        triple_k=90.69,
        triple_bar=0.117,
        critical_k=190.6,
        critical_bar=46.0,
        vaporization_heat=8_190.0,
        sublimation_heat=9_700.0,
        # Mixed with ethane and dissolved nitrogen it stays
        # liquid far below pure methane's 91 K (Titan's
        # polar lakes at ~90 K).
        freezing_c=-200.0
    ),
    "nitrogen": Substance(
        name="nitrogen",
        gas="N2",
        triple_k=63.15,
        triple_bar=0.1253,
        critical_k=126.2,
        critical_bar=34.0,
        vaporization_heat=5_570.0,
        sublimation_heat=6_900.0,
        freezing_c=63.15 - KELVIN
    ),
    "carbon_dioxide": Substance(
        name="carbon_dioxide",
        gas="CO2",
        triple_k=216.58,
        triple_bar=5.185,
        critical_k=304.13,
        critical_bar=73.8,
        vaporization_heat=15_300.0,
        sublimation_heat=25_230.0,
        freezing_c=216.58 - KELVIN
    ),
}

# Ices a surface can be capped with ("none": no frost).
ICES = ("none", *SUBSTANCES)


def substance(
    name: str
) -> Substance | None:

    return SUBSTANCES.get(name)


def vapor_pressure(
    s: Substance,
    temperature_c: float
) -> float:
    """
    Vapor pressure (bar) over the liquid (above the triple
    point) or the ice (below it).
    """

    kelvin = max(temperature_c + KELVIN, 1.0)

    heat = s.vaporization_heat if kelvin >= s.triple_k else s.sublimation_heat

    return s.triple_bar * math.exp(
        -heat / GAS_CONSTANT * (1.0 / kelvin - 1.0 / s.triple_k)
    )


def _curve_temperature(
    s: Substance,
    pressure_bar: float,
    heat: float
) -> float:
    """Temperature (C) where the curve reaches this pressure."""

    inverse = 1.0 / s.triple_k - GAS_CONSTANT / heat * math.log(pressure_bar / s.triple_bar)

    return 1.0 / inverse - KELVIN


def boiling_point(
    s: Substance,
    pressure_bar: float
) -> float | None:
    """
    Boiling point (C) at this surface pressure; None below
    the triple point pressure (no liquid can exist). Above
    the critical pressure, the critical temperature (hotter
    than that there is no liquid surface).
    """

    if pressure_bar < s.triple_bar:
        return None

    if pressure_bar >= s.critical_bar:
        return s.critical_k - KELVIN

    return min(
        _curve_temperature(s, pressure_bar, s.vaporization_heat),
        s.critical_k - KELVIN
    )


def frost_point(
    s: Substance,
    partial_pressure_bar: float
) -> float:
    """
    Temperature (C) below which this gas, at this partial
    pressure, condenses onto the ground as frost / ice.
    """

    if partial_pressure_bar <= 0.0:
        return -KELVIN

    if partial_pressure_bar >= s.triple_bar:
        # Condenses as a liquid first (rain, seas); ice where
        # colder than the freezing point (snow).
        return s.freezing_c

    return _curve_temperature(s, partial_pressure_bar, s.sublimation_heat)


@dataclass(frozen=True, slots=True)
class LiquidRange:

    # C. Seas freeze below `freezing` and boil away above
    # `boiling`. Without a liquid phase at the surface
    # pressure (below the triple point), anything there is
    # ice: freezing is +inf and boiling is where the ice
    # sublimates away.
    freezing: float
    boiling: float

    # False below the triple point pressure.
    can_be_liquid: bool = True


def liquid_range(
    liquid: str,
    pressure_bar: float
) -> LiquidRange | None:
    """
    Where a body's surface liquid stays liquid at this
    pressure. None for no liquid, and for lava (molten from
    volcanic heat, not the climate).
    """

    s = SUBSTANCES.get(liquid)

    if s is None:
        return None

    boiling = boiling_point(s, pressure_bar)

    if boiling is None:

        # Ice that sublimates instead of melting; it lasts
        # where colder than the frost point at a vapor
        # pressure of ~the whole atmosphere.
        return LiquidRange(
            freezing=math.inf,
            boiling=frost_point(s, max(pressure_bar, 1e-12)),
            can_be_liquid=False
        )

    return LiquidRange(freezing=s.freezing_c, boiling=boiling)
