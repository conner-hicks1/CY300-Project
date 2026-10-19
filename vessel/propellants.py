from dataclasses import dataclass


# =========================================================
# Propellants
# =========================================================
#
# Each tank holds one propellant (fuel and oxidizer together)
# and each engine burns one: an engine draws only on tanks of
# its own kind in its section.
#
#   kerolox     kerosene and liquid oxygen: dense and cheap,
#               the boosters' (Isp ~300-350 s); its oxygen
#               boils off slowly
#   methalox    methane and liquid oxygen: a little better,
#               and what a plant can make from Mars's air
#   hydrolox    liquid hydrogen and oxygen: by far the best
#               (~450 s in vacuum) but light (big tanks for
#               its mass) and boiling off fast in sunlight:
#               a quick departure, not a long cruise; made
#               from water ice
#   hypergolic  storable: they ignite on contact, keep for
#               years (~320 s): landers, service modules, a
#               long trip's last burn
#
# Boil-off (a share of what is aboard a day) is for a tank in
# full sunlight at 1 AU; in shadow, or far from the Sun, a
# tank warms less (an inverse square, with a floor for what
# comes through the structure). Insulated tanks boil less; a
# cryocooler (drawing power) keeps some tonnes from boiling
# at all.

# Of the boil-off in sunlight, the share that remains in the
# dark (heat through the structure, the warm vessel).
DARK_SHARE = 0.2


@dataclass(frozen=True, slots=True)
class Propellant:

    id: str
    name: str
    short: str                      # for the HUD: "RP-1/LOX"
    density: float                  # kg/m^3, fuel and oxidizer together
    boil: float                     # share a day, sunlit at 1 AU
    cryogenic: bool


PROPELLANTS = {
    "kerolox": Propellant("kerolox", "Kerosene / liquid oxygen", "RP-1/LOX", 1030.0, 0.001, True),
    "methalox": Propellant("methalox", "Methane / liquid oxygen", "CH4/LOX", 830.0, 0.003, True),
    "hydrolox": Propellant("hydrolox", "Liquid hydrogen / oxygen", "LH2/LOX", 360.0, 0.015, True),
    "hypergolic": Propellant("hypergolic", "Hypergolic (storable)", "MMH/NTO", 1180.0, 0.0, False),
}

DEFAULT = "kerolox"


def of(
    kind: str
) -> Propellant:

    return PROPELLANTS[kind]


def heat_share(
    sunlight: float,
    flux: float
) -> float:
    """How warm a tank gets (1: in full sunlight at 1 AU)."""

    return DARK_SHARE + (1.0 - DARK_SHARE) * max(sunlight, 0.0) * max(flux, 0.0)


def tanks_of(
    vessel,
    kind: str | None = None
) -> list:
    """Tanks (of a propellant: None, any)."""

    return [
        state
        for state in vessel.parts
        if state.part.propellant > 0.0 and (kind is None or state.part.propellant_type == kind)
    ]


def held(
    vessel,
    kind: str
) -> tuple[float, float]:
    """(kg aboard, capacity) of a propellant, every copy counted."""

    tanks = tanks_of(vessel, kind)

    return (
        sum(state.propellant * state.count for state in tanks),
        sum(state.part.propellant * state.count for state in tanks)
    )


def kinds_aboard(
    vessel
) -> list[str]:

    return sorted({state.part.propellant_type for state in tanks_of(vessel)})


def boil_rate(
    vessel,
    sunlight: float,
    flux: float
) -> float:
    """kg a day boiling off now (a cryocooler working, less)."""

    warm = heat_share(sunlight, flux)

    # Cooling, kg of cryogenic propellant kept cold.
    cooling = sum(state.part.cooling * state.count for state in vessel.parts) if vessel.powered else 0.0

    total = 0.0

    # The coldest (fastest boiling) cooled first.
    tanks = sorted(
        (state for state in tanks_of(vessel) if of(state.part.propellant_type).boil > 0.0 and state.propellant > 0.0),
        key=lambda state: -of(state.part.propellant_type).boil
    )

    for state in tanks:

        mass = state.propellant * state.count

        kept = min(cooling, mass)

        cooling -= kept

        total += (mass - kept) * of(state.part.propellant_type).boil * state.part.insulation * warm

    return total


def boil(
    vessel,
    sunlight: float,
    flux: float,
    dt: float
) -> float:
    """`dt` s of boil-off from the tanks (each its own share); kg lost."""

    if dt <= 0.0:
        return 0.0

    warm = heat_share(sunlight, flux)

    cooling = sum(state.part.cooling * state.count for state in vessel.parts) if vessel.powered else 0.0

    days = dt / 86_400.0

    lost = 0.0

    tanks = sorted(
        (state for state in tanks_of(vessel) if of(state.part.propellant_type).boil > 0.0 and state.propellant > 0.0),
        key=lambda state: -of(state.part.propellant_type).boil
    )

    for state in tanks:

        mass = state.propellant * state.count

        kept = min(cooling, mass)

        cooling -= kept

        exposed = (mass - kept) / mass

        rate = of(state.part.propellant_type).boil * state.part.insulation * warm * exposed

        # (Exactly, for long steps: what is left of it.)
        share = 1.0 - (1.0 - min(rate, 1.0)) ** days

        taken = state.propellant * share

        state.propellant -= taken

        lost += taken * state.count

    return lost
