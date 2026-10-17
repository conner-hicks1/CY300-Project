from dataclasses import dataclass


# =========================================================
# Resources (making propellant away from home)
# =========================================================
#
# What a propellant plant (an "isru" part) can make
# propellant from where it lands:
#
#   ice   water ice, split into hydrogen and oxygen: the
#         Moon's and Mercury's shadowed polar craters, the
#         Martian polar caps, Ceres's ice-rich ground, the
#         icy moons' crusts (Europa, Ganymede, Callisto,
#         Enceladus, Titan, Triton), Pluto
#   air   carbon dioxide (Mars, Venus) turned to methane and
#         oxygen with hydrogen made on site, as the Sabatier
#         process does; Titan's methane air: slower
#
# richness scales a plant's rates (1: ordinary).

ICE_LATITUDE = 80.0
MARS_CAP_LATITUDE = 75.0

ICY_BODIES = ("ceres", "europa", "ganymede", "callisto", "enceladus", "titan", "triton", "pluto")

POLAR_ICE = {"moon": ICE_LATITUDE, "mercury": 85.0, "mars": MARS_CAP_LATITUDE}

ICE_SITES = {"lunar_south_pole": 1.5, "mercury_north_pole": 1.2, "planum_boreum": 1.5}

AIRS = {"mars": 1.0, "venus": 0.6, "titan": 0.5}


@dataclass(frozen=True, slots=True)
class Resource:

    kind: str                       # "ice" or "air"
    richness: float
    name: str


def resources_at(
    body: str,
    latitude: float,
    site: str = "",
    has_air: bool = False
) -> list[Resource]:
    """What there is to make propellant from, landed here."""

    found = []

    if site in ICE_SITES:
        found.append(Resource("ice", ICE_SITES[site], "water ice (shadowed craters)"))
    elif body in POLAR_ICE and abs(latitude) >= POLAR_ICE[body]:
        found.append(Resource("ice", 1.0, "polar water ice"))
    elif body in ICY_BODIES:
        found.append(Resource("ice", 0.8 if body == "ceres" else 1.0, "water ice in the ground"))

    if has_air and body in AIRS:
        found.append(Resource("air", AIRS[body], "carbon dioxide air" if body != "titan" else "methane air"))

    return found


def production(
    resources: list[Resource],
    rate_ice: float,
    rate_air: float
) -> float:
    """kg a day a plant makes from the best resource here."""

    best = 0.0

    for resource in resources:

        rate = rate_ice if resource.kind == "ice" else rate_air

        best = max(best, rate * resource.richness)

    return best
