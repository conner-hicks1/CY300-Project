import json
import math

from dataclasses import dataclass
from pathlib import Path


# =========================================================
# Science Sites
# =========================================================
#
# Real places worth going to (assets/science/sites.json):
# Gale crater's old lake bed, Jezero's delta, the Moon's
# shadowed polar craters, Olympus Mons... each with where it
# is (planetocentric latitude, east longitude, its extent),
# why it matters, and how much more experiments there are
# worth. And how much each body is worth overall: the harder
# to reach, the more.

SITES_PATH = Path("assets/science/sites.json")


@dataclass(frozen=True, slots=True)
class Site:

    id: str
    body: str                       # body profile id
    name: str
    latitude: float                 # deg
    longitude: float                # deg east
    radius: float                   # km
    about: str
    value: float = 1.0


def angular_distance(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float
) -> float:
    """Angle between two points on a sphere (rad; haversine)."""

    phi_a = math.radians(latitude_a)
    phi_b = math.radians(latitude_b)

    d_phi = phi_b - phi_a
    d_lambda = math.radians(longitude_b - longitude_a)

    h = math.sin(d_phi / 2.0) ** 2 + math.cos(phi_a) * math.cos(phi_b) * math.sin(d_lambda / 2.0) ** 2

    return 2.0 * math.asin(min(1.0, math.sqrt(h)))


@dataclass(slots=True)
class SiteCatalog:

    sites: dict[str, list[Site]]
    factors: dict[str, float]
    default_factor: float = 5.0

    def factor(
        self,
        body: str
    ) -> float:
        """How much science the body is worth (Earth little, the outer planets a lot)."""

        return self.factors.get(body, self.default_factor)

    def of(
        self,
        body: str
    ) -> list[Site]:

        return self.sites.get(body, [])

    def nearest(
        self,
        body: str,
        latitude: float,
        longitude: float,
        radius_m: float
    ) -> tuple[Site, float] | None:
        """The closest site on the body and how far it is over the surface (m), or None."""

        best = None

        for site in self.of(body):

            distance = angular_distance(latitude, longitude, site.latitude, site.longitude) * radius_m

            if best is None or distance < best[1]:
                best = (site, distance)

        return best

    def site_at(
        self,
        body: str,
        latitude: float,
        longitude: float,
        radius_m: float
    ) -> Site | None:
        """The site the point is in (the smallest, where they overlap), or None."""

        inside = [
            site
            for site in self.of(body)
            if angular_distance(latitude, longitude, site.latitude, site.longitude) * radius_m <= site.radius * 1000.0
        ]

        return min(inside, key=lambda site: site.radius) if inside else None


def load_sites(
    path: Path = SITES_PATH
) -> SiteCatalog:

    data = json.loads(Path(path).read_text(encoding="utf-8"))

    sites: dict[str, list[Site]] = {}

    for raw in data.get("sites", []):

        site = Site(
            id=str(raw["id"]),
            body=str(raw["body"]),
            name=str(raw.get("name", raw["id"])),
            latitude=float(raw["latitude"]),
            longitude=float(raw["longitude"]),
            radius=float(raw.get("radius", 50.0)),
            about=str(raw.get("about", "")),
            value=float(raw.get("value", 1.0))
        )

        if not -90.0 <= site.latitude <= 90.0:
            raise ValueError(f"site '{site.id}': latitude out of range")

        sites.setdefault(site.body, []).append(site)

    return SiteCatalog(
        sites=sites,
        factors={str(key): float(value) for key, value in data.get("factors", {}).items()},
        default_factor=float(data.get("default_factor", 5.0))
    )
