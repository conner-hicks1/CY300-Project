import gzip
import math

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from planet.orbits import equatorial_to_engine


# =========================================================
# The Stars
# =========================================================
#
# The Yale Bright Star Catalog (5th revised edition, Hoffleit
# & Warren 1991, CDS V/50): the 9,110 stars to about
# magnitude 6.5, everything the eye sees on a dark night, at
# their J2000 positions. tools/fetch_stars.py downloads it
# (574 KB); without it the sky gets random stars with the
# same counts by magnitude (so space is not empty, but the
# constellations are not there).
#
# Brightness is physical, in the engine's units: the Sun at
# Earth (magnitude -26.74) is an illuminance of 5 (planet/
# bodies.py sun_intensity), so a star of magnitude m is
#
#     E = 5 * 10^(-0.4 (m + 26.74))
#
# (Sirius ~4e-10, about 8e-11 of sunlight), and the eye's dark
# adaptation (assets/shaders/exposure.frag.glsl) makes it
# visible, the way the night sky is.
#
# Star colors come from the B - V color index through the
# temperature (Ballesteros 2012) and the blackbody color.

STARS_DIRECTORY = Path("data/stars")

CATALOG_URL = "https://cdsarc.cds.unistra.fr/ftp/V/50/"
CATALOG_FILES = ("catalog.gz", "ReadMe")
CATALOG_CREDIT = "Yale Bright Star Catalog, 5th rev. ed. (Hoffleit & Warren 1991), via CDS Strasbourg"

SUN_MAGNITUDE = -26.74
SUN_ILLUMINANCE = 5.0           # engine units at Earth

# The Milky Way's north pole and center (J2000; IAU 1958
# galactic coordinates).
GALACTIC_POLE = (192.85948, 27.12825)       # RA, Dec (deg)
GALACTIC_CENTER = (266.40510, -28.93617)


@dataclass(frozen=True)
class StarCatalog:

    directions: np.ndarray          # (n, 3) unit, engine axes
    magnitudes: np.ndarray          # (n,) visual
    colors: np.ndarray              # (n, 3) linear rgb, luminance 1
    real: bool                      # False: the random stand-in

    def __len__(
        self
    ) -> int:

        return len(self.magnitudes)

    def illuminance(
        self
    ) -> np.ndarray:
        """(n,) engine units."""

        return magnitude_to_illuminance(self.magnitudes)


def magnitude_to_illuminance(
    magnitude
):
    """Illuminance (engine units) from an object of this visual magnitude."""

    return SUN_ILLUMINANCE * 10.0 ** (-0.4 * (np.asarray(magnitude, dtype=np.float64) - SUN_MAGNITUDE))


def illuminance_to_magnitude(
    illuminance
):

    return SUN_MAGNITUDE - 2.5 * np.log10(np.maximum(np.asarray(illuminance, dtype=np.float64), 1e-300) / SUN_ILLUMINANCE)


def temperature_of_color_index(
    b_minus_v
):
    """Effective temperature (K) from B - V (Ballesteros 2012)."""

    bv = np.asarray(b_minus_v, dtype=np.float64)

    return 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62))


def radec_to_engine(
    right_ascension_deg,
    declination_deg
) -> np.ndarray:
    """(n, 3) unit directions in engine axes."""

    ra = np.radians(np.asarray(right_ascension_deg, dtype=np.float64))
    dec = np.radians(np.asarray(declination_deg, dtype=np.float64))

    equatorial = np.stack(
        (np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)),
        axis=-1
    )

    return np.array([equatorial_to_engine(v) for v in np.atleast_2d(equatorial)])


def galactic_basis() -> np.ndarray:
    """
    3x3 matrix whose rows are the galactic x (toward the
    center), y (toward l = 90 deg) and z (north pole) axes in
    engine axes: galactic = basis @ engine direction.
    """

    pole = radec_to_engine(*GALACTIC_POLE)[0]
    center = radec_to_engine(*GALACTIC_CENTER)[0]

    # (The two are perpendicular to ~1e-6.)
    center = center - (center @ pole) * pole
    center /= np.linalg.norm(center)

    return np.stack((center, np.cross(pole, center), pole))


# ---------------------------------------------------------
# Loading
# ---------------------------------------------------------

def catalog_available(
    directory: Path = STARS_DIRECTORY
) -> bool:

    return all((directory / name).is_file() for name in CATALOG_FILES)


def _field(
    line: str,
    first: int,
    last: int
) -> str:
    """Bytes first..last (1-based, inclusive) of a fixed-width record."""

    return line[first - 1:last].strip()


def parse_catalog(
    lines
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    BSC5 records -> (right ascension deg, declination deg,
    V magnitude, B - V). Entries without a J2000 position or a
    magnitude (a few novae and non-stellar objects the
    catalog keeps numbered) are skipped; a missing B - V is
    taken as white (0.6).
    """

    ra = []
    dec = []
    magnitude = []
    color = []

    for line in lines:

        hours = _field(line, 76, 77)
        v = _field(line, 103, 107)

        if not hours or not v:
            continue

        ra.append(
            15.0 * (int(hours) + int(_field(line, 78, 79)) / 60.0 + float(_field(line, 80, 83)) / 3600.0)
        )

        degrees = int(_field(line, 85, 86)) + int(_field(line, 87, 88)) / 60.0 + int(_field(line, 89, 90)) / 3600.0

        dec.append(-degrees if _field(line, 84, 84) == "-" else degrees)

        magnitude.append(float(v))

        bv = _field(line, 110, 114)

        color.append(float(bv) if bv else 0.6)

    return np.array(ra), np.array(dec), np.array(magnitude), np.array(color)


def random_catalog(
    seed: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A stand-in with the real sky's statistics: ~9,000 stars
    to magnitude 6.5, their number rising ~3.4x per magnitude,
    more of them toward the Milky Way's plane. Returns
    (directions in engine axes, magnitudes, B - V).
    """

    rng = np.random.default_rng(seed)

    # Cumulative counts N(< m) ~ 10^(0.53 m) scaled to ~9,100
    # at 6.5.
    count = 9_100

    u = rng.random(count)

    magnitude = 6.5 + np.log10(np.maximum(u, 1e-9)) / 0.53

    magnitude = np.maximum(magnitude, -1.5)

    # Uniform on the sphere, then pulled toward the galactic
    # plane for half of them.
    z = rng.uniform(-1.0, 1.0, count)

    plane = rng.random(count) < 0.5

    z[plane] *= rng.random(plane.sum()) ** 2

    phi = rng.uniform(0.0, 2.0 * math.pi, count)

    r = np.sqrt(1.0 - z * z)

    galactic = np.stack((r * np.cos(phi), r * np.sin(phi), z), axis=-1)

    engine = galactic @ galactic_basis()

    color = np.clip(rng.normal(0.6, 0.45, count), -0.3, 2.0)

    return engine, magnitude, color


@lru_cache(maxsize=2)
def load_catalog(
    directory: Path = STARS_DIRECTORY
) -> StarCatalog:

    from planet.bodies import star_color

    if catalog_available(directory):

        with gzip.open(directory / CATALOG_FILES[0], "rt", encoding="latin-1") as file:
            ra, dec, magnitude, bv = parse_catalog(file)

        directions = radec_to_engine(ra, dec)

        real = True

    else:

        directions, magnitude, bv = random_catalog()

        real = False

    temperatures = temperature_of_color_index(bv)

    colors = np.array([star_color(float(t)) for t in temperatures], dtype=np.float64)

    luminance = colors @ np.array([0.2126, 0.7152, 0.0722])

    colors /= luminance[:, None]

    return StarCatalog(
        directions=directions.astype(np.float32),
        magnitudes=magnitude.astype(np.float32),
        colors=colors.astype(np.float32),
        real=real
    )
