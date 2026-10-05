"""
Download the Yale Bright Star Catalog (planet/stars.py) into
data/stars/: the 9,110 stars the eye sees, for the night sky.

    python tools/fetch_stars.py

574 KB from CDS Strasbourg; files already there are skipped.
"""

import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planet.stars import CATALOG_CREDIT, CATALOG_FILES, CATALOG_URL, STARS_DIRECTORY   # noqa: E402
from tools.fetch_maps import download                                                   # noqa: E402


def main() -> int:

    STARS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print(CATALOG_CREDIT)

    for name in CATALOG_FILES:

        target = STARS_DIRECTORY / name

        if target.is_file():

            print(f"  {name}: already there")

            continue

        print(f"  {name}: downloading")

        download(CATALOG_URL + name, target)

    return 0


if __name__ == "__main__":
    sys.exit(main())
