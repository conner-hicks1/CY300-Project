"""
Download the real elevation and color maps (planet/maps.py)
into data/maps/ and build their cube-sphere caches.

    python tools/fetch_maps.py            # everything (~86 MB)
    python tools/fetch_maps.py mars moon  # some bodies
    python tools/fetch_maps.py --sharp    # also the 4x sharper maps (~1.5 GB)
    python tools/fetch_maps.py --list

Files already there are skipped. Sources are public domain
or free to redistribute; see each dataset's credit.
"""

import sys
import time
import urllib.request

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planet.maps import DATASETS, MAPS_DIRECTORY, load_elevation_map   # noqa: E402


def download(
    url: str,
    target: Path
):

    partial = target.with_suffix(target.suffix + ".part")

    request = urllib.request.Request(url, headers={"User-Agent": "CY300-engine-map-fetcher"})

    with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as out:

        total = int(response.headers.get("Content-Length") or 0)

        done = 0
        last = 0.0

        while True:

            block = response.read(1 << 20)

            if not block:
                break

            out.write(block)

            done += len(block)

            if time.monotonic() - last > 1.0:

                last = time.monotonic()

                share = f" ({100.0 * done / total:.0f}%)" if total else ""

                print(f"    {done / 1e6:6.1f} MB{share}", flush=True)

    partial.replace(target)


def main(
    arguments: list[str]
) -> int:

    if "--list" in arguments:

        for dataset in DATASETS.values():

            state = "downloaded" if dataset.available else "missing"

            print(f"{dataset.id:20} {dataset.body:6} {dataset.kind:9} {dataset.size_hint:>6}  {state}")
            print(f"    {dataset.description}")
            print(f"    {dataset.credit}")

        return 0

    sharp = "--sharp" in arguments

    bodies = {a for a in arguments if not a.startswith("--")}

    MAPS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    failed = False

    for dataset in DATASETS.values():

        if bodies and dataset.body not in bodies:
            continue

        # The sharper maps (~0.5 GB each) only when asked.
        if dataset.kind == "detail" and not sharp:
            continue

        print(f"{dataset.id}: {dataset.description}")

        for url, name in dataset.files:

            target = MAPS_DIRECTORY / name

            if target.is_file():

                print(f"  {name}: already here")

                continue

            print(f"  {name} <- {url}")

            try:
                download(url, target)
            except OSError as error:

                print(f"  FAILED: {error}")

                failed = True

                break

        if dataset.kind == "elevation" and dataset.available:

            started = time.perf_counter()

            if load_elevation_map(dataset.id) is not None:
                print(f"  cube-sphere cache ready ({time.perf_counter() - started:.1f} s)")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
