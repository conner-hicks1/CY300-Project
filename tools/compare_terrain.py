"""
Check the terrain generators against the real bodies: the
same statistics of the measured maps (planet/maps.py) and of
the generated terrain each body gets from its profile.

    python tools/compare_terrain.py              # Mars, Moon, Earth
    python tools/compare_terrain.py mars --image docs/images/terrain_check.png

Statistics (all at the map's resolution):

    hypsometry   how much of the surface lies at each height
                 (percentiles), and the land fraction on
                 worlds with seas
    roughness    RMS height difference between points a
                 distance apart, 10 km to 2000 km (the
                 structure function: how relief grows with
                 scale)
    slopes       median and 95th percentile slope between
                 neighbouring samples
"""

import argparse
import math
import sys

from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planet.bodies import components_for, load_presets                 # noqa: E402
from planet.maps import DATASETS, load_elevation_map                    # noqa: E402
from planet.shape import fibonacci_directions                           # noqa: E402
from planet.tectonics import TectonicField, simulation_for              # noqa: E402
from planet.terrain import Terrain                                      # noqa: E402
from systems.planet_system import terrain_settings_for                  # noqa: E402
from systems.tectonics_system import tectonic_settings_for              # noqa: E402

SCALES_KM = (10, 30, 100, 300, 1000, 2000)
PERCENTILES = (2, 10, 25, 50, 75, 90, 98)


def terrains(
    profile
) -> tuple[Terrain, Terrain]:
    """(measured, generated) terrain of a body, as scenes build them."""

    parts = components_for(profile)

    field = None

    if parts.tectonics is not None:

        simulation = simulation_for(tectonic_settings_for(parts.planet, parts.tectonics))

        field = TectonicField.from_state(simulation.grid, simulation.initial_state(), version=0)

    generated = Terrain(terrain_settings_for(parts.planet), field)

    parts.planet.elevation_map = profile.maps[0]

    measured = Terrain(terrain_settings_for(parts.planet))

    return measured, generated


def heights(
    terrain: Terrain,
    directions: np.ndarray,
    spacing: float
) -> np.ndarray:

    out = np.empty(len(directions))

    for start in range(0, len(directions), 50_000):
        out[start:start + 50_000] = terrain.elevation(directions[start:start + 50_000], spacing)

    return out


def tangent_offsets(
    directions: np.ndarray,
    angle: float,
    rng: np.random.Generator
) -> np.ndarray:
    """Points `angle` (rad) away from each direction, in random directions."""

    random = rng.normal(size=directions.shape)

    tangent = random - np.einsum("ij,ij->i", random, directions)[:, None] * directions
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)

    return directions * math.cos(angle) + tangent * math.sin(angle)


def statistics(
    terrain: Terrain,
    radius: float,
    spacing: float,
    count: int,
    seed: int = 7
) -> dict:

    rng = np.random.default_rng(seed)

    directions = fibonacci_directions(count)

    h = heights(terrain, directions, spacing)

    result = {
        "percentiles": np.percentile(h, PERCENTILES),
        "land": float(np.mean(h > 0.0)),
    }

    sample = directions[:: max(1, count // 40_000)]

    base = heights(terrain, sample, spacing)

    roughness = []

    for scale in SCALES_KM:

        other = heights(terrain, tangent_offsets(sample, scale * 1000.0 / radius, rng), spacing)

        roughness.append(float(np.sqrt(np.mean((other - base) ** 2))))

    result["roughness"] = roughness

    neighbour = heights(terrain, tangent_offsets(sample, 2.0 * spacing / radius, rng), spacing)

    slope = np.degrees(np.arctan(np.abs(neighbour - base) / (2.0 * spacing)))

    result["slopes"] = (float(np.median(slope)), float(np.percentile(slope, 95)))

    return result


def shaded_map(
    terrain: Terrain,
    radius: float,
    spacing: float,
    width: int = 512
) -> np.ndarray:
    """Equirectangular relief map (0..1): height colors, lit from the north-west."""

    height = width // 2

    lat = np.radians(90.0 - (np.arange(height) + 0.5) * 180.0 / height)
    lon = np.radians(-180.0 + (np.arange(width) + 0.5) * 360.0 / width)

    lon_grid, lat_grid = np.meshgrid(lon, lat)

    directions = np.stack(
        (np.cos(lat_grid) * np.sin(lon_grid), np.sin(lat_grid), np.cos(lat_grid) * np.cos(lon_grid)),
        axis=-1
    ).reshape(-1, 3)

    h = heights(terrain, directions, spacing).reshape(height, width)

    # Slopes (m per m): a pixel spans pi R / height north-south.
    pixel = math.pi * radius / height

    north = -np.gradient(h, axis=0) / pixel
    east = np.gradient(h, axis=1) / (pixel * np.maximum(np.cos(lat_grid), 0.05))

    # Exaggerated 15x; light from the north-west, 45 deg up.
    exaggeration = 15.0

    nx, ny, nz = -east * exaggeration, -north * exaggeration, np.ones_like(h)

    light = np.array([-1.0, 1.0, 1.4]) / math.sqrt(1.0 + 1.0 + 1.96)

    shade = (nx * light[0] + ny * light[1] + nz * light[2]) / np.sqrt(nx * nx + ny * ny + nz * nz)

    low, high = np.percentile(h, (2, 98))

    t = np.clip((h - low) / max(high - low, 1.0), 0.0, 1.0)[..., None]

    color = (1.0 - t) * np.array([0.16, 0.3, 0.55]) + t * np.array([0.92, 0.82, 0.62])

    return np.clip(color * (0.35 + 0.85 * np.clip(shade, 0.0, 1.0))[..., None], 0.0, 1.0)


def main(
    arguments: list[str]
) -> int:

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("bodies", nargs="*", default=["mars", "moon", "earth"])
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--image", type=Path, default=None, help="save real vs generated maps")
    options = parser.parse_args(arguments)

    presets = load_presets()

    rows = []

    for body in options.bodies:

        profile = presets[body]

        dataset = profile.maps[0]

        if not dataset or load_elevation_map(dataset) is None:

            print(f"{profile.name}: no measured elevation (python tools/fetch_maps.py {body})")

            continue

        measured, generated = terrains(profile)

        radius = profile.radius_m

        spacing = measured.elevation_map.resolution(radius)

        print(f"\n{profile.name}  ({DATASETS[dataset].description})")
        print(f"  samples {options.samples:,}, spacing {spacing / 1000.0:.1f} km")

        stats = {
            "measured": statistics(measured, radius, spacing, options.samples),
            "generated": statistics(generated, radius, spacing, options.samples),
        }

        print("  hypsometry (km at percentiles " + ", ".join(map(str, PERCENTILES)) + ")")

        for name, s in stats.items():
            print(f"    {name:9}  " + "  ".join(f"{v / 1000.0:6.2f}" for v in s["percentiles"]))

        if profile.liquid != "none":

            for name, s in stats.items():
                print(f"    {name:9}  land {100.0 * s['land']:.0f}%")

        print("  roughness (RMS height difference, km, at " + ", ".join(f"{s} km" for s in SCALES_KM) + ")")

        for name, s in stats.items():
            print(f"    {name:9}  " + "  ".join(f"{v / 1000.0:6.2f}" for v in s["roughness"]))

        print("  slopes (median, 95%, deg)")

        for name, s in stats.items():
            print(f"    {name:9}  {s['slopes'][0]:5.1f}  {s['slopes'][1]:5.1f}")

        rows.append((profile.name, measured, generated, radius, spacing))

    if options.image is not None and rows:

        from PIL import Image

        tiles = []

        for name, measured, generated, radius, spacing in rows:

            tiles.append(np.concatenate(
                (shaded_map(measured, radius, spacing), shaded_map(generated, radius, spacing)),
                axis=1
            ))

        image = np.concatenate(tiles, axis=0)

        options.image.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray((image * 255.0).astype(np.uint8)).save(options.image)

        print(f"\nSaved {options.image} (rows: " + ", ".join(r[0] for r in rows) + "; left measured, right generated)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
