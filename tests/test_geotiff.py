import math

from dataclasses import replace

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

import planet.maps as maps
import planet.terrain as terrain_module

from planet.bodies import load_presets
from planet.maps import (
    DATASETS,
    ElevationMap,
    color_image_from_geotiff,
    cube_from_grid,
    fill_gaps,
    load_elevation_map,
    read_dataset,
    read_geotiff,
    sample_grid
)
from planet.shape import direction_of, fibonacci_directions
from planet.terrain import Terrain, TerrainSettings


RADIUS = 1_000_000.0


def write_geotiff(
    path,
    values,
    center=180.0,
    scale=1.0,
    offset=0.0,
    nodata=None
):
    """A global simple-cylindrical GeoTIFF as USGS Astrogeology writes them."""

    rows, columns = values.shape[-2:]

    meters = 2.0 * math.pi * RADIUS / columns

    keys = [
        1, 1, 0, 3,
        1024, 0, 1, 1,                  # GTModelType: projected
        2057, 34736, 1, 0,              # GeogSemiMajorAxis
        3088, 34736, 1, 1,              # ProjCenterLong
    ]

    extratags = [
        (33550, "d", 3, (meters, meters, 0.0), False),
        (33922, "d", 6, (0.0, 0.0, 0.0, -math.pi * RADIUS, 0.5 * math.pi * RADIUS, 0.0), False),
        (34735, "H", len(keys), keys, False),
        (34736, "d", 2, (RADIUS, center), False),
        (
            42112, "s", 0,
            f'<GDALMetadata><Item name="OFFSET" sample="0" role="offset">{offset}</Item>'
            f'<Item name="SCALE" sample="0" role="scale">{scale}</Item></GDALMetadata>',
            False
        ),
    ]

    if nodata is not None:
        extratags.append((42113, "s", 0, str(nodata), False))

    tifffile.imwrite(
        path,
        values,
        photometric="rgb" if values.ndim == 3 else "minisblack",
        planarconfig="separate" if values.ndim == 3 else None,
        extratags=extratags
    )


def longitudes(columns, left):

    return left + (np.arange(columns) + 0.5) * 360.0 / columns


# =========================================================
# Reading
# =========================================================

def test_bounds_and_values(tmp_path):

    values = np.arange(8 * 16, dtype=np.int16).reshape(8, 16)

    write_geotiff(tmp_path / "a.tif", values, center=180.0, scale=0.5, offset=100.0, nodata=-32768)

    tiff = read_geotiff(tmp_path / "a.tif")

    # x = 0 at the center longitude: from -180 m * pi R ->
    # 0 deg east to 360.
    assert tiff.left == pytest.approx(0.0)
    assert tiff.right == pytest.approx(360.0)
    assert tiff.top == pytest.approx(90.0)
    assert tiff.bottom == pytest.approx(-90.0)

    assert (tiff.scale, tiff.offset, tiff.nodata) == (0.5, 100.0, -32768.0)

    np.testing.assert_array_equal(np.asarray(tiff.values), values)


def test_heights_with_gaps_and_datum(tmp_path, monkeypatch):

    rows, columns = 90, 180

    lon = longitudes(columns, 0.0)
    lat = 90.0 - (np.arange(rows) + 0.5) * 180.0 / rows

    # 1000 m per unit of sin(longitude), stored doubled.
    heights = 1_000.0 * np.sin(np.radians(lon))[None, :] * np.ones((rows, 1))

    stored = np.rint(heights * 2.0).astype(np.int16)

    # An unmapped hole, as on Pluto.
    stored[60:80, 20:50] = -32768

    write_geotiff(tmp_path / "dem.tif", stored, center=180.0, scale=0.5, nodata=-32768)

    monkeypatch.setattr(maps, "MAPS_DIRECTORY", tmp_path)

    dataset = maps.MapDataset(
        id="t", body="t", kind="elevation", format="geotiff",
        files=(("", "dem.tif"),), description="", credit="", datum=-500.0
    )

    grid = read_dataset(dataset)

    assert not np.isnan(grid.values).any()

    # East longitudes, plus the datum (500 m).
    for latitude, longitude in [(10.0, 90.0), (-20.0, -90.0), (45.0, 30.0)]:

        height = sample_grid(grid, direction_of(latitude, longitude)[None, :])[0]

        assert height == pytest.approx(1_000.0 * math.sin(math.radians(longitude)) + 500.0, abs=40.0)

    # The hole is filled from around it, not with -32768.
    hole = sample_grid(grid, direction_of(float(lat[70]), float(lon[35]))[None, :])[0]

    assert -1_000.0 < hole < 2_000.0


def test_fill_gaps_is_smooth():

    values = np.tile(np.linspace(0.0, 100.0, 64), (32, 1))

    holed = values.copy()
    holed[8:24, 20:44] = np.nan

    filled = fill_gaps(holed)

    assert not np.isnan(filled).any()

    np.testing.assert_array_equal(filled[~np.isnan(holed)], values[~np.isnan(holed)])

    # Between the edges' values, with no blocky jumps.
    inside = filled[8:24, 20:44]

    assert inside.min() >= values[0, 18] - 1.0 and inside.max() <= values[0, 45] + 1.0
    assert np.abs(np.diff(inside, axis=1)).max() < 10.0


def test_color_image_layout_and_saturation(tmp_path):

    rows, columns = 64, 128

    # Red east of the prime meridian, blue west of it.
    east = ((longitudes(columns, 0.0) + 180.0) % 360.0 - 180.0) > 0.0

    rgb = np.zeros((3, rows, columns), dtype=np.uint8)
    rgb[0][:, east] = 200
    rgb[2][:, ~east] = 200

    write_geotiff(tmp_path / "c.tif", rgb, center=180.0)

    image = color_image_from_geotiff(tmp_path / "c.tif", 64)

    assert image.shape == (32, 64, 3)

    # The color maps' layout: -180 deg at the left edge.
    assert image[16, 48, 0] > 150 and image[16, 48, 2] < 50          # 90 E
    assert image[16, 16, 2] > 150 and image[16, 16, 0] < 50          # 90 W

    grey = color_image_from_geotiff(tmp_path / "c.tif", 64, saturation=0.0)

    assert np.ptp(grey[16, 48].astype(int)) <= 1


def test_large_heights_keep_their_precision():

    rows, columns = 90, 180

    grid = maps.Grid(np.full((rows, columns), -40_000.0, dtype=np.float32))

    faces = cube_from_grid(grid, size=16)

    assert faces.dtype == np.float32
    assert faces.min() == pytest.approx(-40_000.0)

    assert cube_from_grid(maps.Grid(np.full((rows, columns), 1_234.4, dtype=np.float32)), size=16).dtype == np.int16


# =========================================================
# Terrain on a Map Measured From a Sphere
# =========================================================

def test_shape_is_taken_out_of_sphere_maps(monkeypatch):

    radius = 470_000.0

    def absolute(directions):
        # Flattened like Ceres, with a bump.
        return -20_000.0 * directions[:, 1] ** 2 + 3_000.0 * directions[:, 0]

    grid = maps.Grid(
        np.zeros((180, 360), dtype=np.float32)
    )

    lat = 90.0 - (np.arange(180) + 0.5)
    lon = -180.0 + (np.arange(360) + 0.5)

    lon_grid, lat_grid = np.meshgrid(np.radians(lon), np.radians(lat))

    directions = np.stack(
        (np.cos(lat_grid) * np.sin(lon_grid), np.sin(lat_grid), np.cos(lat_grid) * np.cos(lon_grid)),
        axis=-1
    ).reshape(-1, 3)

    grid.values = absolute(directions).reshape(180, 360).astype(np.float32)

    dataset = replace(DATASETS["ceres_dawn_dtm"], detail="")

    elevation = ElevationMap(dataset, cube_from_grid(grid, size=64))

    monkeypatch.setattr(terrain_module, "load_elevation_map", lambda _: elevation)

    terrain = Terrain(
        TerrainSettings(
            radius=radius,
            has_liquid=False,
            detail_height=0.0,
            crater_density=0.0,
            oblateness=0.04,
            elevation_map="ceres_dawn_dtm"
        )
    )

    points = fibonacci_directions(400)

    # Shape + relief = the measured surface.
    total = terrain.base_height(points) + terrain.elevation(points, 50_000.0)

    np.testing.assert_allclose(total, absolute(points), atol=150.0)

    assert terrain.min_elevation <= (total - terrain.base_height(points)).min()
    assert terrain.max_elevation >= (total - terrain.base_height(points)).max()


# =========================================================
# Every Body Real
# =========================================================

def test_bodies_name_their_maps():

    presets = load_presets()

    expected = {
        "mercury": ("mercury_dem_665m", "mercury_color_665m"),
        "venus": ("venus_magellan_4641m", ""),
        "earth": ("etopo1_5min", "blue_marble_july"),
        "ceres": ("ceres_dawn_dtm", ""),
        "vesta": ("vesta_dawn_dtm", ""),
        "pluto": ("pluto_dem_300m", "pluto_mosaic_300m"),
        "io": ("", "io_mosaic_1km"),
        "europa": ("", "europa_mosaic_500m"),
        "ganymede": ("", "ganymede_mosaic_1km"),
        "callisto": ("", "callisto_mosaic_1km"),
        "titan": ("", "titan_mosaic_4km"),
        "enceladus": ("", "enceladus_mosaic_110m"),
        "triton": ("", "triton_mosaic_600m"),
    }

    for body, maps_ in expected.items():

        assert presets[body].maps == maps_, body

        for dataset in maps_:
            assert not dataset or DATASETS[dataset].body == body


@pytest.mark.parametrize("dataset, latitude, longitude, low, high", [
    ("venus_magellan_4641m", 65.2, 3.3, 8_000.0, 11_500.0),       # Maxwell Montes
    ("venus_magellan_4641m", 0.5, -165.5, 5_000.0, 9_500.0),      # Maat Mons
    ("pluto_dem_300m", 25.0, 180.0, -2_500.0, -500.0),            # Sputnik Planitia
    ("pluto_dem_300m", 25.0, -160.0, 0.0, 2_500.0),               # its eastern rim
])
def test_landmarks(dataset, latitude, longitude, low, high):

    if not DATASETS[dataset].available:
        pytest.skip(f"{dataset} not downloaded (tools/fetch_maps.py)")

    elevation = load_elevation_map(dataset)

    height = elevation.sample(direction_of(latitude, longitude)[None, :])[0]

    assert low < height < high
