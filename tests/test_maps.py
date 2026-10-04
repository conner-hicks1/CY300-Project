import math
import struct

import numpy as np
import pytest

import planet.maps as maps
import planet.terrain as terrain_module

from planet.bodies import ProfileError, load_presets, parse_profile, use_real_maps, components_for
from planet.maps import (
    DATASETS,
    ElevationMap,
    Grid,
    cube_from_grid,
    load_elevation_map,
    parse_pds_label,
    read_netcdf3,
    read_pds_image,
    sample_grid
)
from planet.shape import direction_of, fibonacci_directions
from planet.terrain import Terrain, TerrainSettings


def smooth_field(directions):
    """A smooth height field (m) to resample."""

    return 3_000.0 * directions[:, 1] + 1_000.0 * directions[:, 0] * directions[:, 2]


def grid_of(function, rows=180, columns=360, left=-180.0):

    lat = 90.0 - (np.arange(rows) + 0.5) * 180.0 / rows
    lon = left + (np.arange(columns) + 0.5) * 360.0 / columns

    lon_grid, lat_grid = np.meshgrid(np.radians(lon), np.radians(lat))

    directions = np.stack(
        (np.cos(lat_grid) * np.sin(lon_grid), np.sin(lat_grid), np.cos(lat_grid) * np.cos(lon_grid)),
        axis=-1
    ).reshape(-1, 3)

    return Grid(function(directions).reshape(rows, columns).astype(np.float32), left=left, right=left + 360.0)


# =========================================================
# Archives
# =========================================================

LABEL = """PDS_VERSION_ID = PDS3
^IMAGE = "TEST.IMG"
OBJECT = IMAGE
  LINES = 4
  LINE_SAMPLES = 8
  SAMPLE_TYPE = {kind}
  SAMPLE_BITS = 16
  UNIT = METER
  SCALING_FACTOR = 0.5
  OFFSET = 1000 <METER>
END_OBJECT = IMAGE
OBJECT = IMAGE_MAP_PROJECTION
  MAXIMUM_LATITUDE = 90.0 <DEG>
  MINIMUM_LATITUDE = -90.0 <DEG>
  WESTERNMOST_LONGITUDE = 0.0 <DEG>
  EASTERNMOST_LONGITUDE = 360.0 <DEG>
END_OBJECT = IMAGE_MAP_PROJECTION
END
"""


@pytest.mark.parametrize("kind, order", [("MSB_INTEGER", ">"), ("LSB_INTEGER", "<")])
def test_pds_images(tmp_path, kind, order):

    values = np.arange(32, dtype=np.int16).reshape(4, 8) - 10

    (tmp_path / "test.img").write_bytes(values.astype(order + "i2").tobytes())
    (tmp_path / "test.lbl").write_text(LABEL.format(kind=kind))

    assert parse_pds_label(LABEL.format(kind=kind))["OFFSET"] == "1000"

    grid = read_pds_image(tmp_path / "test.img", tmp_path / "test.lbl", datum=1000.0)

    # DN * scale + offset - datum.
    np.testing.assert_allclose(grid.values, values * 0.5)

    assert (grid.top, grid.bottom, grid.left, grid.right) == (90.0, -90.0, 0.0, 360.0)


def write_netcdf3(path, latitude, longitude, altitude):
    """A minimal classic netCDF file, as ERDDAP serves grids."""

    def name(text):

        data = text.encode()

        return struct.pack(">i", len(data)) + data + b"\0" * ((4 - len(data) % 4) % 4)

    header = b"CDF\x01" + struct.pack(">i", 0)

    header += struct.pack(">ii", 10, 2) + name("latitude") + struct.pack(">i", len(latitude))
    header += name("longitude") + struct.pack(">i", len(longitude))

    header += struct.pack(">ii", 0, 0)      # no global attributes

    variables = [
        ("latitude", (0,), 6, np.asarray(latitude, ">f8").tobytes()),
        ("longitude", (1,), 6, np.asarray(longitude, ">f8").tobytes()),
        ("altitude", (0, 1), 3, np.asarray(altitude, ">i2").tobytes()),
    ]

    def entry(variable, dims, nc_type, begin, size):

        return (
            name(variable)
            + struct.pack(">i", len(dims)) + b"".join(struct.pack(">i", d) for d in dims)
            + struct.pack(">ii", 0, 0)
            + struct.pack(">iii", nc_type, size, begin)
        )

    padded = [data + b"\0" * ((4 - len(data) % 4) % 4) for _, _, _, data in variables]

    # Header size first (offsets depend on it).
    probe = header + struct.pack(">ii", 11, 3) + b"".join(
        entry(v, d, t, 0, len(p)) for (v, d, t, _), p in zip(variables, padded)
    )

    offset = len(probe)

    body = b""
    entries = b""

    for (variable, dims, nc_type, _), data in zip(variables, padded):

        entries += entry(variable, dims, nc_type, offset + len(body), len(data))

        body += data

    path.write_bytes(header + struct.pack(">ii", 11, 3) + entries + body)


def test_netcdf3(tmp_path):

    latitude = np.array([-60.0, 0.0, 60.0])
    longitude = np.array([-120.0, 0.0, 120.0, 180.0])
    altitude = np.arange(12, dtype=np.int16).reshape(3, 4) * 100 - 500

    write_netcdf3(tmp_path / "grid.nc", latitude, longitude, altitude)

    variables = read_netcdf3(tmp_path / "grid.nc")

    np.testing.assert_allclose(variables["latitude"], latitude)
    np.testing.assert_allclose(variables["longitude"], longitude)
    np.testing.assert_array_equal(variables["altitude"], altitude)


# =========================================================
# Sampling
# =========================================================

def test_equirectangular_sampling():

    grid = grid_of(smooth_field)

    directions = fibonacci_directions(2_000)

    np.testing.assert_allclose(sample_grid(grid, directions), smooth_field(directions), atol=40.0)

    # Longitude 0..360 grids wrap the same way.
    shifted = grid_of(smooth_field, left=0.0)

    np.testing.assert_allclose(sample_grid(shifted, directions), smooth_field(directions), atol=40.0)


def test_cube_sphere_cache_is_seamless():

    grid = grid_of(smooth_field, 360, 720)

    elevation = ElevationMap(DATASETS["mola_megdr_16"], cube_from_grid(grid, size=64))

    directions = fibonacci_directions(5_000)

    np.testing.assert_allclose(elevation.sample(directions), smooth_field(directions), atol=60.0)

    # Across a cube edge (x = z, between the +x and +z
    # faces): no step.
    a = np.array([[math.cos(0.3) * math.sin(math.pi / 4 - 1e-4), math.sin(0.3), math.cos(0.3) * math.cos(math.pi / 4 - 1e-4)]])
    b = np.array([[math.cos(0.3) * math.sin(math.pi / 4 + 1e-4), math.sin(0.3), math.cos(0.3) * math.cos(math.pi / 4 + 1e-4)]])

    assert abs(elevation.sample(a)[0] - elevation.sample(b)[0]) < 5.0


# =========================================================
# Terrain from a Map
# =========================================================

def test_terrain_follows_the_map(monkeypatch):

    grid = grid_of(smooth_field, 360, 720)

    elevation = ElevationMap(DATASETS["mola_megdr_16"], cube_from_grid(grid, size=64))

    monkeypatch.setattr(terrain_module, "load_elevation_map", lambda dataset: elevation)

    settings = TerrainSettings(radius=3_389_500.0, has_liquid=False, detail_height=0.0, elevation_map="mola_megdr_16")

    terrain = Terrain(settings, field=object())

    # The map replaces the simulated surface.
    assert terrain.field is None

    directions = fibonacci_directions(500)

    np.testing.assert_allclose(terrain.elevation(directions, spacing=50_000.0), smooth_field(directions), atol=60.0)

    assert terrain.max_elevation >= elevation.max_height
    assert terrain.min_elevation <= elevation.min_height

    # Only craters too small for the map.
    assert terrain.craters is None or terrain.craters.settings.max_diameter <= 4.1 * elevation.resolution(settings.radius)


def test_profiles_name_known_maps():

    presets = load_presets()

    assert presets["mars"].maps == ("mola_megdr_16", "viking_color_1024")
    assert presets["moon"].maps == ("lola_ldem_16", "lroc_color_2k")
    assert presets["earth"].maps[0] == "etopo1_5min"

    bad = {
        "format": 1, "name": "X", "kind": "moon", "physical": {"radius_km": 10, "mass_kg": 1e10, "rotation_hours": 1,
        "solar_day_hours": 1, "axial_tilt_deg": 0, "bond_albedo": 0.1}, "orbit": {"distance_au": 1,
        "eccentricity": 0, "period_days": 1}, "star": {"luminosity_solar": 1, "temperature_k": 5772},
        "surface": {"mean_temperature_c": 0, "liquid": "none", "palette": "mineral", "colors": {
            "low": [0, 0, 0], "high": [0, 0, 0], "steep": [0, 0, 0], "ice": [0, 0, 0]}},
        "terrain": {k: 1.0 for k in ("continent_frequency", "continent_height", "land_bias",
                                      "mountain_frequency", "mountain_height", "detail_height")},
        "geology": {"regime": "none"}, "climate": {"humidity": 0}, "maps": {"elevation": "lroc_color_2k"}
    }

    with pytest.raises(ProfileError, match="elevation map"):
        parse_profile(bad, "x")


def test_real_maps_only_when_downloaded(monkeypatch, tmp_path):

    profile = load_presets()["mars"]

    monkeypatch.setattr(maps, "MAPS_DIRECTORY", tmp_path)
    monkeypatch.setattr(maps, "_AVAILABILITY", {})

    planet = components_for(profile).planet

    use_real_maps(planet, profile)

    assert planet.elevation_map == "" and planet.color_map == ""

    for _, name in DATASETS["mola_megdr_16"].files:
        (tmp_path / name).write_bytes(b"")

    # (Availability is checked on disk every few seconds.)
    maps._AVAILABILITY.clear()

    use_real_maps(planet, profile)

    assert planet.elevation_map == "mola_megdr_16" and planet.color_map == ""


# =========================================================
# The Real Data (when downloaded)
# =========================================================

@pytest.mark.parametrize("dataset, latitude, longitude, low, high", [
    ("mola_megdr_16", 18.65, -133.8, 19_000.0, 22_500.0),     # Olympus Mons
    ("mola_megdr_16", -42.4, 70.5, -8_000.0, -5_000.0),       # Hellas
    ("lola_ldem_16", -53.0, -169.0, -8_000.0, -5_000.0),      # South Pole-Aitken
    ("etopo1_5min", 11.35, 142.2, -11_000.0, -9_000.0),       # Mariana Trench
    ("etopo1_5min", 38.5, -98.0, 300.0, 700.0),               # Kansas
])
def test_landmarks(dataset, latitude, longitude, low, high):

    if not DATASETS[dataset].available:
        pytest.skip(f"{dataset} not downloaded (tools/fetch_maps.py)")

    elevation = load_elevation_map(dataset)

    height = elevation.sample(direction_of(latitude, longitude)[None, :])[0]

    assert low < height < high
