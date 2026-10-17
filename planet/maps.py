import math
import re
import struct
import time

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from core.logger import Logger

from planet.cube_sphere import FACE_NORMALS, FACE_U, FACE_V, face_directions


# =========================================================
# Real Maps
# =========================================================
#
# Measured elevation and color of real bodies, as the base
# of their terrain instead of the generators: Mars from
# MOLA's laser altimeter, the Moon from LRO's LOLA, Earth's
# land and sea floor from ETOPO1, Mercury from MESSENGER,
# Venus from Magellan's radar, Ceres and Vesta from Dawn,
# Pluto from New Horizons; colors from LRO, Viking,
# MESSENGER, Blue Marble and the Voyager, Galileo, Cassini
# and New Horizons mosaics. The files are not part of the
# repository: tools/fetch_maps.py downloads them into
# data/maps/ (about 4.5 GB), and without them a body falls
# back to the generators.
#
# Each elevation map (an equirectangular grid in its
# archive's format) is resampled once onto the cube-sphere
# (6 faces of CUBE_SIZE^2 heights, int16 m) and cached in
# data/cache/, which loads in milliseconds; terrain then
# samples it bilinearly per vertex, with procedural detail
# and small craters added below the map's resolution.
#
# Longitudes are east-positive with 0 at the prime meridian
# (the body frame's +z; planet/orbits.py turns it to face
# the right way at each moment).

MAPS_DIRECTORY = Path("data/maps")
CACHE_DIRECTORY = Path("data/cache")

# Samples per cube-face edge (~1/4 deg: about the maps'
# own resolution).
CUBE_SIZE = 1536

_CACHE_FORMAT = 1

# Color maps made from GeoTIFF mosaics.
_COLOR_CACHE_FORMAT = 2

# (dataset id, directory) -> (when checked, downloaded).
_AVAILABILITY: dict = {}


@dataclass(frozen=True, slots=True)
class MapDataset:

    id: str
    body: str
    kind: str                       # "elevation" or "color"
    format: str                     # "pds", "netcdf", "geotiff" or "image"
    files: tuple[tuple[str, str], ...]   # (url, file name); the first is the data
    description: str
    credit: str

    # Heights: subtracted from the stored values (a radius
    # datum), m.
    datum: float = 0.0

    # Images: longitude at the left edge (deg east).
    left_longitude: float = -180.0

    # Elevation: a sharper dataset of the same body (its id),
    # sampled where the terrain is built finer than this one
    # resolves; "detail" kind datasets are those.
    detail: str = ""

    # Heights measured from a sphere (the body's radius after
    # the datum), not from its own shape: terrain takes the
    # shape (oblateness, lumps) back out (planet/terrain.py).
    above_sphere: bool = False

    # Color: how much of the image's color to keep (0 = grey).
    # Enhanced-color mosaics exaggerate the hues many times.
    saturation: float = 1.0

    @property
    def path(
        self
    ) -> Path:

        return MAPS_DIRECTORY / self.files[0][1]

    @property
    def available(
        self
    ) -> bool:
        """Downloaded (checked on disk at most every few seconds: the editor asks every frame)."""

        now = time.monotonic()

        checked = _AVAILABILITY.get((self.id, MAPS_DIRECTORY))

        if checked is None or now - checked[0] > 3.0:

            checked = (now, all((MAPS_DIRECTORY / name).is_file() for _, name in self.files))

            _AVAILABILITY[(self.id, MAPS_DIRECTORY)] = checked

        return checked[1]

    @property
    def size_hint(
        self
    ) -> str:

        return {
            "mola_megdr_16": "33 MB",
            "lola_ldem_16": "33 MB",
            "etopo1_5min": "19 MB",
            "mola_megdr_64": "531 MB",
            "lola_ldem_64": "531 MB",
            "etopo1_1min": "467 MB",
            "blue_marble_july": "2 MB",
            "mercury_dem_665m": "531 MB",
            "mercury_color_665m": "797 MB",
            "venus_magellan_4641m": "67 MB",
            "ceres_dawn_dtm": "467 MB",
            "vesta_dawn_dtm": "597 MB",
            "pluto_dem_300m": "620 MB",
            "pluto_mosaic_300m": "310 MB",
            "io_mosaic_1km": "197 MB",
            "europa_mosaic_500m": "193 MB",
            "ganymede_mosaic_1km": "137 MB",
            "callisto_mosaic_1km": "115 MB",
            "titan_mosaic_4km": "8 MB",
            "enceladus_mosaic_110m": "104 MB",
            "triton_mosaic_600m": "300 MB",
        }.get(self.id, "<1 MB")


_PDS_MOLA = "https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg016/"
_PDS_MOLA_64 = "https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg064/"
_MOLA_64_TILES = ("megt90n000gb", "megt90n180gb", "megt00n000gb", "megt00n180gb")
_PDS_LOLA = "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/img/"
_USGS = "https://planetarymaps.usgs.gov/mosaic/"
_USGS_CREDIT = "NASA / USGS Astrogeology (planetarymaps.usgs.gov)"


def _usgs(
    id: str,
    body: str,
    kind: str,
    name: str,
    description: str,
    credit: str = _USGS_CREDIT,
    sharp: bool = True,
    **fields
) -> list[MapDataset]:
    """
    A USGS Astrogeology GeoTIFF; an elevation one also as
    its own sharper "detail" dataset (the same file, read in
    place up close) unless it has gaps (sharp=False: those
    would read as pits).
    """

    sharp = sharp and kind == "elevation"

    datasets = [
        MapDataset(
            id=id,
            body=body,
            kind=kind,
            format="geotiff",
            files=((_USGS + name, name),),
            description=description,
            credit=credit,
            detail=id + "_full" if sharp else "",
            **fields
        )
    ]

    if sharp:

        datasets.append(
            MapDataset(
                id=id + "_full",
                body=body,
                kind="detail",
                format="geotiff",
                files=((_USGS + name, name),),
                description=description + ", at full resolution up close",
                credit=credit,
                **fields
            )
        )

    return datasets

DATASETS: dict[str, MapDataset] = {
    dataset.id: dataset
    for dataset in (
        MapDataset(
            id="mola_megdr_16",
            body="mars",
            kind="elevation",
            format="pds",
            files=(
                (_PDS_MOLA + "megt90n000eb.img", "megt90n000eb.img"),
                (_PDS_MOLA + "megt90n000eb.lbl", "megt90n000eb.lbl"),
            ),
            description="Mars topography, MGS MOLA MEGDR, 16 pixels/deg (~3.7 km), above the areoid",
            credit="NASA / MGS MOLA Science Team (PDS Geosciences Node)",
            detail="mola_megdr_64"
        ),
        MapDataset(
            id="mola_megdr_64",
            body="mars",
            kind="detail",
            format="pds_tiles",
            files=tuple(
                (_PDS_MOLA_64 + tile + extension, tile + extension)
                for tile in _MOLA_64_TILES
                for extension in (".img", ".lbl")
            ),
            description="Mars topography, MOLA MEGDR, 64 pixels/deg (~0.9 km): four tiles, sampled up close",
            credit="NASA / MGS MOLA Science Team (PDS Geosciences Node)"
        ),
        MapDataset(
            id="lola_ldem_16",
            body="moon",
            kind="elevation",
            format="pds",
            files=(
                (_PDS_LOLA + "ldem_16.img", "ldem_16.img"),
                (_PDS_LOLA + "ldem_16.lbl", "ldem_16.lbl"),
            ),
            description="Moon topography, LRO LOLA LDEM, 16 pixels/deg (~1.9 km), above 1737.4 km",
            credit="NASA / LRO LOLA Science Team (PDS Geosciences Node)",
            datum=1_737_400.0,
            detail="lola_ldem_64"
        ),
        MapDataset(
            id="lola_ldem_64",
            body="moon",
            kind="detail",
            format="pds_tiles",
            files=(
                (_PDS_LOLA + "ldem_64.img", "ldem_64.img"),
                (_PDS_LOLA + "ldem_64.lbl", "ldem_64.lbl"),
            ),
            description="Moon topography, LRO LOLA LDEM, 64 pixels/deg (~0.5 km), sampled up close",
            credit="NASA / LRO LOLA Science Team (PDS Geosciences Node)",
            datum=1_737_400.0
        ),
        MapDataset(
            id="etopo1_5min",
            body="earth",
            kind="elevation",
            format="netcdf",
            files=(
                (
                    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc"
                    "?altitude%5B(-90.0):5:(90.0)%5D%5B(-180.0):5:(180.0)%5D",
                    "etopo1_5min.nc"
                ),
            ),
            description="Earth relief, ETOPO1 every 5 arc-minutes (~9 km), land and sea floor",
            credit="Amante & Eakins 2009, NOAA NCEI (via NOAA ERDDAP)",
            detail="etopo1_1min"
        ),
        MapDataset(
            id="etopo1_1min",
            body="earth",
            kind="detail",
            format="netcdf",
            files=(
                (
                    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc"
                    "?altitude%5B(-90.0):1:(90.0)%5D%5B(-180.0):1:(180.0)%5D",
                    "etopo1_1min.nc"
                ),
            ),
            description="Earth relief, ETOPO1 every arc-minute (~1.8 km), sampled up close",
            credit="Amante & Eakins 2009, NOAA NCEI (via NOAA ERDDAP)"
        ),
        MapDataset(
            id="lroc_color_2k",
            body="moon",
            kind="color",
            format="image",
            files=(
                (
                    "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_2k.jpg",
                    "lroc_color_2k.jpg"
                ),
            ),
            description="Moon color, LROC WAC mosaic (NASA SVS CGI Moon Kit)",
            credit="NASA's Scientific Visualization Studio / LRO LROC"
        ),
        MapDataset(
            id="viking_color_1024",
            body="mars",
            kind="color",
            format="image",
            files=(
                (
                    "https://astrogeology.usgs.gov/ckan/dataset/dfdc2242-52dc-4126-bc89-03af8253ae79/"
                    "resource/0d7b31dc-0b2e-4ca6-89dc-e3c1404c0232/download/mars_viking_clrmosaic_global_1024.jpg",
                    "mars_viking_clrmosaic_global_1024.jpg"
                ),
            ),
            description="Mars color, Viking global color mosaic",
            credit="NASA / USGS Astrogeology"
        ),
        MapDataset(
            id="blue_marble_july",
            body="earth",
            kind="color",
            format="image",
            files=(
                (
                    "https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74092/"
                    "world.200407.3x5400x2700.jpg",
                    "world.200407.3x5400x2700.jpg"
                ),
            ),
            description="Earth color, Blue Marble Next Generation (July 2004)",
            credit="NASA Earth Observatory / Reto Stockli, NASA GSFC"
        ),
        # Heights of the USGS DEMs: stored value x scale +
        # offset (the file's GDAL metadata), from a sphere of
        # the file's radius (or absolute radii); `datum`
        # brings them to the body's radius in assets/bodies.
        *_usgs(
            "mercury_dem_665m", "mercury", "elevation",
            "Mercury_Messenger_USGS_DEM_Global_665m_v2.tif",
            "Mercury topography, MESSENGER stereo DEM, 665 m/pixel, above 2439.4 km",
            datum=300.0,                    # 2439.7 km - 2439.4 km
            above_sphere=True
        ),
        *_usgs(
            "mercury_color_665m", "mercury", "color",
            "Mercury_MESSENGER_MDIS_Basemap_EnhancedColor_Mosaic_Global_665m.tif",
            "Mercury color, MESSENGER MDIS enhanced-color basemap (toned down to its true greys)",
            saturation=0.04
        ),
        *_usgs(
            "venus_magellan_4641m", "venus", "elevation",
            "Venus_Magellan_Topography_Global_4641m_v02.tif",
            "Venus topography, Magellan radar altimetry, 4.6 km/pixel, above 6051 km",
            datum=800.0,                    # 6051.8 km - 6051.0 km
            above_sphere=True,
            sharp=False
        ),
        *_usgs(
            "ceres_dawn_dtm", "ceres", "elevation",
            "Ceres_Dawn_FC_HAMO_DTM_DLR_Global_60ppd_Oct2016.tif",
            "Ceres topography, Dawn HAMO stereo DTM (DLR), 60 pixels/deg (~140 m)",
            credit="NASA / JPL-Caltech / DLR / USGS Astrogeology",
            datum=482_100.0,                # absolute radii
            above_sphere=True
        ),
        *_usgs(
            "vesta_dawn_dtm", "vesta", "elevation",
            "Vesta_Dawn_HAMO_DTM_DLR_Global_48ppd.tif",
            "Vesta topography, Dawn HAMO stereo DTM (DLR), 48 pixels/deg (~90 m)",
            credit="NASA / JPL-Caltech / DLR / USGS Astrogeology",
            datum=262_700.0,                # absolute radii
            above_sphere=True
        ),
        *_usgs(
            "pluto_dem_300m", "pluto", "elevation",
            "Pluto_NewHorizons_Global_DEM_300m_Jul2017_16bit.tif",
            "Pluto topography, New Horizons stereo DEM, 300 m/pixel (the hemisphere seen up close)",
            credit="NASA / JHUAPL / SwRI / USGS Astrogeology",
            above_sphere=True,
            sharp=False
        ),
        *_usgs(
            "pluto_mosaic_300m", "pluto", "color",
            "Pluto_NewHorizons_Global_Mosaic_300m_Jul2017_8bit.tif",
            "Pluto, New Horizons LORRI/MVIC global mosaic",
            credit="NASA / JHUAPL / SwRI / USGS Astrogeology"
        ),
        *_usgs(
            "io_mosaic_1km", "io", "color",
            "Io_GalileoSSI-Voyager_Global_Mosaic_ClrMerge_1km.tif",
            "Io color, Galileo SSI and Voyager global mosaic"
        ),
        *_usgs(
            "europa_mosaic_500m", "europa", "color",
            "Europa_Voyager_GalileoSSI_global_mosaic_500m.tif",
            "Europa, Voyager and Galileo SSI global mosaic"
        ),
        *_usgs(
            "ganymede_mosaic_1km", "ganymede", "color",
            "Ganymede_Voyager_GalileoSSI_global_mosaic_1km.tif",
            "Ganymede, Voyager and Galileo SSI global mosaic"
        ),
        *_usgs(
            "callisto_mosaic_1km", "callisto", "color",
            "Callisto_Voyager_GalileoSSI_global_mosaic_1km.tif",
            "Callisto, Voyager and Galileo SSI global mosaic"
        ),
        *_usgs(
            "titan_mosaic_4km", "titan", "color",
            "Titan_ISS_P19658_Mosaic_Global_4km.tif",
            "Titan's surface, Cassini ISS near-infrared global mosaic",
            credit="NASA / JPL / Space Science Institute / USGS Astrogeology"
        ),
        *_usgs(
            "enceladus_mosaic_110m", "enceladus", "color",
            "Enceladus_Cassini_mosaic_global_110m.tif",
            "Enceladus, Cassini ISS global mosaic",
            credit="NASA / JPL / Space Science Institute / USGS Astrogeology"
        ),
        *_usgs(
            "triton_mosaic_600m", "triton", "color",
            "Triton_Voyager2_ClrMosaic_GlobalFill_600m.tif",
            "Triton color, Voyager 2 mosaic (unseen areas filled)"
        ),
    )
}


# =========================================================
# Reading Archives
# =========================================================

@dataclass(slots=True)
class Grid:

    # Equirectangular heights (m) or colors, rows from north
    # to south, columns eastward from `left` (deg).
    values: np.ndarray
    top: float = 90.0
    bottom: float = -90.0
    left: float = -180.0
    right: float = 180.0


def parse_pds_label(
    text: str
) -> dict[str, str]:
    """KEY = value pairs of a PDS3 label (units and quotes stripped)."""

    values = {}

    for line in text.splitlines():

        match = re.match(r"\s*([\^A-Z_0-9:]+)\s*=\s*(.+?)\s*$", line)

        if match is None:
            continue

        key, value = match.groups()

        value = re.sub(r"<[^>]*>", "", value).strip().strip('"')

        values.setdefault(key, value)

    return values


def read_pds_image(
    data_path: Path,
    label_path: Path,
    datum: float = 0.0
) -> Grid:

    label = parse_pds_label(Path(label_path).read_text(encoding="latin-1"))

    lines = int(label["LINES"])
    samples = int(label["LINE_SAMPLES"])
    bits = int(label["SAMPLE_BITS"])

    kind = label["SAMPLE_TYPE"]

    order = ">" if kind.startswith(("MSB", "SUN", "MAC")) else "<"

    dtype = {
        ("INTEGER", 16): "i2",
        ("INTEGER", 32): "i4",
        ("REAL", 32): "f4",
    }[("REAL" if "REAL" in kind else "INTEGER", bits)]

    raw = np.fromfile(data_path, dtype=order + dtype, count=lines * samples).reshape(lines, samples)

    scale = float(label.get("SCALING_FACTOR", 1.0))
    offset = float(label.get("OFFSET", 0.0))

    values = raw.astype(np.float32) * scale + (offset - datum)

    return Grid(
        values=values,
        top=float(label.get("MAXIMUM_LATITUDE", 90.0)),
        bottom=float(label.get("MINIMUM_LATITUDE", -90.0)),
        left=float(label.get("WESTERNMOST_LONGITUDE", 0.0)),
        right=float(label.get("EASTERNMOST_LONGITUDE", 360.0))
    )


def read_netcdf3(
    path: Path,
    mapped: bool = False
) -> dict[str, np.ndarray]:
    """
    The non-record variables of a classic netCDF file
    (format 1 or 2), by name: enough for gridded downloads
    without a netCDF library. mapped: the arrays are
    memory-mapped (read from disk as sampled), not loaded.
    """

    # (The header comes first; it is small.)
    data = Path(path).read_bytes() if not mapped else _read_head(path, 1 << 20)

    if data[:3] != b"CDF" or data[3] not in (1, 2):
        raise ValueError(f"{path}: not a classic netCDF file")

    offset_size = 8 if data[3] == 2 else 4

    position = 8

    def read(fmt):

        nonlocal position

        values = struct.unpack_from(">" + fmt, data, position)

        position += struct.calcsize(">" + fmt)

        return values[0] if len(values) == 1 else values

    def name():

        nonlocal position

        length = read("i")

        text = data[position:position + length].decode("utf-8")

        position += (length + 3) // 4 * 4

        return text

    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 4, 6: 8}
    types = {1: "i1", 2: "S1", 3: ">i2", 4: ">i4", 5: ">f4", 6: ">f8"}

    def attributes():

        nonlocal position

        tag, count = read("i"), read("i")

        for _ in range(count if tag == 12 else 0):

            name()

            nc_type, items = read("i"), read("i")

            position += (items * sizes[nc_type] + 3) // 4 * 4

    tag, count = read("i"), read("i")

    dimensions = []

    for _ in range(count if tag == 10 else 0):
        dimensions.append((name(), read("i")))

    attributes()

    tag, count = read("i"), read("i")

    variables = {}

    for _ in range(count if tag == 11 else 0):

        variable = name()

        shape = tuple(dimensions[read("i")][1] for _ in range(read("i")))

        attributes()

        nc_type = read("i")

        read("i")       # vsize

        begin = read("q" if offset_size == 8 else "i")

        if mapped:
            array = np.memmap(path, dtype=types[nc_type], mode="r", offset=begin, shape=shape)
        else:
            array = np.frombuffer(data, dtype=types[nc_type], count=int(np.prod(shape)), offset=begin)

        variables[variable] = array.reshape(shape)

    return variables


# =========================================================
# GeoTIFFs (USGS Astrogeology)
# =========================================================
#
# Uncompressed, so read in place (memory-mapped): simple
# cylindrical grids in meters on a sphere of the file's
# radius, with GDAL's scale and offset for the values.

# Heights of a resampled GeoTIFF: about the cube's own
# spacing (4 samples per cube texel row of a face, around
# the equator).
_GEOTIFF_COLUMNS = 4 * CUBE_SIZE

# Width of a color map built from a GeoTIFF mosaic (the GPU
# texture; 2:1).
COLOR_MAP_WIDTH = 4096


@dataclass(slots=True)
class _GeoTiff:

    values: np.ndarray              # memory-mapped (rows, columns) or (bands, rows, columns)
    scale: float
    offset: float
    nodata: float | None
    top: float                      # cell-edge bounds (deg)
    bottom: float
    left: float
    right: float


def read_geotiff(
    path: Path
) -> _GeoTiff:

    import logging

    import tifffile

    # (It cannot cast float32's lowest value back from its
    # text and says so; parsed below.)
    logging.getLogger("tifffile").setLevel(logging.ERROR)

    with tifffile.TiffFile(path) as tiff:

        page = tiff.pages[0]

        tags = page.tags

        pixel = tags["ModelPixelScaleTag"].value
        tie = tags["ModelTiepointTag"].value

        keys = tiff.geotiff_metadata or {}

        metadata = tags.get("GDAL_METADATA")
        metadata = metadata.value if metadata is not None else ""

        nodata = tags.get("GDAL_NODATA")

    nodata_value = None

    if nodata is not None:
        try:
            nodata_value = float(str(nodata.value).strip("\x00 "))
        except ValueError:
            nodata_value = None

    radius = float(keys.get("GeogSemiMajorAxisGeoKey", 0.0))
    center = float(keys.get("ProjCenterLongGeoKey", keys.get("ProjNatOriginLongGeoKey", 0.0)))

    if radius <= 0.0:
        raise ValueError(f"{path.name}: no body radius in its GeoTIFF keys")

    def item(name: str, default: float) -> float:

        found = re.search(rf'<Item name="{name}" sample="0"[^>]*>([^<]+)</Item>', metadata)

        return float(found.group(1)) if found else default

    values = tifffile.memmap(path, mode="r")

    rows, columns = values.shape[-2:]

    degrees = 180.0 / (math.pi * radius)

    left = center + tie[3] * degrees
    top = tie[4] * degrees

    return _GeoTiff(
        values=values,
        scale=item("SCALE", 1.0),
        offset=item("OFFSET", 0.0),
        nodata=nodata_value,
        top=top,
        bottom=top - rows * pixel[1] * degrees,
        left=left,
        right=left + columns * pixel[0] * degrees
    )


def _block_mean(
    values: np.ndarray,
    factor: int,
    invalid
) -> np.ndarray:
    """
    (rows, columns) averaged over factor x factor blocks,
    read a band of rows at a time; NaN where a block has no
    valid sample (invalid(values) marks those).
    """

    rows, columns = values.shape

    out_rows, out_columns = rows // factor, columns // factor

    out = np.empty((out_rows, out_columns), dtype=np.float32)

    step = max(1, 4096 // factor)

    for start in range(0, out_rows, step):

        stop = min(out_rows, start + step)

        block = np.asarray(
            values[start * factor:stop * factor, :out_columns * factor],
            dtype=np.float64
        ).reshape(stop - start, factor, out_columns, factor)

        bad = invalid(block)

        count = (~bad).sum(axis=(1, 3))

        total = np.where(bad, 0.0, block).sum(axis=(1, 3))

        with np.errstate(invalid="ignore", divide="ignore"):
            out[start:stop] = np.where(count > 0, total / np.maximum(count, 1), np.nan)

    return out


def fill_gaps(
    values: np.ndarray
) -> np.ndarray:
    """
    NaN cells filled smoothly from their surroundings (pull-
    push: average down a pyramid until nothing is missing,
    then fill each level's gaps from the coarser one).
    """

    missing = np.isnan(values)

    if not missing.any():
        return values

    if missing.all():
        return np.zeros_like(values)

    rows, columns = values.shape

    if rows < 2 or columns < 2:
        return np.where(missing, np.nanmean(values), values)

    # Next level down (padded to even sizes).
    padded = np.pad(values, ((0, rows % 2), (0, columns % 2)), mode="edge")

    blocks = padded.reshape(padded.shape[0] // 2, 2, padded.shape[1] // 2, 2)

    valid = ~np.isnan(blocks)

    count = valid.sum(axis=(1, 3))

    total = np.where(valid, blocks, 0.0).sum(axis=(1, 3))

    coarse = fill_gaps(np.where(count > 0, total / np.maximum(count, 1), np.nan))

    return np.where(missing, _upsample(coarse, rows, columns), values)


def _upsample(
    coarse: np.ndarray,
    rows: int,
    columns: int
) -> np.ndarray:
    """A half-size grid back to (rows, columns), bilinear (smooth fills, no blocks)."""

    def axis(fine: int, size: int):

        # Fine cell i's center in coarse cells.
        t = np.clip((np.arange(fine) + 0.5) / 2.0 - 0.5, 0.0, size - 1.0)

        i0 = np.minimum(np.floor(t).astype(np.int64), max(size - 2, 0))
        i1 = np.minimum(i0 + 1, size - 1)

        return i0, i1, t - i0

    y0, y1, fy = axis(rows, coarse.shape[0])
    x0, x1, fx = axis(columns, coarse.shape[1])

    fy = fy[:, None]

    top = coarse[y0][:, x0] * (1.0 - fx) + coarse[y0][:, x1] * fx
    bottom = coarse[y1][:, x0] * (1.0 - fx) + coarse[y1][:, x1] * fx

    return top * (1.0 - fy) + bottom * fy


def _geotiff_grid(
    dataset: MapDataset
) -> Grid:
    """A GeoTIFF's heights (m from the body's radius), averaged down to about the cube's spacing."""

    tiff = read_geotiff(dataset.path)

    if tiff.values.ndim != 2:
        raise ValueError(f"{dataset.id}: not a single-band elevation GeoTIFF")

    factor = max(1, tiff.values.shape[1] // _GEOTIFF_COLUMNS)

    nodata = tiff.nodata

    def invalid(block):

        bad = ~np.isfinite(block) | (block < -1e30)

        if nodata is not None:
            bad |= block == nodata

        return bad

    heights = fill_gaps(_block_mean(tiff.values, factor, invalid))

    heights = heights * tiff.scale + (tiff.offset - dataset.datum)

    rows, columns = heights.shape

    # (The blocks may drop a last partial row or column.)
    return Grid(
        values=heights.astype(np.float32),
        top=tiff.top,
        bottom=tiff.top - (tiff.top - tiff.bottom) * rows * factor / tiff.values.shape[0],
        left=tiff.left,
        right=tiff.left + (tiff.right - tiff.left) * columns * factor / tiff.values.shape[1]
    )


def _read_head(
    path: Path,
    size: int
) -> bytes:

    with open(path, "rb") as file:
        return file.read(size)


def read_dataset(
    dataset: MapDataset
) -> Grid:

    if dataset.format == "pds":

        return read_pds_image(
            MAPS_DIRECTORY / dataset.files[0][1],
            MAPS_DIRECTORY / dataset.files[1][1],
            dataset.datum
        )

    if dataset.format == "geotiff":

        return _geotiff_grid(dataset)

    if dataset.format == "netcdf":

        variables = read_netcdf3(dataset.path)

        latitude = variables["latitude"]
        longitude = variables["longitude"]

        heights = variables["altitude"].astype(np.float32)

        # North first.
        if latitude[0] < latitude[-1]:
            heights = heights[::-1]
            latitude = latitude[::-1]

        # Grid-registered samples (on the edges) -> the
        # cell-centered convention the sampler uses.
        step_lat = abs(float(latitude[0] - latitude[1]))
        step_lon = abs(float(longitude[1] - longitude[0]))

        return Grid(
            values=heights,
            top=float(latitude[0]) + 0.5 * step_lat,
            bottom=float(latitude[-1]) - 0.5 * step_lat,
            left=float(longitude[0]) - 0.5 * step_lon,
            right=float(longitude[-1]) + 0.5 * step_lon
        )

    raise ValueError(f"{dataset.id}: not an elevation archive")


# =========================================================
# Sampling
# =========================================================

def sample_grid(
    grid: Grid,
    directions: np.ndarray
) -> np.ndarray:
    """Bilinear samples of an equirectangular grid at unit directions."""

    rows, columns = grid.values.shape[:2]

    latitude = np.degrees(np.arcsin(np.clip(directions[:, 1], -1.0, 1.0)))
    longitude = np.degrees(np.arctan2(directions[:, 0], directions[:, 2]))

    # Cell centers at (i + 0.5).
    y = (grid.top - latitude) / (grid.top - grid.bottom) * rows - 0.5
    x = ((longitude - grid.left) % 360.0) / (grid.right - grid.left) * columns - 0.5

    y = np.clip(y, 0.0, rows - 1.0)

    y0 = np.minimum(np.floor(y).astype(np.int64), rows - 2)
    x0 = np.floor(x).astype(np.int64)

    fy = y - y0
    fx = x - x0

    x0 %= columns
    x1 = (x0 + 1) % columns

    v = grid.values

    top = v[y0, x0] * (1.0 - fx) + v[y0, x1] * fx
    bottom = v[y0 + 1, x0] * (1.0 - fx) + v[y0 + 1, x1] * fx

    return top * (1.0 - fy) + bottom * fy


class ElevationMap:

    # A body's measured heights on the cube-sphere: six
    # (size + 2)^2 grids of int16 m (one sample of border
    # past each edge, so sampling across faces is seamless).

    def __init__(
        self,
        dataset: MapDataset,
        faces: np.ndarray
    ):

        self.dataset = dataset
        self.faces = faces

        self.size = faces.shape[1] - 2

        self.min_height = float(faces.min())
        self.max_height = float(faces.max())

        # The sharper dataset, if downloaded (opened on first
        # use).
        self._detail = False

    @property
    def detail(
        self
    ) -> "DetailMap | None":

        if self._detail is False:
            self._detail = load_detail_map(self.dataset.detail) if self.dataset.detail else None

        return self._detail

    def resolution(
        self,
        radius: float
    ) -> float:
        """Sample spacing on the body's surface (m)."""

        return math.pi * 0.5 * radius / self.size

    def uses_detail(
        self,
        spacing: float,
        radius: float
    ) -> bool:
        """Whether samples this fine (m) take any of the sharper dataset."""

        return self.dataset.detail != "" and spacing > 0.0 and radius > 0.0 and spacing < self.resolution(radius)

    def sample(
        self,
        directions: np.ndarray,
        spacing: float = 0.0,
        radius: float = 0.0,
        allowed=None
    ) -> np.ndarray:
        """
        Heights (m). spacing: the samples' spacing (m) on a
        body of `radius`: finer than this map resolves, the
        sharper dataset takes over (blended in over a factor
        of two, so a level of detail changes nothing
        abruptly). 0 = this map alone. allowed: given, a
        function of the directions giving how much (0..1) of
        the sharper data each may take (science/survey.py:
        only where mapped).
        """

        heights = self._sample_cube(directions)

        detail = self.detail if spacing > 0.0 and radius > 0.0 else None

        if detail is None:
            return heights

        coarse = self.resolution(radius)

        weight = np.clip((coarse - spacing) / (0.5 * coarse), 0.0, 1.0)

        if weight <= 0.0:
            return heights

        if allowed is not None:

            weight = weight * allowed(directions)

            if not np.any(weight > 0.0):
                return heights

        return heights * (1.0 - weight) + detail.sample(directions) * weight

    def _sample_cube(
        self,
        directions: np.ndarray
    ) -> np.ndarray:

        directions = np.asarray(directions, dtype=np.float64)

        face = np.argmax(directions @ FACE_NORMALS.T, axis=1)

        normal = FACE_NORMALS[face]

        along = np.einsum("ij,ij->i", directions, normal)

        a = np.arctan(np.einsum("ij,ij->i", directions, FACE_U[face]) / along) * (4.0 / math.pi)
        b = np.arctan(np.einsum("ij,ij->i", directions, FACE_V[face]) / along) * (4.0 / math.pi)

        size = self.size

        # Texel i (with border, 0..size+1) has its center at
        # a = -1 + (i - 0.5) * 2 / size.
        x = np.clip((a + 1.0) * 0.5 * size + 0.5, 0.0, size + 1.0)
        y = np.clip((b + 1.0) * 0.5 * size + 0.5, 0.0, size + 1.0)

        x0 = np.minimum(np.floor(x).astype(np.int64), size)
        y0 = np.minimum(np.floor(y).astype(np.int64), size)

        fx = x - x0
        fy = y - y0

        f = self.faces

        h00 = f[face, y0, x0]
        h01 = f[face, y0, x0 + 1]
        h10 = f[face, y0 + 1, x0]
        h11 = f[face, y0 + 1, x0 + 1]

        return (
            (h00 * (1.0 - fx) + h01 * fx) * (1.0 - fy)
            + (h10 * (1.0 - fx) + h11 * fx) * fy
        )


def cube_from_grid(
    grid: Grid,
    size: int = CUBE_SIZE
) -> np.ndarray:
    """
    Resample an equirectangular grid onto the cube faces (m:
    int16, or float32 where heights pass 32 km, as on Ceres
    and Vesta).
    """

    faces = np.empty((6, size + 2, size + 2), dtype=np.float32)

    t = -1.0 + (np.arange(size + 2) - 0.5) * (2.0 / size)

    a, b = np.meshgrid(t, t, indexing="xy")

    for face in range(6):

        directions = face_directions(face, a, b).reshape(-1, 3)

        heights = sample_grid(grid, directions)

        faces[face] = heights.reshape(size + 2, size + 2)

    if faces.min() >= -32768.0 and faces.max() <= 32767.0:
        return np.rint(faces).astype(np.int16)

    return faces


@lru_cache(maxsize=4)
def load_elevation_map(
    dataset_id: str
) -> ElevationMap | None:
    """
    The dataset's cube-sphere heights: from the cache, else
    built from the archive (a few seconds) and cached. None
    when the files have not been downloaded.
    """

    dataset = DATASETS.get(dataset_id)

    if dataset is None or dataset.kind != "elevation":

        Logger.warning("[Maps] Unknown elevation map '%s'.", dataset_id)

        return None

    cache = CACHE_DIRECTORY / f"{dataset.id}_{CUBE_SIZE}_v{_CACHE_FORMAT}.npy"

    if cache.is_file():
        return ElevationMap(dataset, np.load(cache))

    if not dataset.available:

        Logger.warning(
            "[Maps] %s is not downloaded (python tools/fetch_maps.py); using generated terrain.",
            dataset.id
        )

        return None

    Logger.info("[Maps] Resampling %s onto the cube-sphere (once).", dataset.id)

    faces = cube_from_grid(read_dataset(dataset))

    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    np.save(cache, faces)

    return ElevationMap(dataset, faces)


# =========================================================
# Sharper Maps, Read in Place
# =========================================================
#
# The 64 pixels/deg maps are ~0.5 GB each: they are not
# resampled or loaded, but memory-mapped, and only the
# samples the detailed chunks near the camera ask for are
# read from disk (bilinear, per tile).

@dataclass(slots=True)
class _Tile:

    values: np.ndarray          # memory-mapped, rows from north (or south: flip)
    scale: float
    offset: float               # added after scaling (minus the datum)
    top: float                  # cell-edge bounds (deg)
    bottom: float
    left: float
    right: float
    flip: bool = False          # rows stored from the south


class DetailMap:

    def __init__(
        self,
        dataset: MapDataset,
        tiles: list[_Tile]
    ):

        self.dataset = dataset
        self.tiles = tiles

        tile = tiles[0]

        # Degrees per sample.
        self.step = (tile.top - tile.bottom) / tile.values.shape[0]

    def resolution(
        self,
        radius: float
    ) -> float:

        return math.radians(self.step) * radius

    def sample(
        self,
        directions: np.ndarray
    ) -> np.ndarray:

        directions = np.asarray(directions, dtype=np.float64)

        latitude = np.degrees(np.arcsin(np.clip(directions[:, 1], -1.0, 1.0)))
        longitude = np.degrees(np.arctan2(directions[:, 0], directions[:, 2]))

        heights = np.zeros(len(directions))

        for tile in self.tiles:

            span = tile.right - tile.left

            east = (longitude - tile.left) % 360.0

            inside = (latitude <= tile.top) & (latitude >= tile.bottom) & (east <= span)

            if not np.any(inside):
                continue

            heights[inside] = _sample_tile(tile, latitude[inside], east[inside], span >= 359.99)

        return heights


def _sample_tile(
    tile: _Tile,
    latitude: np.ndarray,
    east: np.ndarray,
    wraps: bool
) -> np.ndarray:

    values = tile.values

    rows, columns = values.shape

    y = (tile.top - latitude) / (tile.top - tile.bottom) * rows - 0.5
    x = east / (tile.right - tile.left) * columns - 0.5

    y = np.clip(y, 0.0, rows - 1.0)

    y0 = np.minimum(np.floor(y).astype(np.int64), rows - 2)
    fy = y - y0

    if wraps:
        x0 = np.floor(x).astype(np.int64)
        fx = x - x0
        x0 %= columns
        x1 = (x0 + 1) % columns
    else:
        x = np.clip(x, 0.0, columns - 1.0)
        x0 = np.minimum(np.floor(x).astype(np.int64), columns - 2)
        fx = x - x0
        x1 = x0 + 1

    r0, r1 = y0, y0 + 1

    if tile.flip:
        r0, r1 = rows - 1 - r0, rows - 1 - r1

    # (Sorted reads are kinder to the disk; fancy indexing on
    # a memmap reads just the pages touched.)
    top = values[r0, x0].astype(np.float64) * (1.0 - fx) + values[r0, x1] * fx
    bottom = values[r1, x0].astype(np.float64) * (1.0 - fx) + values[r1, x1] * fx

    return (top * (1.0 - fy) + bottom * fy) * tile.scale + tile.offset


def _pds_tile(
    data_path: Path,
    label_path: Path,
    datum: float
) -> _Tile:

    label = parse_pds_label(Path(label_path).read_text(encoding="latin-1"))

    lines = int(label["LINES"])
    samples = int(label["LINE_SAMPLES"])

    order = ">" if label["SAMPLE_TYPE"].startswith(("MSB", "SUN", "MAC")) else "<"

    values = np.memmap(data_path, dtype=order + "i2", mode="r", shape=(lines, samples))

    return _Tile(
        values=values,
        scale=float(label.get("SCALING_FACTOR", 1.0)),
        offset=float(label.get("OFFSET", 0.0)) - datum,
        top=float(label.get("MAXIMUM_LATITUDE", 90.0)),
        bottom=float(label.get("MINIMUM_LATITUDE", -90.0)),
        left=float(label.get("WESTERNMOST_LONGITUDE", 0.0)),
        right=float(label.get("EASTERNMOST_LONGITUDE", 360.0))
    )


def _netcdf_tile(
    path: Path
) -> _Tile:

    variables = read_netcdf3(path, mapped=True)

    latitude = np.asarray(variables["latitude"])
    longitude = np.asarray(variables["longitude"])

    step_lat = abs(float(latitude[1] - latitude[0]))
    step_lon = abs(float(longitude[1] - longitude[0]))

    flip = latitude[0] < latitude[-1]

    north, south = (latitude[-1], latitude[0]) if flip else (latitude[0], latitude[-1])

    return _Tile(
        values=variables["altitude"],
        scale=1.0,
        offset=0.0,
        top=float(north) + 0.5 * step_lat,
        bottom=float(south) - 0.5 * step_lat,
        left=float(longitude[0]) - 0.5 * step_lon,
        right=float(longitude[-1]) + 0.5 * step_lon,
        flip=bool(flip)
    )


def _geotiff_tile(
    dataset: MapDataset
) -> _Tile:

    tiff = read_geotiff(dataset.path)

    # Gaps would read as deep pits: those maps stay coarse.
    if tiff.nodata is not None and tiff.values.dtype.kind in "iu":

        sample = np.asarray(tiff.values[::97, ::97])

        if np.any(sample == tiff.nodata):
            raise ValueError("has unmapped areas")

    return _Tile(
        values=tiff.values,
        scale=tiff.scale,
        offset=tiff.offset - dataset.datum,
        top=tiff.top,
        bottom=tiff.bottom,
        left=tiff.left,
        right=tiff.right
    )


@lru_cache(maxsize=4)
def load_detail_map(
    dataset_id: str
) -> DetailMap | None:
    """A sharper map, memory-mapped; None until downloaded."""

    dataset = DATASETS.get(dataset_id)

    if dataset is None or dataset.kind != "detail" or not dataset.available:
        return None

    try:

        if dataset.format == "netcdf":
            tiles = [_netcdf_tile(dataset.path)]
        elif dataset.format == "geotiff":
            tiles = [_geotiff_tile(dataset)]
        else:
            names = [name for _, name in dataset.files]
            tiles = [
                _pds_tile(MAPS_DIRECTORY / image, MAPS_DIRECTORY / label, dataset.datum)
                for image, label in zip(names[0::2], names[1::2])
            ]

    except (OSError, ValueError, KeyError) as error:

        Logger.warning("[Maps] %s could not be read: %s", dataset.id, error)

        return None

    return DetailMap(dataset, tiles)


def color_map_path(
    dataset_id: str
) -> Path | None:
    """
    The image file of a color map, if downloaded (a GeoTIFF
    mosaic: an image made from it once and cached).
    """

    dataset = DATASETS.get(dataset_id)

    if dataset is None or dataset.kind != "color":
        return None

    if dataset.format != "geotiff":
        return dataset.path if dataset.available else None

    cache = CACHE_DIRECTORY / f"{dataset.id}_{COLOR_MAP_WIDTH}_v{_COLOR_CACHE_FORMAT}.png"

    if cache.is_file():
        return cache

    if not dataset.available:
        return None

    try:
        image = color_image_from_geotiff(dataset.path, COLOR_MAP_WIDTH, dataset.saturation)
    except (OSError, ValueError, KeyError) as error:

        Logger.warning("[Maps] %s could not be read: %s", dataset.id, error)

        return None

    from PIL import Image

    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    Image.fromarray(image).save(cache)

    return cache


def color_image_from_geotiff(
    path: Path,
    width: int,
    saturation: float = 1.0
) -> np.ndarray:
    """
    (width / 2, width, 3) uint8 RGB of a GeoTIFF mosaic, in
    the color maps' layout (north up, -180 deg at the left
    edge); unmapped areas filled from around them.
    """

    Logger.info("[Maps] Building a %d px color map from %s (once).", width, path.name)

    tiff = read_geotiff(path)

    values = tiff.values

    bands = [values] if values.ndim == 2 else [values[band] for band in range(min(values.shape[0], 3))]

    rows, columns = bands[0].shape

    factor = max(1, columns // (2 * width))

    def invalid(block):
        return block == (tiff.nodata if tiff.nodata is not None else -1.0)

    averaged = [fill_gaps(_block_mean(band, factor, invalid)) for band in bands]

    grid_rows, grid_columns = averaged[0].shape

    height = width // 2

    latitude = 90.0 - (np.arange(height) + 0.5) * 180.0 / height
    longitude = -180.0 + (np.arange(width) + 0.5) * 360.0 / width

    span_y = (tiff.top - tiff.bottom) * grid_rows * factor / rows
    span_x = (tiff.right - tiff.left) * grid_columns * factor / columns

    y = np.clip(((tiff.top - latitude) / span_y * grid_rows - 0.5), 0.0, grid_rows - 1.0)
    x = ((longitude - tiff.left) % 360.0) / span_x * grid_columns - 0.5

    y0 = np.minimum(np.floor(y).astype(np.int64), grid_rows - 2)[:, None]
    x0 = np.floor(x).astype(np.int64)[None, :]

    fy = y[:, None] - y0
    fx = x[None, :] - x0

    # (Around the globe: the right edge meets the left.)
    x0 %= grid_columns
    x1 = (x0 + 1) % grid_columns

    channels = []

    for grid in averaged:

        top = grid[y0, x0] * (1.0 - fx) + grid[y0, x1] * fx
        bottom = grid[y0 + 1, x0] * (1.0 - fx) + grid[y0 + 1, x1] * fx

        channels.append(top * (1.0 - fy) + bottom * fy)

    while len(channels) < 3:
        channels.append(channels[0])

    rgb = np.stack(channels, axis=-1)

    if saturation != 1.0:

        grey = rgb @ np.array([0.2126, 0.7152, 0.0722])

        rgb = grey[..., None] + (rgb - grey[..., None]) * saturation

    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
