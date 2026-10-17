import base64
import math
import threading

import numpy as np


# =========================================================
# Orbital Survey
# =========================================================
#
# Which parts of a body have been mapped from orbit, on a
# grid of CELL_DEGREES cells of latitude and longitude. A
# survey camera marks the sunlit ground in its swath as the
# vessel flies over (systems/science_system.py).
#
# Mapping unlocks the sharper elevation data (planet/maps.py
# "detail" datasets): where a body's mask gates it, terrain
# blends the sharp heights in only over mapped cells (by a
# smooth weight between cell centers, so chunk edges agree
# and nothing steps). Bodies without a registered mask, or
# whose mask does not gate (home), use it everywhere.
#
# The terrain is built on worker threads: masks are found
# through a registry by body id, and read without locks
# (a cell going from unmapped to mapped mid-build only
# changes that chunk's key next time; the system rebuilds
# fine chunks over newly mapped ground).

CELL_DEGREES = 1.0

ROWS = int(round(180.0 / CELL_DEGREES))
COLUMNS = int(round(360.0 / CELL_DEGREES))


def latitude_longitude(
    directions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Body-frame unit directions -> (latitude, east longitude) deg, as planet/maps.py."""

    directions = np.atleast_2d(np.asarray(directions, dtype=np.float64))

    latitude = np.degrees(np.arcsin(np.clip(directions[:, 1], -1.0, 1.0)))
    longitude = np.degrees(np.arctan2(directions[:, 0], directions[:, 2]))

    return latitude, longitude


def direction_of(
    latitude: float,
    longitude: float
) -> np.ndarray:
    """The body-frame unit direction of a point (the inverse of latitude_longitude)."""

    phi = math.radians(latitude)
    lam = math.radians(longitude)

    return np.array([math.cos(phi) * math.sin(lam), math.sin(phi), math.cos(phi) * math.cos(lam)])


# Each cell's area as a share of the sphere's.
_ROW_LATITUDES = 90.0 - (np.arange(ROWS) + 0.5) * CELL_DEGREES

_CELL_AREA = (
    np.sin(np.radians(_ROW_LATITUDES + 0.5 * CELL_DEGREES))
    - np.sin(np.radians(_ROW_LATITUDES - 0.5 * CELL_DEGREES))
) / (2.0 * COLUMNS)


class SurveyMask:

    def __init__(
        self,
        cells: np.ndarray | None = None,
        gating: bool = True
    ):

        # Rows from the north (row 0: 90..89 N), columns from
        # 180 W eastward.
        self.cells = np.zeros((ROWS, COLUMNS), dtype=bool) if cells is None else cells.astype(bool)

        # Whether it holds the sharp data back where unmapped.
        self.gating = gating

        # Bumped whenever cells are marked.
        self.version = 0

        self._weights: np.ndarray | None = None

    # -----------------------------------------------------
    # Marking
    # -----------------------------------------------------

    def mark(
        self,
        latitudes,
        longitudes,
        half_width: float
    ) -> int:
        """
        Mark every cell whose center is within `half_width`
        rad of any of the points (deg). Returns how many
        cells were newly mapped.
        """

        latitudes = np.atleast_1d(np.asarray(latitudes, dtype=np.float64))
        longitudes = np.atleast_1d(np.asarray(longitudes, dtype=np.float64))

        if not len(latitudes):
            return 0

        reach = math.degrees(half_width) + 0.5 * CELL_DEGREES

        before = int(self.cells.sum())

        for latitude, longitude in zip(latitudes, longitudes):

            top = min(int((90.0 - (latitude + reach)) / CELL_DEGREES), ROWS - 1)
            bottom = min(int((90.0 - (latitude - reach)) / CELL_DEGREES), ROWS - 1)

            rows = np.arange(max(top, 0), max(bottom, 0) + 1)

            row_latitudes = _ROW_LATITUDES[rows]

            # Columns within reach at each row: wider toward
            # the poles (all of them near one).
            cosine = np.cos(np.radians(row_latitudes))

            span = np.where(cosine > 1e-3, reach / np.maximum(cosine, 1e-3), 360.0)

            phi = math.radians(latitude)

            for row, row_latitude, width in zip(rows, row_latitudes, span):

                if width >= 180.0:

                    columns = np.arange(COLUMNS)

                else:

                    first = int(math.floor((longitude - width + 180.0) / CELL_DEGREES))
                    last = int(math.floor((longitude + width + 180.0) / CELL_DEGREES))

                    columns = np.arange(first, last + 1) % COLUMNS

                # Exactly: the centers within the half width.
                centers = -180.0 + (columns + 0.5) * CELL_DEGREES

                cos_angle = (
                    math.sin(phi) * math.sin(math.radians(row_latitude))
                    + math.cos(phi) * math.cos(math.radians(row_latitude)) * np.cos(np.radians(centers - longitude))
                )

                near = cos_angle >= math.cos(min(half_width + math.radians(0.5 * CELL_DEGREES), math.pi))

                self.cells[row, columns[near]] = True

        newly = int(self.cells.sum()) - before

        if newly:
            self.version += 1
            self._weights = None

        return newly

    def coverage(
        self
    ) -> float:
        """Share of the body's surface mapped (0..1, by area)."""

        return float((self.cells * _CELL_AREA[:, None]).sum())

    def is_mapped(
        self,
        latitude: float,
        longitude: float
    ) -> bool:

        row, column = _cell_of(np.array([latitude]), np.array([longitude]))

        return bool(self.cells[row[0], column[0]])

    # -----------------------------------------------------
    # Terrain
    # -----------------------------------------------------

    def weights(
        self,
        directions: np.ndarray
    ) -> np.ndarray:
        """
        0..1 at body-frame directions: 1 well inside mapped
        ground, 0 in unmapped, blended between cell centers.
        """

        grid = self._weights

        if grid is None:

            # One cell of border each way (longitude wraps;
            # the poles repeat their row).
            cells = self.cells.astype(np.float32)

            grid = np.concatenate([cells[:, -1:], cells, cells[:, :1]], axis=1)
            grid = np.concatenate([grid[:1], grid, grid[-1:]], axis=0)

            self._weights = grid

        latitude, longitude = latitude_longitude(directions)

        # Fractional cell coordinates, centers at .5 (+1 for
        # the border).
        y = (90.0 - latitude) / CELL_DEGREES - 0.5 + 1.0
        x = (longitude + 180.0) / CELL_DEGREES - 0.5 + 1.0

        x0 = np.clip(np.floor(x).astype(np.int64), 0, COLUMNS)
        y0 = np.clip(np.floor(y).astype(np.int64), 0, ROWS)

        fx = np.clip(x - x0, 0.0, 1.0)
        fy = np.clip(y - y0, 0.0, 1.0)

        return (
            (grid[y0, x0] * (1.0 - fx) + grid[y0, x0 + 1] * fx) * (1.0 - fy)
            + (grid[y0 + 1, x0] * (1.0 - fx) + grid[y0 + 1, x0 + 1] * fx) * fy
        ).astype(np.float64)

    def signature(
        self,
        directions: np.ndarray
    ) -> bytes:
        """What the weights at these directions depend on (the cells round them), for cache keys."""

        latitude, longitude = latitude_longitude(directions)

        rows, columns = _cell_of(latitude, longitude)

        touched = set()

        for row, column in zip(rows.tolist(), columns.tolist()):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    touched.add((min(max(row + dr, 0), ROWS - 1), (column + dc) % COLUMNS))

        ordered = sorted(touched)

        flags = np.array([self.cells[row, column] for row, column in ordered], dtype=bool)

        index = np.array(ordered, dtype=np.int32).tobytes()

        return index + np.packbits(flags).tobytes()

    # -----------------------------------------------------
    # Saving
    # -----------------------------------------------------

    def to_text(
        self
    ) -> str:

        return base64.b64encode(np.packbits(self.cells).tobytes()).decode("ascii")

    @classmethod
    def from_text(
        cls,
        text: str,
        gating: bool = True
    ) -> "SurveyMask":

        bits = np.frombuffer(base64.b64decode(text), dtype=np.uint8)

        cells = np.unpackbits(bits)[:ROWS * COLUMNS].reshape(ROWS, COLUMNS)

        return cls(cells, gating)


def _cell_of(
    latitude: np.ndarray,
    longitude: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    rows = np.clip(((90.0 - latitude) / CELL_DEGREES).astype(np.int64), 0, ROWS - 1)
    columns = (np.floor((longitude + 180.0) / CELL_DEGREES).astype(np.int64)) % COLUMNS

    return rows, columns


# ---------------------------------------------------------
# Registry (terrain looks masks up by body id)
# ---------------------------------------------------------

_MASKS: dict[str, SurveyMask] = {}

_LOCK = threading.Lock()


def register(
    body: str,
    mask: SurveyMask
):

    with _LOCK:
        _MASKS[body] = mask


def mask_for(
    body: str
) -> SurveyMask | None:

    return _MASKS.get(body)


def clear():

    with _LOCK:
        _MASKS.clear()
