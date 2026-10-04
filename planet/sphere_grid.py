from functools import lru_cache

import numpy as np

from planet.cube_sphere import (
    FACE_NORMALS,
    FACE_U,
    FACE_V,
    face_directions
)


_FOUR_OVER_PI = 4.0 / np.pi


class SphereGrid:

    # =====================================================
    # Global Simulation Grid
    # =====================================================
    #
    # The sphere as 6 faces of n x n cells, using the same
    # equal-angle cube-sphere mapping as the terrain chunks
    # (planet/cube_sphere.py). Cell (face, i, j) is centered
    # at face coordinates
    #
    #     a = -1 + (i + 0.5) * 2 / n,   b = -1 + (j + 0.5) * 2 / n
    #
    # and has flat index face * n * n + i * n + j.
    #
    # Provides what the plate simulation needs:
    #
    #   * cell_of(directions): nearest cell, vectorized;
    #   * neighbors: the 4 adjacent cells of every cell,
    #     across face edges too;
    #   * sample(): bilinear interpolation of a per-cell
    #     field at arbitrary directions, continuous across
    #     face edges (faces are padded with one ring of
    #     "ghost" cells copied from their neighbors).

    def __init__(
        self,
        n: int
    ):

        self.n = int(n)

        self.cell_count = 6 * self.n * self.n

        face, i, j = np.meshgrid(
            np.arange(6),
            np.arange(self.n),
            np.arange(self.n),
            indexing="ij"
        )

        self.face = face.ravel()
        self.i = i.ravel()
        self.j = j.ravel()

        self.directions = face_directions(
            self.face,
            self._coordinate(self.i),
            self._coordinate(self.j)
        )

        # Angular size of a cell at the face center.
        self.cell_angle = (np.pi / 2.0) / self.n

        self.neighbors = self._build_neighbors()

        self._ghost_sources = self._build_ghost_sources()

    def _coordinate(
        self,
        index
    ):

        return -1.0 + (np.asarray(index, dtype=np.float64) + 0.5) * (2.0 / self.n)

    def __reduce__(self):

        # Pickled (disk cache) as its size: the shared grid.
        return (sphere_grid, (self.n,))

    # =====================================================
    # Lookups
    # =====================================================

    @staticmethod
    def face_coordinates(
        directions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(face, a, b) for (m, 3) directions."""

        directions = np.asarray(directions, dtype=np.float64)

        face = np.argmax(directions @ FACE_NORMALS.T, axis=1)

        along = np.einsum("ij,ij->i", directions, FACE_NORMALS[face])

        a = np.arctan(np.einsum("ij,ij->i", directions, FACE_U[face]) / along) * _FOUR_OVER_PI
        b = np.arctan(np.einsum("ij,ij->i", directions, FACE_V[face]) / along) * _FOUR_OVER_PI

        return face, a, b

    def cell_of(
        self,
        directions: np.ndarray
    ) -> np.ndarray:
        """Index of the cell containing each direction."""

        face, a, b = self.face_coordinates(directions)

        n = self.n

        i = np.clip(((a + 1.0) * 0.5 * n).astype(np.int64), 0, n - 1)
        j = np.clip(((b + 1.0) * 0.5 * n).astype(np.int64), 0, n - 1)

        return (face * n + i) * n + j

    # =====================================================
    # Sampling
    # =====================================================

    def pad(
        self,
        values: np.ndarray
    ) -> np.ndarray:
        """Per-cell values -> (6, n + 2, n + 2) with ghost cells."""

        values = np.asarray(values)

        return values[self._ghost_sources]

    def sample(
        self,
        padded: np.ndarray,
        directions: np.ndarray
    ) -> np.ndarray:
        """Bilinear interpolation of a pad()-ed field."""

        face, a, b = self.face_coordinates(directions)

        n = self.n

        # Continuous cell coordinates; +0.5 shifts into the
        # padded array (cell centers at padded index + 0.5).
        x = (a + 1.0) * 0.5 * n + 0.5
        y = (b + 1.0) * 0.5 * n + 0.5

        x0 = np.clip(np.floor(x).astype(np.int64), 0, n)
        y0 = np.clip(np.floor(y).astype(np.int64), 0, n)

        fx = np.clip(x - x0, 0.0, 1.0)
        fy = np.clip(y - y0, 0.0, 1.0)

        v00 = padded[face, x0, y0]
        v10 = padded[face, x0 + 1, y0]
        v01 = padded[face, x0, y0 + 1]
        v11 = padded[face, x0 + 1, y0 + 1]

        top = v00 + (v10 - v00) * fx
        bottom = v01 + (v11 - v01) * fx

        return top + (bottom - top) * fy

    # =====================================================
    # Construction
    # =====================================================

    def _build_neighbors(
        self
    ) -> np.ndarray:
        """(cells, 4): the cells one step along +-u, +-v."""

        n = self.n

        face, i, j = self.face, self.i, self.j

        columns = []

        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):

            ni = i + di
            nj = j + dj

            inside = (ni >= 0) & (ni < n) & (nj >= 0) & (nj < n)

            index = np.empty(self.cell_count, dtype=np.int64)

            index[inside] = (face[inside] * n + ni[inside]) * n + nj[inside]

            # Across a face edge: the cell at the stepped
            # position (just past this face's edge).
            outside = ~inside

            if outside.any():

                directions = face_directions(
                    face[outside],
                    self._coordinate(ni[outside]),
                    self._coordinate(nj[outside])
                )

                index[outside] = self.cell_of(directions)

            columns.append(index)

        return np.stack(columns, axis=1)

    def _build_ghost_sources(
        self
    ) -> np.ndarray:
        """(6, n + 2, n + 2) cell index feeding each padded slot."""

        n = self.n

        face, pi, pj = np.meshgrid(
            np.arange(6),
            np.arange(n + 2),
            np.arange(n + 2),
            indexing="ij"
        )

        directions = face_directions(
            face.ravel(),
            self._coordinate(pi.ravel() - 1),
            self._coordinate(pj.ravel() - 1)
        )

        return self.cell_of(directions).reshape(6, n + 2, n + 2)


@lru_cache(maxsize=4)
def sphere_grid(
    n: int
) -> SphereGrid:
    """Shared grid per resolution (building one takes ~0.1 s)."""

    return SphereGrid(n)
