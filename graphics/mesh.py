import numpy as np

from core.assertions import engine_assert
from core.logger import Logger

from graphics.geometry_pool import (
    GeometryPool,
    PoolAllocation
)
from graphics.mesh_data import MeshData


class Mesh:

    # =====================================================
    # Mesh
    # =====================================================
    #
    # A range of vertices and indices in the shared
    # GeometryPool (one VAO for every mesh), so meshes can
    # be drawn in batches with multi-draw indirect.
    #
    # Keeps its MeshData for CPU-side queries: bounds for
    # picking, and a bounding sphere for frustum culling.

    @classmethod
    def from_data(
        cls,
        data: MeshData,
        pool: GeometryPool | None = None
    ) -> "Mesh":

        engine_assert(
            data is not None,
            "Mesh.from_data() requires MeshData."
        )

        return cls(
            data,
            pool or GeometryPool.instance()
        )

    def __init__(
        self,
        data: MeshData,
        pool: GeometryPool
    ):

        self.data = data

        self._pool = pool

        self.allocation: PoolAllocation | None = pool.allocate(
            data.vertices,
            data.indices
        )

        # Local-space axis-aligned bounds (min, max).
        self.bounds: tuple[np.ndarray, np.ndarray] = data.bounds

        low, high = self.bounds

        # Local bounding sphere (center, radius) around the
        # box: cheap to transform and test per frame.
        self.bounding_center = (
            (np.asarray(low, dtype=np.float64) + np.asarray(high, dtype=np.float64))
            * 0.5
        )

        self.bounding_radius = float(
            np.linalg.norm(
                np.asarray(high, dtype=np.float64)
                - np.asarray(low, dtype=np.float64)
            ) * 0.5
        )

        Logger.debug(
            "[Mesh] Allocated %d vertices, %d indices.",
            self.allocation.vertex_count,
            self.allocation.index_count
        )

    # =====================================================
    # Queries
    # =====================================================

    @property
    def vertex_count(
        self
    ) -> int:

        return self.allocation.vertex_count

    @property
    def index_count(
        self
    ) -> int:

        return self.allocation.index_count

    @property
    def triangle_count(
        self
    ) -> int:

        return self.allocation.index_count // 3

    # =====================================================
    # Cleanup
    # =====================================================

    def delete(self):

        if self.allocation is None:
            return

        Logger.debug(
            "[Mesh] Freeing %d vertices.",
            self.allocation.vertex_count
        )

        self._pool.free(
            self.allocation
        )

        self.allocation = None
