import time

import numpy as np

from core.jobs import JobSystem
from core.logger import Logger

from graphics.scatter import BLOCK_SIZE, ScatterFrame, pack_layer

from math3d import quaternion

from planet.cube_sphere import ChunkKey
from planet.scatter import (
    BIOME_ROCK,
    LAYERS,
    ScatterLayer,
    SurfaceSettings,
    cell_depth,
    cells_around,
    generate_cell
)


# =========================================================
# Scatter Streaming
# =========================================================
#
# Keeps the rocks, trees and grass (planet/scatter.py)
# around the camera on the planet it stands on: which cells
# each layer needs within its reach, building them on the
# workers, dropping the ones left behind, and packing them
# for the renderer (graphics/scatter.py), also on a worker,
# around an anchor near the camera (instances are stored
# relative to it, so float32 stays exact; it moves when the
# camera has gone far from it).

class ScatterSystem:

    # Cells built at once (workers).
    MAX_JOBS = 6

    # The anchor follows the camera past this distance (m).
    ANCHOR_RANGE = 1_500.0

    # Cells are kept until this far beyond a layer's reach.
    KEEP = 1.4

    def __init__(
        self,
        jobs: JobSystem
    ):

        self._jobs = jobs

        self.enabled = True

        self._entity = None
        self._terrain = None
        self._surface: SurfaceSettings | None = None

        self._cells: dict[tuple[str, ChunkKey], object] = {}
        self._pending: dict[tuple[str, ChunkKey], object] = {}

        self._wanted: dict[str, set] = {}
        self._wanted_at: dict[str, np.ndarray] = {}

        self._anchor: np.ndarray | None = None

        self._packed = None
        self._packing = None
        self._dirty = False

        self._version = 0

        self.frame: ScatterFrame | None = None

    # =====================================================
    # Per Frame
    # =====================================================

    def update(
        self,
        entity,
        terrain,
        surface: SurfaceSettings,
        world: np.ndarray,
        camera_local: np.ndarray,
        camera_position: np.ndarray,
        altitude: float,
        body_values: dict,
        busy: bool = False
    ):
        """
        The planet the camera is on (None: nothing nearby).
        world: its rigid world matrix; camera_local: the
        camera in its frame (m); altitude: above the ground.
        busy: the terrain is still streaming (it goes first:
        meanwhile one cell at a time).
        """

        if not self.enabled or entity is None:

            self.clear()

            return

        if entity != self._entity or terrain is not self._terrain or surface != self._surface:

            self.clear()

            self._entity = entity
            self._terrain = terrain
            self._surface = surface

        radius = terrain.settings.radius

        direction = camera_local / max(float(np.linalg.norm(camera_local)), 1.0)

        ground = direction * float(np.linalg.norm(camera_local) - altitude)

        if self._anchor is None or float(np.linalg.norm(ground - self._anchor)) > self.ANCHOR_RANGE:

            self._anchor = ground.copy()
            self._dirty = True

        # -------------------------------------------------
        # Which cells, per layer
        # -------------------------------------------------

        for layer in LAYERS:

            if altitude > layer.reach:

                if self._wanted.get(layer.name):
                    self._wanted[layer.name] = set()
                    self._drop(layer, set())

                continue

            depth = cell_depth(radius, layer.cell)

            last = self._wanted_at.get(layer.name)

            if last is None or float(np.linalg.norm(ground - last)) > 0.25 * layer.cell:

                self._wanted[layer.name] = cells_around(radius, direction, layer.reach, depth)
                self._wanted_at[layer.name] = ground.copy()

                keep = cells_around(radius, direction, self.KEEP * layer.reach, depth)

                self._drop(layer, keep)

        # -------------------------------------------------
        # Build what is missing (nearest layers first)
        # -------------------------------------------------

        started = 0

        for layer in sorted(LAYERS, key=lambda l: l.reach):

            for key in self._wanted.get(layer.name, ()):

                # While the terrain streams, one cell at a time.
                limit = 1 if busy else self.MAX_JOBS

                if started >= limit or len(self._pending) >= (1 if busy else 2 * self.MAX_JOBS):
                    break

                ident = (layer.name, key)

                if ident in self._cells or ident in self._pending:
                    continue

                self._start(layer, key)

                started += 1

        # -------------------------------------------------
        # Pack for drawing (on a worker)
        # -------------------------------------------------

        if self._dirty and self._packing is None:
            self._pack()

        # The frame the renderer draws: the latest packing,
        # placed for this frame's camera.
        if self._packed is not None:

            version, anchor, layers = self._packed

            rotation = np.asarray(world, dtype=np.float64)[:3, :3]

            self.frame = ScatterFrame(
                version=version,
                layers=layers,
                anchor=(np.asarray(world, dtype=np.float64) @ np.append(anchor, 1.0))[:3] - camera_position,
                frame=np.asarray(quaternion.from_matrix3(rotation), dtype=np.float64),
                camera=camera_local - anchor,
                body_values=body_values
            )

    def clear(self):

        for job in self._pending.values():
            job.cancel()

        if self._packing is not None:
            self._packing.cancel()

        self._entity = None
        self._terrain = None
        self._surface = None
        self._cells = {}
        self._pending = {}
        self._wanted = {}
        self._wanted_at = {}
        self._anchor = None
        self._packed = None
        self._packing = None
        self._dirty = False
        self.frame = None

    # =====================================================
    # Cells
    # =====================================================

    def _start(
        self,
        layer: ScatterLayer,
        key: ChunkKey
    ):

        terrain = self._terrain
        surface = self._surface

        ident = (layer.name, key)

        def on_complete(cell):

            self._pending.pop(ident, None)

            if terrain is not self._terrain or key not in self._wanted.get(layer.name, ()):
                return

            self._cells[ident] = cell
            self._dirty = True

        def on_error(error):

            self._pending.pop(ident, None)

            Logger.error("[Scatter] %s cell %s failed: %s: %s", layer.name, key, type(error).__name__, error)

        self._pending[ident] = self._jobs.submit(
            lambda: generate_cell(terrain, surface, layer, key),
            on_complete=on_complete,
            on_error=on_error,
            name=f"scatter {layer.name}"
        )

    def _drop(
        self,
        layer: ScatterLayer,
        keep: set
    ):

        stale = [ident for ident in self._cells if ident[0] == layer.name and ident[1] not in keep]

        for ident in stale:
            del self._cells[ident]

        if stale:
            self._dirty = True

    def _pack(self):

        anchor = self._anchor.copy()

        cells = {layer.name: [c for (name, _), c in self._cells.items() if name == layer.name] for layer in LAYERS}

        self._dirty = False

        self._version += 1

        version = self._version

        def work():

            return {name: pack_layer(group, anchor, BLOCK_SIZE[name]) for name, group in cells.items()}

        def on_complete(layers):

            self._packing = None

            if self._anchor is None:
                return

            self._packed = (version, anchor, layers)

        def on_error(error):

            self._packing = None

            Logger.error("[Scatter] Packing failed: %s: %s", type(error).__name__, error)

        self._packing = self._jobs.submit(work, on_complete=on_complete, on_error=on_error, name="scatter pack")


def surface_settings(
    component
) -> SurfaceSettings | None:
    """A planet's surface for the scatter; None for giants."""

    if component.palette == "bands":
        return None

    mineral = component.palette == "mineral"

    return SurfaceSettings(
        life=bool(component.life),
        has_liquid=component.liquid != "none",
        mineral=mineral,
        frost_point=float(component.frost_point),
        rock_color=tuple(float(c) for c in component.color_steep) if mineral else BIOME_ROCK,
        seed=int(component.seed)
    )
