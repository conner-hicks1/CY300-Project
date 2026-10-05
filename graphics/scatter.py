import ctypes
import math

from dataclasses import dataclass, field

import numpy as np

from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_DRAW_INDIRECT_BUFFER,
    GL_DYNAMIC_DRAW,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_FALSE,
    GL_FLOAT,
    GL_STATIC_DRAW,
    GL_TRIANGLES,
    GL_UNSIGNED_INT,
    glBindBuffer,
    glBindVertexArray,
    glBufferData,
    glBufferSubData,
    glDeleteBuffers,
    glDeleteVertexArrays,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glMultiDrawElementsIndirect,
    glVertexAttribDivisor,
    glVertexAttribPointer
)

from graphics.draw_list import frustum_planes, spheres_in_frustum
from graphics.geometry_pool import (
    COLOR_SIZE,
    NORMAL_SIZE,
    POSITION_SIZE,
    TANGENT_SIZE,
    UV_SIZE,
    GeometryPool
)
from graphics.mesh import Mesh
from graphics.scatter_meshes import build_meshes

from planet.scatter import LAYERS


# =========================================================
# Scatter Rendering
# =========================================================
#
# The rocks, trees and grass of planet/scatter.py, drawn
# instanced: one buffer of instances per layer, sorted into
# blocks (one kind, a few tens of meters of ground each), so
# that each frame one multi-draw per layer covers the blocks
# in view, each block with the mesh for its distance (trees:
# near or far detail). Instanced attributes follow each
# command's baseInstance, so no per-instance work happens in
# Python per frame.

FLOATS = 12                 # per instance

# Block size (m) per layer, and where trees switch to their
# far meshes (m).
BLOCK_SIZE = {"trees": 80.0, "rocks": 40.0, "grass": 16.0}
NEAR_TREES = 160.0

# Foliage sway (radians at the top) per layer.
SWAY = {"trees": 0.012, "rocks": 0.0, "grass": 0.08}

# How far each layer casts shadows (m): the near cascades,
# where its shadows are big enough to see (grass: none).
SHADOW_REACH = {"trees": 300.0, "rocks": 120.0, "grass": 0.0}

INSTANCE_LOCATIONS = (6, 7, 8)


@dataclass(slots=True)
class LayerData:
    """A layer's instances, packed (CPU), with its blocks."""

    instances: np.ndarray       # (n, FLOATS) float32
    kinds: np.ndarray           # (b,) per block
    starts: np.ndarray          # (b,) first instance
    counts: np.ndarray          # (b,)
    centers: np.ndarray         # (b, 3) planet frame, from the anchor
    radii: np.ndarray           # (b,)


@dataclass(slots=True)
class ScatterFrame:
    """What the planet system hands the renderer."""

    version: int
    layers: dict[str, LayerData]

    # The anchor (camera-relative world m), the planet's
    # frame (quaternion: planet frame -> world), the camera in
    # the planet frame from the anchor (m).
    anchor: np.ndarray
    frame: np.ndarray
    camera: np.ndarray

    # The planet's material values (uBodyCenter, uBodyFrame).
    body_values: dict = field(default_factory=dict)


def quaternions_from_up(
    up: np.ndarray,
    yaw: np.ndarray
) -> np.ndarray:
    """(n, 4) xyzw: local +y onto `up`, turned by `yaw` about it."""

    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up, axis=1, keepdims=True)

    # Yaw about y first.
    half = 0.5 * np.asarray(yaw, dtype=np.float64)

    yaw_q = np.stack((np.zeros_like(half), np.sin(half), np.zeros_like(half), np.cos(half)), axis=1)

    # Shortest arc y -> up.
    y = np.array([0.0, 1.0, 0.0])

    axis = np.cross(np.broadcast_to(y, up.shape), up)

    dot = up[:, 1]

    align = np.concatenate((axis, (1.0 + dot)[:, None]), axis=1)

    # (Upside down: half a turn about x.)
    flip = dot < -0.9999
    align[flip] = (1.0, 0.0, 0.0, 0.0)

    align /= np.linalg.norm(align, axis=1, keepdims=True)

    return _multiply(align, yaw_q)


def _multiply(
    a: np.ndarray,
    b: np.ndarray
) -> np.ndarray:

    ax, ay, az, aw = a.T
    bx, by, bz, bw = b.T

    return np.stack((
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ), axis=1)


def pack_layer(
    cells,
    anchor: np.ndarray,
    block: float
) -> LayerData:
    """ScatterCells of one layer -> instances sorted into blocks around `anchor` (planet frame, m)."""

    cells = [c for c in cells if len(c)]

    if not cells:
        return LayerData(
            np.zeros((0, FLOATS), np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64),
            np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros(0)
        )

    absolute = np.vstack([c.positions for c in cells])

    positions = absolute - anchor

    up = np.vstack([c.up for c in cells])
    yaw = np.concatenate([c.yaw for c in cells])
    scale = np.concatenate([c.scale for c in cells])
    kind = np.concatenate([c.kind for c in cells]).astype(np.int64)
    tint = np.vstack([c.tint for c in cells])

    blocks = np.floor(positions / block).astype(np.int64)

    order = np.lexsort((blocks[:, 2], blocks[:, 1], blocks[:, 0], kind))

    positions, up, yaw, scale, kind, tint, blocks, absolute = (
        positions[order], up[order], yaw[order], scale[order], kind[order], tint[order], blocks[order], absolute[order]
    )

    key = np.column_stack((kind, blocks))

    change = np.any(key[1:] != key[:-1], axis=1)

    starts = np.concatenate(([0], np.flatnonzero(change) + 1))
    counts = np.diff(np.concatenate((starts, [len(kind)])))

    instances = np.zeros((len(kind), FLOATS), dtype=np.float32)

    instances[:, 0:3] = positions
    instances[:, 3] = scale
    instances[:, 4:8] = quaternions_from_up(up, yaw)
    instances[:, 8:11] = tint

    # A seed per instance from where it stands (its surface
    # detail; the same wherever the anchor is).
    instances[:, 11] = np.mod(np.floor(absolute * 10.0) @ np.array([0.1031, 0.1030, 0.0973]), 1.0)

    centers = np.add.reduceat(positions, starts, axis=0) / counts[:, None]

    extent = np.linalg.norm(positions - np.repeat(centers, counts, axis=0), axis=1) + scale

    radii = np.maximum.reduceat(extent, starts)

    return LayerData(instances, kind[starts], starts, counts, centers, radii)


class _LayerBuffers:

    def __init__(self):

        self.vao = int(glGenVertexArrays(1))
        self.instances = int(glGenBuffers(1))
        self.commands = int(glGenBuffers(1))

        self.command_capacity = 0
        self.pool_buffers = None
        self.count = 0

    def configure(
        self,
        pool: GeometryPool
    ):
        """(Re)bind the pool's vertex and index buffers (they move when the pool grows)."""

        buffers = (pool.vertex_buffer, pool.index_buffer)

        if buffers == self.pool_buffers:
            return

        self.pool_buffers = buffers

        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, pool.vertex_buffer)

        offset = 0

        for location, size in enumerate((POSITION_SIZE, NORMAL_SIZE, COLOR_SIZE, UV_SIZE, TANGENT_SIZE)):

            glEnableVertexAttribArray(location)
            glVertexAttribPointer(location, size, GL_FLOAT, GL_FALSE, pool.vertex_stride, ctypes.c_void_p(offset * 4))

            offset += size

        glBindBuffer(GL_ARRAY_BUFFER, self.instances)

        for i, location in enumerate(INSTANCE_LOCATIONS):

            glEnableVertexAttribArray(location)
            glVertexAttribPointer(location, 4, GL_FLOAT, GL_FALSE, FLOATS * 4, ctypes.c_void_p(16 * i))
            glVertexAttribDivisor(location, 1)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pool.index_buffer)

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def upload(
        self,
        instances: np.ndarray
    ):

        glBindBuffer(GL_ARRAY_BUFFER, self.instances)
        glBufferData(GL_ARRAY_BUFFER, max(instances.nbytes, 16), instances if len(instances) else None, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        self.count = len(instances)

    def draw(
        self,
        commands: np.ndarray
    ):

        if len(commands) == 0:
            return

        data = np.ascontiguousarray(commands, dtype=np.uint32)

        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, self.commands)

        if data.nbytes > self.command_capacity:

            capacity = max(4096, self.command_capacity)

            while capacity < data.nbytes:
                capacity *= 2

            glBufferData(GL_DRAW_INDIRECT_BUFFER, capacity, None, GL_DYNAMIC_DRAW)

            self.command_capacity = capacity

        glBufferSubData(GL_DRAW_INDIRECT_BUFFER, 0, data.nbytes, data)

        glBindVertexArray(self.vao)

        glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, ctypes.c_void_p(0), len(data), 0)

        glBindVertexArray(0)
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0)

    def delete(self):

        glDeleteBuffers(2, [self.instances, self.commands])
        glDeleteVertexArrays(1, [self.vao])


class ScatterRenderer:

    def __init__(self):

        self._meshes = {key: Mesh.from_data(data) for key, data in build_meshes().items()}

        self._layers = {layer.name: _LayerBuffers() for layer in LAYERS}

        self._reach = {layer.name: layer.reach for layer in LAYERS}

        self._version = None

        self._table = None

        self.stats = {"instances": 0, "drawn": 0}

    def _mesh_table(
        self
    ) -> np.ndarray:
        """(kinds, 2 details, 3): index count, first index, base vertex; near, far."""

        if self._table is None:

            kinds = max(kind for kind, _ in self._meshes) + 1

            table = np.zeros((kinds, 2, 3), dtype=np.int64)

            for kind in range(kinds):
                for column, detail in enumerate(("near", "far")):

                    mesh = self._meshes.get((kind, detail)) or self._meshes[(kind, "near")]

                    allocation = mesh.allocation

                    table[kind, column] = (allocation.index_count, allocation.first_index, allocation.base_vertex)

            self._table = table

        return self._table

    def update(
        self,
        frame: ScatterFrame
    ):

        if frame.version == self._version:
            return

        self._version = frame.version

        total = 0

        for name, buffers in self._layers.items():

            data = frame.layers.get(name)

            buffers.upload(data.instances if data is not None else np.zeros((0, FLOATS), np.float32))

            total += buffers.count

        self.stats["instances"] = total

    def commands(
        self,
        frame: ScatterFrame,
        clip_matrix: np.ndarray | None,
        shadows: bool = False
    ) -> dict[str, np.ndarray]:
        """
        Per layer: the indirect commands for the blocks within
        reach (and in view, given a render-space clip matrix;
        then also the stats).
        """

        rotation = _rotation_matrix(frame.frame)

        planes = frustum_planes(clip_matrix) if clip_matrix is not None else None

        table = self._mesh_table()

        result = {}

        drawn = 0

        for name, data in frame.layers.items():

            if data is None or len(data.starts) == 0:
                result[name] = np.zeros((0, 5), np.uint32)
                continue

            distance = np.linalg.norm(data.centers - frame.camera, axis=1) - data.radii

            reach = SHADOW_REACH.get(name, 0.0) if shadows else self._reach[name]

            keep = distance < reach

            if planes is not None and np.any(keep):

                world = data.centers @ rotation.T + frame.anchor

                keep &= spheres_in_frustum(planes, world, data.radii)

            blocks = np.flatnonzero(keep)

            far = (distance[blocks] >= NEAR_TREES) if name == "trees" else np.zeros(len(blocks), dtype=bool)

            meshes = table[data.kinds[blocks], far.astype(np.int64)]

            commands = np.empty((len(blocks), 5), dtype=np.uint32)

            commands[:, 0] = meshes[:, 0]
            commands[:, 1] = data.counts[blocks]
            commands[:, 2] = meshes[:, 1]
            commands[:, 3] = meshes[:, 2]
            commands[:, 4] = data.starts[blocks]

            drawn += int(data.counts[blocks].sum())

            result[name] = commands

        if planes is not None:
            self.stats["drawn"] = drawn

        return result

    def draw(
        self,
        shader,
        frame: ScatterFrame,
        commands: dict[str, np.ndarray],
        time_seconds: float
    ):
        """With `shader` bound and set up (lit, or depth)."""

        pool = GeometryPool.instance()

        shader.set_vec4("uScatterFrame", tuple(float(v) for v in frame.frame))
        shader.set_vec3("uScatterAnchor", tuple(float(v) for v in frame.anchor))
        shader.set_vec3("uScatterCamera", tuple(float(v) for v in frame.camera))

        if shader.has_uniform("uTime"):
            shader.set_float("uTime", float(time_seconds % 10_000.0))

        for name, buffers in self._layers.items():

            layer_commands = commands.get(name)

            if layer_commands is None or len(layer_commands) == 0:
                continue

            buffers.configure(pool)

            shader.set_float("uScatterReach", float(self._reach[name]))

            if shader.has_uniform("uScatterSway"):
                shader.set_float("uScatterSway", SWAY.get(name, 0.0))

            buffers.draw(layer_commands)

    def delete(self):

        for buffers in self._layers.values():
            buffers.delete()

        for mesh in self._meshes.values():
            mesh.delete()


def _rotation_matrix(
    q
) -> np.ndarray:

    x, y, z, w = (float(v) for v in q)

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
