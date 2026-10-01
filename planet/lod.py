from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from planet.cube_sphere import (
    FACE_NORMALS,
    FACE_U,
    FACE_V,
    ROOT_KEYS,
    ChunkKey,
    edge_length,
    face_directions
)


# =========================================================
# Level-of-Detail Selection
# =========================================================
#
# Walks the six face quadtrees each frame and decides which
# nodes to draw. A node splits into its four children when
# the camera is within split_factor x its edge length of
# it, but children replace it only once all four are built;
# until then the parent keeps drawing (no holes while
# chunks stream in). Likewise a node that should merge
# keeps drawing its children until it is built itself.
#
# Nodes entirely below the horizon are skipped: the planet
# hides them (Cesium-style horizon culling against the
# sea-level sphere).
#
# Each depth level is processed as one numpy batch rather
# than node by node, which keeps the walk well under a
# millisecond for hundreds of nodes.

# A split node stays split until the camera is this much
# farther away than the split distance, so nodes near the
# threshold do not flicker between levels.
MERGE_HYSTERESIS = 1.25


@dataclass(slots=True)
class LodSelection:

    # Nodes to draw this frame (all are ready).
    draw: list[ChunkKey] = field(default_factory=list)

    # Nodes that should be built, most important first.
    wanted: list[ChunkKey] = field(default_factory=list)

    # Every node visited (drawn, split or wanted), for
    # keeping their chunks alive.
    visited: int = 0


class LodSelector:

    def __init__(
        self,
        radius: float,
        max_elevation: float,
        max_depth: int,
        split_factor: float
    ):

        self.radius = radius
        self.max_elevation = max_elevation
        self.max_depth = max_depth
        self.split_factor = split_factor

        self._split: set[ChunkKey] = set()

    def select(
        self,
        camera: np.ndarray,
        ready: Callable[[ChunkKey], bool]
    ) -> LodSelection:
        """
        camera: camera position in planet space (planet
            center at the origin).
        ready: whether a node's chunk is built.
        """

        camera = np.asarray(camera, dtype=np.float64)

        selection = LodSelection()

        wanted: list[tuple[int, float, ChunkKey]] = []

        split: set[ChunkKey] = set()

        level = list(ROOT_KEYS)

        while level:

            distances, visible = self._measure(level, camera)

            next_level: list[ChunkKey] = []

            for key, distance, is_visible in zip(level, distances, visible):

                if not is_visible:
                    continue

                selection.visited += 1

                key_ready = ready(key)

                threshold = (
                    self.split_factor
                    * edge_length(self.radius, key.depth)
                )

                if key in self._split:
                    threshold *= MERGE_HYSTERESIS

                want_split = (
                    key.depth < self.max_depth
                    and distance < threshold
                )

                if want_split or not key_ready:

                    children = key.children()

                    if key.depth < self.max_depth and all(map(ready, children)):

                        # Children cover this node.
                        split.add(key)

                        next_level.extend(children)

                        if not key_ready and not want_split:
                            wanted.append((key.depth, distance, key))

                        continue

                    if want_split:

                        for child in children:

                            if not ready(child):
                                wanted.append((child.depth, distance, child))

                if key_ready:
                    selection.draw.append(key)

                else:
                    wanted.append((key.depth, distance, key))

            level = next_level

        self._split = split

        # Coarse before fine, near before far.
        wanted.sort(key=lambda item: (item[0], item[1]))

        selection.wanted = [key for _, _, key in wanted]

        return selection

    # =====================================================
    # Geometry (one depth level at a time)
    # =====================================================

    def _measure(
        self,
        keys: list[ChunkKey],
        camera: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Per node: distance from the camera to the node's
        nearest surface point, and whether any part of it
        can be above the horizon.
        """

        count = len(keys)

        face = np.fromiter((k.face for k in keys), dtype=np.int64, count=count)
        depth = np.fromiter((k.depth for k in keys), dtype=np.int64, count=count)
        x = np.fromiter((k.x for k in keys), dtype=np.float64, count=count)
        y = np.fromiter((k.y for k in keys), dtype=np.float64, count=count)

        size = 2.0 / np.left_shift(1, depth)

        a0 = -1.0 + x * size
        b0 = -1.0 + y * size
        a1 = a0 + size
        b1 = b0 + size

        # -------------------------------------------------
        # Nearest point
        # -------------------------------------------------
        #
        # Project the camera direction onto each node's face
        # and clamp into the node's square. Cameras in the
        # opposite hemisphere of a face use the node center.

        camera_length = float(np.linalg.norm(camera))

        camera_direction = (
            camera / camera_length
            if camera_length > 0.0
            else np.array([0.0, 1.0, 0.0])
        )

        along = FACE_NORMALS[face] @ camera_direction

        facing = along > 1e-6

        safe_along = np.where(facing, along, 1.0)

        cam_a = np.arctan((FACE_U[face] @ camera_direction) / safe_along) * (4.0 / np.pi)
        cam_b = np.arctan((FACE_V[face] @ camera_direction) / safe_along) * (4.0 / np.pi)

        near_a = np.where(facing, np.clip(cam_a, a0, a1), (a0 + a1) * 0.5)
        near_b = np.where(facing, np.clip(cam_b, b0, b1), (b0 + b1) * 0.5)

        nearest = face_directions(face, near_a, near_b) * self.radius

        distances = np.linalg.norm(nearest - camera, axis=1)

        # -------------------------------------------------
        # Horizon
        # -------------------------------------------------
        #
        # Sample points at the highest possible terrain: the
        # nearest point, the center and the four corners. The
        # node is hidden only if all are behind the sea-level
        # sphere.

        sample_a = np.stack((near_a, (a0 + a1) * 0.5, a0, a1, a0, a1), axis=1)
        sample_b = np.stack((near_b, (b0 + b1) * 0.5, b0, b0, b1, b1), axis=1)

        samples = face_directions(
            face[:, None],
            sample_a,
            sample_b
        ) * (self.radius + self.max_elevation)

        visible = _above_horizon(
            samples,
            camera,
            self.radius
        ).any(axis=1)

        return distances, visible


def _above_horizon(
    points: np.ndarray,
    camera: np.ndarray,
    occluder_radius: float
) -> np.ndarray:
    """
    Whether each point can be seen past a sphere of
    `occluder_radius` at the origin (it is visible unless it
    is both beyond the horizon plane and inside the sphere's
    shadow cone).
    """

    to_center = -camera

    horizon_squared = float(to_center @ to_center) - occluder_radius ** 2

    if horizon_squared <= 0.0:

        # Camera inside the sphere: nothing is occluded.
        return np.ones(points.shape[:-1], dtype=bool)

    to_point = points - camera

    projection = to_point @ to_center

    beyond_plane = projection > horizon_squared

    inside_cone = (
        projection * projection
        / np.einsum("...i,...i->...", to_point, to_point)
    ) > horizon_squared

    return ~(beyond_plane & inside_cone)
