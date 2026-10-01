from dataclasses import dataclass

import numpy as np

from core.handle import Handle
from core.jobs import Job, JobSystem
from core.logger import Logger

from ecs.components import (
    CameraControllerComponent,
    PlanetComponent,
    TransformComponent
)
from ecs.entity import Entity

from graphics.draw_list import DrawItem
from graphics.mesh import Mesh

from math3d.matrices import translation

from planet.chunk import ChunkData, build_chunk
from planet.cube_sphere import ChunkKey, edge_length
from planet.lod import LodSelector
from planet.terrain import Terrain, TerrainSettings

from resources.resources import Resources
from scene.scene import Scene


@dataclass(slots=True)
class PlanetStats:

    planets: int = 0
    chunks_drawn: int = 0
    chunks_loaded: int = 0
    chunks_building: int = 0
    chunks_queued: int = 0


class _Chunk:

    __slots__ = ("mesh", "offset", "job", "last_used")

    def __init__(self):

        self.mesh: Mesh | None = None

        # Translation from planet center to vertex origin.
        self.offset: np.ndarray | None = None

        self.job: Job | None = None

        self.last_used = 0


class _Planet:

    # Streaming state of one planet entity.

    def __init__(
        self,
        component: PlanetComponent
    ):

        self.config = _config_of(component)

        settings = _terrain_settings(component)

        self.terrain = Terrain(settings)
        self.resolution = component.resolution

        self.selector = LodSelector(
            radius=settings.radius,
            max_elevation=settings.max_elevation,
            max_depth=component.max_depth,
            split_factor=component.split_factor
        )

        self.chunks: dict[ChunkKey, _Chunk] = {}

        self.draw: list[ChunkKey] = []
        self.wanted: list[ChunkKey] = []

        # Built chunks the last selection relied on (drawn,
        # or covering a split); never evicted while in use.
        self.in_use: set[ChunkKey] = set()

        # Finest vertex spacing, for ground queries that
        # should match the rendered surface.
        self.finest_spacing = edge_length(
            settings.radius,
            component.max_depth
        ) / (component.resolution - 1)

        # Reuse the last selection while the camera is
        # still and no chunk has arrived.
        self.last_camera: np.ndarray | None = None
        self.dirty = True

        self.ground_cache: tuple[np.ndarray, float] | None = None

        self.alive = True

    def ground_elevation(
        self,
        direction: np.ndarray
    ) -> float:
        """
        Terrain elevation under a unit direction. Cached:
        re-evaluated (about a millisecond) only after
        moving a quarter of the finest vertex spacing.
        """

        cached = self.ground_cache

        tolerance = 0.25 * self.finest_spacing / self.terrain.settings.radius

        if cached is not None and float(np.linalg.norm(direction - cached[0])) < tolerance:
            return cached[1]

        elevation = float(
            self.terrain.elevation(
                direction[None, :],
                spacing=self.finest_spacing
            )[0]
        )

        self.ground_cache = (direction.copy(), elevation)

        return elevation

    def ready(
        self,
        key: ChunkKey
    ) -> bool:

        chunk = self.chunks.get(key)

        return chunk is not None and chunk.mesh is not None


class PlanetSystem:

    # =====================================================
    # Planet Streaming
    # =====================================================
    #
    # Per frame, for every entity with a PlanetComponent:
    #
    #   1. LodSelector picks the quadtree nodes to draw for
    #      the camera position (in planet space).
    #   2. Missing nodes are built on worker threads
    #      (planet/chunk.py), at most MAX_BUILDING at a
    #      time, coarse and near first.
    #   3. Finished chunks are uploaded to the GPU on the
    #      main thread (JobSystem completion callbacks).
    #   4. Chunks unused for a while are freed.
    #
    # Chunks are not ECS entities: they are a cache owned by
    # this system and reach the renderer as DrawItems.

    # Concurrent chunk builds. Python threads share one
    # interpreter lock, so more mainly steals time from the
    # main thread.
    MAX_BUILDING = 3

    # Frames an unused chunk is kept for (moving back and
    # forth should not rebuild), and how many may be kept.
    KEEP_FRAMES = 300
    MAX_CHUNKS = 2500

    MATERIAL_KEY = "planet"

    def __init__(
        self,
        resources: Resources,
        jobs: JobSystem,
        material: Handle
    ):

        self._resources = resources
        self._jobs = jobs
        self._material = material

        self._planets: dict[Entity, _Planet] = {}

        self._frame = 0

        self._items: list[DrawItem] = []

        self.stats = PlanetStats()

    # =====================================================
    # Update
    # =====================================================

    def update(
        self,
        scene: Scene,
        camera_position
    ):
        """camera_position: world-space camera position."""

        self._frame += 1

        camera_position = np.asarray(camera_position, dtype=np.float64)

        material = self._resources.materials.get(self._material)

        items: list[DrawItem] = []

        seen = set()

        building = sum(
            1
            for planet in self._planets.values()
            for chunk in planet.chunks.values()
            if chunk.job is not None
        )

        queued = 0

        for entity, transform, component in scene.registry.view_with(
            TransformComponent,
            PlanetComponent
        ):

            seen.add(entity)

            planet = self._planets.get(entity)

            if planet is None or planet.config != _config_of(component):

                if planet is not None:

                    Logger.info("[Planet] Settings changed; rebuilding.")

                    building -= self._release(planet)

                planet = _Planet(component)

                self._planets[entity] = planet

            world = _rigid(transform.world_matrix)

            camera_local = (
                np.linalg.inv(world)
                @ np.append(camera_position, 1.0)
            )[:3]

            self._select(planet, camera_local)

            # ---------------------------------------------
            # Build requests
            # ---------------------------------------------

            for key in planet.wanted:

                chunk = planet.chunks.get(key)

                if chunk is not None and (chunk.mesh is not None or chunk.job is not None):
                    continue

                if building >= self.MAX_BUILDING:

                    queued += 1

                    continue

                self._request(planet, key)

                building += 1

            # ---------------------------------------------
            # Draw items
            # ---------------------------------------------

            frame = self._frame

            for key in planet.in_use:
                planet.chunks[key].last_used = frame

            for key in planet.draw:

                chunk = planet.chunks[key]

                items.append(
                    DrawItem(
                        mesh=chunk.mesh,
                        material=material,
                        world_matrix=world @ translation(chunk.offset),
                        casts_shadows=True
                    )
                )

            self._evict(planet)

        # Planets whose entity is gone.
        for entity in [e for e in self._planets if e not in seen]:

            self._release(self._planets.pop(entity))

        self._items = items

        self.stats = PlanetStats(
            planets=len(self._planets),
            chunks_drawn=len(items),
            chunks_loaded=sum(
                1
                for planet in self._planets.values()
                for chunk in planet.chunks.values()
                if chunk.mesh is not None
            ),
            chunks_building=building,
            chunks_queued=queued
        )

    @property
    def draw_items(
        self
    ) -> list[DrawItem]:

        return self._items

    # =====================================================
    # Selection
    # =====================================================

    @staticmethod
    def _select(
        planet: _Planet,
        camera_local: np.ndarray
    ):

        last = planet.last_camera

        if (
            not planet.dirty
            and last is not None
            and float(np.linalg.norm(camera_local - last)) < 1e-3
        ):
            return

        in_use = set()

        def ready(key):

            if planet.ready(key):

                in_use.add(key)

                return True

            return False

        selection = planet.selector.select(
            camera_local,
            ready
        )

        planet.draw = selection.draw
        planet.wanted = selection.wanted
        planet.in_use = in_use

        planet.last_camera = camera_local
        planet.dirty = False

    # =====================================================
    # Chunk Builds
    # =====================================================

    def _request(
        self,
        planet: _Planet,
        key: ChunkKey
    ):

        chunk = planet.chunks.setdefault(key, _Chunk())

        chunk.last_used = self._frame

        terrain = planet.terrain
        resolution = planet.resolution

        def on_complete(data: ChunkData):

            chunk.job = None

            if not planet.alive or planet.chunks.get(key) is not chunk:
                return

            chunk.mesh = Mesh.from_data(data.mesh)
            chunk.offset = data.center

            planet.dirty = True

        def on_error(error: BaseException):

            chunk.job = None

            Logger.error(
                "[Planet] Building chunk %s failed: %s: %s",
                key,
                type(error).__name__,
                error
            )

        chunk.job = self._jobs.submit(
            lambda: build_chunk(key, terrain, resolution),
            on_complete=on_complete,
            on_error=on_error,
            name=f"planet chunk {key.face}/{key.depth}/{key.x},{key.y}"
        )

    def _evict(
        self,
        planet: _Planet
    ):

        frame = self._frame

        in_use = planet.in_use

        stale = [
            (chunk.last_used, key)
            for key, chunk in planet.chunks.items()
            if key not in in_use and chunk.job is None
        ]

        over = len(planet.chunks) - self.MAX_CHUNKS

        stale.sort(key=lambda item: item[0])

        for index, (last_used, key) in enumerate(stale):

            if frame - last_used <= self.KEEP_FRAMES and index >= over:
                break

            chunk = planet.chunks.pop(key)

            if chunk.mesh is not None:
                chunk.mesh.delete()

            planet.dirty = True

    def _release(
        self,
        planet: _Planet
    ) -> int:
        """Free a planet's chunks; returns jobs cancelled."""

        planet.alive = False

        cancelled = 0

        for chunk in planet.chunks.values():

            if chunk.job is not None:

                chunk.job.cancel()
                chunk.job = None

                cancelled += 1

            if chunk.mesh is not None:

                chunk.mesh.delete()
                chunk.mesh = None

        planet.chunks.clear()

        return cancelled

    # =====================================================
    # Ground
    # =====================================================

    def update_camera_ground(
        self,
        scene: Scene
    ):
        """
        Point every planet-mode camera controller at the
        nearest planet: planet_center becomes its center and
        planet_radius the ground radius under the camera
        (sea level over oceans). The controller's altitude,
        speed scaling and min_altitude then follow the
        terrain.
        """

        planets = []

        for entity, transform, _ in scene.registry.view_with(
            TransformComponent,
            PlanetComponent
        ):

            planet = self._planets.get(entity)

            if planet is not None:
                planets.append((_rigid(transform.world_matrix), planet))

        if not planets:
            return

        for _, transform, controller in scene.registry.view_with(
            TransformComponent,
            CameraControllerComponent
        ):

            if not controller.planet_mode:
                continue

            position = np.append(transform.transform.position, 1.0)

            world, planet = min(
                planets,
                key=lambda entry: float(np.linalg.norm(entry[0][:3, 3] - position[:3]))
            )

            local = (np.linalg.inv(world) @ position)[:3]

            distance = float(np.linalg.norm(local))

            if distance <= 0.0:
                continue

            elevation = planet.ground_elevation(
                local / distance
            )

            controller.planet_center = tuple(float(v) for v in world[:3, 3])

            controller.planet_radius = (
                planet.terrain.settings.radius
                + max(elevation, 0.0)
            )

    # =====================================================
    # Shutdown
    # =====================================================

    def shutdown(self):

        for planet in self._planets.values():
            self._release(planet)

        self._planets.clear()

        self._items = []


def _terrain_settings(
    component: PlanetComponent
) -> TerrainSettings:

    return TerrainSettings(
        seed=component.seed,
        radius=component.radius,
        continent_frequency=component.continent_frequency,
        continent_height=component.continent_height,
        land_bias=component.land_bias,
        mountain_frequency=component.mountain_frequency,
        mountain_height=component.mountain_height,
        detail_height=component.detail_height
    )


def _config_of(
    component: PlanetComponent
) -> tuple:
    """Everything that, when changed, requires a rebuild."""

    return (
        _terrain_settings(component),
        component.resolution,
        component.max_depth,
        component.split_factor
    )


def _rigid(
    matrix: np.ndarray
) -> np.ndarray:
    """Rotation + translation of a world matrix (scale dropped)."""

    matrix = np.asarray(matrix, dtype=np.float64)

    rigid = np.eye(4)

    basis = matrix[:3, :3]

    rigid[:3, :3] = basis / np.linalg.norm(basis, axis=0, keepdims=True)
    rigid[:3, 3] = matrix[:3, 3]

    return rigid
