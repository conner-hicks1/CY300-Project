import math
import time

from functools import lru_cache

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from core.handle import Handle
from core.jobs import Job, JobSystem
from core.logger import Logger

from ecs.components import (
    AtmosphereComponent,
    CameraControllerComponent,
    PlanetComponent,
    TransformComponent
)
from ecs.entity import Entity

from graphics.atmosphere import AtmosphereParameters
from graphics.draw_list import DrawItem
from graphics.mesh import Mesh
from graphics.texture import Texture2D

from math3d import quaternion
from math3d.matrices import translation

from planet.chunk import ChunkData, build_cached_chunk
from planet.cube_sphere import ChunkKey, edge_length
from planet.lod import LodSelector
from planet.phases import SUBSTANCES
from planet.maps import color_map_path
from planet.shape import shape_settings
from planet.spawn import SpawnPoint, find_spawn
from planet.terrain import Terrain, TerrainSettings, body_shape

from resources.resources import Resources
from scene.scene import Scene


# Terrain view modes (assets/shaders/include/terrain.glsl
# terrainOverlay; index = uTerrainView).
TERRAIN_VIEWS: tuple[tuple[str, str], ...] = (
    ("Natural", "Biome colors."),
    ("Elevation", "Height map with 500 m contour lines; coast in black."),
    ("Slope", "Flat ground bright, cliffs dark."),
    ("Temperature", "Annual mean, blue (-30 C) to red (35 C); dark line at freezing."),
    ("Rainfall", "Yearly rain, dry (tan) to wet (blue), log scale."),
    ("Detail level", "Quadtree depth of each chunk."),
    ("Plates", "Tectonic plates, outlined at their boundaries."),
    ("Crust age", "Sea floor from young (red, at ridges) to old (blue); continents grey."),
    ("Crust type", "Continental (tan) or oceanic (blue) crust."),
    ("Boundaries", "Plates converging (red) or pulling apart (blue)."),
    ("Biomes", "Ice, tundra, taiga, forest, grassland, desert, savanna, rainforest."),
)


@dataclass(slots=True)
class PlanetStats:

    planets: int = 0
    chunks_drawn: int = 0
    chunks_loaded: int = 0
    chunks_building: int = 0
    chunks_queued: int = 0


class _Chunk:

    __slots__ = ("mesh", "offset", "local_matrix", "job", "last_used")

    def __init__(self):

        self.mesh: Mesh | None = None

        # Translation from planet center to vertex origin,
        # and as a matrix (built once).
        self.offset: np.ndarray | None = None
        self.local_matrix: np.ndarray | None = None

        self.job: Job | None = None

        self.last_used = 0


class _Planet:

    # Streaming state of one planet entity.

    def __init__(
        self,
        component: PlanetComponent,
        field=None,
        climate=None
    ):

        self.config = _config_of(component)

        # Tectonic field and climate the terrain was built
        # from (None = noise continents / latitude climate),
        # and their versions, for change checks.
        self.field = field
        self.climate = climate
        self.data_version = _data_version(field, climate)

        settings = terrain_settings_for(component, climate)

        self.terrain = Terrain(settings, field, climate)

        self.created = time.monotonic()

        # The planet this one replaces: it keeps drawing
        # until this one has streamed in (no popping to
        # coarse terrain on every rebuild).
        self.previous: "_Planet | None" = None
        self.resolution = component.resolution

        base_low, base_high = self.terrain.base_bounds

        self.selector = LodSelector(
            radius=settings.radius,
            max_elevation=self.terrain.max_elevation + base_high,
            max_depth=component.max_depth,
            split_factor=component.split_factor,
            min_elevation=self.terrain.min_elevation + base_low,
            shape=self.terrain.base_height if self.terrain.shape is not None else None
        )

        self.chunks: dict[ChunkKey, _Chunk] = {}

        self.draw: list[ChunkKey] = []
        self.wanted: list[ChunkKey] = []
        self.wanted_sizes: list[float] = []

        # Bumped whenever `draw` is replaced.
        self.draw_version = 0

        # Draw items for `draw`, and what they were built for.
        self.items: list[DrawItem] = []
        self.items_key: tuple | None = None

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

        self.ground_cache: tuple[np.ndarray, float, float] | None = None

        self.spawn: SpawnPoint | None = None

        self.alive = True

    def ground(
        self,
        direction: np.ndarray
    ) -> tuple[float, float]:
        """
        (base height of the body's shape, visible surface
        above it: the liquid's over seas) under a unit
        direction, in m. Cached: re-evaluated (about a
        millisecond) only after moving a quarter of the
        finest vertex spacing.
        """

        cached = self.ground_cache

        tolerance = 0.25 * self.finest_spacing / self.terrain.settings.radius

        if cached is not None and float(np.linalg.norm(direction - cached[0])) < tolerance:
            return cached[1], cached[2]

        terrain = self.terrain

        elevation = float(
            terrain.elevation(
                direction[None, :],
                spacing=self.finest_spacing
            )[0]
        )

        if terrain.settings.has_liquid:
            elevation = max(elevation, 0.0)

        base = float(terrain.base_height(direction[None, :])[0])

        self.ground_cache = (direction.copy(), base, elevation)

        return base, elevation

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

    # Terrain settings must be unchanged this long before
    # the planet rebuilds, so dragging a slider does not
    # restart the whole build every frame.
    REBUILD_DELAY = 0.35

    # A changed tectonic field (simulation running) rebuilds
    # the planet at most this often (seconds).
    FIELD_REBUILD_INTERVAL = 1.0

    # A replaced planet keeps drawing until its successor
    # has streamed in, or at most this long (seconds).
    SWAP_TIMEOUT = 4.0

    def __init__(
        self,
        resources: Resources,
        jobs: JobSystem,
        material: Handle,
        field_provider: Callable[[Entity], object] | None = None,
        climate_provider: Callable[[Entity], object] | None = None
    ):
        """
        field_provider: entity -> tectonic field (or None),
            e.g. TectonicsSystem.field.
        climate_provider: entity -> climate field (or None),
            e.g. ClimateSystem.field.
        """

        self._resources = resources
        self._jobs = jobs
        self._material = material
        self._field_provider = field_provider or (lambda entity: None)
        self._climate_provider = climate_provider or (lambda entity: None)

        self._planets: dict[Entity, _Planet] = {}

        self._frame = 0

        self._items: list[DrawItem] = []

        # Per planet entity: its terrain material.
        self._materials: dict[Entity, object] = {}

        self.stats = PlanetStats()

        # Index into TERRAIN_VIEWS.
        self.view_mode = 0

        # entity -> (config waiting to be applied, since).
        self._pending_config: dict[Entity, tuple[tuple, float]] = {}

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

        base_material = self._resources.materials.get(self._material)

        items: list[DrawItem] = []

        seen = set()

        building = sum(
            1
            for planet in self._planets.values()
            for chunk in planet.chunks.values()
            if chunk.job is not None
        )

        queued = 0

        requests = []

        for entity, transform, component in scene.registry.view_with(
            TransformComponent,
            PlanetComponent
        ):

            seen.add(entity)

            planet = self._planets.get(entity)

            config = _config_of(component)

            field = self._field_provider(entity)
            climate = self._climate_provider(entity)

            data_version = _data_version(field, climate)

            if planet is None:

                planet = _Planet(component, field, climate)

                self._planets[entity] = planet

            elif (
                planet.config != config
                and self._settled(entity, config)
            ) or (
                planet.config == config
                and planet.data_version != data_version
                and time.monotonic() - planet.created >= self.FIELD_REBUILD_INTERVAL
            ):

                building -= self._replace(entity, planet, component, field, climate)

                planet = self._planets[entity]

            material = self._material_for(entity, base_material, component, planet)

            world = _rigid(transform.world_matrix)

            # Which body the terrain belongs to (lit.frag.glsl):
            # where it is, its frame, and whether it is a
            # (flattened) sphere.
            shape = planet.terrain.shape

            material.set_vec4(
                "uBodyCenter",
                (
                    *((world[:3, 3] - camera_position) / 1000.0),
                    1.0 if shape is None or shape.spheroid else 2.0
                )
            )

            material.set_vec4("uBodyFrame", world_to_body(world))

            camera_local = (
                np.linalg.inv(world)
                @ np.append(camera_position, 1.0)
            )[:3]

            self._select(planet, camera_local)

            # ---------------------------------------------
            # Build requests (all planets share the workers)
            # ---------------------------------------------

            for key, size in zip(planet.wanted, planet.wanted_sizes):

                chunk = planet.chunks.get(key)

                if chunk is not None and (chunk.mesh is not None or chunk.job is not None):
                    continue

                # Every body whole first (its coarsest levels),
                # then detail where it looks largest.
                requests.append((key.depth > 1, -size, len(requests), planet, key))

            # ---------------------------------------------
            # Draw items
            # ---------------------------------------------

            frame = self._frame

            for key in planet.in_use:
                planet.chunks[key].last_used = frame

            # Rebuilt only when the drawn set or the planet's
            # placement changes (selection is cached too).
            cache_key = (planet.draw_version, world.tobytes(), id(material))

            if planet.items_key != cache_key:

                planet.items = [
                    DrawItem(
                        mesh=planet.chunks[key].mesh,
                        material=material,
                        world_matrix=world @ planet.chunks[key].local_matrix,
                        casts_shadows=True
                    )
                    for key in planet.draw
                ]

                planet.items_key = cache_key

            previous = planet.previous

            if previous is not None and (
                not planet.wanted
                or time.monotonic() - planet.created >= self.SWAP_TIMEOUT
            ):

                self._release(previous)

                planet.previous = previous = None

            # Below a giant's cloud tops, the deck is above:
            # the fog in the atmosphere pass is all there is.
            if planet.terrain.settings.bands and _below_cloud_tops(planet, camera_local):
                self._evict(planet)
                continue

            items.extend(previous.items if previous is not None else planet.items)

            self._evict(planet)

        requests.sort(key=lambda request: request[:3])

        for _, _, _, planet, key in requests:

            if building >= self.MAX_BUILDING:

                queued += 1

                continue

            self._request(planet, key)

            building += 1

        # Planets whose entity is gone.
        for entity in [e for e in self._planets if e not in seen]:

            self._release_all(self._planets.pop(entity))

            self._pending_config.pop(entity, None)

            self._materials.pop(entity, None)

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

    # Without an atmosphere to model: how deep below a giant's
    # cloud tops the camera may go, as a share of the radius.
    GIANT_DESCENT = 0.004

    # Liquid / palette names -> shader codes
    # (include/terrain.glsl).
    _LIQUID_CODES = {"none": 0.0, "water": 1.0, "methane": 2.0, "lava": 3.0}
    _PALETTE_CODES = {"biomes": 0.0, "mineral": 1.0, "bands": 2.0}

    def _material_for(
        self,
        entity: Entity,
        base,
        component: PlanetComponent,
        planet: "_Planet | None" = None
    ):
        """
        The planet's own copy of the terrain material, with
        its surface colors and liquid (planets can differ).
        Updated every frame, so color edits apply at once.
        """

        liquid = component.liquid

        # Seas boiled away (the terrain was built dry).
        if planet is not None and not planet.terrain.settings.has_liquid:
            liquid = "none"

        phase = self._liquid_phase(component, planet)

        material = self._materials.get(entity)

        if material is None:

            material = base.copy()

            self._materials[entity] = material

        material.set_float("uTerrainView", float(self.view_mode))
        material.set_float("uSurfacePalette", self._PALETTE_CODES.get(component.palette, 0.0))
        material.set_float("uLiquid", self._LIQUID_CODES.get(liquid, 1.0))
        material.set_float("uLiquidFreezing", phase)
        material.set_float("uLife", 1.0 if component.life else 0.0)
        material.set_vec3("uColorLow", component.color_low)
        material.set_vec3("uColorHigh", component.color_high)
        material.set_vec3("uColorSteep", component.color_steep)
        material.set_vec3("uColorIce", component.color_ice)
        material.set_float("uFrostPoint", component.frost_point)

        # Giants: bands, storms, polar hexagon.
        material.set_float("uBandCount", float(max(component.bands, 1)))
        material.set_vec4(
            "uStorm",
            (
                math.radians(component.storm_latitude),
                math.radians(component.storm_longitude),
                max(component.storm_size, 0.0) / max(component.radius, 1.0),
                float(component.storm_strength) if component.storm_size > 0.0 else 0.0
            )
        )
        material.set_vec3("uStormColor", component.storm_color)
        material.set_float("uOvals", float(max(component.ovals, 0.0)))
        material.set_float("uPolarHexagon", 1.0 if component.polar_hexagon else 0.0)

        self._apply_color_map(material, component)

        return material

    def _apply_color_map(
        self,
        material,
        component: PlanetComponent
    ):
        """
        A real color map (planet/maps.py) as the surface's
        albedo, brought to the body's own brightness (map
        images are stretched for display).
        """

        path = color_map_path(component.color_map) if component.color_map else None

        if path is None:

            material.set_float("uColorMapStrength", 0.0)

            return

        handle = self._resources.textures.load(
            f"maps/{component.color_map}",
            lambda: Texture2D(str(path), srgb=True)
        )

        material.set_texture("uColorMap", handle)
        material.set_float("uColorMapStrength", 1.0)

        target = 0.5 * (
            _luminance(component.color_low) + _luminance(component.color_high)
        )

        material.set_float("uColorMapScale", target / max(_map_luminance(str(path)), 1e-4))

    @staticmethod
    def _liquid_phase(
        component: PlanetComponent,
        planet: "_Planet | None"
    ) -> float:
        """
        Freezing point (C) of the planet's seas for the
        shader: below it they ice over. Liquid that cannot
        exist at the surface pressure is always ice.
        """

        climate = planet.climate if planet is not None else None

        if climate is not None and climate.liquid_state == "frozen":
            return 1.0e4

        substance = SUBSTANCES.get(component.liquid)

        if substance is None:
            return -1.0e4

        return substance.freezing_c

    def _settled(
        self,
        entity: Entity,
        config: tuple
    ) -> bool:
        """True once `config` has been unchanged for REBUILD_DELAY."""

        now = time.monotonic()

        pending = self._pending_config.get(entity)

        if pending is None or pending[0] != config:

            self._pending_config[entity] = (config, now)

            return self.REBUILD_DELAY <= 0.0

        if now - pending[1] >= self.REBUILD_DELAY:

            del self._pending_config[entity]

            return True

        return False

    def rebuild_pending(
        self,
        entity: Entity
    ) -> bool:
        """Settings changed and a rebuild is about to start."""

        return entity in self._pending_config

    # =====================================================
    # Queries (editor)
    # =====================================================

    def terrain(
        self,
        entity: Entity
    ) -> Terrain | None:

        planet = self._planets.get(entity)

        return planet.terrain if planet is not None else None

    def spawn_point(
        self,
        entity: Entity
    ) -> SpawnPoint | None:
        """
        A scenic starting place on the planet (planet/spawn.py),
        computed once per terrain (~0.1 s).
        """

        planet = self._planets.get(entity)

        if planet is None:
            return None

        if planet.spawn is None:
            planet.spawn = find_spawn(planet.terrain)

        return planet.spawn

    def ground(
        self,
        entity: Entity,
        direction
    ) -> tuple[float, float] | None:
        """
        (base height of the shape, visible surface above it),
        m, under a planet-space direction: sea level is at
        radius + base, the ground at radius + base + surface.
        """

        planet = self._planets.get(entity)

        if planet is None:
            return None

        direction = np.asarray(direction, dtype=np.float64)

        # Shares the camera-ground cache.
        return planet.ground(direction / np.linalg.norm(direction))

    def extent(
        self,
        entity: Entity
    ) -> float | None:
        """Farthest point of the planet's surface from its center (m)."""

        planet = self._planets.get(entity)

        if planet is None:
            return None

        terrain = planet.terrain

        return terrain.settings.radius + terrain.base_bounds[1] + max(terrain.max_elevation, 0.0)

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
        planet.draw_version += 1
        planet.wanted = selection.wanted
        planet.wanted_sizes = selection.wanted_sizes
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
            chunk.local_matrix = translation(data.center)

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
            lambda: build_cached_chunk(key, terrain, resolution),
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

    def _replace(
        self,
        entity: Entity,
        planet: _Planet,
        component: PlanetComponent,
        field,
        climate=None
    ) -> int:
        """
        Start a new build of the planet; the current one
        keeps drawing until it is ready. Returns the number
        of chunk builds cancelled.
        """

        if planet.config != _config_of(component):
            Logger.info("[Planet] Settings changed; rebuilding.")

        cancelled = 0

        # Only one generation of fallback: if the planet being
        # replaced was itself still waiting on its
        # predecessor, that older one stays on screen.
        if planet.previous is not None:

            cancelled += self._release(planet)

            fallback = planet.previous

        else:

            fallback = planet

            cancelled += self._cancel_jobs(planet)

        replacement = _Planet(component, field, climate)

        replacement.previous = fallback

        self._planets[entity] = replacement

        return cancelled

    def _cancel_jobs(
        self,
        planet: _Planet
    ) -> int:

        cancelled = 0

        for key in [key for key, chunk in planet.chunks.items() if chunk.job is not None]:

            chunk = planet.chunks.pop(key)

            chunk.job.cancel()
            chunk.job = None

            cancelled += 1

        return cancelled

    def _release_all(
        self,
        planet: _Planet
    ):

        if planet.previous is not None:

            self._release(planet.previous)

            planet.previous = None

        self._release(planet)

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
                planets.append((_rigid(transform.world_matrix), planet, entity))

        if not planets:
            return

        for _, transform, controller in scene.registry.view_with(
            TransformComponent,
            CameraControllerComponent
        ):

            if not controller.planet_mode:
                continue

            position = np.append(transform.transform.position, 1.0)

            # The planet whose surface is nearest (between a
            # planet and its moon, the one the camera is
            # flying over).
            world, planet, entity = min(
                planets,
                key=lambda entry: (
                    float(np.linalg.norm(entry[0][:3, 3] - position[:3]))
                    - entry[1].terrain.settings.radius
                )
            )

            local = (np.linalg.inv(world) @ position)[:3]

            distance = float(np.linalg.norm(local))

            if distance <= 0.0:
                continue

            # The ground: the liquid's surface over seas, the
            # basin floor on dry worlds, on the body's shape.
            base, elevation = planet.ground(local / distance)

            controller.planet_center = tuple(float(v) for v in world[:3, 3])

            controller.planet_radius = planet.terrain.settings.radius + base + elevation

            # Giants: no ground, only ever-thicker air below the
            # cloud tops, down to the bottom of the rendered model
            # (where it is hot enough to glow; not a limit of the
            # world, only of what is drawn).
            controller.descent = (
                self._giant_depth(scene, entity, planet)
                if planet.terrain.settings.bands
                else 0.0
            )

    def _giant_depth(
        self,
        scene: Scene,
        entity: Entity,
        planet
    ) -> float:
        """How far below a giant's cloud tops its air is modeled (m)."""

        atmosphere = scene.registry.try_get(entity, AtmosphereComponent)

        if atmosphere is None or atmosphere.height <= 0.0:
            return self.GIANT_DESCENT * planet.terrain.settings.radius

        parameters = AtmosphereParameters.from_components(planet.terrain.settings.radius, atmosphere)

        # A little short of it: the floor stays in front.
        return 0.98 * parameters.floor_depth * 1000.0

    # =====================================================
    # Shutdown
    # =====================================================

    def shutdown(self):

        for planet in self._planets.values():
            self._release_all(planet)

        self._planets.clear()

        self._items = []


def terrain_settings_for(
    component: PlanetComponent,
    climate=None
) -> TerrainSettings:
    """
    climate: the planet's climate field, if any: seas that
    have boiled away there leave dry basins.
    """

    return TerrainSettings(
        seed=component.seed,
        radius=component.radius,
        continent_frequency=component.continent_frequency,
        continent_height=component.continent_height,
        land_bias=component.land_bias,
        mountain_frequency=component.mountain_frequency,
        mountain_height=component.mountain_height,
        detail_height=component.detail_height,
        has_liquid=(
            component.liquid != "none"
            and not getattr(climate, "liquid_boiled", False)
        ),
        bands=max(0, int(component.bands)) if component.palette == "bands" else 0,
        oblateness=float(component.oblateness),
        elevation_map=component.elevation_map,
        shape=shape_settings(
            axes=component.shape_axes,
            main_center=component.shape_center,
            lobe_center=component.lobe_center,
            lobe_axes=component.lobe_axes,
            lumpiness=component.lumpiness,
            basins=component.basins,
            seed=component.seed
        ),
        crater_density=max(0.0, float(component.crater_density)),
        surface_age=float(component.surface_age),
        crater_erosion=max(0.0, float(component.crater_erosion)),
        crater_min_diameter=max(0.0, float(component.crater_min_diameter)),
        crater_transition=max(100.0, float(component.crater_transition)),
        crater_rays=bool(component.crater_rays),
        volcanism=float(min(max(component.volcanism, 0.0), 1.0)),
        volcano_max_height=max(100.0, float(component.volcano_max_height)),
        rivers=bool(component.rivers),
        dune_density=max(0.0, float(component.dune_density)),
        dune_amplitude=max(0.0, float(component.dune_amplitude)),
        dune_wavelength=max(50.0, float(component.dune_wavelength)),
        dune_linear=bool(component.dune_linear),
        dune_max_latitude=float(component.dune_max_latitude),
        dune_darkening=float(min(max(component.dune_darkening, 0.0), 1.0))
    )


def planet_extent(
    component: PlanetComponent
) -> float:
    """Farthest point of a planet's shape from its center (m)."""

    shape = body_shape(terrain_settings_for(component))

    return component.radius if shape is None else shape.extent


def _data_version(
    field,
    climate
) -> tuple:
    """Identifies the simulation data a planet is built from."""

    return (
        field.version if field is not None else None,
        climate.version if climate is not None else None,
    )


def _config_of(
    component: PlanetComponent
) -> tuple:
    """Everything that, when changed, requires a rebuild."""

    return (
        terrain_settings_for(component),
        component.resolution,
        component.max_depth,
        component.split_factor
    )


def _luminance(
    color
) -> float:

    r, g, b = color

    return 0.2126 * r + 0.7152 * g + 0.0722 * b


@lru_cache(maxsize=8)
def _map_luminance(
    path: str
) -> float:
    """Mean linear luminance of a color map image (area-weighted)."""

    from PIL import Image

    image = np.asarray(Image.open(path).convert("RGB").resize((256, 128)), dtype=np.float64) / 255.0

    linear = np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)

    luminance = linear @ np.array([0.2126, 0.7152, 0.0722])

    # Rows nearer the poles cover less ground.
    weights = np.cos(np.radians(90.0 - (np.arange(128) + 0.5) * 180.0 / 128.0))[:, None]

    return float((luminance * weights).sum() / (weights.sum() * 256))


def world_to_body(
    world: np.ndarray
) -> tuple[float, float, float, float]:
    """Quaternion (xyzw) taking world directions into a rigid world matrix's frame."""

    q = quaternion.conjugate(quaternion.from_matrix3(np.asarray(world, dtype=np.float64)[:3, :3]))

    return tuple(float(v) for v in q)


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


def _below_cloud_tops(
    planet: "_Planet",
    camera_local: np.ndarray
) -> bool:
    """Is the camera under a giant planet's cloud tops?"""

    distance = float(np.linalg.norm(camera_local))

    if distance <= 0.0:
        return True

    direction = (camera_local / distance)[None, :]

    tops = planet.terrain.settings.radius + float(planet.terrain.base_height(direction)[0])

    return distance < tops - 50.0
