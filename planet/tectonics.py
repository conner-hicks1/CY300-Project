import time

from dataclasses import dataclass, field

import numpy as np

from core.disk_cache import cache_key, cached

from planet.acceleration import accelerator
from planet.noise import Perlin, fbm
from planet.sphere_grid import SphereGrid, sphere_grid


# =========================================================
# Plate Tectonics
# =========================================================
#
# A kinematic plate model on the global grid
# (planet/sphere_grid.py), in the spirit of Cortial et al.,
# "Procedural Tectonic Planets" (2019), simplified to run in
# numpy in a fraction of a second per step:
#
#   * Each plate is a rigid cap rotating about its own axis
#     (Euler pole) at a few cm/year.
#   * Every step (default 5 Myr) the crust moves: each cell
#     asks every plate "would your rotation have carried
#     crust from elsewhere to me?" (a backward,
#     semi-Lagrangian lookup).
#       - No plate arrives: plates are pulling apart. New
#         ocean floor forms at a mid-ocean ridge (age 0).
#       - Several arrive: a convergent boundary. Continental
#         crust never sinks; between oceanic plates the
#         older, denser one does. The surviving crust is
#         lifted: island arcs (ocean-ocean), Andes-style
#         ranges (ocean under continent), Himalaya-style
#         ranges (continent-continent collision).
#   * Ocean floor deepens as it ages (half-space cooling,
#     depth ~ sqrt(age)); continents erode toward low
#     plains and their relief diffuses outward.
#   * Every so often a large plate rifts in two, opening a
#     new ocean (a Wilson cycle).
#
# Everything is deterministic for a given seed: the random
# generator of each step is seeded by (seed, step number),
# so a saved planet can be re-simulated to the same state.
#
# Units: meters, millions of years (Myr), radians.


@dataclass(frozen=True, slots=True)
class TectonicSettings:

    seed: int = 1
    plate_count: int = 12

    # Fraction of the surface that starts as continent.
    land_fraction: float = 0.3

    # Typical plate speed (cm / year; Earth: 2-10).
    plate_speed: float = 5.0

    # Simulated time per step (Myr).
    time_step: float = 5.0

    # Grid cells per cube-face edge (6 * n^2 cells).
    resolution: int = 128

    radius: float = 6_371_000.0

    # A large plate rifts apart about this often (Myr).
    rift_interval: float = 150.0

    # How the outer shell moves (planet/regimes.py):
    # plate_tectonics, stagnant_lid, episodic_resurfacing,
    # heat_pipe or ice_shell.
    regime: str = "plate_tectonics"

    # Relief multiplier for the other regimes (weaker
    # gravity holds up taller mountains).
    relief_scale: float = 1.0

    # Stagnant lids: how strong the crustal dichotomy is
    # (1 = the default step), and how much of the surface
    # lies in the low hemisphere (Mars's northern lowlands:
    # a third).
    dichotomy: float = 1.0
    lowlands: float = 0.5


# Uplift per 5 Myr at convergent boundaries (m).
UPLIFT_COLLISION = 700.0
UPLIFT_SUBDUCTION = 450.0
UPLIFT_ISLAND_ARC = 300.0

MAX_CONTINENT_ELEVATION = 7_500.0

# Height of the mountain belts the starting continents
# already have (m).
INITIAL_BELT_HEIGHT = 1_600.0

# Two plates whose continents have collided along this
# many cells in total (recent steps, fading with
# COLLISION_MEMORY Myr) weld into one plate (suturing), as
# India did with Asia. Without this, colliding continents
# would grind into each other forever.
SUTURE_CELLS = 500
COLLISION_MEMORY = 30.0

# Continents relax toward this height (m) with this time
# constant (Myr) as they erode.
CONTINENT_BASE = 350.0
EROSION_TIME = 120.0

# Recent uplift ("orogeny") fades with this time constant;
# it marks young mountain belts for terrain detail.
OROGENY_TIME = 60.0

# Fraction of the difference to the neighbor mean removed
# per step (relief spreading outward).
DIFFUSION = 0.12

RIDGE_DEPTH = -2_600.0
ABYSSAL_DEPTH = -5_600.0

# The oldest sea floor at the start (Myr).
MAX_OCEAN_AGE = 180.0

# Sea floor reaches most of its final depth within a few
# times this many Myr (the "plate model" of cooling: deep
# but flattening, unlike unbounded sqrt(age)).
COOLING_TIME = 55.0

# Continental crust is recycled in collisions; new crust
# grows at active margins (island arcs, subduction zones).
# Up to this fraction of the shortfall is regrown per step.
ACCRETION_RATE = 0.5

# 1 cm/yr = 10 km/Myr.
METERS_PER_MYR_PER_CM_PER_YEAR = 10_000.0


@dataclass(slots=True)
class TectonicState:

    # Per cell.
    plate: np.ndarray           # int32 plate index
    continental: np.ndarray     # float32 0 = oceanic .. 1 = continental
    land_height: np.ndarray     # float32 height of continental crust (m)
    age: np.ndarray             # float32 crust age (Myr)
    orogeny: np.ndarray         # float32 recent uplift (m)
    activity: np.ndarray        # float32 boundary closing speed (cm/yr; < 0 = pulling apart)

    # Per plate.
    axes: np.ndarray            # (plates, 3) unit rotation axes
    speeds: np.ndarray          # (plates,) angular speed (rad / Myr)

    time: float = 0.0
    step_index: int = 0

    # (plate a, plate b) with a < b -> accumulated collision
    # cells (see SUTURE_CELLS).
    collisions: dict[tuple[int, int], float] = field(default_factory=dict)

    # Wall-clock seconds of the last step (diagnostics).
    step_seconds: float = 0.0

    # Regimes other than plates (planet/regimes.py): the
    # surface height per cell (m) directly, and whatever
    # they need to remember between steps.
    height: np.ndarray | None = None
    memory: dict = field(default_factory=dict)

    regime: str = "plate_tectonics"

    # Identifies the state for the disk cache: its settings
    # and every step that led here ("" = unknown history).
    key: str = ""

    @property
    def plate_total(
        self
    ) -> int:

        return len(self.speeds)


@dataclass(frozen=True, slots=True)
class PlateInfo:

    index: int
    cell_fraction: float
    continental_fraction: float
    speed_cm_per_year: float


# =========================================================
# Derived Fields
# =========================================================

def ocean_depth(
    age
) -> np.ndarray:
    """Sea-floor depth from crust age (cooling, flattening with age)."""

    age = np.maximum(age, 0.0)

    return RIDGE_DEPTH + (ABYSSAL_DEPTH - RIDGE_DEPTH) * (1.0 - np.exp(-age / COOLING_TIME))


def surface_elevation(
    state: TectonicState
) -> np.ndarray:
    """Elevation (m) per cell: ocean floor blended into land."""

    if state.height is not None:
        return state.height

    land = _smoothstep(0.35, 0.65, state.continental)

    ocean = ocean_depth(state.age) + state.orogeny * (1.0 - land) * 0.8

    return ocean + (state.land_height - ocean) * land


def rotated_cells(
    directions: np.ndarray,
    rotations: np.ndarray,
    resolution: int
) -> np.ndarray:
    """
    (rotations, cells) sphere-grid cell of each direction
    rotated by each 3x3 rotation. (The reference for the GPU
    version, graphics/gpu_simulation.py.)
    """

    grid = sphere_grid(resolution)

    return np.stack([grid.cell_of(directions @ rotation.T) for rotation in rotations])


def plate_velocities(
    state: TectonicState,
    directions: np.ndarray,
    plates: np.ndarray
) -> np.ndarray:
    """Surface velocity at unit directions, in radians / Myr (x radius for m / Myr)."""

    omega = state.axes[plates] * state.speeds[plates, None]

    return np.cross(omega, directions)


# =========================================================
# Simulation
# =========================================================

class CachedStates:

    # Simulations whose states come from the disk cache when
    # they were computed before (core/disk_cache.py): the
    # initial state by its settings, each step by the key of
    # the state it followed. Subclasses implement
    # _initial_state() and _step().

    def initial_state(
        self
    ) -> TectonicState:
        """The starting state (deterministic for the settings)."""

        key = cache_key("tectonics", self.settings)

        state = cached("tectonics", key, self._initial_state)

        state.key = key

        return state

    def step(
        self,
        state: TectonicState
    ) -> TectonicState:
        """One time step; returns a new state (the input is untouched)."""

        if not state.key:
            return self._step(state)

        key = cache_key("tectonics step", self.settings, state.key)

        new = cached("tectonics", key, lambda: self._step(state))

        new.key = key

        return new


class TectonicSimulation(CachedStates):

    def __init__(
        self,
        settings: TectonicSettings
    ):

        self.settings = settings

        self.grid: SphereGrid = sphere_grid(settings.resolution)

    # -----------------------------------------------------
    # Initial state
    # -----------------------------------------------------

    def _initial_state(
        self
    ) -> TectonicState:

        s = self.settings
        grid = self.grid

        rng = np.random.default_rng([s.seed, 0])

        directions = grid.directions

        # Plates: nearest of random seed points, measured
        # on a warped sphere so boundaries wander.
        warp = np.stack(
            [
                fbm(Perlin(s.seed * 31 + axis), directions * 2.0, 4)
                for axis in range(3)
            ],
            axis=1
        )

        warped = directions + 0.35 * warp
        warped /= np.linalg.norm(warped, axis=1, keepdims=True)

        count = max(2, int(s.plate_count))

        seeds = rng.normal(size=(count, 3))
        seeds /= np.linalg.norm(seeds, axis=1, keepdims=True)

        similarity = warped @ seeds.T

        plate = np.argmax(similarity, axis=1).astype(np.int32)

        # How far inside its plate each cell is: the gap
        # between the nearest and second-nearest seed.
        ranked = np.sort(similarity, axis=1)

        interior = ranked[:, -1] - ranked[:, -2]

        axes, speeds = self._random_motions(rng, count)

        # Continents: the highest land_fraction of a
        # low-frequency noise field, kept inside plates (on
        # Earth, continents ride within plates; a boundary
        # through a continent would tear it apart at once).
        noise = fbm(Perlin(s.seed * 7919), directions * 1.3, 6)

        noise = noise - 0.6 * (1.0 - _smoothstep(0.02, 0.12, interior))

        threshold = np.quantile(noise, 1.0 - np.clip(s.land_fraction, 0.02, 0.95))

        continental = _smoothstep(threshold - 0.04, threshold + 0.04, noise).astype(np.float32)

        # Ancient mountain belts (old orogens) winding across
        # the continents, so there is relief before the
        # first collision.
        ridge = 1.0 - np.abs(fbm(Perlin(s.seed * 4241), directions * 2.5, 4))

        belts = _smoothstep(0.93, 0.99, ridge) * continental

        # Low coastal plains rising inland (on Earth, half
        # the land is below ~450 m, and coasts are lowest).
        inland = np.clip((noise - threshold) * 3.0, 0.0, 1.0)

        land_height = (
            40.0
            + 1_100.0 * inland ** 1.5
            + INITIAL_BELT_HEIGHT * belts
        ).astype(np.float32)

        # Ocean floor of mixed ages, mostly young (as on
        # Earth, where the area of sea floor falls off
        # steadily with age up to ~180 Myr: half of it is
        # younger than ~55 Myr). Continents are old.
        age_noise = fbm(Perlin(s.seed * 104_729), directions * 2.5, 3)

        rank = np.argsort(np.argsort(age_noise)) / max(len(age_noise) - 1, 1)

        age = np.where(
            continental > 0.5,
            1_000.0,
            MAX_OCEAN_AGE * (1.0 - np.sqrt(1.0 - rank))
        ).astype(np.float32)

        zeros = np.zeros(grid.cell_count, dtype=np.float32)

        state = TectonicState(
            plate=plate,
            continental=continental,
            land_height=land_height,
            age=age,
            orogeny=(INITIAL_BELT_HEIGHT * belts).astype(np.float32),
            activity=zeros.copy(),
            axes=axes,
            speeds=speeds
        )

        state.activity = self._boundary_activity(state)

        return state

    def _random_motions(
        self,
        rng: np.random.Generator,
        count: int
    ) -> tuple[np.ndarray, np.ndarray]:

        s = self.settings

        axes = rng.normal(size=(count, 3))
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)

        surface_speed = (
            s.plate_speed
            * rng.uniform(0.4, 1.6, size=count)
            * METERS_PER_MYR_PER_CM_PER_YEAR
        )

        return axes, surface_speed / s.radius

    # -----------------------------------------------------
    # Step
    # -----------------------------------------------------

    def _step(
        self,
        state: TectonicState
    ) -> TectonicState:

        started = time.perf_counter()

        s = self.settings
        grid = self.grid

        dt = float(s.time_step)
        scale = dt / 5.0

        rng = np.random.default_rng([s.seed, state.step_index + 1])

        directions = grid.directions

        cells = grid.cell_count
        plates = state.plate_total

        # -------------------------------------------------
        # 1. Who arrives where
        # -------------------------------------------------
        #
        # For plate k, the crust now at p came from
        # R_k(-w_k dt) p. It is plate k's if that source
        # cell belonged to plate k.

        sources = np.full((plates, cells), -1, dtype=np.int64)

        rotations = {
            k: _rotation_matrix(state.axes[k], -state.speeds[k] * dt)
            for k in range(plates)
            if np.any(state.plate == k)
        }

        if rotations:

            # Every plate at once (on the GPU when there is
            # one: planet/acceleration.py).
            gpu = accelerator()

            found = (gpu.rotated_cells if gpu is not None else rotated_cells)(
                directions,
                np.stack(list(rotations.values())),
                grid.n
            )

            for row, k in enumerate(rotations):

                source = found[row]

                sources[k] = np.where(state.plate[source] == k, source, -1)

        claimed = sources >= 0

        arrivals = claimed.sum(axis=0)

        # -------------------------------------------------
        # 2. Who stays on top
        # -------------------------------------------------
        #
        # Continental crust is buoyant and never subducts;
        # between oceanic crusts the younger (lighter) wins.

        safe = np.where(claimed, sources, 0)

        source_continental = state.continental[safe]
        source_age = state.age[safe]

        # Ties (e.g. two continents) go to the plate already
        # holding the cell, so collision fronts stay put
        # instead of flickering between plates cell by cell.
        incumbent = np.arange(plates)[:, None] == state.plate[None, :]

        score = np.where(
            claimed,
            source_continental * 1_000.0 - source_age * 0.01 + incumbent * 0.05,
            -np.inf
        )

        winner = np.argmax(score, axis=0)

        column = np.arange(cells)

        source = safe[winner, column]

        moved = arrivals > 0

        plate = np.where(moved, winner, state.plate).astype(np.int32)

        # Continental fraction from the nearest source cell
        # (keeps coastlines crisp); smooth fields are
        # interpolated at the exact source position, which
        # avoids the stair-step streaks that repeated
        # rounding to cells leaves in the ocean floor.
        continental = np.where(moved, state.continental[source], 0.0).astype(np.float32)

        land_height = np.full(cells, CONTINENT_BASE, dtype=np.float32)
        age = np.zeros(cells, dtype=np.float32)
        orogeny = np.zeros(cells, dtype=np.float32)

        padded_land = grid.pad(state.land_height)
        padded_age = grid.pad(state.age)
        padded_orogeny = grid.pad(state.orogeny)

        for k, rotation in rotations.items():

            members = np.flatnonzero(moved & (winner == k))

            if len(members) == 0:
                continue

            origin = directions[members] @ rotation.T

            land_height[members] = grid.sample(padded_land, origin)
            age[members] = grid.sample(padded_age, origin) + dt
            orogeny[members] = grid.sample(padded_orogeny, origin)

        # Gaps (plates pulling apart) join the plate most of
        # their neighbors belong to: each side of a new
        # ridge accretes to its own plate.
        gaps = np.flatnonzero(~moved)

        if len(gaps):
            plate[gaps] = _majority(plate[grid.neighbors[gaps]], plate[gaps])

        # Stray cells surrounded by another plate (rounding
        # noise at boundaries) join it.
        plate = _clean_plate_ids(plate, grid.neighbors)

        # -------------------------------------------------
        # 3. Convergent boundaries: uplift
        # -------------------------------------------------

        converging = arrivals >= 2

        if converging.any():

            # Most continental crust among the losers.
            loser_continental = np.where(
                claimed & (np.arange(plates)[:, None] != winner[None, :]),
                source_continental,
                0.0
            ).max(axis=0)

            winner_land = continental > 0.5
            loser_land = loser_continental > 0.5

            collision = converging & winner_land & loser_land
            subduction = converging & winner_land & ~loser_land
            island_arc = converging & ~winner_land

            uplift = scale * (
                collision * UPLIFT_COLLISION
                + subduction * UPLIFT_SUBDUCTION
                + island_arc * UPLIFT_ISLAND_ARC
            )

            land_height = np.minimum(land_height + uplift, MAX_CONTINENT_ELEVATION).astype(np.float32)

            # The losing plate carrying the most continental
            # crust at each collision cell.
            loser_plate = np.argmax(
                np.where(
                    claimed & (np.arange(plates)[:, None] != winner[None, :]),
                    source_continental,
                    -1.0
                ),
                axis=0
            )

            collisions = _accumulate_collisions(
                state.collisions,
                winner[collision],
                loser_plate[collision],
                plates,
                np.exp(-dt / COLLISION_MEMORY)
            )

            orogeny = (orogeny + uplift).astype(np.float32)

            # Colliding continents weld together; island
            # arcs slowly build new continental crust.
            continental = np.where(
                collision,
                np.maximum(continental, loser_continental),
                continental
            )

            continental = np.where(
                island_arc,
                np.minimum(continental + 0.04 * scale, 1.0),
                continental
            ).astype(np.float32)

        else:

            collisions = _accumulate_collisions(
                state.collisions,
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                plates,
                np.exp(-dt / COLLISION_MEMORY)
            )

        # -------------------------------------------------
        # 4. Erosion, relaxation, spreading of relief
        # -------------------------------------------------

        decay = np.exp(-dt / EROSION_TIME)

        land_height = CONTINENT_BASE + (land_height - CONTINENT_BASE) * decay

        orogeny = orogeny * np.exp(-dt / OROGENY_TIME)

        neighbors = grid.neighbors

        land_height = land_height + DIFFUSION * (land_height[neighbors].mean(axis=1) - land_height)

        orogeny = orogeny + DIFFUSION * (orogeny[neighbors].mean(axis=1) - orogeny)

        # A little smoothing keeps coastlines from turning
        # into grid-aligned staircases as crust moves.
        continental = continental + 0.05 * (continental[neighbors].mean(axis=1) - continental)

        continental = self._accrete(continental, converging, scale)

        new = TectonicState(
            plate=plate,
            continental=continental.astype(np.float32),
            land_height=land_height.astype(np.float32),
            age=age,
            orogeny=orogeny.astype(np.float32),
            activity=state.activity,
            axes=state.axes.copy(),
            speeds=state.speeds.copy(),
            time=state.time + dt,
            step_index=state.step_index + 1,
            collisions=collisions
        )

        target = max(2, int(s.plate_count))

        minimum = max(2, (2 * target) // 3)

        for keep, absorbed in _suture_candidates(new.collisions):

            if len(np.unique(new.plate)) <= minimum:
                break

            _merge_plates(new, keep, absorbed)

        # -------------------------------------------------
        # 5. Rifting: a large plate splits in two
        # -------------------------------------------------
        #
        # Regularly, and sooner while there are fewer plates
        # than the target (after sutures).

        steps_per_rift = max(1, int(round(s.rift_interval / dt)))

        if len(np.unique(new.plate)) < target:
            steps_per_rift = max(1, steps_per_rift // 10)

        if new.step_index % steps_per_rift == 0:
            self._rift(new, rng)

        new.activity = self._boundary_activity(new)

        new.step_seconds = time.perf_counter() - started

        return new

    def _accrete(
        self,
        continental: np.ndarray,
        converging: np.ndarray,
        scale: float
    ) -> np.ndarray:
        """
        Grow continental crust at active margins to make up
        for crust recycled in collisions, keeping the total
        near the starting amount.
        """

        target = np.clip(self.settings.land_fraction, 0.02, 0.95) * self.grid.cell_count

        shortfall = target - float(continental.sum())

        if shortfall <= 0.0:
            return continental

        # Convergent cells and their neighbors, next to
        # existing continent: continents grow at their
        # active edges rather than as scattered slivers.
        neighbors = self.grid.neighbors

        margin = converging | converging[neighbors].any(axis=1)

        margin &= (continental[neighbors] > 0.5).any(axis=1)

        room = np.where(margin, 1.0 - continental, 0.0)

        capacity = float(room.sum())

        if capacity <= 0.0:
            return continental

        grow = min(1.0, ACCRETION_RATE * scale * shortfall / capacity)

        return (continental + room * grow).astype(np.float32)

    def _rift(
        self,
        state: TectonicState,
        rng: np.random.Generator
    ):
        """Split the largest plate along a great circle through its middle."""

        if len(np.unique(state.plate)) >= 2 * max(2, self.settings.plate_count):
            return

        directions = self.grid.directions

        sizes = np.bincount(state.plate, minlength=state.plate_total)

        largest = int(np.argmax(sizes))

        members = state.plate == largest

        if members.sum() < 50:
            return

        centroid = directions[members].mean(axis=0)
        centroid /= np.linalg.norm(centroid)

        normal = rng.normal(size=3)
        normal -= np.dot(normal, centroid) * centroid
        normal /= np.linalg.norm(normal)

        new_index = state.plate_total

        state.plate = np.where(
            members & (directions @ normal > 0.0),
            new_index,
            state.plate
        ).astype(np.int32)

        _, speed = self._random_motions(rng, 1)

        # The new half moves away from the old one: with the
        # pole at centroid x normal, its velocity at the
        # centroid (pole x centroid) points along +normal,
        # across the rift.
        axis = np.cross(centroid, normal)
        axis /= np.linalg.norm(axis)

        state.axes = np.vstack((state.axes, axis[None, :]))
        state.speeds = np.concatenate((state.speeds, speed))

    def _boundary_activity(
        self,
        state: TectonicState
    ) -> np.ndarray:
        """
        Per cell: closing speed (cm/yr) with the strongest
        neighboring plate; > 0 converging, < 0 diverging,
        0 inside a plate.
        """

        grid = self.grid

        directions = grid.directions
        neighbors = grid.neighbors

        own = plate_velocities(state, directions, state.plate)

        activity = np.zeros(grid.cell_count, dtype=np.float32)

        for column in range(neighbors.shape[1]):

            other = neighbors[:, column]

            different = state.plate[other] != state.plate

            if not different.any():
                continue

            toward = directions[other] - directions
            toward /= np.maximum(np.linalg.norm(toward, axis=1, keepdims=True), 1e-12)

            theirs = plate_velocities(state, directions, state.plate[other])

            closing = np.einsum("ij,ij->i", own - theirs, toward)

            # Unit-sphere velocities (rad / Myr) -> m / Myr -> cm / yr.
            closing = (
                np.where(different, closing, 0.0)
                * self.settings.radius
                / METERS_PER_MYR_PER_CM_PER_YEAR
            )

            stronger = np.abs(closing) > np.abs(activity)

            activity = np.where(stronger, closing, activity).astype(np.float32)

        return activity

    # -----------------------------------------------------
    # Summaries
    # -----------------------------------------------------

    def plates(
        self,
        state: TectonicState
    ) -> list[PlateInfo]:

        cells = self.grid.cell_count

        sizes = np.bincount(state.plate, minlength=state.plate_total)

        land = np.bincount(
            state.plate,
            weights=(state.continental > 0.5).astype(np.float64),
            minlength=state.plate_total
        )

        surface = state.speeds * self.settings.radius / METERS_PER_MYR_PER_CM_PER_YEAR

        return [
            PlateInfo(
                index=index,
                cell_fraction=float(sizes[index] / cells),
                continental_fraction=float(land[index] / sizes[index]) if sizes[index] else 0.0,
                speed_cm_per_year=float(surface[index])
            )
            for index in range(state.plate_total)
            if sizes[index] > 0
        ]


# =========================================================
# Field (what the terrain reads)
# =========================================================

@dataclass(frozen=True, slots=True, eq=False)
class TectonicField:

    # Immutable snapshot of a state, padded for sampling.
    # Terrain chunks built on worker threads read it while
    # the simulation computes the next state.

    grid: SphereGrid

    elevation: np.ndarray       # padded, m
    continental: np.ndarray     # padded, 0..1
    age: np.ndarray             # padded, Myr
    orogeny: np.ndarray         # padded, m
    activity: np.ndarray        # padded, cm/yr

    plate: np.ndarray           # per cell (nearest lookup)

    time: float
    version: int

    regime: str = "plate_tectonics"

    # Crust age shown as "old" in the Crust age view (Myr).
    age_scale: float = 400.0

    # The state's cache key (TectonicState.key).
    content_key: str = ""

    @classmethod
    def from_state(
        cls,
        grid: SphereGrid,
        state: TectonicState,
        version: int
    ) -> "TectonicField":

        from planet.regimes import AGE_SCALES

        pad = grid.pad

        age_scale = AGE_SCALES.get(state.regime, 400.0)

        return cls(
            grid=grid,
            elevation=pad(surface_elevation(state).astype(np.float32)),
            continental=pad(state.continental),
            age=pad(np.minimum(state.age, age_scale)),
            orogeny=pad(state.orogeny),
            activity=pad(state.activity),
            plate=state.plate.copy(),
            time=state.time,
            version=version,
            regime=state.regime,
            age_scale=age_scale,
            content_key=state.key
        )

    def sample(
        self,
        name: str,
        directions: np.ndarray
    ) -> np.ndarray:

        return self.grid.sample(getattr(self, name), directions)

    def plate_at(
        self,
        directions: np.ndarray
    ) -> np.ndarray:

        return self.plate[self.grid.cell_of(directions)]


def simulation_for(
    settings: TectonicSettings
):
    """
    The simulation for the settings' regime: plates
    (TectonicSimulation) or one of planet/regimes.py.
    """

    from planet.regimes import REGIME_SIMULATIONS

    kind = REGIME_SIMULATIONS.get(settings.regime)

    if kind is None:
        return TectonicSimulation(settings)

    return kind(settings)


# =========================================================
# Helpers
# =========================================================

def _majority(
    neighbor_plates: np.ndarray,
    fallback: np.ndarray
) -> np.ndarray:
    """Most common value per row of (m, 4) neighbor plate ids."""

    best = fallback.copy()
    best_count = np.zeros(len(fallback), dtype=np.int64)

    for column in range(neighbor_plates.shape[1]):

        candidate = neighbor_plates[:, column]

        count = (neighbor_plates == candidate[:, None]).sum(axis=1)

        better = count > best_count

        best = np.where(better, candidate, best)
        best_count = np.where(better, count, best_count)

    return best


def _clean_plate_ids(
    plate: np.ndarray,
    neighbors: np.ndarray
) -> np.ndarray:
    """A cell whose neighbors (3 of 4 or more) agree on another plate joins it."""

    neighbor_plates = plate[neighbors]

    majority = _majority(neighbor_plates, plate)

    agreeing = (neighbor_plates == majority[:, None]).sum(axis=1)

    return np.where((agreeing >= 3) & (majority != plate), majority, plate).astype(np.int32)


def _accumulate_collisions(
    previous: dict[tuple[int, int], float],
    winners: np.ndarray,
    losers: np.ndarray,
    plates: int,
    fade: float
) -> dict[tuple[int, int], float]:
    """Fade old collision tallies and add this step's cells."""

    tallies = {
        pair: value * fade
        for pair, value in previous.items()
        if value * fade >= 1.0
    }

    if len(winners):

        codes = np.minimum(winners, losers) * plates + np.maximum(winners, losers)

        unique, counts = np.unique(codes, return_counts=True)

        for code, count in zip(unique, counts):

            a, b = divmod(int(code), plates)

            if a != b:
                tallies[(a, b)] = tallies.get((a, b), 0.0) + float(count)

    return tallies


def _suture_candidates(
    tallies: dict[tuple[int, int], float]
) -> list[tuple[int, int]]:
    """Pairs over SUTURE_CELLS, strongest first, each plate once."""

    pairs = []
    used = set()

    for (a, b), value in sorted(tallies.items(), key=lambda item: -item[1]):

        if value < SUTURE_CELLS:
            break

        if a in used or b in used:
            continue

        used.update((a, b))

        pairs.append((a, b))

    return pairs


def _merge_plates(
    state: TectonicState,
    keep: int,
    absorbed: int
):
    """Weld plate `absorbed` into `keep`; they move as one (area-weighted)."""

    sizes = np.bincount(state.plate, minlength=state.plate_total).astype(np.float64)

    total = sizes[keep] + sizes[absorbed]

    if total <= 0:
        return

    omega = (
        state.axes[keep] * state.speeds[keep] * sizes[keep]
        + state.axes[absorbed] * state.speeds[absorbed] * sizes[absorbed]
    ) / total

    speed = float(np.linalg.norm(omega))

    if speed > 0.0:

        state.axes[keep] = omega / speed
        state.speeds[keep] = speed

    state.plate = np.where(state.plate == absorbed, keep, state.plate).astype(np.int32)

    # Tallies involving the absorbed plate are void.
    state.collisions = {
        pair: value
        for pair, value in state.collisions.items()
        if absorbed not in pair and pair != (min(keep, absorbed), max(keep, absorbed))
    }


def _rotation_matrix(
    axis: np.ndarray,
    angle: float
) -> np.ndarray:
    """Rodrigues rotation matrix (right-handed, radians)."""

    x, y, z = axis

    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c

    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


def _smoothstep(
    edge0: float,
    edge1: float,
    x
) -> np.ndarray:

    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)

    return t * t * (3.0 - 2.0 * t)
