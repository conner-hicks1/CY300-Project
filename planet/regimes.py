import math
import time

import numpy as np

from planet.noise import Perlin, fbm
from planet.sphere_grid import SphereGrid, sphere_grid
from planet.tectonics import PlateInfo, TectonicSettings, TectonicState


# =========================================================
# Tectonic Regimes Beyond Plates
# =========================================================
#
# Earth is the only body known to have plate tectonics.
# Elsewhere the outer shell moves (or doesn't) in other
# ways, and each leaves its own kind of surface:
#
#   stagnant lid          one rigid shell over the whole
#   (Mars, Mercury,       planet. Its relief is ancient: a
#   the Moon)             crustal dichotomy, giant impact
#                         basins, and a few volcanic
#                         provinces above long-lived mantle
#                         plumes (Tharsis); lava floods the
#                         lowest basins (the lunar maria).
#                         Activity fades as the interior
#                         cools.
#   episodic resurfacing  a stagnant lid whose heat builds up
#   (Venus)               until the lid founders and lava
#                         floods nearly everything at once
#                         (Venus's surface is ~500 Myr old
#                         everywhere). Only high, rugged
#                         plateaus (tesserae) survive; rifts,
#                         volcanic rises and coronae (ring-
#                         shaped upwellings) form between
#                         events.
#   heat-pipe             heat escapes through countless
#   (Io)                  volcanoes, burying the surface in
#                         lava so fast that nothing is older
#                         than a few Myr: flat plains, lava-
#                         filled calderas (paterae), and
#                         tall isolated mountain blocks
#                         thrust up as the crust subsides.
#   ice shell             a thin ice shell over an ocean,
#   (Europa, Ganymede,    cracked by tides: crisscrossing
#   Triton)               ridges, spreading bands and
#                         chaos terrain on a young surface
#                         of little relief (ice flows).
#
# Each produces the same TectonicState as the plate model
# (one plate, no motion), with the surface height in
# `height`, so the terrain, data views and editor work the
# same. "continental" carries the crust type the mineral
# palette colors by: 1 = bright (highlands, ice plains,
# sulfur plains), 0 = dark (maria, lineae, lava).
#
# Heights scale with relief_scale (lower gravity supports
# taller relief). Angular sizes are fractions of the
# planet. Everything is deterministic for a seed.


# Default simulated time per step (Myr) by regime: their
# surfaces change at very different rates.
DEFAULT_TIME_STEPS = {
    "plate_tectonics": 5.0,
    "stagnant_lid": 50.0,
    "episodic_resurfacing": 25.0,
    "heat_pipe": 0.5,
    "ice_shell": 1.0,
}

# Crust ages normalized for the "Crust age" view (Myr).
AGE_SCALES = {
    "plate_tectonics": 400.0,
    "stagnant_lid": 4_500.0,
    "episodic_resurfacing": 1_000.0,
    "heat_pipe": 5.0,
    "ice_shell": 100.0,
}


class RegimeSimulation:

    # Shared by the regimes: the grid, the state's
    # bookkeeping, and geometry helpers.

    regime = ""

    def __init__(
        self,
        settings: TectonicSettings
    ):

        self.settings = settings

        self.grid: SphereGrid = sphere_grid(settings.resolution)

        self.directions = self.grid.directions

        self.relief = float(np.clip(settings.relief_scale, 0.1, 5.0))

    # -----------------------------------------------------
    # Simulation interface (planet/tectonics.py)
    # -----------------------------------------------------

    def initial_state(
        self
    ) -> TectonicState:
        """The surface as it is today."""

        rng = np.random.default_rng([self.settings.seed, 0])

        n = self.grid.cell_count

        state = TectonicState(
            plate=np.zeros(n, dtype=np.int32),
            continental=np.ones(n, dtype=np.float32),
            land_height=np.zeros(n, dtype=np.float32),
            age=np.zeros(n, dtype=np.float32),
            orogeny=np.zeros(n, dtype=np.float32),
            activity=np.zeros(n, dtype=np.float32),
            axes=np.array([[0.0, 1.0, 0.0]]),
            speeds=np.zeros(1),
            height=np.zeros(n, dtype=np.float32),
            regime=self.regime
        )

        self._initial(state, rng)

        self._finish(state)

        return state

    def step(
        self,
        state: TectonicState
    ) -> TectonicState:
        """One time step; returns a new state (the input is untouched)."""

        started = time.perf_counter()

        dt = float(self.settings.time_step)

        rng = np.random.default_rng([self.settings.seed, state.step_index + 1])

        new = TectonicState(
            plate=state.plate,
            continental=state.continental.astype(np.float64),
            land_height=state.land_height,
            age=state.age.astype(np.float64) + dt,
            orogeny=state.orogeny.astype(np.float64),
            activity=state.activity.astype(np.float64),
            axes=state.axes,
            speeds=state.speeds,
            height=state.height.astype(np.float64),
            time=state.time + dt,
            step_index=state.step_index + 1,
            memory=dict(state.memory),
            regime=self.regime
        )

        self._advance(new, rng, dt)

        self._finish(new)

        new.step_seconds = time.perf_counter() - started

        return new

    def plates(
        self,
        state: TectonicState
    ) -> list[PlateInfo]:
        """One plate: the whole shell."""

        return [
            PlateInfo(
                index=0,
                cell_fraction=1.0,
                continental_fraction=float(np.mean(state.continental > 0.5)),
                speed_cm_per_year=0.0
            )
        ]

    # -----------------------------------------------------
    # Regime
    # -----------------------------------------------------

    def _initial(
        self,
        state: TectonicState,
        rng: np.random.Generator
    ):
        raise NotImplementedError

    def _advance(
        self,
        state: TectonicState,
        rng: np.random.Generator,
        dt: float
    ):
        raise NotImplementedError

    def _finish(
        self,
        state: TectonicState
    ):

        state.height = state.height.astype(np.float32)
        state.land_height = state.height
        state.continental = np.clip(state.continental, 0.0, 1.0).astype(np.float32)
        state.age = np.maximum(state.age, 0.0).astype(np.float32)
        state.orogeny = np.maximum(state.orogeny, 0.0).astype(np.float32)
        state.activity = state.activity.astype(np.float32)

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------

    def noise(
        self,
        salt: int,
        frequency: float,
        octaves: int
    ) -> np.ndarray:

        return fbm(
            Perlin(self.settings.seed * 101 + salt),
            self.directions * frequency,
            octaves
        )

    def angle_to(
        self,
        center: np.ndarray
    ) -> np.ndarray:
        """Angular distance (rad) of every cell from a direction."""

        return np.arccos(np.clip(self.directions @ center, -1.0, 1.0))

    def arc(
        self,
        normal: np.ndarray,
        center: np.ndarray,
        half_length: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        A great-circle arc (through `center`, perpendicular
        to `normal`): every cell's angular offset from its
        circle, and a 0..1 taper that is 1 along the arc and
        fades past its ends.
        """

        side = np.cross(normal, center)

        offset = np.arcsin(np.clip(self.directions @ normal, -1.0, 1.0))

        along = np.arctan2(self.directions @ side, self.directions @ center)

        taper = _smoothstep(half_length, half_length * 0.75, np.abs(along))

        return offset, taper

    def diffuse(
        self,
        values: np.ndarray,
        rate: float,
        iterations: int = 1
    ) -> np.ndarray:
        """Spread relief out (lava, ice and rock all flow)."""

        neighbors = self.grid.neighbors

        for _ in range(iterations):
            values = values + rate * (values[neighbors].mean(axis=1) - values)

        return values

    @property
    def cell_angle(
        self
    ) -> float:

        return self.grid.cell_angle


# =========================================================
# Stagnant Lid (Mars, Mercury, the Moon)
# =========================================================

class StagnantLid(RegimeSimulation):

    regime = "stagnant_lid"

    # Volcanism dies down as the interior cools (Myr).
    COOLING_TIME = 1_500.0

    def _initial(
        self,
        state: TectonicState,
        rng: np.random.Generator
    ):

        s = self.relief

        # Crustal dichotomy: one hemisphere higher, its
        # boundary wandering (Mars's southern highlands).
        axis = _random_direction(rng)

        dichotomy = np.tanh(3.0 * (0.6 * (self.directions @ axis) + self.noise(1, 1.2, 4)))

        height = 1_500.0 * s * dichotomy + 500.0 * s * self.noise(2, 3.0, 5)

        highland = _smoothstep(-0.3, 0.3, dichotomy)

        age = 4_000.0 - 400.0 * (1.0 - highland)

        # Giant impact basins (Hellas, the lunar
        # South Pole-Aitken): bowls with raised rims.
        for _ in range(int(rng.integers(3, 7))):

            center = _random_direction(rng)

            radius = rng.uniform(0.12, 0.4)

            depth = rng.uniform(3_000.0, 5_500.0) * s

            distance = self.angle_to(center)

            height += _basin(distance, radius, depth)

            age = np.where(distance < radius * 1.2, np.minimum(age, rng.uniform(3_800.0, 4_100.0)), age)

        # Highlands are rugged, but mostly from craters.
        orogeny = 400.0 * highland

        # Volcanic provinces over mantle plumes (Tharsis):
        # broad rises, younger than the crust around them.
        activity = np.zeros_like(height)

        for _ in range(int(rng.integers(1, 4))):

            center = _random_direction(rng)

            radius = rng.uniform(0.2, 0.5)

            rise = _dome(self.angle_to(center), radius)

            height += rng.uniform(2_000.0, 4_500.0) * s * rise

            age = np.where(rise > 0.05, np.minimum(age, rng.uniform(800.0, 3_000.0)), age)

            orogeny *= 1.0 - rise

            activity += rise

        # Lava floods the lowest ground: flat, dark plains
        # (maria, Mars's northern plains).
        level = float(np.percentile(height, rng.uniform(12.0, 30.0)))

        flooded = _smoothstep(level + 300.0 * s, level - 300.0 * s, height)

        height = height + (level + 0.08 * (height - level) - height) * flooded

        age = age + (rng.uniform(3_200.0, 3_700.0) - age) * flooded

        state.height = self.diffuse(height, 0.25, 2)

        # Bright highlands, mid-toned lowlands, dark lava.
        state.continental = (0.6 + 0.4 * highland) * (1.0 - flooded)
        state.age = age
        state.orogeny = orogeny * (1.0 - flooded)
        state.activity = activity * 0.5

    def _advance(
        self,
        state: TectonicState,
        rng: np.random.Generator,
        dt: float
    ):

        s = self.relief

        vigor = math.exp(-state.time / self.COOLING_TIME)

        state.activity *= math.exp(-dt / 400.0)

        # Now and then a plume erupts: near the old volcanic
        # provinces while they last, anywhere otherwise.
        if rng.random() < 1.0 - math.exp(-dt / 120.0 * vigor):

            weights = state.activity + 1e-3

            cell = int(rng.choice(len(weights), p=weights / weights.sum()))

            center = _jitter(self.directions[cell], rng, 0.1)

            radius = rng.uniform(0.04, 0.15)

            rise = _dome(self.angle_to(center), radius)

            state.height += rng.uniform(500.0, 2_500.0) * s * rise
            state.age = np.where(rise > 0.05, 0.0, state.age)
            state.activity += rise
            state.continental *= 1.0 - 0.5 * rise

        # Relief relaxes very slowly in a cold lid.
        state.height = self.diffuse(state.height, 0.002 * dt / 50.0)


# =========================================================
# Episodic Resurfacing (Venus)
# =========================================================

class EpisodicResurfacing(RegimeSimulation):

    regime = "episodic_resurfacing"

    # Time between global resurfacing events (Myr).
    INTERVAL = (400.0, 800.0)

    def _initial(
        self,
        state: TectonicState,
        rng: np.random.Generator
    ):

        s = self.relief

        # Volcanic plains of rolling lowlands and uplands.
        height = 700.0 * s * self.noise(1, 1.5, 5)

        age = rng.uniform(350.0, 650.0) + 120.0 * self.noise(2, 2.0, 3)

        # Tesserae: high, rugged plateaus that survived the
        # last resurfacing (Ovda, Thetis, Alpha Regio).
        tessera = _smoothstep(0.22, 0.32, self.noise(3, 2.0, 5))

        height += 2_200.0 * s * tessera
        age = age + (rng.uniform(900.0, 1_500.0) - age) * tessera
        orogeny = 2_500.0 * tessera

        # A highland plateau ringed by mountains (Ishtar
        # Terra and Maxwell Montes).
        center = _random_direction(rng)

        radius = rng.uniform(0.15, 0.25)

        distance = self.angle_to(center)

        plateau = _smoothstep(radius, radius * 0.8, distance)

        height += 3_000.0 * s * plateau
        tessera = np.maximum(tessera, plateau)

        normal = _perpendicular(center, rng)

        offset, taper = self.arc(normal, _jitter(center, rng, radius * 0.5), radius)

        belt = np.exp(-(offset / (radius * 0.12)) ** 2) * taper * _smoothstep(radius * 1.4, radius * 0.6, distance)

        height += 5_000.0 * s * belt
        orogeny = np.maximum(orogeny, 6_000.0 * belt)

        activity = np.zeros_like(height)

        height, age, activity = self._volcanism(height, age, activity, rng, rises=3, rifts=6, coronae=40)

        state.height = self.diffuse(height, 0.25, 2)
        state.continental = tessera
        state.age = age
        state.orogeny = orogeny
        state.activity = activity

        # Time of the next resurfacing (Myr from now).
        state.memory["next_resurfacing"] = float(rng.uniform(150.0, 450.0))

    def _volcanism(
        self,
        height: np.ndarray,
        age: np.ndarray,
        activity: np.ndarray,
        rng: np.random.Generator,
        rises: int,
        rifts: int,
        coronae: int
    ):
        """Volcanic rises, rift zones and coronae, all young."""

        s = self.relief

        for _ in range(rises):

            rise = _dome(self.angle_to(_random_direction(rng)), rng.uniform(0.15, 0.3))

            height = height + rng.uniform(1_200.0, 2_500.0) * s * rise
            age = np.where(rise > 0.1, np.minimum(age, rng.uniform(0.0, 100.0)), age)
            activity = activity + rise

        # Rifts (chasmata): troughs with raised flanks.
        for _ in range(rifts):

            center = _random_direction(rng)

            offset, taper = self.arc(_perpendicular(center, rng), center, rng.uniform(0.15, 0.45))

            width = max(rng.uniform(0.01, 0.02), self.cell_angle * 1.2)

            height = height + s * taper * (
                -1_800.0 * np.exp(-(offset / width) ** 2)
                + 700.0 * np.exp(-(offset / (width * 4.0)) ** 2)
            )

            activity = activity - 2.0 * taper * np.exp(-(offset / (width * 3.0)) ** 2)

        # Coronae: rings over small upwellings, the center
        # raised or sunken.
        for _ in range(coronae):

            center = _random_direction(rng)

            radius = rng.uniform(0.02, 0.07)

            x = self.angle_to(center) / radius

            near = x < 1.6

            if not np.any(near):
                continue

            ring = np.exp(-((x[near] - 1.0) / 0.22) ** 2)

            interior = np.clip(1.0 - x[near] ** 2, 0.0, None) * rng.choice((-1.0, 1.0))

            height[near] += s * (700.0 * ring + 450.0 * interior)
            age[near] = np.minimum(age[near], np.where(x[near] < 1.3, rng.uniform(0.0, 200.0), age[near]))
            activity[near] += 0.5 * ring

        return height, age, activity

    def _advance(
        self,
        state: TectonicState,
        rng: np.random.Generator,
        dt: float
    ):

        next_event = state.memory.get("next_resurfacing", 300.0) - dt

        state.activity *= math.exp(-dt / 100.0)

        if next_event <= 0.0:

            # The lid founders: lava floods everything but
            # the high tesserae; then volcanism starts anew.
            s = self.relief

            plains = 700.0 * s * fbm(
                Perlin(self.settings.seed * 101 + 1000 + state.step_index),
                self.directions * 1.5,
                5
            )

            survives = _smoothstep(0.3, 0.7, state.continental) * _smoothstep(
                1_000.0 * s, 2_500.0 * s, state.height
            )

            flooded = 1.0 - survives

            state.height = state.height + (plains - state.height) * flooded
            state.age = state.age * survives
            state.orogeny = state.orogeny * survives
            state.continental = state.continental * survives
            state.activity = 3.0 * flooded

            next_event = float(rng.uniform(*self.INTERVAL))

        elif rng.random() < 1.0 - math.exp(-dt / 30.0):

            # Between events: a new corona or rift now and then.
            state.height, state.age, state.activity = self._volcanism(
                state.height, state.age, state.activity, rng,
                rises=0,
                rifts=int(rng.random() < 0.3),
                coronae=int(rng.integers(1, 3))
            )

        state.memory["next_resurfacing"] = next_event


# =========================================================
# Heat-Pipe Volcanism (Io)
# =========================================================

class HeatPipe(RegimeSimulation):

    regime = "heat_pipe"

    PATERAE = 140
    MOUNTAINS = 60

    # Fresh lava darkens the plains; sulfur frost brightens
    # them again over this many Myr.
    FROST_TIME = 1.5

    def _initial(
        self,
        state: TectonicState,
        rng: np.random.Generator
    ):

        s = self.relief

        # Lava plains, all a few Myr old at most.
        height = 400.0 * s + 250.0 * s * self.noise(1, 3.0, 5)
        age = np.clip(1.5 + 1.5 * self.noise(2, 4.0, 3), 0.0, None)
        bright = np.ones_like(height)
        orogeny = np.zeros_like(height)
        activity = np.zeros_like(height)

        # Mountains: isolated crustal blocks thrust up and
        # tilted as the lava-loaded crust subsides.
        for _ in range(self.MOUNTAINS):

            center = _random_direction(rng)

            radius = rng.uniform(0.02, 0.06)

            distance = self.angle_to(center)

            near = distance < radius

            if not np.any(near):
                continue

            tilt_axis = _perpendicular(center, rng)

            tilt = np.clip(0.5 + (self.directions[near] @ tilt_axis) / radius * 0.5, 0.0, 1.0)

            block = _smoothstep(radius, radius * 0.75, distance[near]) * (0.35 + 0.65 * tilt)

            peak = rng.uniform(2_000.0, 5_000.0) * s

            height[near] += peak * block
            orogeny[near] = np.maximum(orogeny[near], peak * block)
            bright[near] = np.minimum(bright[near], 1.0 - 0.4 * block)
            age[near] = np.maximum(age[near], 10.0 * block)

        state.height = height
        state.continental = bright
        state.age = age
        state.orogeny = orogeny
        state.activity = activity

        # Calderas: (direction, radius, depth, active).
        paterae = [self._new_patera(rng) for _ in range(self.PATERAE)]

        for patera in paterae:
            self._erupt(state, patera, rng, 1.0)

        state.memory["paterae"] = paterae

    def _new_patera(
        self,
        rng: np.random.Generator
    ) -> tuple:

        return (
            tuple(_random_direction(rng)),
            float(rng.uniform(0.006, 0.025)),
            float(rng.uniform(300.0, 1_500.0) * self.relief),
            bool(rng.random() < 0.6)
        )

    def _erupt(
        self,
        state: TectonicState,
        patera: tuple,
        rng: np.random.Generator,
        strength: float
    ):
        """Sink the caldera (its floor below the lava level) and spread dark flows."""

        center, radius, depth, active = patera

        distance = self.angle_to(np.asarray(center))

        near = distance < radius * 4.0

        if not np.any(near):
            return

        d = distance[near]

        floor = _smoothstep(radius, radius * 0.8, d)

        state.height[near] = np.minimum(
            state.height[near],
            state.height[near] * (1.0 - floor) - depth * floor
        )

        state.continental[near] *= 1.0 - 0.9 * floor

        if active:

            # Lava flows: dark and fresh around the vent.
            flows = _smoothstep(radius * 4.0, radius, d) * strength

            state.continental[near] *= 1.0 - 0.6 * flows
            state.age[near] *= 1.0 - flows
            state.activity[near] = np.maximum(state.activity[near], 3.0 * floor + flows)

    def _advance(
        self,
        state: TectonicState,
        rng: np.random.Generator,
        dt: float
    ):

        paterae = list(state.memory.get("paterae", []))

        # Sulfur frost re-brightens old flows; mountains
        # slump and erode.
        state.continental += (1.0 - state.continental) * (1.0 - math.exp(-dt / self.FROST_TIME))
        state.activity *= math.exp(-dt / 0.5)

        mountains = state.orogeny > 0.0

        slump = 1.0 - math.exp(-dt / 60.0)

        state.height[mountains] -= state.orogeny[mountains] * slump
        state.orogeny *= 1.0 - slump

        # Calderas die (filled with lava) and new ones open;
        # active ones keep erupting.
        survivors = []

        for patera in paterae:

            if rng.random() < 1.0 - math.exp(-dt / 8.0):

                center, radius, _, _ = patera

                fill = _smoothstep(radius * 1.2, radius * 0.8, self.angle_to(np.asarray(center)))

                state.height += (400.0 * self.relief - state.height) * fill * (state.height < 0.0)
                state.age *= 1.0 - fill

                continue

            survivors.append(patera)

        while len(survivors) < self.PATERAE:
            survivors.append(self._new_patera(rng))

        for patera in survivors:
            self._erupt(state, patera, rng, min(1.0, dt))

        # A new mountain block now and then.
        if rng.random() < 1.0 - math.exp(-dt / 3.0):

            center = _random_direction(rng)

            radius = rng.uniform(0.02, 0.05)

            block = _smoothstep(radius, radius * 0.75, self.angle_to(center))

            peak = rng.uniform(1_500.0, 4_000.0) * self.relief

            state.height += peak * block
            state.orogeny = np.maximum(state.orogeny, peak * block)

        state.memory["paterae"] = survivors


# =========================================================
# Ice Shell (Europa, Ganymede, Triton)
# =========================================================

class IceShell(RegimeSimulation):

    regime = "ice_shell"

    # Counts at the default 128-cell grid. Lines are at
    # least ~1.5 cells wide, so coarser grids get fewer
    # (wider) ones and the dark share stays the same.
    RIDGES = 140
    BANDS = 20
    CHAOS = 18

    def __init__(
        self,
        settings: TectonicSettings
    ):

        super().__init__(settings)

        # Noise that bends the cracks (built on first use).
        self._wobble: tuple[np.ndarray, np.ndarray] | None = None

    def _initial(
        self,
        state: TectonicState,
        rng: np.random.Generator
    ):

        s = self.relief

        state.height = 150.0 * s * self.noise(1, 2.0, 4)
        state.continental = np.ones_like(state.height, dtype=np.float64)
        state.age = 50.0 + 20.0 * self.noise(2, 2.0, 3)
        state.orogeny = np.zeros_like(state.height, dtype=np.float64)
        state.activity = np.zeros_like(state.height, dtype=np.float64)

        density = min(self.settings.resolution / 128.0, 1.5)

        for _ in range(int(self.BANDS * density)):
            self._band(state, rng, rng.uniform(0.0, 40.0))

        for _ in range(int(self.RIDGES * density)):
            self._ridge(state, rng, rng.uniform(0.0, 60.0))

        for _ in range(self.CHAOS):
            self._chaos(state, rng, rng.uniform(0.0, 20.0))

    def _cracks(
        self,
        rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Offset and taper of a tidal crack: a great-circle arc,
        wobbling as the tidal stress it follows turns (Europa's
        cycloidal ridges).
        """

        center = _random_direction(rng)

        offset, taper = self.arc(_perpendicular(center, rng), center, rng.uniform(0.1, 1.0))

        if self._wobble is None:
            self._wobble = (self.noise(11, 5.0, 3), self.noise(12, 9.0, 2))

        slow, fast = self._wobble

        wobble = rng.uniform(-0.04, 0.04) * slow + rng.uniform(-0.015, 0.015) * fast

        return offset + wobble, taper

    def _ridge(
        self,
        state: TectonicState,
        rng: np.random.Generator,
        age: float
    ):
        """A double ridge along a crack: dark lineae."""

        offset, taper = self._cracks(rng)

        # At least ~1.5 grid cells wide, so lines stay
        # continuous when sampled.
        width = self.cell_angle * rng.uniform(1.4, 2.0)

        line = np.exp(-(offset / width) ** 2) * taper

        near = line > 0.01

        state.height[near] += rng.uniform(100.0, 300.0) * self.relief * line[near]
        state.continental[near] *= 1.0 - 0.65 * line[near]
        state.age[near] = np.minimum(state.age[near], np.where(line[near] > 0.3, age, state.age[near]))
        state.activity[near] += 0.5 * line[near]

    def _band(
        self,
        state: TectonicState,
        rng: np.random.Generator,
        age: float
    ):
        """A spreading band: the shell pulled apart and refilled."""

        offset, taper = self._cracks(rng)

        width = self.cell_angle * rng.uniform(2.0, 4.0)

        band = _smoothstep(width, width * 0.6, np.abs(offset)) * taper

        near = band > 0.01

        state.height[near] += (-60.0 * self.relief - state.height[near] * 0.7) * band[near]
        state.continental[near] = state.continental[near] * (1.0 - band[near]) + 0.45 * band[near]
        state.age[near] = state.age[near] * (1.0 - band[near]) + age * band[near]
        state.activity[near] -= band[near]

    def _chaos(
        self,
        state: TectonicState,
        rng: np.random.Generator,
        age: float
    ):
        """Chaos terrain: the shell broken into tilted blocks over warm ice."""

        center = _random_direction(rng)

        radius = rng.uniform(0.03, 0.09)

        patch = _smoothstep(radius, radius * 0.7, self.angle_to(center))

        near = patch > 0.0

        hummocks = fbm(
            Perlin(self.settings.seed * 101 + 77 + int(rng.integers(1 << 20))),
            self.directions[near] * 60.0,
            3
        )

        state.height[near] += self.relief * (-80.0 + 250.0 * hummocks) * patch[near]
        state.continental[near] *= 1.0 - 0.7 * patch[near]
        state.age[near] = state.age[near] * (1.0 - patch[near]) + age * patch[near]
        state.orogeny[near] = np.maximum(state.orogeny[near], 400.0 * patch[near])

    def _advance(
        self,
        state: TectonicState,
        rng: np.random.Generator,
        dt: float
    ):

        density = min(self.settings.resolution / 128.0, 1.5)

        for _ in range(int(rng.poisson(4.0 * dt * density))):
            self._ridge(state, rng, 0.0)

        if rng.random() < 1.0 - math.exp(-dt / 2.0):
            self._band(state, rng, 0.0)

        if rng.random() < 1.0 - math.exp(-dt / 4.0):
            self._chaos(state, rng, 0.0)

        state.activity *= math.exp(-dt / 2.0)

        # Ice flows: relief relaxes.
        state.height = self.diffuse(state.height, min(0.02 * dt, 0.2))


# =========================================================
# Factory
# =========================================================

REGIME_SIMULATIONS = {
    StagnantLid.regime: StagnantLid,
    EpisodicResurfacing.regime: EpisodicResurfacing,
    HeatPipe.regime: HeatPipe,
    IceShell.regime: IceShell,
}


# =========================================================
# Geometry
# =========================================================

def _random_direction(
    rng: np.random.Generator
) -> np.ndarray:

    v = rng.normal(size=3)

    return v / np.linalg.norm(v)


def _perpendicular(
    direction: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """A random unit vector perpendicular to `direction`."""

    v = np.cross(direction, _random_direction(rng))

    return v / np.linalg.norm(v)


def _jitter(
    direction: np.ndarray,
    rng: np.random.Generator,
    angle: float
) -> np.ndarray:

    v = direction + rng.normal(size=3) * angle

    return v / np.linalg.norm(v)


def _dome(
    distance: np.ndarray,
    radius: float
) -> np.ndarray:
    """Smooth rise: 1 at the center, 0 at `radius`."""

    x = np.clip(distance / radius, 0.0, 1.0)

    return (1.0 - x * x) ** 2


def _basin(
    distance: np.ndarray,
    radius: float,
    depth: float
) -> np.ndarray:
    """An impact basin: a bowl with a raised rim."""

    x = distance / radius

    bowl = -depth * np.clip(1.0 - x * x, 0.0, None)

    rim = 0.2 * depth * np.exp(-((x - 1.0) / 0.25) ** 2)

    return bowl + rim


def _smoothstep(
    edge0: float,
    edge1: float,
    x
) -> np.ndarray:

    t = np.clip((np.asarray(x) - edge0) / (edge1 - edge0), 0.0, 1.0)

    return t * t * (3.0 - 2.0 * t)
