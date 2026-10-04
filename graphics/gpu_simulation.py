import time

import numpy as np

from core.logger import Logger

from graphics.compute import ComputeShader, GpuQueue, GpuQueueClosed, StorageBuffer

from planet.climate import solve_temperature
from planet.hydrology import drainage
from planet.tectonics import rotated_cells


# =========================================================
# Simulations on the GPU
# =========================================================
#
# The planet/acceleration.py accelerator: the simulations'
# heaviest steps as compute shaders (assets/shaders/
# compute/), called from the simulation threads through the
# GpuQueue:
#
#   temperature    the climate's radiative-diffusive balance:
#                  Newton steps, each a full conjugate-
#                  gradient solve in one dispatch
#   drainage       the hydrology's basin filling, routing and
#                  discharge accumulation
#   rotated cells  where each tectonic plate's crust came from
#
# Each returns what the NumPy reference does (to float32
# precision). If a kernel fails (an old driver), the
# accelerator logs it, switches itself off and the NumPy
# version answers from then on.

SHADERS = "assets/shaders/compute"

# Work per dispatch, a few ms at most (OpenGL draws nothing
# while a compute dispatch runs): fill and accumulation
# sweeps, conjugate-gradient iterations.
SWEEPS_PER_DISPATCH = 64
CG_PER_DISPATCH = 64

# Newton steps and CG iterations per step, as on the CPU.
NEWTON_STEPS = 40
CG_ITERATIONS = 2000


class GpuSimulation:

    def __init__(
        self,
        queue: GpuQueue
    ):
        """Create on the GL thread (compiles the kernels)."""

        self._queue = queue

        self._shaders = {
            name: ComputeShader(f"{SHADERS}/{name}.comp.glsl")
            for name in ("temperature", "drainage", "rotated_cells")
        }

        self._buffers: dict[str, StorageBuffer] = {}

        self._donors: tuple[int, np.ndarray] | None = None

        self.enabled = True

        # Diagnostics: kernel name -> (calls, seconds of wall time).
        self.timings: dict[str, list[float]] = {}

    # =====================================================
    # Accelerator
    # =====================================================

    def solve_temperature(self, absorbed, emissivity, conductance, neighbors, initial_kelvin):

        return self._call(
            "temperature",
            lambda: self._temperature(absorbed, emissivity, conductance, neighbors, initial_kelvin),
            lambda: solve_temperature(absorbed, emissivity, conductance, neighbors, initial_kelvin)
        )

    def drainage(self, elevation, ocean, neighbors, runoff):

        return self._call(
            "drainage",
            lambda: self._drainage(elevation, ocean, neighbors, runoff),
            lambda: drainage(elevation, ocean, neighbors, runoff)
        )

    def rotated_cells(self, directions, rotations, resolution):

        return self._call(
            "rotated_cells",
            lambda: self._rotated_cells(directions, rotations, resolution),
            lambda: rotated_cells(directions, rotations, resolution)
        )

    def _call(
        self,
        name: str,
        gpu,
        cpu
    ):

        if not self.enabled:
            return cpu()

        started = time.perf_counter()

        try:

            result = self._queue.run(gpu)

        except GpuQueueClosed:

            # Shutting down: the simulation gives up (its job
            # fails quietly) rather than holding up the exit
            # with a long CPU solve.
            raise

        except Exception as error:

            Logger.error(
                "[GPU] %s kernel failed (%s: %s); simulations use the CPU from now on.",
                name,
                type(error).__name__,
                error
            )

            self.enabled = False

            return cpu()

        timing = self.timings.setdefault(name, [0, 0.0])

        timing[0] += 1
        timing[1] += time.perf_counter() - started

        return result

    # =====================================================
    # Kernels (GL thread; generators: one dispatch per
    # next(), so long solves spread over frames)
    # =====================================================

    def _buffer(
        self,
        name: str,
        data=None,
        nbytes: int | None = None
    ) -> StorageBuffer:

        buffer = self._buffers.get(name)

        if buffer is None:

            buffer = StorageBuffer(data if data is not None else int(nbytes))

            self._buffers[name] = buffer

        elif data is not None:

            buffer.upload(data)

        elif buffer.nbytes != nbytes:

            buffer.allocate(nbytes)

        return buffer

    def _temperature(self, absorbed, emissivity, conductance, neighbors, initial_kelvin):

        cells = len(absorbed)

        shader = self._shaders["temperature"]

        bindings = [
            self._buffer("absorbed", np.asarray(absorbed, np.float32)),
            self._buffer("emissivity", np.asarray(emissivity, np.float32)),
            self._buffer("conductance", np.asarray(conductance, np.float32).ravel()),
            self._buffer("neighbors", np.asarray(neighbors, np.int32).ravel()),
            self._buffer("kelvin", np.full(cells, initial_kelvin, np.float32)),
        ]

        for name in ("step", "residual", "direction", "applied", "slope", "diagonal"):
            bindings.append(self._buffer(name, nbytes=cells * 4))

        bindings.append(self._buffer("status", np.zeros(8, np.float32)))

        for newton in range(NEWTON_STEPS):

            phase = 0

            while True:

                shader.bind()

                shader.set_uint("uCells", cells)
                shader.set_uint("uMaxIterations", CG_PER_DISPATCH)
                shader.set_uint("uTotalIterations", CG_ITERATIONS)
                shader.set_uint("uPhase", phase)

                for binding, buffer in enumerate(bindings):
                    buffer.bind(binding)

                shader.dispatch(1)

                # Next frame: the GPU has finished by then.
                yield

                status = bindings[-1].read(np.float32, 8)

                if status[2] > 0.5:
                    break

                phase = 1

            largest_step = float(status[0])

            if not np.isfinite(largest_step):
                raise FloatingPointError("temperature solve diverged")

            if largest_step < 0.01:
                break

        return bindings[4].read(np.float32, cells).astype(np.float64)

    def _donor_table(
        self,
        neighbors: np.ndarray
    ) -> np.ndarray:
        """Per cell, the cells listing it as a neighbor (-1 padded)."""

        key = (neighbors.shape, hash(neighbors.tobytes()))

        if self._donors is not None and self._donors[0] == key:
            return self._donors[1]

        cells = len(neighbors)

        lists: list[list[int]] = [[] for _ in range(cells)]

        for cell, row in enumerate(neighbors.tolist()):
            for neighbor in row:
                if cell not in lists[neighbor]:
                    lists[neighbor].append(cell)

        width = max(len(entries) for entries in lists)

        table = np.full((cells, width), -1, dtype=np.int32)

        for cell, entries in enumerate(lists):
            table[cell, :len(entries)] = entries

        self._donors = (key, table)

        return table

    def _drainage(self, elevation, ocean, neighbors, runoff):

        cells = len(elevation)

        donors = self._donor_table(np.asarray(neighbors))

        shader = self._shaders["drainage"]

        bindings = [
            self._buffer("elevation", np.asarray(elevation, np.float32)),
            self._buffer("ocean", np.asarray(ocean, np.uint32)),
            self._buffer("donors", donors.ravel()),
            self._buffer("runoff", np.asarray(runoff, np.float32)),
            self._buffer("filled", nbytes=cells * 4),
            self._buffer("receiver", nbytes=cells * 4),
            self._buffer("discharge", nbytes=cells * 4),
            self._buffer("previous", nbytes=cells * 4),
            self._buffer("status", np.zeros(2, np.float32)),
            self._buffer("neighbors", np.asarray(neighbors, np.int32).ravel()),
        ]

        def run(phase):

            shader.bind()

            shader.set_uint("uCells", cells)
            shader.set_uint("uDonors", donors.shape[1])
            shader.set_uint("uPhase", phase)
            shader.set_uint("uSweeps", SWEEPS_PER_DISPATCH)

            for binding, buffer in enumerate(bindings):
                buffer.bind(binding)

            shader.dispatch(1)

        # Fill, then route and accumulate, each until settled.
        for first, more in ((0, 1), (2, 3)):

            phase = first

            for _ in range(cells):

                run(phase)

                yield

                if bindings[8].read(np.float32, 2)[0] == 0.0:
                    break

                phase = more

        filled = bindings[4].read(np.float32, cells).astype(np.float64)
        receiver = bindings[5].read(np.int32, cells).astype(np.int64)
        discharge = bindings[6].read(np.float32, cells).astype(np.float64)

        # Never reached from the sea (a basin with no outlet
        # on the grid): as on the CPU, left as it is.
        unreached = filled >= 1e29

        filled[unreached] = np.asarray(elevation)[unreached]

        return filled, receiver, discharge

    def _rotated_cells(self, directions, rotations, resolution):

        cells = len(directions)
        count = len(rotations)

        shader = self._shaders["rotated_cells"]

        bindings = [
            self._buffer("directions", np.asarray(directions, np.float32).ravel()),
            self._buffer("rotations", np.asarray(rotations, np.float32).reshape(count, 9).ravel()),
            self._buffer("cells", nbytes=cells * count * 4),
        ]

        shader.bind()

        shader.set_uint("uCells", cells)
        shader.set_uint("uRotations", count)
        shader.set_int("uResolution", resolution)

        for binding, buffer in enumerate(bindings):
            buffer.bind(binding)

        shader.dispatch((cells * count + 255) // 256)

        yield

        return bindings[2].read(np.int32, cells * count).astype(np.int64).reshape(count, cells)

    # =====================================================
    # Shutdown
    # =====================================================

    def delete(self):

        for buffer in self._buffers.values():
            buffer.delete()

        for shader in self._shaders.values():
            shader.delete()

        self._buffers.clear()
