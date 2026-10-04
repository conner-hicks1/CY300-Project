from typing import Protocol

import numpy as np


# =========================================================
# Acceleration
# =========================================================
#
# The simulations' heaviest data-parallel steps can run on
# the GPU (graphics/gpu_simulation.py); the application
# installs that accelerator when compute shaders are
# available. planet/ never touches OpenGL itself: it asks
# here, and without an accelerator (tests, tools, an old
# GPU) it runs its own NumPy versions, which are also the
# reference the GPU kernels are tested against.
#
# Results agree to float32 precision (the GPU works in
# single precision): within a run everything stays
# deterministic, but a CPU and a GPU run may differ in the
# last digits.


class Accelerator(Protocol):

    def solve_temperature(
        self,
        absorbed: np.ndarray,
        emissivity: np.ndarray,
        conductance: np.ndarray,
        neighbors: np.ndarray,
        initial_kelvin: float
    ) -> np.ndarray:
        """
        Radiative-diffusive balance (planet/climate.py):
        absorbed = emissivity sigma T^4 + sum_j c_ij (T_i - T_j)
        per cell; returns T (K).
        """

    def drainage(
        self,
        elevation: np.ndarray,
        ocean: np.ndarray,
        neighbors: np.ndarray,
        runoff: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        (filled surface, receiver per cell (-1 = none),
        accumulated discharge): every basin filled to its
        spill point, each land cell draining to its lowest
        neighbor, discharge summed downstream
        (planet/hydrology.py).
        """

    def rotated_cells(
        self,
        directions: np.ndarray,
        rotations: np.ndarray,
        resolution: int
    ) -> np.ndarray:
        """
        (len(rotations), cells) sphere-grid cell index of
        every direction rotated by each 3x3 rotation
        (planet/tectonics.py: where each plate's crust came
        from).
        """


_accelerator: Accelerator | None = None


def set_accelerator(
    accelerator: Accelerator | None
):

    global _accelerator

    _accelerator = accelerator


def accelerator() -> Accelerator | None:

    return _accelerator
