from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from core.jobs import Job, JobSystem
from core.logger import Logger

from ecs.components import BodyComponent, ClimateComponent, PlanetComponent
from ecs.entity import Entity

from planet.bodies import climate_physics
from planet.climate import (
    ClimateField,
    ClimateModel,
    ClimateSettings
)
from planet.terrain import Terrain

from scene.scene import Scene

from systems.planet_system import terrain_settings_for


@dataclass(slots=True)
class ClimateStatus:

    ready: bool = False
    computing: bool = False
    compute_seconds: float = 0.0
    field: ClimateField | None = None


class _Run:

    def __init__(self):

        self.key: tuple | None = None

        self.job: Job | None = None
        self.job_key: tuple | None = None

        self.field: ClimateField | None = None
        self.version = 0

        self.compute_seconds = 0.0

        self.alive = True


class ClimateSystem:

    # =====================================================
    # Climate Runner
    # =====================================================
    #
    # Keeps each planet's ClimateField in step with its land:
    # whenever the inputs change (climate settings, terrain
    # settings, or a new tectonic field while the plates
    # move), the climate is recomputed on a worker thread
    # (~0.3 s). While a computation runs, further changes
    # wait; the next run uses the latest inputs, so a fast
    # tectonic simulation skips intermediate states instead
    # of queuing them.

    def __init__(
        self,
        jobs: JobSystem,
        tectonic_field: Callable[[Entity], object] | None = None
    ):
        """tectonic_field: entity -> TectonicField or None."""

        self._jobs = jobs
        self._tectonic_field = tectonic_field or (lambda entity: None)

        self._runs: dict[Entity, _Run] = {}

        # Field versions are unique across runs and primed
        # fields, so consumers can compare them.
        self._next_version = 1

    # =====================================================
    # Queries
    # =====================================================

    def field(
        self,
        entity: Entity
    ) -> ClimateField | None:

        run = self._runs.get(entity)

        return run.field if run is not None else None

    def status(
        self,
        entity: Entity
    ) -> ClimateStatus:

        run = self._runs.get(entity)

        if run is None:
            return ClimateStatus()

        return ClimateStatus(
            ready=run.field is not None,
            computing=run.job is not None,
            compute_seconds=run.compute_seconds,
            field=run.field
        )

    def prime(
        self,
        entity: Entity,
        planet: PlanetComponent,
        component: ClimateComponent,
        field: ClimateField,
        tectonic_field=None,
        body: BodyComponent | None = None
    ):
        """Start from a field computed elsewhere (the demo)."""

        run = _Run()

        run.key = _key(planet, component, tectonic_field, body)
        run.field = field
        run.version = field.version

        self._next_version = max(self._next_version, field.version + 1)

        old = self._runs.get(entity)

        if old is not None:
            self._stop(old)

        self._runs[entity] = run

    # =====================================================
    # Update
    # =====================================================

    def update(
        self,
        scene: Scene
    ):

        seen = set()

        for entity, planet, component in scene.registry.view_with(
            PlanetComponent,
            ClimateComponent
        ):

            seen.add(entity)

            run = self._runs.get(entity)

            if run is None:

                run = _Run()

                self._runs[entity] = run

            tectonic_field = self._tectonic_field(entity)

            body = scene.registry.try_get(entity, BodyComponent)

            key = _key(planet, component, tectonic_field, body)

            if run.job is None and key != run.key:
                self._start(run, key, planet, component, tectonic_field, body)

        for entity in [e for e in self._runs if e not in seen]:
            self._stop(self._runs.pop(entity))

    def _start(
        self,
        run: _Run,
        key: tuple,
        planet: PlanetComponent,
        component: ClimateComponent,
        tectonic_field,
        body: BodyComponent | None = None
    ):

        settings = _climate_settings(component, planet, body)
        terrain_settings = terrain_settings_for(planet)

        version = self._next_version

        self._next_version += 1

        def work():

            return compute_climate(
                settings,
                Terrain(terrain_settings, tectonic_field),
                version=version
            )

        def done(result: tuple[ClimateField, float]):

            run.job = None

            if not run.alive:
                return

            field, seconds = result

            run.field = field
            run.version = field.version
            run.key = key
            run.compute_seconds = seconds

        def failed(error: BaseException):

            run.job = None

            # Do not retry the same inputs every frame.
            run.key = key

            Logger.error(
                "[Climate] Computation failed: %s: %s",
                type(error).__name__,
                error
            )

        run.job_key = key

        run.job = self._jobs.submit(
            work,
            on_complete=done,
            on_error=failed,
            name="climate"
        )

    @staticmethod
    def _stop(
        run: _Run
    ):

        run.alive = False

        if run.job is not None:

            run.job.cancel()

            run.job = None

    def shutdown(self):

        for run in self._runs.values():
            self._stop(run)

        self._runs.clear()


# =========================================================
# Helpers
# =========================================================

def compute_climate(
    settings: ClimateSettings,
    terrain: Terrain,
    version: int = 1
) -> tuple[ClimateField, float]:
    """
    Climate for a terrain: elevation sampled on the climate
    grid, then the model. Returns (field, seconds).
    """

    model = ClimateModel(settings)

    grid = model.grid

    spacing = grid.cell_angle * terrain.settings.radius

    elevation = terrain.elevation(grid.directions, spacing)

    state = model.compute(elevation)

    field = ClimateField.from_state(
        grid,
        state,
        version,
        land=elevation > 0.0,
        lapse_rate=settings.lapse_rate
    )

    return field, state.compute_seconds


def _climate_settings(
    component: ClimateComponent,
    planet: PlanetComponent | None = None,
    body: BodyComponent | None = None
) -> ClimateSettings:
    """
    Climate settings for a planet: the climate component's
    tilt, offset and humidity, plus the body's physics
    (sunlight, albedo, greenhouse, pressure, lapse rate);
    Earth's physics without a body.
    """

    humidity = max(0.0, float(component.humidity))

    physics = {}

    if body is not None:

        physics = climate_physics(
            body,
            planet.liquid if planet is not None else "water",
            humidity
        )

    elif planet is not None and planet.liquid != "water":

        physics = {"ocean": False}

    return ClimateSettings(
        axial_tilt=float(np.clip(component.axial_tilt, 0.0, 90.0)),
        temperature_offset=float(component.temperature_offset),
        humidity=humidity,
        resolution=max(16, int(component.resolution)),
        **physics
    )


def climate_settings_for(
    component: ClimateComponent,
    planet: PlanetComponent | None = None,
    body: BodyComponent | None = None
) -> ClimateSettings:

    return _climate_settings(component, planet, body)


def _key(
    planet: PlanetComponent,
    component: ClimateComponent,
    tectonic_field,
    body: BodyComponent | None = None
) -> tuple:

    return (
        _climate_settings(component, planet, body),
        terrain_settings_for(planet),
        tectonic_field.version if tectonic_field is not None else None,
    )
