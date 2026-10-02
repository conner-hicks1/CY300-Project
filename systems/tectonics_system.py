import time

from dataclasses import dataclass

from core.jobs import Job, JobSystem
from core.logger import Logger

from ecs.components import PlanetComponent, TectonicsComponent
from ecs.entity import Entity

from planet.tectonics import (
    PlateInfo,
    TectonicField,
    TectonicSettings,
    TectonicSimulation,
    TectonicState
)

from scene.scene import Scene


@dataclass(slots=True)
class TectonicsStatus:

    ready: bool = False
    playing: bool = False

    # Re-simulating up to the saved time after a load or
    # a reset.
    catching_up: bool = False

    time: float = 0.0
    target_time: float = 0.0

    step_seconds: float = 0.0

    plates: tuple[PlateInfo, ...] = ()

    continental_fraction: float = 0.0


class _Run:

    # One planet's simulation.

    def __init__(
        self,
        settings: TectonicSettings
    ):

        self.settings = settings

        self.simulation = TectonicSimulation(settings)

        self.state: TectonicState | None = None
        self.field: TectonicField | None = None

        self.job: Job | None = None

        self.playing = False

        # Steps requested while paused (Step button).
        self.pending_steps = 0

        self.last_step_started = 0.0

        self.version = 0

        self.plates: tuple[PlateInfo, ...] = ()

        self.alive = True


class TectonicsSystem:

    # =====================================================
    # Plate Tectonics Runner
    # =====================================================
    #
    # Runs each planet's TectonicSimulation on the job
    # system, one step per job, and publishes an immutable
    # TectonicField after every step. The planet's terrain
    # (PlanetSystem, via field()) rebuilds from new fields.
    #
    # TectonicsComponent.simulated_time is the saved
    # position in time. A freshly loaded (or reset) planet is
    # re-simulated from its seed up to that time; while
    # playing, the component follows the simulation.

    # Simulated million years per real second at most while
    # playing (steps otherwise run back to back).
    DEFAULT_RATE = 50.0

    # Settings must be unchanged this long before the
    # simulation restarts with them (slider drags).
    RESTART_DELAY = 0.4

    def __init__(
        self,
        jobs: JobSystem
    ):

        self._jobs = jobs

        self._runs: dict[Entity, _Run] = {}

        # entity -> (settings waiting to apply, since).
        self._pending: dict[Entity, tuple[TectonicSettings, float]] = {}

        self.rate = self.DEFAULT_RATE

    # =====================================================
    # Queries / Commands (editor)
    # =====================================================

    def field(
        self,
        entity: Entity
    ) -> TectonicField | None:

        run = self._runs.get(entity)

        return run.field if run is not None else None

    def status(
        self,
        entity: Entity,
        component: TectonicsComponent
    ) -> TectonicsStatus:

        run = self._runs.get(entity)

        if run is None or run.state is None:

            return TectonicsStatus(target_time=component.simulated_time)

        state = run.state

        return TectonicsStatus(
            ready=True,
            playing=run.playing,
            catching_up=state.time + 1e-6 < component.simulated_time and not run.playing,
            time=state.time,
            target_time=component.simulated_time,
            step_seconds=state.step_seconds,
            plates=run.plates,
            continental_fraction=float((state.continental > 0.5).mean())
        )

    def prime(
        self,
        entity: Entity,
        planet: PlanetComponent,
        component: TectonicsComponent,
        state: TectonicState
    ):
        """
        Start from a state computed elsewhere (the demo
        builds the initial state up front so its terrain is
        ready on the first frame).
        """

        run = _Run(tectonic_settings_for(planet, component))

        self._publish(run, state)

        old = self._runs.get(entity)

        if old is not None:
            self._stop(old)

        self._runs[entity] = run

    def play(
        self,
        entity: Entity,
        playing: bool = True
    ):

        run = self._runs.get(entity)

        if run is not None:
            run.playing = playing

    def step(
        self,
        entity: Entity,
        count: int = 1
    ):

        run = self._runs.get(entity)

        if run is not None:
            run.pending_steps += count

    def reset(
        self,
        entity: Entity,
        component: TectonicsComponent
    ):
        """Back to time 0 (same seed and settings)."""

        component.simulated_time = 0.0

        run = self._runs.get(entity)

        if run is None:
            return

        self._stop(run)

        fresh = _Run(run.settings)

        fresh.field = run.field

        self._runs[entity] = fresh

        self._start(fresh)

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
            TectonicsComponent
        ):

            seen.add(entity)

            settings = tectonic_settings_for(planet, component)

            run = self._runs.get(entity)

            if run is None or (run.settings != settings and self._settled(entity, settings)):

                previous_field = None

                if run is not None:

                    Logger.info("[Tectonics] Settings changed; restarting from time 0.")

                    self._stop(run)

                    component.simulated_time = 0.0

                    previous_field = run.field

                run = _Run(settings)

                # Keep showing the old continents until the
                # new initial state is ready.
                run.field = previous_field

                self._runs[entity] = run

                self._start(run)

                continue

            if run.settings != settings:
                continue

            if run.job is not None or run.state is None:
                continue

            # Follow the saved time (catch up after a load),
            # play, or perform requested single steps.
            behind = run.state.time + 1e-6 < component.simulated_time

            if run.playing:

                interval = settings.time_step / max(self.rate, 1e-3)

                if time.monotonic() - run.last_step_started >= interval:
                    self._advance(run, component)

            elif behind:

                self._advance(run, component, follow=False)

            elif run.pending_steps > 0:

                run.pending_steps -= 1

                self._advance(run, component)

        for entity in [e for e in self._runs if e not in seen]:

            self._stop(self._runs.pop(entity))

            self._pending.pop(entity, None)

    def _settled(
        self,
        entity: Entity,
        settings: TectonicSettings
    ) -> bool:

        now = time.monotonic()

        pending = self._pending.get(entity)

        if pending is None or pending[0] != settings:

            self._pending[entity] = (settings, now)

            return self.RESTART_DELAY <= 0.0

        if now - pending[1] >= self.RESTART_DELAY:

            del self._pending[entity]

            return True

        return False

    # =====================================================
    # Jobs
    # =====================================================

    def _start(
        self,
        run: _Run
    ):

        simulation = run.simulation

        def done(state: TectonicState):

            run.job = None

            if run.alive:
                self._publish(run, state)

        run.job = self._jobs.submit(
            simulation.initial_state,
            on_complete=done,
            on_error=lambda error: self._failed(run, error),
            name="tectonics: initial state"
        )

    def _advance(
        self,
        run: _Run,
        component: TectonicsComponent,
        follow: bool = True
    ):
        """
        follow: the component's time tracks the simulation
            (playing / stepping); False while catching up to
            it.
        """

        state = run.state
        simulation = run.simulation

        run.last_step_started = time.monotonic()

        def done(new_state: TectonicState):

            run.job = None

            if not run.alive:
                return

            self._publish(run, new_state)

            if follow:
                component.simulated_time = max(component.simulated_time, new_state.time)

        run.job = self._jobs.submit(
            lambda: simulation.step(state),
            on_complete=done,
            on_error=lambda error: self._failed(run, error),
            name=f"tectonics: step to {state.time + run.settings.time_step:.0f} Myr"
        )

    def _publish(
        self,
        run: _Run,
        state: TectonicState
    ):

        run.state = state

        run.version += 1

        run.field = TectonicField.from_state(
            run.simulation.grid,
            state,
            run.version
        )

        run.plates = tuple(run.simulation.plates(state))

    def _failed(
        self,
        run: _Run,
        error: BaseException
    ):

        run.job = None
        run.playing = False

        Logger.error(
            "[Tectonics] Simulation step failed: %s: %s",
            type(error).__name__,
            error
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


def tectonic_settings_for(
    planet: PlanetComponent,
    component: TectonicsComponent
) -> TectonicSettings:

    return TectonicSettings(
        seed=int(component.seed),
        plate_count=max(2, int(component.plate_count)),
        land_fraction=float(component.land_fraction),
        plate_speed=float(component.plate_speed),
        time_step=max(0.5, float(component.time_step)),
        resolution=max(16, int(component.resolution)),
        radius=float(planet.radius)
    )
