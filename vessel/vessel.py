import math

from dataclasses import dataclass, field

from vessel.parts import PartDefinition, VesselDesign


# Standard gravity: defines the specific impulse unit.
G0 = 9.80665

# Sea-level pressure (bar) at which isp_sea_level applies.
ATMOSPHERE_BAR = 1.01325

# Drag coefficient of a rocket nose-first (a cone on a
# cylinder, ~0.3-0.5).
BODY_DRAG_COEFFICIENT = 0.45

# Seconds a parachute takes to open fully.
PARACHUTE_OPENING = 3.0


# =========================================================
# A Vessel's Parts
# =========================================================
#
# The stack as it flies: which parts are still attached,
# how much propellant each tank holds, which engines burn,
# and what that adds up to (mass, center of mass, thrust,
# moments of inertia). Physics is in vessel/physics.py.
#
# Stages fire in order with Space: an engine ignites at
# its stage; a separator fires at its stage and drops
# everything below it. Separators also split the
# propellant: engines burn only the tanks of their own
# section (between separators), all of them evenly.
#
# Positions along the stack are heights above the bottom of
# the lowest attached part (the vessel's origin).


@dataclass(slots=True)
class PartState:

    part: PartDefinition
    stage: int | None
    propellant: float = 0.0
    ignited: bool = False

    # Height of the part's bottom above the vessel's origin.
    bottom: float = 0.0

    # Parachutes: staged (waits for thick, slow enough air),
    # and how far open, 0..1.
    armed: bool = False
    deployed: float = 0.0


@dataclass(frozen=True, slots=True)
class StageStats:
    """One stage as the editor shows it (vessel.stage_stats)."""

    stage: int
    start_mass: float               # kg when it fires
    end_mass: float                 # kg when its propellant is gone
    delta_v_vacuum: float           # m/s
    delta_v_surface: float          # m/s at the launch body's surface pressure
    twr_surface: float              # thrust / weight at its start, on the launch body's surface
    burn_time: float                # s at full throttle
    engines: int


@dataclass(slots=True)
class Vessel:

    name: str
    parts: list[PartState]

    # The stage Space fires next.
    next_stage: int = 0

    # 0..1: how hard the burning engines push.
    throttle: float = 0.0

    # Parts dropped by separators, for the renderer (cleared
    # by whoever takes them).
    dropped: list[PartState] = field(default_factory=list)

    @classmethod
    def from_design(
        cls,
        design: VesselDesign
    ) -> "Vessel":

        vessel = cls(
            design.name,
            [PartState(entry.part, entry.stage, propellant=entry.part.propellant) for entry in design.stack]
        )

        vessel._restack()

        return vessel

    def _restack(self):

        height = 0.0

        for state in reversed(self.parts):

            state.bottom = height

            height += state.part.height

    # -----------------------------------------------------
    # Sums
    # -----------------------------------------------------

    @property
    def mass(
        self
    ) -> float:
        """kg, with propellant."""

        return sum(state.part.mass + state.propellant for state in self.parts)

    @property
    def height(
        self
    ) -> float:

        return sum(state.part.height for state in self.parts)

    @property
    def radius(
        self
    ) -> float:

        return max(state.part.radius for state in self.parts)

    @property
    def center_of_mass(
        self
    ) -> float:
        """Height above the origin (m)."""

        total = self.mass

        return sum(
            (state.part.mass + state.propellant) * (state.bottom + state.part.height * 0.5)
            for state in self.parts
        ) / total

    @property
    def moments_of_inertia(
        self
    ) -> tuple[float, float]:
        """
        (pitch / yaw about the center of mass, roll), kg m^2:
        each part a solid cylinder of its own mass.
        """

        com = self.center_of_mass

        pitch = roll = 0.0

        for state in self.parts:

            mass = state.part.mass + state.propellant
            radius = state.part.radius
            length = state.part.height

            offset = state.bottom + length * 0.5 - com

            pitch += mass * (3.0 * radius * radius + length * length) / 12.0 + mass * offset * offset
            roll += 0.5 * mass * radius * radius

        return pitch, roll

    @property
    def propellant(
        self
    ) -> float:

        return sum(state.propellant for state in self.parts)

    @property
    def propellant_capacity(
        self
    ) -> float:

        return sum(state.part.propellant for state in self.parts)

    @property
    def crash_speed(
        self
    ) -> float:
        """What the bottom part survives hitting the ground at (m/s)."""

        return self.parts[-1].part.crash_speed

    @property
    def stages_left(
        self
    ) -> int:

        stages = [state.stage for state in self.parts if state.stage is not None]

        return sum(1 for stage in set(stages) if stage >= self.next_stage)

    # -----------------------------------------------------
    # Sections and engines
    # -----------------------------------------------------

    def _section_of(
        self,
        index: int
    ) -> int:
        """Separators passed from the top down to this part."""

        return sum(1 for state in self.parts[:index] if state.part.kind == "decoupler")

    def _section_tanks(
        self,
        index: int
    ) -> list[PartState]:

        section = self._section_of(index)

        return [
            state
            for i, state in enumerate(self.parts)
            if state.part.propellant > 0.0 and self._section_of(i) == section
        ]

    def burning_engines(
        self
    ) -> list[tuple[int, PartState]]:
        """Ignited engines with propellant to burn."""

        return [
            (index, state)
            for index, state in enumerate(self.parts)
            if state.ignited
            and state.part.engine is not None
            and any(tank.propellant > 0.0 for tank in self._section_tanks(index))
        ]

    def section_propellant(
        self
    ) -> tuple[float, float]:
        """(left, capacity) kg in the tanks the burning (or next) engines draw on."""

        engines = self.burning_engines()

        if engines:
            index = engines[0][0]
        else:
            index = max(
                (i for i, state in enumerate(self.parts) if state.part.engine is not None),
                default=len(self.parts) - 1
            )

        tanks = self._section_tanks(index)

        return sum(t.propellant for t in tanks), sum(t.part.propellant for t in tanks)

    @staticmethod
    def isp(
        state: PartState,
        pressure_bar: float
    ) -> float:
        """The engine's specific impulse (s) at an outside pressure."""

        engine = state.part.engine

        share = pressure_bar / ATMOSPHERE_BAR

        return max(engine.isp_vacuum + (engine.isp_sea_level - engine.isp_vacuum) * share, 1.0)

    def thrust(
        self,
        pressure_bar: float = 0.0
    ) -> tuple[float, float]:
        """
        (thrust N, propellant flow kg/s) at the throttle.
        The flow is fixed by the engine; the push it gives
        falls with the outside pressure (Isp).
        """

        thrust = flow = 0.0

        for _, state in self.burning_engines():

            engine = state.part.engine

            mass_flow = engine.thrust_vacuum / (engine.isp_vacuum * G0) * self.throttle

            flow += mass_flow
            thrust += mass_flow * self.isp(state, pressure_bar) * G0

        return thrust, flow

    def max_thrust(
        self,
        pressure_bar: float = 0.0
    ) -> float:

        throttle = self.throttle

        self.throttle = 1.0

        try:
            return self.thrust(pressure_bar)[0]
        finally:
            self.throttle = throttle

    def control_torque(
        self,
        pressure_bar: float = 0.0
    ) -> float:
        """
        N m the vessel can turn itself with: reaction wheels
        plus the burning engines swinging their nozzles
        (thrust x sin(gimbal) x distance to the center of
        mass).
        """

        torque = sum(state.part.reaction_wheel_torque for state in self.parts)

        com = self.center_of_mass

        for _, state in self.burning_engines():

            engine = state.part.engine

            mass_flow = engine.thrust_vacuum / (engine.isp_vacuum * G0) * self.throttle

            push = mass_flow * self.isp(state, pressure_bar) * G0

            torque += push * math.sin(math.radians(engine.gimbal)) * max(com - state.bottom, 0.0)

        return torque

    def burn(
        self,
        dt: float,
        pressure_bar: float = 0.0
    ):
        """Drain `dt` seconds of the engines' propellant from their sections' tanks, evenly."""

        for index, state in self.burning_engines():

            engine = state.part.engine

            need = engine.thrust_vacuum / (engine.isp_vacuum * G0) * self.throttle * dt

            tanks = [tank for tank in self._section_tanks(index) if tank.propellant > 0.0]

            while need > 1e-12 and tanks:

                share = need / len(tanks)

                need = 0.0

                for tank in tanks:

                    taken = min(share, tank.propellant)

                    tank.propellant -= taken

                    need += share - taken

                tanks = [tank for tank in tanks if tank.propellant > 0.0]

    # -----------------------------------------------------
    # Staging
    # -----------------------------------------------------

    def stage(
        self
    ) -> int | None:
        """
        Fire the next stage: ignite its engines, fire its
        separators (dropping everything below each). Returns
        the stage fired, None when none are left.
        """

        stages = sorted({state.stage for state in self.parts if state.stage is not None and state.stage >= self.next_stage})

        if not stages:
            return None

        stage = stages[0]

        self.next_stage = stage + 1

        # The highest separator of the stage drops all below.
        cut = next(
            (
                index
                for index, state in enumerate(self.parts)
                if state.stage == stage and state.part.kind == "decoupler"
            ),
            None
        )

        if cut is not None:

            self.dropped.extend(self.parts[cut:])

            self.parts = self.parts[:cut]

            self._restack()

        for state in self.parts:

            if state.stage == stage and state.part.engine is not None:
                state.ignited = True

            if state.stage == stage and state.part.parachute is not None:
                state.armed = True

        return stage

    # -----------------------------------------------------
    # Drag and parachutes
    # -----------------------------------------------------

    def open_parachutes(
        self,
        pressure_bar: float,
        airspeed: float,
        dt: float
    ):
        """Staged parachutes open over PARACHUTE_OPENING s once the air is thick enough and slow enough."""

        for state in self.parts:

            chute = state.part.parachute

            if chute is None or not state.armed or state.deployed >= 1.0:
                continue

            if state.deployed > 0.0 or (pressure_bar >= chute.min_pressure and airspeed <= chute.max_speed):
                state.deployed = min(1.0, state.deployed + dt / PARACHUTE_OPENING)

    def drag_area(
        self
    ) -> float:
        """Drag coefficient x area (m^2): the stack nose-first, plus open parachutes."""

        area = BODY_DRAG_COEFFICIENT * math.pi * self.radius ** 2

        for state in self.parts:

            chute = state.part.parachute

            if chute is not None and state.deployed > 0.0:
                area += chute.drag_coefficient * chute.area * state.deployed

        return area

    @property
    def parachutes(
        self
    ) -> list[PartState]:

        return [state for state in self.parts if state.part.parachute is not None]

    def _copy(
        self
    ) -> "Vessel":
        """A working copy (nothing done to it changes this one)."""

        return Vessel(
            self.name,
            [PartState(s.part, s.stage, s.propellant, s.ignited, s.bottom, s.armed, s.deployed) for s in self.parts],
            self.next_stage,
            1.0
        )

    def stage_stats(
        self,
        surface_pressure: float = 0.0,
        gravity: float = 9.80665
    ) -> list[StageStats]:
        """
        Every stage still to fire, in order: what it weighs,
        its delta-v in vacuum and at the surface pressure,
        its thrust to weight on the surface (gravity m/s^2),
        how long it burns. Stages that only separate or arm
        parachutes have no delta-v.
        """

        copy = self._copy()

        stats = []

        for _ in range(len(copy.parts) + 8):

            fired = copy.stage()

            if fired is None:
                break

            engines = copy.burning_engines()

            start = copy.mass

            if not engines:

                stats.append(StageStats(fired, start, start, 0.0, 0.0, 0.0, 0.0, 0))

                continue

            vacuum, flow = copy.thrust(0.0)
            surface, _ = copy.thrust(surface_pressure)

            left, _ = copy.section_propellant()

            end = start - left

            ratio = math.log(start / end) if end > 0.0 else 0.0

            stats.append(StageStats(
                stage=fired,
                start_mass=start,
                end_mass=end,
                delta_v_vacuum=vacuum / flow * ratio,
                delta_v_surface=surface / flow * ratio,
                twr_surface=surface / (start * gravity),
                burn_time=left / flow if flow > 0.0 else 0.0,
                engines=len(engines)
            ))

            for tank in copy._section_tanks(engines[0][0]):
                tank.propellant = 0.0

        return stats

    def delta_v(
        self,
        pressure_bar: float = 0.0
    ) -> float:
        """
        m/s the vessel has left, stage by stage (each burns
        its own section's propellant, then drops it).
        """

        # Work on a copy so nothing changes.
        copy = self._copy()

        total = 0.0

        for _ in range(len(copy.parts) + 1):

            engines = copy.burning_engines()

            if engines:

                thrust, flow = copy.thrust(pressure_bar)

                isp = thrust / (flow * G0)

                start = copy.mass

                left, _ = copy.section_propellant()

                total += isp * G0 * math.log(start / (start - left))

                for tank in copy._section_tanks(engines[0][0]):
                    tank.propellant = 0.0

            if copy.stage() is None:
                break

        return total
