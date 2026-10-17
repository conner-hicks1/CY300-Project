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
# The vessel as it flies: which parts are still attached,
# how much propellant each tank holds, which engines burn,
# and what that adds up to (mass, center of mass, thrust,
# moments of inertia). Physics is in vessel/physics.py.
#
# The core stack runs along the vessel's axis. Radial
# groups hang on the side of a core part, `count` copies
# evenly round it (side boosters): one PartState stands for
# all the copies (they are identical and fly identically,
# so thrust stays on the axis). Parts are listed top down,
# each core part followed by its radial groups' parts.
#
# Stages fire in order with Space: an engine ignites at its
# stage; a core separator drops everything below it (radial
# groups on what it drops go too); a radial separator drops
# its group. Separators split the propellant: an engine
# burns only the tanks of its own section, all evenly: the
# core between separators, or each radial copy its own
# group's tanks.
#
# Positions are heights above the bottom of the lowest core
# part (the vessel's origin); radial copies stand `distance`
# m off the axis.


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

    # Radial parts: how many copies, which group (None: the
    # core stack), the group's height above the bottom of
    # the part it hangs on, and how far off the axis it
    # stands (set by restacking).
    count: int = 1
    group: int | None = None
    offset: float = 0.0
    distance: float = 0.0

    # Where it is in the design: (core index,) or (core
    # index, group, part) or (core index, group, "separator").
    source: tuple = ()

    # Its skin's temperature (K; vessel/heating.py).
    temperature: float = 290.0

    @property
    def radial(
        self
    ) -> bool:

        return self.group is not None


@dataclass(frozen=True, slots=True)
class StageStats:
    """One stage as the editor shows it (vessel.stage_stats)."""

    stage: int
    start_mass: float               # kg when it fires
    end_mass: float                 # kg when it stops burning (next burnout)
    delta_v_vacuum: float           # m/s
    delta_v_surface: float          # m/s at the launch body's surface pressure
    twr_surface: float              # thrust / weight at its start, on the launch body's surface
    burn_time: float                # s at full throttle
    engines: int                    # engine copies burning


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

        parts = []

        group = 0

        for index, entry in enumerate(design.stack):

            parts.append(PartState(entry.part, entry.stage, propellant=entry.part.propellant, source=(index,)))

            for number, radial in enumerate(entry.radial):

                if radial.decoupler is not None:
                    parts.append(PartState(
                        radial.decoupler.part,
                        radial.decoupler.stage,
                        count=radial.symmetry,
                        group=group,
                        offset=radial.offset,
                        source=(index, number, "separator")
                    ))

                for position, member in enumerate(radial.stack):
                    parts.append(PartState(
                        member.part,
                        member.stage,
                        propellant=member.part.propellant,
                        count=radial.symmetry,
                        group=group,
                        offset=radial.offset,
                        source=(index, number, position)
                    ))

                group += 1

        vessel = cls(design.name, parts)

        vessel._restack()

        return vessel

    def _restack(self):
        """Heights of every part (and radial copies' distance off the axis) from the stack as it is now."""

        height = 0.0

        for state in reversed(self.parts):

            if not state.radial:

                state.bottom = height

                height += state.part.height

        # Radial groups: on the core part listed before them.
        groups: dict[int, tuple[PartState, list[PartState]]] = {}

        parent = None

        for state in self.parts:

            if not state.radial:
                parent = state
            elif parent is not None:
                groups.setdefault(state.group, (parent, []))[1].append(state)

        for parent, members in groups.values():

            stacked = [member for member in members if member.part.kind != "decoupler"]
            separators = [member for member in members if member.part.kind == "decoupler"]

            reach = max((member.part.radius for member in stacked), default=0.0)

            distance = parent.part.radius + reach

            level = parent.bottom + (members[0].offset if members else 0.0)

            for member in reversed(stacked):

                member.bottom = level
                member.distance = distance

                level += member.part.height

            span = level - parent.bottom - (members[0].offset if members else 0.0)

            for separator in separators:

                # Between the core and the group, halfway up.
                separator.bottom = parent.bottom + separator.offset + 0.5 * (span - separator.part.height)
                separator.distance = parent.part.radius + separator.part.radius

    # -----------------------------------------------------
    # Sums
    # -----------------------------------------------------

    @property
    def core(
        self
    ) -> list[PartState]:

        return [state for state in self.parts if not state.radial]

    @property
    def mass(
        self
    ) -> float:
        """kg, with propellant."""

        return sum((state.part.mass + state.propellant) * state.count for state in self.parts)

    @property
    def height(
        self
    ) -> float:
        """The core stack's (m)."""

        return sum(state.part.height for state in self.core)

    @property
    def radius(
        self
    ) -> float:
        """Farthest any part reaches from the axis (m)."""

        return max(
            (state.distance + state.part.radius if state.radial else state.part.radius for state in self.parts),
            default=0.0
        )

    @property
    def center_of_mass(
        self
    ) -> float:
        """Height above the origin (m): symmetric copies keep it on the axis."""

        total = self.mass

        if total <= 0.0:
            return 0.0

        return sum(
            (state.part.mass + state.propellant) * state.count * (state.bottom + state.part.height * 0.5)
            for state in self.parts
        ) / total

    @property
    def moments_of_inertia(
        self
    ) -> tuple[float, float]:
        """
        (pitch / yaw about the center of mass, roll), kg m^2:
        each part a solid cylinder of its own mass; radial
        copies also by their distance off the axis.
        """

        com = self.center_of_mass

        pitch = roll = 0.0

        for state in self.parts:

            mass = state.part.mass + state.propellant
            radius = state.part.radius
            length = state.part.height

            offset = state.bottom + length * 0.5 - com

            own_pitch = mass * (3.0 * radius * radius + length * length) / 12.0 + mass * offset * offset
            own_roll = 0.5 * mass * radius * radius

            if state.radial:

                d2 = state.distance * state.distance

                # A ring of copies: half its d^2 about either
                # pitch axis, all of it about the roll axis.
                pitch += state.count * (own_pitch + 0.5 * mass * d2)
                roll += state.count * (own_roll + mass * d2)

            else:

                pitch += own_pitch
                roll += own_roll

        return pitch, roll

    @property
    def propellant(
        self
    ) -> float:

        return sum(state.propellant * state.count for state in self.parts)

    @property
    def propellant_capacity(
        self
    ) -> float:

        return sum(state.part.propellant * state.count for state in self.parts)

    @property
    def crash_speed(
        self
    ) -> float:
        """What the bottom core part survives hitting the ground at (m/s)."""

        core = self.core

        return core[-1].part.crash_speed if core else 8.0

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
    ) -> tuple:
        """The core's separators passed from the top down; or the part's radial group."""

        state = self.parts[index]

        if state.radial:
            return ("radial", state.group)

        return ("core", sum(1 for other in self.parts[:index] if other.part.kind == "decoupler" and not other.radial))

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
        """Ignited engines with propellant to burn (one entry for all a radial group's copies)."""

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
        """(left, capacity) kg in the tanks the burning (or next) engines draw on, every copy counted."""

        engines = self.burning_engines()

        if engines:
            indices = [index for index, _ in engines]
        else:
            indices = [
                max(
                    (i for i, state in enumerate(self.parts) if state.part.engine is not None and not state.radial),
                    default=len(self.parts) - 1
                )
            ] if self.parts else []

        tanks = {}

        for index in indices:
            for tank in self._section_tanks(index):
                tanks[id(tank)] = tank

        return (
            sum(tank.propellant * tank.count for tank in tanks.values()),
            sum(tank.part.propellant * tank.count for tank in tanks.values())
        )

    @staticmethod
    def isp(
        state: PartState,
        pressure_bar: float
    ) -> float:
        """The engine's specific impulse (s) at an outside pressure."""

        engine = state.part.engine

        share = pressure_bar / ATMOSPHERE_BAR

        return max(engine.isp_vacuum + (engine.isp_sea_level - engine.isp_vacuum) * share, 1.0)

    @staticmethod
    def _flow(
        state: PartState
    ) -> float:
        """One copy's propellant flow at full throttle (kg/s)."""

        engine = state.part.engine

        return engine.thrust_vacuum / (engine.isp_vacuum * G0)

    def thrust(
        self,
        pressure_bar: float = 0.0
    ) -> tuple[float, float]:
        """
        (thrust N, propellant flow kg/s) at the throttle,
        every copy counted. The flow is fixed by the engine;
        the push it gives falls with the outside pressure.
        """

        thrust = flow = 0.0

        for _, state in self.burning_engines():

            mass_flow = self._flow(state) * self.throttle * state.count

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

        torque = sum(state.part.reaction_wheel_torque * state.count for state in self.parts)

        com = self.center_of_mass

        for _, state in self.burning_engines():

            push = self._flow(state) * self.throttle * self.isp(state, pressure_bar) * G0 * state.count

            torque += push * math.sin(math.radians(state.part.engine.gimbal)) * max(com - state.bottom, 0.0)

        return torque

    def burn(
        self,
        dt: float,
        pressure_bar: float = 0.0
    ):
        """
        Drain `dt` seconds of the engines' propellant from
        their sections' tanks, evenly (per copy: each radial
        copy feeds its own engine).
        """

        for index, state in self.burning_engines():

            need = self._flow(state) * self.throttle * dt

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
        Fire the next stage: ignite its engines, arm its
        parachutes, fire its separators (a core one drops
        everything below it, a radial one its group).
        Returns the stage fired, None when none are left.
        """

        stages = sorted({state.stage for state in self.parts if state.stage is not None and state.stage >= self.next_stage})

        if not stages:
            return None

        stage = stages[0]

        self.next_stage = stage + 1

        dropped = []

        # The highest core separator of the stage drops all
        # below (radial groups on those parts with them).
        cut = next(
            (
                index
                for index, state in enumerate(self.parts)
                if state.stage == stage and state.part.kind == "decoupler" and not state.radial
            ),
            None
        )

        if cut is not None:

            dropped.extend(self.parts[cut:])

            self.parts = self.parts[:cut]

        groups = {
            state.group
            for state in self.parts
            if state.radial and state.stage == stage and state.part.kind == "decoupler"
        }

        if groups:

            dropped.extend(state for state in self.parts if state.group in groups)

            self.parts = [state for state in self.parts if state.group not in groups]

        if dropped:

            self.dropped.extend(dropped)

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
        """
        Drag coefficient x area (m^2): the core nose-first,
        each radial copy's own front, plus open parachutes.
        """

        core = max((state.part.radius for state in self.core), default=0.0)

        front = math.pi * core ** 2

        widest: dict[int, tuple[float, int]] = {}

        for state in self.parts:

            if state.radial:

                radius, count = widest.get(state.group, (0.0, state.count))

                widest[state.group] = (max(radius, state.part.radius), count)

        front += sum(math.pi * radius ** 2 * count for radius, count in widest.values())

        area = BODY_DRAG_COEFFICIENT * front

        for state in self.parts:

            chute = state.part.parachute

            if chute is not None and state.deployed > 0.0:
                area += chute.drag_coefficient * chute.area * state.deployed * state.count

        return area

    @property
    def parachutes(
        self
    ) -> list[PartState]:

        return [state for state in self.parts if state.part.parachute is not None]

    # -----------------------------------------------------
    # Delta-v
    # -----------------------------------------------------

    def _copy(
        self
    ) -> "Vessel":
        """A working copy (nothing done to it changes this one)."""

        return Vessel(
            self.name,
            [
                PartState(
                    s.part, s.stage, s.propellant, s.ignited, s.bottom, s.armed, s.deployed,
                    s.count, s.group, s.offset, s.distance, s.source, s.temperature
                )
                for s in self.parts
            ],
            self.next_stage,
            1.0
        )

    def _burn_to_burnout(
        self,
        surface_pressure: float
    ) -> tuple[float, float, float] | None:
        """
        Full throttle until the first burning section runs
        dry: (delta-v in vacuum, at the surface pressure,
        seconds). None when nothing burns.
        """

        engines = self.burning_engines()

        if not engines:
            return None

        # Each section's flow (per copy) and propellant left
        # (per copy): the first to run out ends the burn.
        sections: dict[tuple, list[float]] = {}

        for index, state in engines:

            key = self._section_of(index)

            if key not in sections:
                sections[key] = [0.0, sum(tank.propellant for tank in self._section_tanks(index))]

            sections[key][0] += self._flow(state)

        time = min(left / flow for flow, left in sections.values() if flow > 0.0)

        vacuum, flow = self.thrust(0.0)
        surface, _ = self.thrust(surface_pressure)

        start = self.mass
        end = start - flow * time

        ratio = math.log(start / end) if end > 0.0 else 0.0

        self.burn(time)

        # (Floating-point dregs: the section that ran out is
        # empty.)
        for index, _ in engines:
            for tank in self._section_tanks(index):
                if tank.propellant < 1e-6:
                    tank.propellant = 0.0

        return vacuum / flow * ratio, surface / flow * ratio, time

    def stage_stats(
        self,
        surface_pressure: float = 0.0,
        gravity: float = 9.80665
    ) -> list[StageStats]:
        """
        Every stage still to fire, in order: what it weighs,
        its delta-v in vacuum and at the surface pressure,
        its thrust to weight on the surface (gravity m/s^2),
        how long it burns: from its firing until the next
        engine runs dry (when the next stage would fire).
        Engines still burning after the last stage add to it.
        Stages that only separate or arm parachutes, with
        nothing burning, have no delta-v.
        """

        copy = self._copy()

        stats = []

        for _ in range(len(copy.parts) + 16):

            fired = copy.stage()

            if fired is None:
                break

            start = copy.mass

            surface_thrust = copy.thrust(surface_pressure)[0]

            engines = sum(state.count for _, state in copy.burning_engines())

            burned = copy._burn_to_burnout(surface_pressure)

            if burned is None:

                stats.append(StageStats(fired, start, start, 0.0, 0.0, 0.0, 0.0, 0))

                continue

            vacuum, surface, time = burned

            stats.append(StageStats(
                stage=fired,
                start_mass=start,
                end_mass=copy.mass,
                delta_v_vacuum=vacuum,
                delta_v_surface=surface,
                twr_surface=surface_thrust / (start * gravity),
                burn_time=time,
                engines=engines
            ))

        # Engines still burning after the last stage: theirs.
        while stats:

            burned = copy._burn_to_burnout(surface_pressure)

            if burned is None:
                break

            last = stats[-1]

            stats[-1] = StageStats(
                last.stage,
                last.start_mass,
                copy.mass,
                last.delta_v_vacuum + burned[0],
                last.delta_v_surface + burned[1],
                last.twr_surface,
                last.burn_time + burned[2],
                last.engines
            )

        return stats

    def delta_v(
        self,
        pressure_bar: float = 0.0
    ) -> float:
        """
        m/s the vessel has left: what burns now, then stage by
        stage (vacuum unless a pressure is given).
        """

        copy = self._copy()

        total = 0.0

        # What is burning already, to its burnouts first.
        while True:

            burned = copy._burn_to_burnout(pressure_bar)

            if burned is None:
                break

            total += burned[1] if pressure_bar else burned[0]

        for stage in copy.stage_stats(pressure_bar):
            total += stage.delta_v_surface if pressure_bar else stage.delta_v_vacuum

        return total
