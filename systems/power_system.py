import math

import numpy as np

from math3d import quaternion

from scene.scene import Scene

from vessel import kepler, power, propellants


# =========================================================
# Power, Each Frame
# =========================================================
#
# Every vessel's power over the frame's simulated time
# (vessel/power.py): what its wings and generators make,
# what it draws, its batteries. The active vessel's wings
# unfold and fold, and tear off in thick fast air.
#
# Sunlight: the Sun at the origin of the solar system's
# frame, any body in between casting its shadow (a vessel
# on or near the ground: its own body's horizon instead),
# thinner through air. In time warp a step can be longer
# than an orbit, or a day landed: then the light is the
# average over the step, sampled round the orbit (the
# vessel holding its attitude, as on rails) or round the
# day (the Sun going round the sky).
#
# The same sunlight warms the tanks: cryogenic propellant
# boils off (vessel/propellants.py), once launched (on the
# pad it is topped off).

SUN = None


class PowerSystem:

    # A step longer than this (s) is sampled.
    SAMPLE_SPAN = 30.0
    SAMPLES = 32

    def __init__(self):

        # The Sun's direction in the active vessel's frame
        # (None: in shadow), for its wings to face.
        self.sun_local: np.ndarray | None = None

    def update(
        self,
        scene: Scene,
        vessel_system,
        orbit_system,
        dt: float
    ) -> list[str]:
        """
        Each vessel's power over `dt` simulated s (its
        component's `power`). Events: "power out:<name>",
        "power back:<name>", "torn:<what>" (the active
        vessel's wings).
        """

        events: list[str] = []

        bodies = vessel_system.soi_bodies(scene)

        # Where every body is now: (key, center, radius).
        shades = []

        for key, body in bodies.items():

            if key is SUN:
                continue

            reference = vessel_system.reference_state(orbit_system, key)

            if reference is not None:
                shades.append((key, np.asarray(reference[0], dtype=np.float64), body.radius))

        found = vessel_system.find(scene)

        active = found[0] if found is not None else None

        self.sun_local = None

        for entity, component in vessel_system.vessels(scene):

            state = component.state

            if state.destroyed:
                continue

            reference = vessel_system.reference_state(orbit_system, component.body)

            if reference is None:
                continue

            vessel = component.vessel

            physics = vessel_system._body_physics(scene, component.body, orbit_system)

            if dt > 0.0:
                power.move_panels(vessel, dt)

            if entity == active and dt > 0.0:

                for name in power.tear_panels(vessel, self._dynamic_pressure(state, physics)):
                    events.append(f"torn:{name}")

            center = np.asarray(reference[0], dtype=np.float64)

            others = [(where, radius) for key, where, radius in shades if key != component.body]

            own = (physics.radius if component.body is not SUN else None)

            sunlight, light, sun_local, flux = self._sunlight(vessel, state, physics, center, others, own, dt)

            generation = power.generator_power(vessel) + light

            was = vessel.powered

            status = power.settle(vessel, generation, dt)

            status.sunlight = sunlight
            status.flux = flux
            status.sun_local = sun_local

            if component.paid:

                status.boiloff = propellants.boil_rate(vessel, sunlight, flux)

                propellants.boil(vessel, sunlight, flux, dt)

            component.power = status

            if entity == active:
                self.sun_local = sun_local

            if was and not vessel.powered:
                events.append(f"power out:{vessel.name}")
            elif not was and vessel.powered:
                events.append(f"power back:{vessel.name}")

        return events

    # -----------------------------------------------------
    # Light
    # -----------------------------------------------------

    @staticmethod
    def _pressure(
        physics,
        radius: float
    ) -> float:

        column = physics.column

        if column is None:
            return 0.0

        return max(column.pressure((radius - physics.radius) / 1000.0), 0.0)

    def _dynamic_pressure(
        self,
        state,
        physics
    ) -> float:
        """Pa: half rho v^2 through the air."""

        column = physics.column

        if column is None or state.landed:
            return 0.0

        radius = float(np.linalg.norm(state.position))

        altitude = (radius - physics.radius) / 1000.0

        pressure = self._pressure(physics, radius)

        if pressure <= 0.0:
            return 0.0

        density = pressure * 1e5 / (column.gas_constant * column.temperature(altitude))

        airspeed = state.velocity - np.cross(physics.spin, state.position)

        return 0.5 * density * float(airspeed @ airspeed)

    def _sunlight(
        self,
        vessel,
        state,
        physics,
        center: np.ndarray,
        others: list,
        own_radius: float | None,
        dt: float
    ) -> tuple[float, float, np.ndarray | None, float]:
        """
        (share of the step in sunlight, W from the wings on
        average, the Sun's direction in the vessel's frame
        now or None, the Sun's strength here).
        """

        # Vessel -> inertial.
        frame = quaternion.to_matrix3(state.orientation)

        positions = [state.position]
        turns = [0.0]

        if dt > self.SAMPLE_SPAN:

            if state.landed:

                spin = float(np.linalg.norm(physics.spin))

                if spin > 0.0:

                    span = min(dt, 2.0 * math.pi / spin)

                    turns = list((np.arange(self.SAMPLES) + 0.5) * span / self.SAMPLES)

                    positions = [state.position] * len(turns)

            else:

                shape = kepler.orbit_shape(state.position, state.velocity, physics.mu)

                span = dt if shape.period is None else min(dt, shape.period)

                times = (np.arange(self.SAMPLES) + 0.5) * span / self.SAMPLES

                sampled, _ = kepler.propagate_many(state.position, state.velocity, physics.mu, times)

                good = np.isfinite(sampled).all(axis=1)

                if good.any():

                    positions = list(sampled[good])
                    turns = [0.0] * len(positions)

        return self._average(vessel, physics, center, others, own_radius, frame, positions, turns)

    def _average(
        self,
        vessel,
        physics,
        center,
        others,
        own_radius,
        frame,
        positions,
        turns
    ):

        lit = 0
        light = 0.0

        now_local = None

        flux = 0.0

        spin = physics.spin

        spin_rate = float(np.linalg.norm(spin))

        axis = spin / spin_rate if spin_rate > 0.0 else np.array([0.0, 0.0, 1.0])

        for index, (position, turn) in enumerate(zip(positions, turns)):

            where = center + np.asarray(position, dtype=np.float64)

            distance = float(np.linalg.norm(where))

            if distance <= 0.0:
                continue

            to_sun = -where / distance

            # Landed, the day going by: the Sun round the sky
            # (the vessel turning with the ground the other
            # way).
            if turn:
                to_sun = _rotate(to_sun, axis, -spin_rate * turn)

            radius = float(np.linalg.norm(position))

            up = np.asarray(position, dtype=np.float64) / max(radius, 1.0)

            sun_height = float(up @ to_sun)

            strength = power.strength(distance)

            flux = strength if index == 0 else flux

            dark = False

            # Its own body: near the ground, the horizon
            # (terrain can sit under the sphere's radius);
            # higher, its shadow.
            if own_radius is not None:

                if radius < own_radius * 1.02:
                    dark = sun_height < -0.01
                else:
                    dark = _behind(where, center, own_radius, to_sun)

            if not dark and others:
                dark = power.shadowed(where, np.zeros(3), others)

            if dark:
                continue

            share = power.through_air(self._pressure(physics, radius), sun_height) if own_radius is not None else 1.0

            local = frame.T @ to_sun

            if index == 0:
                now_local = local

            lit += 1

            light += power.solar_power(vessel, local, strength * share)

        count = max(len(positions), 1)

        return lit / count, light / count, now_local, flux


def _rotate(
    vector: np.ndarray,
    axis: np.ndarray,
    angle: float
) -> np.ndarray:
    """Rodrigues: `vector` turned `angle` rad about the unit `axis`."""

    cos, sin = math.cos(angle), math.sin(angle)

    return vector * cos + np.cross(axis, vector) * sin + axis * float(axis @ vector) * (1.0 - cos)


def _behind(
    where: np.ndarray,
    center: np.ndarray,
    radius: float,
    to_sun: np.ndarray
) -> bool:
    """In a sphere's shadow (a cylinder behind it, away from the Sun)."""

    offset = where - center

    along = float(offset @ to_sun)

    if along >= 0.0:
        return False

    across = offset - to_sun * along

    return float(np.linalg.norm(across)) < radius
