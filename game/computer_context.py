import math

import numpy as np

from vessel import maneuver
from vessel.sequence import FlightContext, hover_throttle


# =========================================================
# The Flight Computer's View of the Game
# =========================================================
#
# vessel/sequence.py's FlightContext for the application:
# what the computer reads (the vessel's readout, its orbit,
# the landing burn, the node) and does (throttle, staging,
# SAS, science), on the vessel itself: no light delay, it
# is aboard. Transmitting needs a link home.

class GameFlightContext(FlightContext):

    def __init__(
        self,
        app,
        component
    ):

        self.app = app
        self.component = component

        self._readout = None
        self._readout_read = False

    # -----------------------------------------------------
    # Reading
    # -----------------------------------------------------

    def _vessel(self):

        if not self._readout_read:

            app = self.app

            self._readout = app.vessel_system.readout(app.scene, app.orbit_system, app.planet_system)
            self._readout_read = True

        return self._readout

    def time(self) -> float:

        return self.app.orbit_system.time

    def situation(self) -> str:

        readout = self._vessel()

        return readout.situation if readout is not None else ""

    def _environment(self):

        app = self.app

        return app.vessel_system.environment(app.scene, self.component.body, app.orbit_system, app.planet_system)

    def quantity(self, name: str) -> float | None:

        app = self.app

        readout = self._vessel()

        if readout is None:
            return None

        state = self.component.state

        if name == "altitude":

            environment = self._environment()

            radius = float(np.linalg.norm(state.position))

            return radius - environment.ground(state.position / radius)

        if name == "apoapsis":
            return readout.apoapsis

        if name == "periapsis":
            return readout.periapsis

        if name == "speed":
            return readout.surface_speed

        if name == "vertical":
            return readout.vertical_speed

        if name == "orbital":
            return readout.orbital_speed

        if name == "pressure":

            environment = self._environment()

            radius = float(np.linalg.norm(state.position))

            return environment.air(radius, state.position / radius)[0]

        if name == "fuel":
            return readout.propellant / readout.propellant_capacity if readout.propellant_capacity > 0.0 else 0.0

        if name == "heat":
            return readout.heat[0] / readout.heat[1] if readout.heat is not None else 0.0

        if name == "slope":
            return readout.slope

        if name in ("dust", "wind"):
            return None if readout.weather is None else (readout.weather.tau if name == "dust" else readout.weather.wind)

        if name == "charge":

            vessel = self.component.vessel

            capacity = vessel.charge_capacity

            return vessel.charge / capacity if capacity > 0.0 else (1.0 if vessel.powered else 0.0)

        if name == "brake":

            braking = app.vessel_system.braking(app.scene, app.orbit_system, app.planet_system)

            return braking.start_in if braking is not None else None

        if name == "node":

            node = self.component.node

            return node.time - self.time() if node is not None else None

        return None

    def event_time(self, name: str) -> float | None:

        if name == "node":

            node = self.component.node

            return node.time if node is not None else None

        app = self.app

        bodies = app.vessel_system.soi_bodies(app.scene)

        body = bodies.get(self.component.body)

        state = self.component.state

        if body is None or state.landed:
            return None

        wait = maneuver.time_to_apsis(state.position, state.velocity, body.mu, name)

        return None if wait is None else self.time() + wait

    def has_node(self) -> bool:

        return self.component.node is not None

    def node_start(self) -> float | None:

        node = self.component.node

        if node is None:
            return None

        burn = self.app.vessel_system.node_burn_time(self.component)

        return node.time - 0.5 * (burn if math.isfinite(burn) else 0.0)

    # -----------------------------------------------------
    # Doing
    # -----------------------------------------------------

    def set_throttle(self, value: float):

        self.component.vessel.throttle = min(max(float(value), 0.0), 1.0)

    def hover(self, sink_speed: float | None):

        self.component.hover = sink_speed

    def stage(self):

        self.app.stage_vessel(self.component)

    def sas(self, word: str):

        controls = self.component.controls

        if word == "off":
            controls.sas = False
        elif word == "on":
            controls.sas = True
        else:
            controls.sas = True
            controls.sas_mode = word

    def science(self):

        self.app.run_science_aboard(self.component)

    def transmit(self):

        self.app.transmit_all(self.component)

    def panels(self, out: bool):

        self.app.set_panels(self.component, out)

    def legs(self, down: bool):

        self.app.set_legs(self.component, down)


def hold_sink_rate(
    app,
    component
):
    """Each frame while the computer hovers or descends: the throttle that holds the sinking speed."""

    if component.hover is None or component.state.destroyed:
        return

    vessels = app.vessel_system

    state = component.state

    if state.landed:
        return

    environment = vessels.environment(app.scene, component.body, app.orbit_system, app.planet_system)

    radius = float(np.linalg.norm(state.position))

    up = state.position / radius

    pressure, _ = environment.air(radius, up)

    relative = state.velocity - np.cross(environment.spin, state.position)

    vessel = component.vessel

    component.vessel.throttle = hover_throttle(
        component.hover,
        float(relative @ up),
        environment.mu / radius ** 2,
        vessel.mass,
        vessel.max_thrust(pressure),
        float(state.up @ up)
    )
