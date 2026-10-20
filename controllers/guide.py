import math
import time

import numpy as np

from game.guide import Moment


# =========================================================
# The Application: Flight School and Hints
# =========================================================
#
# A few times a second: the flight as a Moment (game/
# guide.py) for the lesson's steps and the hints.
#
# A part of Application (application.py), apart for its
# size: its methods run on the application (self is it).

class GuideController:

    # s (real) between looks at the flight.
    GUIDE_EVERY = 0.25

    def _moment(self) -> Moment | None:
        """The flown vessel's flight now, or None (none flown)."""

        found = self.vessel_system.find(self.scene)

        if found is None:
            return None

        readout = self.vessel_system.readout(self.scene, self.orbit_system, self.planet_system)

        if readout is None:
            return None

        component = found[1]

        vessel = component.vessel
        state = component.state

        radius = float(np.linalg.norm(state.position))

        tilt = math.degrees(math.acos(min(max(float(state.up @ state.position) / radius, -1.0), 1.0))) if radius > 0.0 else 0.0

        planned = self._node_readout(component)

        hot = readout.heat

        return Moment(
            situation=readout.situation,
            body=readout.body,
            height=readout.height,
            throttle=readout.throttle,
            staged=vessel.next_stage > 0,
            thrust=readout.thrust,
            sas=readout.sas,
            sas_mode=readout.sas_mode,
            tilt=tilt,
            vertical_speed=readout.vertical_speed,
            surface_speed=readout.surface_speed,
            apoapsis=readout.apoapsis,
            periapsis=readout.periapsis,
            air_top=readout.air_top,
            air=readout.situation == "Flying",
            stage_dry=readout.propellant_capacity > 0.0 and readout.propellant <= 1e-3 * readout.propellant_capacity and readout.stages_left > 0,
            heat=hot[0] / hot[1] if hot is not None and hot[1] > 0.0 else 0.0,
            parachute=readout.parachute if vessel.parachutes else "none",
            legs=readout.legs,
            map_open=self.map_view.open,
            warp=self.time_warp.rate,
            node=component.node is not None,
            node_in=planned[1] if planned is not None else None,
            career=not self.sandbox
        )

    def _update_guide(
        self,
        paused: bool
    ):

        # (Paused: the lesson waits, a hint stays.)
        if paused:
            return

        now = time.monotonic()

        if now - self._guided < self.GUIDE_EVERY:
            return

        elapsed = min(now - self._guided, 1.0)

        self._guided = now

        if not self.school.active and not self.hints.enabled and self.hints.showing is None:
            return

        moment = self._moment()

        if moment is None:
            return

        self.school.update(moment)

        # (No hints over the lesson: it says it all.)
        if not self.school.active:
            self.hints.update(moment, elapsed)
