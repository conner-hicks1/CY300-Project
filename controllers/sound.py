import numpy as np

from ecs.components import TransformComponent

from audio.sound import Scene

from vessel import rover


# =========================================================
# The Application: Sound
# =========================================================
#
# Each frame: the flown vessel's sound from where the camera
# is (audio/sound.py); its events as one-shots.
#
# A part of Application (application.py), apart for its
# size: its methods run on the application (self is it).

class SoundController:

    # Faster than this, the sounds of flight go quiet (on
    # rails).
    QUIET_WARP = 5.0

    def _update_sound(
        self,
        paused: bool
    ):

        sound = self.sound

        found = self.vessel_system.find(self.scene)

        if found is None or found[1].state.destroyed:

            sound.update(Scene(0.0, 0.0, 0.0, 0.0, 1e9, 0.0, 0.0, False, paused=True))

            return

        entity, component = found

        vessel = component.vessel
        state = component.state

        density = pressure = 0.0
        dynamic = 0.0

        if self.vessel_system.reference_state(self.orbit_system, component.body) is not None:

            environment = self.vessel_system.environment(self.scene, component.body, self.orbit_system, self.planet_system)

            radius = float(np.linalg.norm(state.position))

            pressure, density = environment.air(radius, state.position / radius)

            airspeed = state.velocity - np.cross(environment.spin, state.position)

            dynamic = 0.5 * density * float(airspeed @ airspeed)

        where = np.asarray(self.scene.get_component(entity, TransformComponent).transform.position, dtype=np.float64)

        distance = float(np.linalg.norm(where - self.render_system.camera_position(self.scene)))

        top = rover.top_speed(vessel)

        sound.update(Scene(
            thrust=vessel.thrust(pressure)[0],
            density=density,
            dynamic_pressure=dynamic,
            heat_flux=vessel.heat_flux,
            distance=distance,
            rotor=vessel.throttle if rover.rotors(vessel) and vessel.powered else 0.0,
            wheels=abs(state.drive) / top if top > 0.0 and state.landed else 0.0,
            rcs=component.controls.translating and vessel.rcs_thrust > 0.0,
            paused=paused,
            warping=self.time_warp.rate > self.QUIET_WARP
        ))

    def _sound_events(
        self,
        events: list[str]
    ):
        """The vessel's events as one-shots."""

        sound = self.sound

        for event in events:

            kind, _, rest = event.partition(":")

            if kind == "destroyed":
                sound.play("crash", 1.0)
            elif kind == "toppled":
                sound.play("crash", 0.5)
            elif kind == "landed" and not rest:
                sound.play("thud", 0.9)
            elif kind == "bump":
                sound.play("bump", min(0.4 + 0.15 * float(rest.rpartition(":")[2] or 0.0), 1.0))
