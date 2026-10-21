import time

import numpy as np

from ecs.components import TransformComponent

from audio.sound import Scene

from vessel import rover


# =========================================================
# The Application: Sound
# =========================================================
#
# Each frame: the flown vessel's sound from where the camera
# is (audio/sound.py); its events as one-shots; the alarms
# (the landing burn to start now, debris passing close);
# the music (the title screen, quiet space).
#
# A part of Application (application.py), apart for its
# size: its methods run on the application (self is it).

class SoundController:

    # Faster than this, the sounds of flight go quiet (on
    # rails).
    QUIET_WARP = 5.0

    # The music's level: on the title screen; out in space,
    # coasting.
    TITLE_MUSIC = 0.8
    SPACE_MUSIC = 0.55

    # s between the alarms while they hold (BRAKE NOW, a
    # debris pass).
    BRAKE_EVERY = 1.2
    PROXIMITY_EVERY = 8.0

    # Icy worlds: the wheels crunch ice, not gravel.
    ICY = frozenset({"europa", "enceladus", "ganymede", "callisto", "titan", "triton", "pluto", "haumea", "churyumov_gerasimenko", "halley"})

    def _update_sound(
        self,
        paused: bool
    ):

        sound = self.sound

        # (No sound made: headless, or not ready yet.)
        if sound.mixer is None:
            return

        found = self.vessel_system.find(self.scene)

        title = self.title.open

        if found is None or found[1].state.destroyed or title:

            sound.update(Scene(0.0, 0.0, 0.0, 0.0, 1e9, 0.0, 0.0, False, paused=True, music=self.TITLE_MUSIC if title else 0.0))

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

        thrust = vessel.thrust(pressure)[0]

        # Music out in space, coasting (not over a launch's
        # roar or a reentry).
        quiet = density < 1e-4 and thrust <= 0.0 and vessel.heat_flux < 1e5 and not state.landed

        self._music = getattr(self, "_music", 0.0)

        if not paused:
            self._music = self.SPACE_MUSIC if quiet else 0.0

        body = self.science_system.body_id(self.scene, component.body)

        sound.update(Scene(
            thrust=thrust,
            density=density,
            dynamic_pressure=dynamic,
            heat_flux=vessel.heat_flux,
            distance=distance,
            rotor=vessel.throttle if rover.rotors(vessel) and vessel.powered else 0.0,
            wheels=abs(state.drive) / top if top > 0.0 and state.landed else 0.0,
            rcs=component.controls.translating and vessel.rcs_thrust > 0.0,
            paused=paused,
            warping=self.time_warp.rate > self.QUIET_WARP,
            surface="ice" if body in self.ICY else "gravel",
            music=self._music
        ))

        if not paused:
            self._sound_alarms(component)

    def _sound_alarms(
        self,
        component
    ):
        """BRAKE NOW (the landing burn's moment), and debris coming close: beeps while they hold."""

        now = time.monotonic()

        braking = self.vessel_system.braking(self.scene, self.orbit_system, self.planet_system)

        if braking is not None and braking.start_in <= 0.5:

            if now - getattr(self, "_brake_beeped", 0.0) > self.BRAKE_EVERY:

                self._brake_beeped = now

                self.sound.play("brake", 0.9, alert=True)

        passing = None if component.state.landed else self.vessel_system.conjunction(self.scene, component)

        if passing is None:

            self._pinged = 0.0

        elif now - getattr(self, "_pinged", 0.0) > self.PROXIMITY_EVERY:

            self._pinged = now

            self.sound.play("proximity", 0.8, alert=True)

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
