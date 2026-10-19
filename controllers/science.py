import numpy as np

from ecs.components import PlanetComponent

from game.science_panel import ScienceInfo
from game import format as readable

from vessel import power


# =========================================================
# The Application: Science
# =========================================================
#
# Science: where the vessel is, running experiments,
# recovering and sending results, mapping, the Science
# panel.
#
# A part of Application (application.py), apart for its
# size: its methods run on the application (self is it).

class ScienceController:

    def run_science_aboard(
        self,
        component
    ):
        """Run the experiments (the pilot's R arriving, or the computer)."""

        place = self._science_place()

        if place is None:
            return

        # Every instrument's run, at once (vessel/power.py).
        energy = sum(
            state.part.experiment.energy * state.count
            for state in component.vessel.parts
            if state.part.experiment is not None
        )

        if not power.take(component.vessel, energy):

            self.hud.notify(f"Not enough power to run the experiments ({energy:,.0f} Wh)")

            return

        before = len(component.science)

        for notice in self.science_system.run(component, place, self._has_air()):
            self.hud.notify(notice)

        # A scientist aboard gets more out of each experiment.
        if self.programme_system.has_role(component.vessel, self.programme_system.career.corps, "scientist"):
            for result in component.science[before:]:
                result.cap *= 1.25

    def transmit_all(
        self,
        component
    ):
        """Send every result aboard home (half value), if there is a link."""

        if not component.science:
            return

        if not self.link.connected:

            self.hud.notify(f"No link to transmit ({self.link.reason})")

            return

        gained = 0.0

        while component.science:

            if not power.take(component.vessel, power.TRANSMIT_ENERGY):

                self.hud.notify(f"Not enough power to transmit ({power.TRANSMIT_ENERGY:,.0f} Wh a result)")

                break

            gained += self.science_system.transmit(component, 0)

        self.hud.notify(f"Transmitted: +{gained:,.1f} science")

    @property
    def current_site(self) -> str | None:
        """The science site the vessel is in (its id), or None."""

        return self._site

    def _science_place(self):

        return self.science_system.place(self.scene, self.vessel_system, self.orbit_system, self.planet_system)

    def _has_air(self) -> bool:
        """Air outside the vessel now."""

        found = self.vessel_system.find(self.scene)

        if found is None or self.vessel_system.reference_state(self.orbit_system, found[1].body) is None:
            return False

        environment = self.vessel_system.environment(self.scene, found[1].body, self.orbit_system, self.planet_system)

        state = found[1].state

        radius = float(np.linalg.norm(state.position))

        return environment.air(radius, state.position / radius)[0] > 0.0

    def _run_science(self):
        """R: run the experiments aboard (a probe: when the command gets there)."""

        found = self.vessel_system.find(self.scene)

        if found is None:
            return

        if self.link.crewed:

            self.run_science_aboard(found[1])

        elif self.link.connected:

            self._uplink(found[1], "science", None)

            if self.link.delay > 1.0:
                self.hud.notify(f"Run experiments: arrives in {readable.countdown(self.link.delay)}")

        else:

            self.hud.notify(f"No link ({self.link.reason}): the probe cannot hear you")

    def _update_science(
        self,
        events: list[str],
        simulated: float
    ):
        """Each frame: the camera maps; landing home recovers what is aboard; sites passed are named."""

        found = self.vessel_system.find(self.scene)

        if found is None:
            return

        _, component = found

        # (The camera needs power.)
        if component.vessel.powered:
            for notice in self.science_system.map_ground(self.scene, self.vessel_system, self.orbit_system, simulated):
                self.hud.notify(notice)

        if "landed" in events and component.body == self.vessel_system.launch_body and component.science:

            gained = self.science_system.recover(component)

            self.hud.notify(f"Science recovered: +{gained:,.1f}")

        # Arriving at a named site (landed or low over it).
        place = self._science_place() if events or self.timer.frame_count % 15 == 0 else None

        if place is not None:

            site = place.site.id if place.site is not None else None

            # (Not at high warp: an orbit passes a site every
            # few frames.)
            if site != self._site and site is not None and self.time_warp.rate <= 10.0:

                verb = "Landed at" if place.situation == "landed" else "Over"

                self.hud.notify(f"{verb} {place.site.name}")

            self._site = site

        self.science_system.upkeep(self.planet_system)

    def _draw_science(self):

        found = self.vessel_system.find(self.scene)

        if not self.science_panel.open or found is None:
            return

        _, component = found

        science = self.science_system

        place = self._science_place()

        info = ScienceInfo(points=science.record.points)

        launch = self.vessel_system.launch_body

        info.home = self.vessel_system.body_name(self.scene, launch) if launch is not None else "home"

        if place is not None:

            info.here = place.description

            if place.site is not None:
                info.about = place.site.about

            if place.nearest is not None and place.site is None:
                info.nearest = (place.nearest[0].name, place.nearest[1])

            info.experiments = science.status(component, place, self._has_air())

            mask = science.record.surveys.get(place.body_id)

            planet = self.scene.try_get_component(place.body, PlanetComponent) if place.body is not None else None

            if mask is not None and planet is not None and planet.palette != "bands":

                info.mapped = mask.coverage()
                info.body = place.body_name
                info.mapping = science.mapping_status(self.scene, self.vessel_system, self.orbit_system)

        info.results = [
            (result.title, science.record.worth(result), science.record.worth(result, transmitted=True))
            for result in component.science
        ]

        for action in self.science_panel.draw(info):

            if action[0] == "run":

                self._run_science()

            elif action[0] == "transmit":

                if not self.link.connected:

                    self.hud.notify(f"No link to transmit ({self.link.reason})")

                    continue

                if not power.take(component.vessel, power.TRANSMIT_ENERGY):

                    self.hud.notify(f"Not enough power to transmit ({power.TRANSMIT_ENERGY:,.0f} Wh)")

                    continue

                gained = science.transmit(component, action[1])

                self.hud.notify(f"Transmitted: +{gained:,.1f} science")
