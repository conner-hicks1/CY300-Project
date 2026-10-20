from core.logger import Logger

from game import format as readable

from vessel import power


# =========================================================
# The Application: Power
# =========================================================
#
# Each vessel's power over the frame (systems/power_system.py,
# vessel/power.py), the solar wings (U: a probe's, a light
# time late), notices when power runs out and comes back.
#
# A part of Application (application.py), apart for its
# size: its methods run on the application (self is it).

class PowerController:

    def _update_power(
        self,
        simulated: float
    ):

        for event in self.power_system.update(self.scene, self.vessel_system, self.orbit_system, simulated):

            kind, _, name = event.partition(":")

            if kind == "power out":

                Logger.info("[Power] %s ran out of power.", name)

                self.hud.notify(f"{name} is out of power: no reaction wheels, no link, no computer")

                self.sound.play("power down", 0.8, alert=True)

                if name == self._flown_name():
                    self.time_warp.stop()

            elif kind == "power back":

                Logger.info("[Power] %s has power again.", name)

                self.hud.notify(f"{name} has power again")

            elif kind == "torn":

                Logger.info("[Power] Solar wings torn off (%s).", name)

                self.hud.notify(f"Solar wings torn off in the air ({name})")

    def _update_crew(
        self,
        simulated: float
    ):
        """Every crew's supplies and radiation (systems/crew_system.py); losses and doses to the programme."""

        losses, notices = self.crew_system.update(self.scene, self.vessel_system, self.orbit_system, simulated)

        for notice in notices:
            self.hud.notify(notice)

        programme = self.programme_system

        if losses:

            for loss in losses:
                Logger.info("[Crew] The crew of %s %s.", loss.vessel, loss.why)

            programme.crew_losses(losses)

            self.time_warp.stop()

        programme.add_doses(self.crew_system.doses)

    def _flown_name(self) -> str:

        found = self.vessel_system.find(self.scene)

        return found[1].vessel.name if found is not None else ""

    def _toggle_panels(self):
        """U: unfold the solar wings, or fold them (a probe: when the command gets there)."""

        found = self.vessel_system.find(self.scene)

        if found is None:
            return

        _, component = found

        vessel = component.vessel

        wings = [state for state in power.solar_parts(vessel) if not state.broken]

        if not wings:

            self.hud.notify("No solar wings" if not power.solar_parts(vessel) else "The solar wings are gone")

            return

        out = not any(state.extending for state in wings)

        if self.link.crewed:

            self.set_panels(component, out)

        elif self.link.connected:

            self._uplink(component, "panels", out)

            if self.link.delay > 1.0:
                self.hud.notify(f"Solar wings: arrives in {readable.countdown(self.link.delay)}")

        else:

            self.hud.notify(f"No link ({self.link.reason}): the probe cannot hear you")

    def set_panels(
        self,
        component,
        out: bool
    ):
        """Unfold (or fold) the vessel's solar wings (the pilot's U arriving, the computer)."""

        if not power.solar_parts(component.vessel):
            return

        power.set_panels(component.vessel, out)

        self.hud.notify("Solar wings unfolding" if out else "Solar wings folding")
