from game import format as readable

from vessel import landing


# =========================================================
# The Application: Landing
# =========================================================
#
# The landing legs (vessel/landing.py): G lowers or raises
# them (a probe's a light time late).
#
# A part of Application (application.py), apart for its
# size: its methods run on the application (self is it).

class LandingController:

    def _toggle_legs(self):
        """G: lower the landing legs, or raise them."""

        found = self.vessel_system.find(self.scene)

        if found is None:
            return

        _, component = found

        found_legs = landing.legs(component.vessel)

        if not found_legs:

            self.hud.notify("No landing legs")

            return

        down = not any(state.extending for state in found_legs)

        if self.link.crewed:

            self.set_legs(component, down)

        elif self.link.connected:

            self._uplink(component, "legs", down)

            if self.link.delay > 1.0:
                self.hud.notify(f"Landing legs: arrives in {readable.countdown(self.link.delay)}")

        else:

            self.hud.notify(f"No link ({self.link.reason}): the probe cannot hear you")

    def set_legs(
        self,
        component,
        down: bool
    ):
        """Lower (or raise) the vessel's legs (the pilot's G arriving, the computer)."""

        if not landing.legs(component.vessel):
            return

        landing.set_legs(component.vessel, down)

        self.hud.notify("Landing legs down" if down else "Landing legs up")
