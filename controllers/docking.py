import numpy as np

from ecs.components import VesselComponent

from vessel import docking

from game import format as readable


# =========================================================
# The Application: Docking
# =========================================================
#
# Docking: the approach to the targeted vessel,
# auto-docking (B), joining and undocking.
#
# A part of Application (application.py), apart for its
# size: its methods run on the application (self is it).

class DockingController:

    # Auto-docking starts within this of the target (m).
    DOCKING_RANGE = 2_000.0

    def _toggle_docking(self):
        """B: dock with the targeted vessel automatically (again: stop)."""

        if self._docking is not None:

            self._docking = None

            found = self.vessel_system.find(self.scene)

            if found is not None:
                found[1].controls.translate = np.zeros(3)

            self.hud.notify("Auto-docking off")

            return

        target = self.map_view.target

        found = self.vessel_system.find(self.scene)

        if not (isinstance(target, tuple) and target and target[0] == "vessel") or found is None:

            self.hud.notify("Target a vessel to dock with first (Vessels: L)")

            return

        pair = self.vessel_system.docking_pair(self.scene, found[1], target[1])

        if pair is None:

            self.hud.notify("No free docking ports to join (one at an end of each stack)")

            return

        if found[1].vessel.rcs_thrust <= 0.0:

            self.hud.notify("No translation thrusters (RCS) aboard")

            return

        distance = float(np.linalg.norm(pair[1][0] - pair[3][0]))

        if distance > self.DOCKING_RANGE:

            self.hud.notify(f"Too far to dock: {readable.distance(distance)} (within {readable.distance(self.DOCKING_RANGE)})")

            return

        self._docking = target[1]

        self.hud.notify("Auto-docking: the nose turns, the thrusters bring it in")

    def _update_docking(
        self,
        paused: bool
    ):
        """Each frame: the approach to the targeted vessel's port (the HUD's line; auto-docking flies it); docked when they touch."""

        self._docking_command = None

        found = self.vessel_system.find(self.scene)

        target = self.map_view.target

        if found is None or not (isinstance(target, tuple) and target and target[0] == "vessel"):

            self._docking = None

            return

        entity, component = found

        pair = self.vessel_system.docking_pair(self.scene, component, target[1])

        if pair is None:

            self._docking = None

            return

        own, own_point, theirs, their_point, other = pair

        if float(np.linalg.norm(own_point[0] - their_point[0])) > 5.0 * self.DOCKING_RANGE:
            return

        command = docking.docking_command(
            own_point,
            component.state.velocity,
            component.state.orientation,
            own.end,
            their_point,
            other.state.velocity,
            component.vessel.rcs_thrust / component.vessel.mass
        )

        self._docking_command = command

        if docking.can_capture(own_point, component.state.velocity, their_point, other.state.velocity):

            name = other.vessel.name

            if not self.vessel_system.dock(self.scene, entity, own.end, target[1], theirs.end, self.orbit_system, own.copy, theirs.copy):

                self._docking = None

                self.hud.notify(docking.can_hang(component.vessel if theirs.end == "side" else other.vessel) or "Could not dock")

                return

            component.controls.translate = np.zeros(3)

            self._docking = None
            self.map_view.target = None

            self.hud.notify(f"Docked with {name}")

            self.sound.play("dock", 1.0)

            return

        if self._docking != target[1] or paused:
            return

        controls = component.controls

        controls.sas = True
        controls.sas_mode = "maneuver"
        controls.maneuver = command.up

        component.vessel.throttle = 0.0

        # (Translating only once pointing near enough.)
        controls.translate = command.translate if command.misalignment < 10.0 else np.zeros(3)

    def undock(self):
        """Split the flown vessel at its last docking joint."""

        found = self.vessel_system.find(self.scene)

        if found is None:
            return

        other = self.vessel_system.undock(self.scene, found[0], self.orbit_system)

        if other is None:

            self.hud.notify("Nothing docked to undock")

            return

        name = self.scene.get_component(other, VesselComponent).vessel.name

        self.hud.notify(f"Undocked: {name} drifts away")

    def _docking_line(self) -> tuple[str, str] | None:
        """For the HUD: the approach to the target's port, or None."""

        command = self._docking_command

        if command is None:
            return None

        text = (
            f"Dock: {readable.distance(command.distance)}, closing {command.closing:+.2f} m/s, "
            f"{command.misalignment:.0f} deg off, {command.lateral:.1f} m off axis"
        )

        if self._docking is not None:
            return text + "  (auto, B: stop)", "good"

        return text + "  (B: auto-dock)", "warn"
