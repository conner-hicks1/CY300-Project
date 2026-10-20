import time

import numpy as np

from core.logger import Logger

from ecs.components import VesselComponent

from game.chase_camera import ChaseCamera
from game.vessels_panel import VesselRow, VesselsInfo

from systems.comms_system import crew as crew_of
from systems.programme_system import move_propellant

from programme import fleet

from vessel import docking


# =========================================================
# The Application: Fleet
# =========================================================
#
# The vessels in flight: saving them, switching between
# them, the Vessels list, recovered vessels leaving.
#
# A part of Application (application.py), apart for its
# size: its methods run on the application (self is it).

class FleetController:

    # A recovered vessel stays this long (s, real) before it
    # is taken away.
    RECOVERY_PAUSE = 3.0

    def _take_recovered(self):
        """Recovered vessels go (their flight is over); the flown one's place: a fresh rocket on the pad."""

        programme = self.programme_system

        now = time.monotonic()

        waiting = []

        for entity, when in programme.recovered:

            component = self.scene.try_get_component(entity, VesselComponent)

            # Gone, or off again (it pays again to fly).
            if component is None or not component.state.landed or component.paid:
                continue

            if now - when < self.RECOVERY_PAUSE:

                waiting.append((entity, when))

                continue

            found = self.vessel_system.find(self.scene)

            if found is not None and found[0] == entity:

                # (Rebuild takes the unpaid one off the pad.)
                self.vessel_system.rebuild(self.scene, self.vessel_design, self.orbit_system)

                self.chase_camera = ChaseCamera()

                self.hud.notify(f"{self.vessel_design.name} on the pad")

            else:

                self.vessel_system.remove(self.scene, entity)

        programme.recovered = waiting

    def save_fleet(self):

        if not self._fleet_restored or self.fleet_path is None:
            return

        try:
            fleet.save(
                fleet.snapshot(self.scene, self.vessel_system, self.science_system, self.orbit_system.time),
                self.fleet_path
            )
        except OSError as error:
            Logger.error("[Application] Could not save the fleet: %s", error)

    def switch_vessel(
        self,
        entity
    ) -> bool:
        """Fly another vessel (the camera follows it)."""

        why = self.vessel_system.switch(self.scene, entity, self.orbit_system, self.planet_system)

        if why:

            self.hud.notify(why)

            return False

        found = self.vessel_system.find(self.scene)

        # (Not a target of itself.)
        if found is not None and self.map_view.target == ("vessel", found[0]):
            self.map_view.target = None

        self.chase_camera = ChaseCamera()
        self._sent = None
        self._site = None
        self._window = None
        self._arrival = None
        self._warp_until = None

        if found is not None:
            self.hud.notify(f"Flying {found[1].vessel.name}")

        return True

    def _cycle_vessel(
        self,
        step: int
    ):

        vessels = [entity for entity, component in self.vessel_system.vessels(self.scene) if not component.state.destroyed]

        if len(vessels) < 2:
            return

        found = self.vessel_system.find(self.scene)

        index = vessels.index(found[0]) if found is not None and found[0] in vessels else 0

        self.switch_vessel(vessels[(index + step) % len(vessels)])

    def _draw_vessels(self):

        if not self.vessels_panel.open:
            return

        vessels = self.vessel_system

        found = vessels.find(self.scene)

        info = VesselsInfo()

        here = vessels.position_of(found[1], self.orbit_system) if found is not None else None

        for entity, component in vessels.vessels(self.scene):

            if component.state.destroyed:
                continue

            where = vessels.position_of(component, self.orbit_system)

            active = found is not None and entity == found[0]

            info.rows.append(VesselRow(
                entity=entity,
                name=component.vessel.name,
                body=vessels.body_name(self.scene, component.body),
                situation=vessels.brief_situation(self.scene, component, self.orbit_system),
                distance=None if active or here is None or where is None else float(np.linalg.norm(where - here)),
                active=active,
                targeted=self.map_view.target == ("vessel", entity),
                crewed=crew_of(component.vessel) > 0,
                propellant=component.vessel.propellant,
                capacity=component.vessel.propellant_capacity,
                docked=docking.modules(component.vessel)
            ))

        if found is not None:

            info.propellant = found[1].vessel.propellant
            info.capacity = found[1].vessel.propellant_capacity

            info.nearby = vessels.alongside(self.scene, found[1], self.orbit_system)

        for action in self.vessels_panel.draw(info):

            if action[0] == "fly":

                self.switch_vessel(action[1])

            elif action[0] == "target":

                self.map_view.target = ("vessel", action[1]) if action[1] is not None else None

            elif action[0] == "end":

                if found is not None and action[1] == found[0]:

                    self.hud.notify("Fly another vessel first")

                    continue

                self.vessel_system.remove(self.scene, action[1])

                self.hud.notify("Mission ended")

            elif action[0] == "undock":

                self.undock()

            elif action[0] == "transfer" and found is not None:

                other = self.scene.try_get_component(action[1], VesselComponent)

                if other is None:
                    continue

                amount = action[2]

                giver, taker = (other, found[1]) if amount > 0 else (found[1], other)

                moved = move_propellant(giver.vessel, taker.vessel, abs(amount))

                self.hud.notify(f"Moved {moved / 1000.0:,.1f} t of propellant to {taker.vessel.name}")
