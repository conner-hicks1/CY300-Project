import math

import numpy as np

from game.maneuver_panel import NodeInfo

from vessel import maneuver
from vessel.maneuver import ManeuverNode

from game import format as readable


# =========================================================
# The Application: Maneuver
# =========================================================
#
# Maneuver nodes: the Maneuver panel, suggested burns made
# nodes, the HUD's countdown.
#
# A part of Application (application.py), apart for its
# size: its methods run on the application (self is it).

class ManeuverController:

    def _node_readout(
        self,
        component
    ) -> tuple[float, float, float] | None:
        """For the HUD: (m/s left, s until the burn should start, burn s), or None."""

        if component.node is None or self.maneuver_dv is None:
            return None

        burn = self.vessel_system.node_burn_time(component)

        start = component.node.time - 0.5 * (burn if math.isfinite(burn) else 0.0)

        return self.maneuver_dv, start - self.orbit_system.time, burn

    def _plan_node(
        self,
        component,
        planned,
        suggestions: list
    ):
        """The maneuver panel: show the node (or how to add one), carry out what was clicked."""

        vessels = self.vessel_system

        state = component.state

        now = self.orbit_system.time

        info = NodeInfo(can_plan=not state.landed and not state.destroyed)

        bodies = vessels.soi_bodies(self.scene)

        mu = bodies[component.body].mu if component.body in bodies else None

        if mu is not None and info.can_plan:

            info.apsides = tuple(
                apsis
                for apsis in ("apoapsis", "periapsis")
                if maneuver.time_to_apsis(state.position, state.velocity, mu, apsis) is not None
            )

        node = component.node

        info.suggestions = [(item.label, item.why) for item in suggestions]

        if node is not None:

            remaining = vessels.node_remaining(component)

            info.exists = True
            info.time_to = node.time - now
            info.components = {"prograde": node.prograde, "normal": node.normal, "radial": node.radial}
            info.remaining = float(np.linalg.norm(remaining))
            info.burn_time = vessels.node_burn_time(component)
            info.delta_v = component.vessel.delta_v()
            info.after = self._after_node(component, planned, bodies)

            if planned is None:
                info.note = "Past the path's next sphere: move it sooner"

        for action in self.maneuver_panel.draw(info):
            self._node_action(component, action, suggestions, mu)

    def _node_action(
        self,
        component,
        action: tuple,
        suggestions: list,
        mu: float | None
    ):

        vessels = self.vessel_system

        state = component.state

        now = self.orbit_system.time

        kind = action[0]

        def apsis_time(which: str) -> float | None:

            if mu is None:
                return None

            wait = maneuver.time_to_apsis(state.position, state.velocity, mu, which)

            return None if wait is None else now + max(wait, 30.0)

        if kind == "new":

            time = now + 300.0 if action[1] == "soon" else apsis_time(action[1])

            if time is not None:
                vessels.set_node(self.scene, self.orbit_system, ManeuverNode(time))

        elif kind == "suggest":

            item = suggestions[action[1]]

            where = vessels.node_state(self.scene, self.orbit_system, item.time)

            if where is not None:

                prograde, normal, radial = maneuver.from_inertial(*where, item.vector)

                vessels.set_node(self.scene, self.orbit_system, ManeuverNode(item.time, prograde, normal, radial))

        elif component.node is not None:

            node = component.node

            if kind == "delete":

                vessels.set_node(self.scene, self.orbit_system, None)

                self._warp_until = None

            elif kind == "warp":

                self._warp_until = node.time - 0.5 * min(vessels.node_burn_time(component), 1.0e7) - vessels.NODE_LEAD

            else:

                moved = ManeuverNode(node.time, node.prograde, node.normal, node.radial)

                if kind == "time":
                    moved.time = max(node.time + action[1], now + 1.0)
                elif kind == "time_to":
                    moved.time = apsis_time(action[1]) or node.time
                elif kind == "set":
                    setattr(moved, action[1], float(action[2]))

                if not vessels.set_node(self.scene, self.orbit_system, moved):
                    self.hud.notify("No orbit there to burn on")

    def _after_node(
        self,
        component,
        planned,
        bodies: dict
    ) -> list:
        """What the orbit becomes after the node, for the panel."""

        if planned is None or not planned.patches:
            return []

        first = planned.patches[0]

        body = bodies.get(first.body)

        if body is None:
            return []

        lines = []

        sea = body.radius

        if first.end == "orbit":

            distances = np.linalg.norm(first.points, axis=1)

            lines.append((f"After: Ap {readable.distance(float(distances.max()) - sea)}  Pe {readable.distance(first.periapsis - sea)}", None))

        elif first.end == "impact":

            lines.append((f"After: comes down on {body.name}", (1.0, 0.6, 0.4, 1.0)))

        elif first.end == "escape":

            lines.append((f"After: escapes {body.name}", (1.0, 0.6, 1.0, 1.0)))

        if first.end == "encounter" and len(planned.patches) > 1:

            inner = planned.patches[1]

            goal = bodies.get(inner.body)

            if goal is not None:
                lines.append((f"After: meets {goal.name}, Pe {readable.distance(inner.periapsis - goal.radius)}", self.GOOD))

        return lines
