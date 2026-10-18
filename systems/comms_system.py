import math

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np

from scene.scene import Scene

from science.survey import direction_of


# =========================================================
# Communications
# =========================================================
#
# The vessel talks to home through ground stations on the
# body it launched from: the Deep Space Network's three,
# a third of the world apart so one always faces the sky
# (Goldstone, Madrid, Canberra). A link needs
#
#   * the vessel above a station's horizon,
#   * no body in between (behind the Moon, behind Mars, or
#     landed on its far side: no link), and
#   * the vessel within its best antenna's range (built
#     into command parts: about the Moon's distance; a
#     medium-gain antenna reaches the inner planets near,
#     a high-gain dish the outer ones).
#
# On or close to home (in its air, or landed) mission
# control is always in touch. Signals go at the speed of
# light: 1.3 s to the Moon, 3 to 22 minutes to Mars. A
# crewed vessel flies itself; a probe's commands arrive
# that much later, and without a link not at all: it flies
# what it was sent (vessel/sequence.py). Without power
# (vessel/power.py) its radio is dead: no link, and no
# relaying for others.

SPEED_OF_LIGHT = 299_792_458.0

GROUND_STATIONS = (
    ("Goldstone", 35.43, -116.89),
    ("Madrid", 40.43, -4.25),
    ("Canberra", -35.40, 148.98),
)

# Close to home this high (m): mission control, always.
HOME_HEIGHT = 200_000.0


@dataclass(frozen=True, slots=True)
class Link:

    connected: bool
    station: str = ""               # which ground station (or "Mission control")
    distance: float = 0.0           # m
    delay: float = 0.0              # s, one way
    range: float = 0.0              # the vessel's best antenna's (m)
    reason: str = ""                # why not, when not
    crewed: bool = False


def crew(
    vessel
) -> int:

    return sum(state.part.crew * state.count for state in vessel.parts)


def antenna_range(
    vessel
) -> float:

    return max((state.part.antenna_range for state in vessel.parts), default=0.0)


def blocked_by(
    start: np.ndarray,
    end: np.ndarray,
    center: np.ndarray,
    radius: float
) -> bool:
    """Whether the sphere stands between two points (the segment passes through it)."""

    segment = end - start

    length2 = float(segment @ segment)

    if length2 <= 0.0:
        return False

    t = float(np.clip((center - start) @ segment / length2, 0.0, 1.0))

    # (Only between them: an end touching it does not count.)
    if t <= 1e-9 or t >= 1.0 - 1e-9:
        return False

    closest = start + segment * t

    return float(np.linalg.norm(center - closest)) < radius


class CommsSystem:

    def __init__(self):

        # (clock time, Link) for the frame.
        self._cached: tuple[float, Link] | None = None

    def link(
        self,
        scene: Scene,
        vessel_system,
        orbit_system,
        planet_system=None
    ) -> Link:
        """The vessel's link home now."""

        found = vessel_system.find(scene)

        if found is None:
            return Link(False, reason="No vessel")

        key = (orbit_system.time, found[0])

        if self._cached is not None and self._cached[0] == key:
            return self._cached[1]

        link = self._find(scene, vessel_system, orbit_system, planet_system, found[1])

        self._cached = (key, link)

        return link

    def _find(
        self,
        scene: Scene,
        vessel_system,
        orbit_system,
        planet_system,
        component
    ) -> Link:

        vessel = component.vessel
        state = component.state

        crewed = crew(vessel) > 0

        reach = antenna_range(vessel)

        home = vessel_system.launch_body

        if state.destroyed:
            return Link(False, reason="Destroyed", crewed=crewed)

        if not vessel.powered:
            return Link(False, reason="No power", range=reach, crewed=crewed)

        bodies = vessel_system.soi_bodies(scene)

        if home is None or home not in bodies or vessel_system.reference_state(orbit_system, home) is None:
            return Link(True, "Mission control", crewed=crewed, range=reach)

        reference = vessel_system.reference_state(orbit_system, component.body)

        if reference is None:
            return Link(False, reason="Lost", crewed=crewed)

        where = np.asarray(reference[0], dtype=np.float64) + state.position

        center, frame = vessel_system.reference_state(orbit_system, home)

        center = np.asarray(center, dtype=np.float64)

        radius = bodies[home].radius

        # Close to home: mission control hears it directly.
        if component.body == home:

            height = float(np.linalg.norm(state.position)) - radius

            air_top = 0.0

            if planet_system is not None:
                air_top = vessel_system.environment(scene, home, orbit_system, planet_system).air_top - radius

            if state.landed or height < max(HOME_HEIGHT, air_top):
                return Link(True, "Mission control", max(height, 0.0), max(height, 0.0) / SPEED_OF_LIGHT, reach, crewed=crewed)

        if reach <= 0.0:
            return Link(False, reason="No antenna", range=0.0, crewed=crewed)

        home_frame = (home, center, frame, radius)

        best, why, nearest = self._reach_home(vessel_system, orbit_system, bodies, home_frame, component, where, reach)

        if best is not None:
            return Link(True, best[0], best[1], best[1] / SPEED_OF_LIGHT, reach, crewed=crewed)

        # Through another vessel in flight that hears home: a
        # relay (one hop).
        relayed = None

        for other_entity, other in vessel_system.vessels(scene):

            if other is component or other.state.destroyed or not other.vessel.powered:
                continue

            other_reach = antenna_range(other.vessel)

            other_reference = vessel_system.reference_state(orbit_system, other.body)

            if other_reach <= 0.0 or other_reference is None:
                continue

            other_where = np.asarray(other_reference[0], dtype=np.float64) + other.state.position

            hop = float(np.linalg.norm(other_where - where))

            if hop > min(reach, other_reach):
                continue

            if self._blocked(vessel_system, orbit_system, bodies, where, other_where):
                continue

            onward, _, _ = self._reach_home(vessel_system, orbit_system, bodies, home_frame, other, other_where, other_reach)

            if onward is None:
                continue

            total = hop + onward[1]

            if relayed is None or total < relayed[1]:
                relayed = (f"{onward[0]} via {other.vessel.name}", total)

        if relayed is not None:
            return Link(True, relayed[0], relayed[1], relayed[1] / SPEED_OF_LIGHT, reach, crewed=crewed)

        if nearest > reach:
            why = "Out of range"

        return Link(False, reason=why, distance=nearest, delay=nearest / SPEED_OF_LIGHT, range=reach, crewed=crewed)

    @staticmethod
    def _blocked(
        vessel_system,
        orbit_system,
        bodies: dict,
        start: np.ndarray,
        end: np.ndarray
    ) -> bool:
        """Any body between two points in space (just inside its surface: landed ends do not count)."""

        for key, body in bodies.items():

            position = vessel_system.reference_state(orbit_system, key)

            if position is not None and blocked_by(start, end, np.asarray(position[0], dtype=np.float64), body.radius * 0.999):
                return True

        return False

    @staticmethod
    def _reach_home(
        vessel_system,
        orbit_system,
        bodies: dict,
        home_frame: tuple,
        component,
        where: np.ndarray,
        reach: float
    ) -> tuple[tuple | None, str, float]:
        """
        The best ground station a vessel at `where` (about the
        star) reaches directly: ((station, m) or None, why
        not, the nearest station's distance).
        """

        home, center, frame, radius = home_frame

        state = component.state

        own = bodies.get(component.body)

        # At (or just above) its own body's surface, its
        # horizon decides; higher, the body is a sphere in the
        # way like any other.
        on_surface = (
            own is not None
            and component.body is not None
            and component.body != home
            and float(np.linalg.norm(state.position)) < own.radius * 1.02
        )

        # The bodies that could be in the way: every one but
        # home (its horizon decides) and, at its surface, the
        # vessel's own.
        others = []

        for key, body in bodies.items():

            # (Round the Sun it can still be in the way.)
            if key == home or (key == component.body and key is not None and on_surface):
                continue

            position = vessel_system.reference_state(orbit_system, key)

            if position is not None:
                others.append((body.name, np.asarray(position[0], dtype=np.float64), body.radius))

        best = None
        why = "Below every station's horizon"
        nearest = math.inf

        for name, latitude, longitude in GROUND_STATIONS:

            up = frame @ direction_of(latitude, longitude)

            station = center + up * radius

            line = where - station

            distance = float(np.linalg.norm(line))

            nearest = min(nearest, distance)

            if distance > reach:
                continue

            if float(line @ up) < -0.0175 * distance:
                continue

            # From the vessel at the surface: its own body's horizon.
            if on_surface:

                vessel_up = state.position / float(np.linalg.norm(state.position))

                if float(-line @ vessel_up) < -0.0175 * distance:

                    why = f"Behind {own.name}"

                    continue

            in_the_way = next((body for body, position, size in others if blocked_by(station, where, position, size)), None)

            if in_the_way is not None:

                why = f"Behind {in_the_way}"

                continue

            if best is None or distance < best[1]:
                best = (name, distance)

        return best, why, nearest
