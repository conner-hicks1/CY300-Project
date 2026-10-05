import math

import numpy as np

from ecs.components import (
    BodyComponent,
    ClockComponent,
    DirectionalLightComponent,
    HierarchyComponent,
    NameComponent,
    OrbitComponent,
    PlanetComponent,
    StarComponent,
    TransformComponent
)
from ecs.entity import Entity

from math3d import quaternion

from planet.bodies import AU_M, star_color, sun_intensity
from planet.orbits import (
    DAY,
    OrbitElements,
    body_frame,
    ecliptic_to_engine,
    equator_node,
    moon_offset,
    orbit_offset,
    pole_direction,
    prime_meridian_angle
)

from scene.scene import Scene


# Reference frames of orbit elements: (node origin, 90 deg
# ahead, north) in inertial coordinates.
_ECLIPTIC_BASIS = np.stack(
    (ecliptic_to_engine((1.0, 0.0, 0.0)), ecliptic_to_engine((0.0, 1.0, 0.0)), ecliptic_to_engine((0.0, 0.0, 1.0))),
    axis=1
)


class OrbitSystem:

    # =====================================================
    # Orbits, Spin and the Clock
    # =====================================================
    #
    # Per frame: advance the clock (ClockComponent on the
    # sun), work out every body's position and orientation
    # at that time (planet/orbits.py), and place their
    # entities.
    #
    # One body, the anchor, stays put in the world: the one
    # the camera is nearest (its surface). Everything else
    # is placed relative to it, turning with it, so a camera
    # standing on the ground stays there while the sky
    # wheels overhead: the sun rises and sets, the Moon
    # crosses the sky. When the camera flies to another body
    # that one becomes the anchor, seamlessly (at that
    # moment it keeps exactly where it was).
    #
    # The sun entity's light points from the star, with the
    # brightness and color of the star at the anchor's
    # distance.

    # The camera must be this much nearer another body's
    # surface than the anchor's before the anchor changes
    # (no flip-flopping midway).
    SWITCH_RATIO = 0.7

    def __init__(self):

        self.anchor: Entity | None = None

        # The anchor's fixed place in the world: its center
        # and the world rotation of its frame.
        self._anchor_center = np.zeros(3)
        self._anchor_rotation = np.eye(3)

        # entity -> (inertial position m, body rotation)
        # at the current time.
        self.states: dict[Entity, tuple[np.ndarray, np.ndarray]] = {}

        # Inertial -> world rotation (None before any body).
        self.inertial_to_world: np.ndarray | None = None

        self._star_radius = 6.957e8

        self.time = 0.0

        # Set by anchor_to: keep that anchor for this update.
        self._pinned = False

    # =====================================================
    # Queries
    # =====================================================

    def star_distance(
        self,
        entity: Entity
    ) -> tuple[float, float] | None:
        """(distance to the star m, star radius m) of a body, live."""

        state = self.states.get(entity)

        if state is None:
            return None

        return float(np.linalg.norm(state[0])), self._star_radius

    def star_position(
        self
    ) -> np.ndarray | None:
        """
        Where the star is in the world (m), live: body
        positions are relative to it, and the anchor's world
        place fixes it. None without orbits.
        """

        if self.anchor is None or self.inertial_to_world is None or self.anchor not in self.states:
            return None

        anchor_position, _ = self.states[self.anchor]

        return self._anchor_center - self.inertial_to_world @ np.asarray(anchor_position, dtype=np.float64)

    def anchor_to(
        self,
        scene: Scene,
        entity: Entity
    ):
        """
        Make a body the anchor where it stands, and place the
        others around it now (a new planetary system, whose
        other bodies are not placed yet).
        """

        transform = scene.get_component(entity, TransformComponent).transform

        self.anchor = entity

        self._anchor_center = np.asarray(transform.position, dtype=np.float64).copy()
        self._anchor_rotation = quaternion.to_matrix3(transform.orientation)

        self._pinned = True

        self.update(scene, 0.0, self._anchor_center)

    # =====================================================
    # Update
    # =====================================================

    def update(
        self,
        scene: Scene,
        delta_time: float,
        camera_position
    ):

        registry = scene.registry

        clock = next((c for _, c in registry.view_with(ClockComponent)), None)

        if clock is not None and clock.rate != 0.0:
            clock.time += clock.rate * delta_time

        self.time = clock.time if clock is not None else 0.0

        bodies = [
            (entity, transform, orbit)
            for entity, transform, orbit in registry.view_with(TransformComponent, OrbitComponent)
            if not registry.has(entity, HierarchyComponent)
        ]

        if not bodies:

            self.states = {}
            self.inertial_to_world = None
            self.anchor = None

            return

        self.states = body_states(scene, bodies, self.time)

        if self._pinned and self.anchor in self.states:
            self._pinned = False
        else:
            self._choose_anchor(scene, bodies, camera_position)

        anchor_position, anchor_frame = self.states[self.anchor]

        # Inertial -> world: the anchor's frame stays where
        # it is in the world.
        to_world = self._anchor_rotation @ anchor_frame.T

        self.inertial_to_world = to_world

        for entity, transform, _ in bodies:

            position, frame = self.states[entity]

            transform.transform.position = self._anchor_center + to_world @ (position - anchor_position)
            transform.transform.orientation = quaternion.from_matrix3(to_world @ frame)

        self._update_sun(scene, to_world, anchor_position)

    def _choose_anchor(
        self,
        scene: Scene,
        bodies,
        camera_position
    ):

        registry = scene.registry

        camera = np.asarray(camera_position, dtype=np.float64)

        def surface_distance(entity, transform):

            planet = registry.try_get(entity, PlanetComponent)

            radius = planet.radius if planet is not None else 0.0

            return float(np.linalg.norm(np.asarray(transform.transform.position) - camera)) - radius

        distances = {entity: surface_distance(entity, transform) for entity, transform, _ in bodies}

        nearest = min(distances, key=distances.get)

        current = self.anchor

        moved = False

        if current is not None and current in distances:

            # Not where it was left (a scene loaded, undo, an
            # edit): it stays where it now is instead.
            transform = scene.get_component(current, TransformComponent).transform

            moved = (
                float(np.linalg.norm(np.asarray(transform.position) - self._anchor_center)) > 1.0
                or float(np.abs(quaternion.to_matrix3(transform.orientation) - self._anchor_rotation).max()) > 1e-6
            )

        if moved:
            nearest = current

        if (
            moved
            or current is None
            or current not in distances
            or (nearest != current and distances[nearest] < self.SWITCH_RATIO * distances[current])
        ):

            transform = scene.get_component(nearest, TransformComponent).transform

            self.anchor = nearest

            self._anchor_center = np.asarray(transform.position, dtype=np.float64).copy()
            self._anchor_rotation = quaternion.to_matrix3(transform.orientation)

    def _update_sun(
        self,
        scene: Scene,
        to_world: np.ndarray,
        anchor_position: np.ndarray
    ):

        registry = scene.registry

        for entity, transform, light in registry.view_with(TransformComponent, DirectionalLightComponent):

            star = registry.try_get(entity, StarComponent)

            if star is None:
                continue

            self._star_radius = float(star.radius)

            distance = float(np.linalg.norm(anchor_position))

            if distance <= 0.0:
                continue

            toward_star = to_world @ (-anchor_position / distance)

            transform.transform.orientation = sun_orientation(toward_star)

            light.color = star_color(star.temperature)
            light.intensity = sun_intensity(star.luminosity / (distance / AU_M) ** 2)

            break


def sun_orientation(
    toward_sun: np.ndarray
) -> np.ndarray:
    """Orientation of a directional light shining from `toward_sun`."""

    travel = -np.asarray(toward_sun, dtype=np.float64)

    hint = np.array([0.0, 1.0, 0.0]) if abs(travel[1]) < 0.99 else np.array([0.0, 0.0, -1.0])

    return quaternion.look_rotation(travel, hint)


# =========================================================
# State at a Time
# =========================================================

def body_states(
    scene: Scene,
    bodies,
    seconds: float
) -> dict[Entity, tuple[np.ndarray, np.ndarray]]:
    """
    entity -> (inertial position m, body-to-inertial
    rotation) for (entity, transform, orbit) bodies.
    """

    registry = scene.registry

    by_name = {}

    for entity, _, orbit in bodies:

        name = registry.try_get(entity, NameComponent)

        if name is not None:
            by_name.setdefault(name.name, entity)

    orbits = {entity: orbit for entity, _, orbit in bodies}

    states: dict[Entity, tuple[np.ndarray, np.ndarray]] = {}

    def resolve(entity, depth=0):

        if entity in states:
            return states[entity]

        orbit = orbits[entity]

        parent = by_name.get(orbit.parent) if orbit.parent else None

        if parent == entity or depth > 8:
            parent = None

        elements = elements_of(orbit, seconds)

        if parent is not None:

            parent_position, _ = resolve(parent, depth + 1)

            basis = equator_basis(orbits[parent]) if orbit.plane == "equator" else _ECLIPTIC_BASIS

        else:

            parent_position = np.zeros(3)

            basis = _ECLIPTIC_BASIS

            # A moon without its planet in the scene: placed
            # where its planet would be around the star.
            body = registry.try_get(entity, BodyComponent)

            if orbit.parent and body is not None:

                elements = OrbitElements(
                    semi_major_axis=body.orbit_distance_au * AU_M,
                    eccentricity=0.0,
                    inclination=0.0,
                    ascending_node=0.0,
                    periapsis=0.0,
                    mean_anomaly=0.0,
                    period=max(body.year_days, 1.0) * DAY
                )

        if orbit.theory == "moon" and parent is not None:
            offset = moon_offset(seconds)
        else:
            offset = basis @ orbit_offset(elements, seconds)

        position = parent_position + offset

        if orbit.tidally_locked and parent is not None:

            # The orbit's pole, prime meridian facing the planet.
            i, node = elements.inclination, elements.ascending_node

            normal = basis @ np.array([math.sin(i) * math.sin(node), -math.sin(i) * math.cos(node), math.cos(i)])

            frame = body_frame(normal, -offset)

        else:

            frame = spin_frame(orbit, seconds)

        states[entity] = (position, frame)

        return states[entity]

    for entity, _, _ in bodies:
        resolve(entity)

    # A planet's elements are its barycenter's: the planet
    # itself swings around it opposite its moons (Earth by
    # ~4,700 km, which moves where eclipses fall).
    for entity, _, orbit in bodies:

        if orbit.parent and by_name.get(orbit.parent) in states:
            continue

        mass = registry.try_get(entity, BodyComponent)

        if mass is None:
            continue

        family = [
            other for other, _, other_orbit in bodies
            if other_orbit.parent and by_name.get(other_orbit.parent) == entity
        ]

        moons = [(m, registry.try_get(m, BodyComponent)) for m in family]
        moons = [(m, b) for m, b in moons if b is not None]

        if not moons:
            continue

        total = mass.mass + sum(b.mass for _, b in moons)

        center = states[entity][0]

        shift = -sum(b.mass * (states[m][0] - center) for m, b in moons) / total

        for member in [entity] + [m for m, _ in moons]:

            position, frame = states[member]

            states[member] = (position + shift, frame)

    return states


def elements_of(
    orbit: OrbitComponent,
    seconds: float = 0.0
) -> OrbitElements:
    """The orbit's elements at a time (precession applied)."""

    years = seconds / (365.25 * DAY)

    return OrbitElements(
        semi_major_axis=max(float(orbit.semi_major_axis), 1.0),
        eccentricity=min(max(float(orbit.eccentricity), 0.0), 0.99),
        inclination=math.radians(orbit.inclination),
        ascending_node=math.radians(orbit.ascending_node + orbit.node_rate * years),
        periapsis=math.radians(orbit.periapsis + orbit.periapsis_rate * years),
        mean_anomaly=math.radians(orbit.mean_anomaly),
        period=max(float(orbit.period), 1.0)
    )


def equator_basis(
    orbit: OrbitComponent
) -> np.ndarray:
    """A body's equatorial frame (node, 90 deg ahead, pole)."""

    pole = pole_direction(orbit.pole_ra, orbit.pole_dec)

    node = equator_node(orbit.pole_ra)

    node = node - np.dot(node, pole) * pole
    node /= np.linalg.norm(node)

    return np.stack((node, np.cross(pole, node), pole), axis=1)


def spin_frame(
    orbit: OrbitComponent,
    seconds: float
) -> np.ndarray:
    """Body-to-inertial rotation of a freely spinning body."""

    pole = pole_direction(orbit.pole_ra, orbit.pole_dec)

    node = equator_node(orbit.pole_ra)

    w = prime_meridian_angle(orbit.prime_meridian, orbit.rotation_period, seconds)

    meridian = node * math.cos(w) + np.cross(pole, node) * math.sin(w)

    return body_frame(pole, meridian)
