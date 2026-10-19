import math

import numpy as np

from imgui_bundle import imgui

from core.input import Input

from ecs.components import CameraComponent, TransformComponent, VesselComponent

from game.flight import FlightContext
from game.map_view import ScreenProjection

from systems.vessel_system import Suggestion

from planet import weather
from programme import events

from science.survey import direction_of as survey_direction

from vessel import kepler, maneuver
from vessel.transfer import (
    closest_approach,
    correction,
    find_window,
    plan_transfer
)

from game import format as readable


# =========================================================
# The Application: Navigation
# =========================================================
#
# The map: its camera, the orbit lines, the navigation
# panel (windows, encounters, course corrections,
# rendezvous).
#
# A part of Application (application.py), apart for its
# size: its methods run on the application (self is it).

class NavigationController:

    def _map_camera(
        self,
        paused: bool
    ):
        """The map's camera, far round the body the vessel flies about."""

        context = FlightContext.find(self.scene)

        found = self.vessel_system.find(self.scene)

        if found is None or context.camera is None:
            return

        _, component = found

        scroll = 0.0 if self.imgui_layer.wants_mouse or paused else Input.scroll_delta()[1]

        self.map_view.orbit(Input.mouse_delta() if self._looking and not paused else (0.0, 0.0), scroll)

        center, axis, radius = self._map_body(component.body, component)

        prediction = self.vessel_system.prediction(self.scene, self.orbit_system)

        fit = radius * 2.0

        if prediction is not None and prediction.patches:

            reach = float(np.linalg.norm(prediction.patches[0].points, axis=1).max())

            # (Round a planet, no farther than its sphere of
            # influence; round the Sun, the whole path.)
            bodies = self.vessel_system.soi_bodies(self.scene)

            limit = bodies[component.body].soi if component.body in bodies else math.inf

            fit = max(fit, min(reach, limit if math.isfinite(limit) else reach))

        self.map_view.place_camera(context.camera_transform(), center, axis, fit, radius * 1.3)

    def _map_body(
        self,
        key,
        component
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """(world center, spin axis, radius) of a body in patched conics (None: the Sun)."""

        bodies = self.vessel_system.soi_bodies(self.scene)

        body = bodies.get(key)

        radius = body.radius if body is not None else 1.0

        if key is None:

            center = self.orbit_system.star_position()

            pole = np.cross(component.state.position, component.state.velocity)

            axis = self.orbit_system.inertial_to_world @ (pole / max(float(np.linalg.norm(pole)), 1e-9))

            return (center if center is not None else np.zeros(3)), axis, radius

        transform = self.scene.get_component(key, TransformComponent)

        matrix = np.asarray(transform.world_matrix, dtype=np.float64)

        return matrix[:3, 3].copy(), matrix[:3, 1] / np.linalg.norm(matrix[:3, 1]), radius

    def _draw_map(self):
        """The map's orbit lines over the view."""

        found = self.vessel_system.find(self.scene)

        context = FlightContext.find(self.scene)

        if not self.map_view.open or found is None or context.camera is None:
            return

        entity, component = found

        camera = self.scene.get_component(context.camera, CameraComponent)

        io = imgui.get_io()

        projection = ScreenProjection.of(context.camera_transform(), camera.fov, io.display_size.x, io.display_size.y)

        bodies = self.vessel_system.soi_bodies(self.scene)

        reference = component.body

        keys = [reference] + [
            key for key, body in bodies.items()
            if key is not None and body.parent == reference and key != reference
        ]

        body_world = {}
        body_radius = {}
        names = {}

        for key in keys:

            center, _, radius = self._map_body(key, component)

            body_world[key] = center
            body_radius[key] = radius
            names[key] = bodies[key].name if key in bodies else ""

        # The moons' orbits about the reference body (redrawn
        # every simulated hour).
        hour = int(self.orbit_system.time // 3600.0)

        children = [key for key in keys[1:] if key in self.orbit_system.states]

        relative = self.vessel_system.relative_states(self.scene, self.orbit_system, reference, children)

        moon_orbits = {}

        for key in children:

            cached = self._moon_orbits.get(key)

            if cached is None or cached[0] != hour:

                position, velocity = relative[key]

                shape = kepler.orbit_shape(position, velocity, bodies[reference].mu)

                span = shape.period if shape.period is not None else 30.0 * 86_400.0

                points, _ = kepler.propagate_many(position, velocity, bodies[reference].mu, np.linspace(0.0, span, 129))

                cached = (hour, points)

                self._moon_orbits[key] = cached

            moon_orbits[key] = cached[1]

        prediction = self.vessel_system.prediction(self.scene, self.orbit_system)

        planned = self.vessel_system.planned(self.scene, self.orbit_system)

        node_world = None

        if component.node is not None:

            where = self.vessel_system.node_state(self.scene, self.orbit_system, component.node.time)

            if where is not None and reference in body_world:
                node_world = body_world[reference] + self.orbit_system.inertial_to_world @ where[0]

        # The other vessels round the body (orbits redrawn every
        # simulated minute).
        others = []

        if reference in body_world:

            minute = int(self.orbit_system.time // 60.0)

            for other_entity, other in self.vessel_system.vessels(self.scene):

                if other_entity == entity or other.body != reference or other.state.destroyed:
                    continue

                point = body_world[reference] + self.orbit_system.inertial_to_world @ other.state.position

                orbit = None

                if not other.state.landed and reference in bodies:

                    cached = self._other_orbits.get(other_entity)

                    if cached is None or cached[0] != minute:

                        shape = kepler.orbit_shape(other.state.position, other.state.velocity, bodies[reference].mu)

                        span = shape.period if shape.period is not None and (shape.apoapsis or 0.0) < bodies[reference].soi else 86_400.0

                        points, _ = kepler.propagate_many(other.state.position, other.state.velocity, bodies[reference].mu, np.linspace(0.0, span, 129))

                        cached = (minute, points[np.isfinite(points).all(axis=1)])

                        self._other_orbits[other_entity] = cached

                    orbit = body_world[reference] + cached[1] @ self.orbit_system.inertial_to_world.T

                others.append((point, other.vessel.name, orbit, self.map_view.target == ("vessel", other_entity)))

        # Apophis, near its pass by the Earth.
        now = self.orbit_system.time

        if events.apophis_near(now) and reference is not None and reference in body_world and self.science_system.body_id(self.scene, reference) == "earth":

            to_world = self.orbit_system.inertial_to_world

            here = events.apophis_position(now)

            point = body_world[reference] + to_world @ here

            distance = None

            found = self.vessel_system.find(self.scene)

            if found is not None and found[1].body == reference:
                distance = float(np.linalg.norm(here - found[1].state.position))

            label = "Apophis" + (f" ({distance / 1000.0:,.0f} km)" if distance is not None else "")

            others.append((point, label, body_world[reference] + events.apophis_path(now) @ to_world.T, False))

        # The science sites on the body, where they are now.
        sites = []

        body_id = self.science_system.body_id(self.scene, reference)

        if reference is not None and reference in self.orbit_system.states and reference in body_world:

            _, frame = self.orbit_system.states[reference]

            for site in self.science_system.catalog.of(body_id):

                local = survey_direction(site.latitude, site.longitude)

                sites.append((
                    body_world[reference] + self.orbit_system.inertial_to_world @ (frame @ local) * body_radius[reference],
                    site.name
                ))

        # Mars's dust storms, where they are now.
        storms = []

        if reference is not None and reference in self.orbit_system.states and reference in body_world and body_id == "mars":

            _, frame = self.orbit_system.states[reference]

            radius = body_radius[reference]

            to_world = self.orbit_system.inertial_to_world

            for storm, strength in weather.active(self.orbit_system.time):

                label = f"{storm.name.split(',')[0]} (dust {storm.tau * strength:.1f})"

                center = body_world[reference] + to_world @ (frame @ survey_direction(storm.latitude, storm.longitude)) * radius * 1.01

                if storm.global_storm:

                    storms.append((None, f"Global dust storm (dust {storm.tau * strength:.1f})", body_world[reference] + to_world @ (frame @ np.array([0.0, 1.0, 0.0])) * radius * 1.15))

                    continue

                # Its edge: a ring its radius from its center.
                ring = []

                angle = storm.radius / weather.MARS_RADIUS

                pole = survey_direction(storm.latitude, storm.longitude)

                hint = np.array([0.0, 1.0, 0.0]) if abs(pole[1]) < 0.9 else np.array([1.0, 0.0, 0.0])

                across = np.cross(pole, hint)
                across /= np.linalg.norm(across)

                along = np.cross(pole, across)

                for k in range(49):

                    turn = 2.0 * np.pi * k / 48

                    local = pole * np.cos(angle) + (across * np.cos(turn) + along * np.sin(turn)) * np.sin(angle)

                    ring.append(body_world[reference] + to_world @ (frame @ local) * radius * 1.01)

                storms.append((np.array(ring), label, center))

        self.map_view.draw(
            projection,
            self.orbit_system.inertial_to_world,
            body_world,
            body_radius,
            names,
            prediction,
            np.asarray(self.scene.get_component(entity, TransformComponent).transform.position, dtype=np.float64),
            moon_orbits,
            body_radius.get(reference, 0.0),
            planned=planned,
            node_world=node_world,
            node_label=f"Node {self.maneuver_dv:,.0f} m/s" if self.maneuver_dv is not None else "",
            sites=sites,
            others=others,
            storms=storms
        )

        targets = sorted(
            (
                (key, body.name)
                for key, body in bodies.items()
                if key is not None and key != reference and key in self.orbit_system.states
            ),
            # Planets first, then moons (by name).
            key=lambda item: (bodies[item[0]].parent is not None, item[1])
        )

        # Other vessels too (a rendezvous).
        targets += [
            (("vessel", other_entity), f"{other.vessel.name} (vessel)")
            for other_entity, other in self.vessel_system.vessels(self.scene)
            if other_entity != entity and not other.state.destroyed
        ]

        # (Planning a burn, the encounter it makes shows.)
        lines, burn, suggested = self._navigation(component, bodies, planned or prediction)

        self.map_view.draw_navigation(targets, lines, burn, projection)

        self._plan_node(
            component,
            planned,
            suggested + self.vessel_system.suggested_burns(self.scene, self.orbit_system, self.planet_system)
        )

    def _update_maneuver(self):
        """
        Each frame: the maneuver SAS mode points along what is
        left of the node's burn (the HUD counts it down); and
        round the Sun with a planet targeted, the course
        correction that reaches it (vessel/transfer.py), for
        the navigation panel to suggest as a node.
        """

        self.maneuver_dv = None
        self._correction = None

        found = self.vessel_system.find(self.scene)

        if found is None:
            return

        _, component = found

        remaining = self.vessel_system.node_remaining(component)

        component.controls.maneuver = remaining

        if remaining is not None:
            self.maneuver_dv = float(np.linalg.norm(remaining))

        # Warping to the node: as fast as allowed (the node
        # holds it back as the burn comes), until there.
        if self._warp_until is not None:

            if component.node is None or self.orbit_system.time >= self._warp_until:
                self._warp_until = None
            else:
                self.time_warp.index = len(self.time_warp.rates) - 1

        self._correction = self._course_correction(component)

    def _rendezvous(
        self,
        component,
        bodies: dict,
        prediction,
        target_entity
    ) -> list:
        """Navigation lines toward another vessel: how close the paths come, how fast past each other."""

        other = self.scene.try_get_component(target_entity, VesselComponent)

        if other is None or other.state.destroyed:

            self.map_view.target = None

            return [("That vessel is gone.", self.NOTE)]

        name = other.vessel.name

        if other.body != component.body:
            return [(f"{name} is round {self.vessel_system.body_name(self.scene, other.body)}: get there first.", self.NOTE)]

        distance = float(np.linalg.norm(other.state.position - component.state.position))
        speed = float(np.linalg.norm(other.state.velocity - component.state.velocity))

        lines = [(f"{name}: {readable.distance(distance)} away, {speed:,.1f} m/s apart", None)]

        near = self.vessel_system.alongside(self.scene, component, self.orbit_system)

        if near is not None and near[0] == target_entity:

            lines.append(("Alongside: move propellant in Vessels (L)", self.GOOD))

            return lines

        reference = bodies.get(component.body)

        if prediction is None or not prediction.patches or reference is None or other.state.landed or component.state.landed:

            lines.append(("Both in flight round the same body for an approach.", self.NOTE))

            return lines

        patch = prediction.patches[0]

        where, moving = kepler.propagate_many(other.state.position, other.state.velocity, reference.mu, patch.times)

        gap, when = closest_approach(patch.points, patch.times, where)

        index = int(np.argmin(np.abs(patch.times - when)))

        _, own = kepler.propagate(component.state.position, component.state.velocity, reference.mu, float(when))

        relative = float(np.linalg.norm(moving[index] - own)) if np.isfinite(moving[index]).all() else float("nan")

        lines.append((f"Closest approach: {readable.distance(gap)} in {readable.countdown(when)}", self.GOOD if gap < 5_000.0 else None))
        lines.append((f"Passing at {relative:,.0f} m/s: brake to it there (a node at the time)", self.NOTE))

        return lines

    def _course_correction(
        self,
        component
    ) -> np.ndarray | None:
        """Round the Sun, the map's target a planet: the velocity change now that meets it, or None."""

        target = self.map_view.target

        bodies = self.vessel_system.soi_bodies(self.scene)

        if (
            component.body is not None
            or target not in bodies
            or bodies[target].parent is not None
            or target not in self.orbit_system.states
            or component.state.destroyed
        ):
            return None

        now = self.orbit_system.time

        clock_now = self.timer.time_ns / 1e9 if self.timer is not None else 0.0

        # When it gets closest, from the predicted path (found
        # again every few seconds; the aim is that moment).
        if self._arrival is None or self._arrival[0] != target or clock_now - self._arrival_found > 3.0:

            prediction = self.vessel_system.prediction(self.scene, self.orbit_system)

            if prediction is None or not prediction.patches:
                return None

            # Already meeting it: trim inside its sphere instead.
            if any(patch.body == target for patch in prediction.patches[1:]):

                self._arrival = (target, None)

            else:

                patch = prediction.patches[0]

                relative = self.vessel_system.relative_states(self.scene, self.orbit_system, None, [target])[target]

                where, _ = kepler.propagate_many(*relative, bodies[None].mu, patch.times)

                _, when = closest_approach(patch.points, patch.times, where)

                self._arrival = (target, now + when if when > 86_400.0 else None)

            self._arrival_found = clock_now

        arrival = self._arrival[1]

        if arrival is None or arrival - now < 86_400.0:
            return None

        velocities = self.orbit_system.velocities(self.scene)

        # The cheapest of a few arrival times round the closest
        # approach (near half a turn away the transfer's plane
        # is ill-defined, and one exact time can cost a lot).
        candidates = []

        for factor in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15):

            try:
                candidates.append(correction(
                    bodies[None].mu,
                    component.state.position,
                    component.state.velocity,
                    self.orbit_system.states[target][0],
                    velocities[target],
                    (arrival - now) * factor
                ))
            except (ValueError, ZeroDivisionError, OverflowError):
                continue

        if not candidates:
            return None

        return min(candidates, key=lambda vector: float(np.linalg.norm(vector)))

    def _navigation(
        self,
        component,
        bodies: dict,
        prediction
    ) -> tuple[list, np.ndarray | None, list]:
        """
        (lines, burn point in the world or None, Suggestions)
        for the map's navigation panel: how to get from here
        to the target, and the burns to plan for it.
        """

        target = self.map_view.target

        if isinstance(target, tuple) and target and target[0] == "vessel":
            return self._rendezvous(component, bodies, prediction, target[1]), None, []

        if target is None or target not in bodies:
            return [("Choose a target: how to get there shows here.", self.NOTE)], None, []

        reference = component.body

        state = component.state

        here = bodies.get(reference)
        goal = bodies[target]

        days = lambda seconds: seconds / 86_400.0

        # Encountering it already?
        if prediction is not None:

            for patch in prediction.patches[1:]:

                if patch.body == target:

                    return [
                        (f"Encounter with {goal.name} in {days(patch.start):,.1f} days", self.GOOD),
                        (f"Closest: {readable.distance(patch.periapsis - goal.radius)} above it", None),
                        ("Tune it with small burns; the map follows.", self.NOTE),
                        ("Inside its sphere, plan the capture at Pe.", self.NOTE),
                    ], None, []

        # There already.
        if target == reference:

            if state.landed:
                return [(f"Landed on {goal.name}.", self.GOOD)], None, []

            return [
                (f"In {goal.name}'s sphere of influence.", self.GOOD),
                ("Capture at the periapsis to stay (Maneuver),", self.NOTE),
                ("or come down through the air (heat shield first).", self.NOTE),
            ], None, []

        # From a planet's orbit to another planet: the window.
        if (
            here is not None
            and reference is not None
            and here.parent is None
            and goal.parent is None
            and not state.landed
        ):

            velocities = self.orbit_system.velocities(self.scene)

            now = self.orbit_system.time

            cached = self._window

            # The cheapest departure (~0.1 s to find): found
            # again for a new target, once it has passed, or
            # every couple of months (the orbits drift from
            # their conics).
            if (
                cached is None
                or cached[:2] != (reference, target)
                or now > cached[2] + cached[3].departure + 2.0 * 86_400.0
                or now - cached[2] > 60.0 * 86_400.0
            ):

                window = find_window(
                    bodies[None].mu,
                    (self.orbit_system.states[reference][0], velocities[reference]),
                    (self.orbit_system.states[target][0], velocities[target])
                )

                cached = self._window = (reference, target, now, window)

            plan = plan_transfer(
                bodies[None].mu,
                self.orbit_system.states[reference][0],
                velocities[reference],
                self.orbit_system.states[target][0],
                velocities[target],
                here.mu,
                state.position,
                state.velocity,
                window=cached[3],
                window_wait_time=cached[2] + cached[3].departure - now
            )

            wait = days(plan.wait)

            lines = [
                (f"To {goal.name}: {days(plan.transfer_time):,.0f} days on the way", None),
                (f"{goal.name} leads by {plan.phase_now:+.0f} deg (window at {plan.phase_wanted:+.0f})", None),
            ]

            # (Within a day either side.)
            if abs(wait) < 1.0:
                lines.append(("Window: now", self.GOOD))
            else:
                lines.append((f"Window in {wait:,.0f} days (time warp: . )", None))

            minutes, seconds = divmod(int(plan.time_to_burn_point), 60)

            lines += [
                (f"At the window: ~{plan.departure_burn:,.0f} m/s prograde", None),
                (f"Burning now at the marker: {plan.burn_now:,.0f} m/s", self.GOOD if plan.burn_now < plan.departure_burn * 1.1 else None),
                (f"(over it in {minutes}:{seconds:02d})", self.NOTE),
                (f"Arrive {plan.arrival_excess:,.0f} m/s fast: burn or brake in its air", self.NOTE),
            ]

            center, _, _ = self._map_body(reference, component)

            burn = center + self.orbit_system.inertial_to_world @ (
                plan.burn_direction * float(np.linalg.norm(state.position))
            )

            suggested = []

            # The window open: a node over the marker.
            if abs(wait) < 1.0:

                when = now + plan.time_to_burn_point

                where = self.vessel_system.node_state(self.scene, self.orbit_system, when)

                if where is not None:
                    suggested.append(Suggestion(
                        f"Burn for {goal.name}",
                        f"{plan.burn_now:,.0f} m/s prograde over the marker",
                        when,
                        maneuver.to_inertial(*where, plan.burn_now, 0.0, 0.0)
                    ))

            return lines, burn, suggested

        # Both round the same body: how close the path comes.
        if goal.parent == reference and prediction is not None and prediction.patches:

            patch = prediction.patches[0]

            relative = self.vessel_system.relative_states(self.scene, self.orbit_system, reference, [target])[target]

            where, _ = kepler.propagate_many(*relative, bodies[reference].mu if reference in bodies else here.mu, patch.times)

            gap, when = closest_approach(patch.points, patch.times, where)

            lines = [(f"Closest approach: {readable.distance(gap)} in {days(when):,.1f} days", None)]

            if gap < goal.soi:
                lines.append(("Inside its sphere of influence: an encounter", self.GOOD))
            else:
                lines.append((f"(its sphere of influence: {readable.distance(goal.soi)})", self.NOTE))

            suggested = []

            if self._correction is not None:

                size = float(np.linalg.norm(self._correction))

                lines.append((f"Course correction: {size:,.0f} m/s (Maneuver: suggested)", self.GOOD))

                suggested.append(Suggestion(
                    "Course correction",
                    f"{size:,.0f} m/s to meet {goal.name}",
                    self.orbit_system.time + 10.0,
                    self._correction
                ))

            return lines, None, suggested

        parent = bodies.get(goal.parent)

        return [(f"Reach {parent.name if parent is not None else 'orbit'} first.", self.NOTE)], None, []
