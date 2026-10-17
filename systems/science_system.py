import math
import time

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np

from core.logger import Logger

from ecs.components import BodyComponent, PlanetComponent

from planet.maps import load_elevation_map

from scene.scene import Scene

from science import survey
from science.record import (
    REGIONAL,
    SITUATION_NAMES,
    ScienceRecord,
    ScienceResult,
    situation_of,
    subject_key
)
from science.regions import GLOBAL, Region, elevation_levels, region_of
from science.sites import Site, SiteCatalog

from vessel import kepler


@dataclass(frozen=True, slots=True)
class Place:
    """Where the vessel is, for science."""

    body: Hashable                  # its key (entity; None: the Sun)
    body_id: str                    # profile id ("mars")
    body_name: str
    situation: str                  # science/record.py SITUATIONS
    region: Region
    site: Site | None               # the named site it is in, if any
    latitude: float                 # deg, under the vessel
    longitude: float
    altitude: float                 # m above the body's radius
    nearest: tuple[Site, float] | None   # the closest site and its distance (m)

    @property
    def description(
        self
    ) -> str:
        """'landed at Gale crater, Mars', 'flying over Mars's highlands', 'in space high over Mars'."""

        where = SITUATION_NAMES[self.situation]

        body = self.body_name

        if self.situation not in REGIONAL:
            return f"{where} {body}"

        if self.site is not None:

            if self.situation == "space_low":
                return f"{where} {body}, over {self.site.name}"

            return f"{where} {'at' if self.situation == 'landed' else 'over'} {self.site.name}, {body}"

        region = self.region.name.lower()

        if not region:
            return f"{where} {body}"

        possessive = f"{body}'" if body.endswith("s") else f"{body}'s"

        if self.situation == "space_low":
            return f"{where} {body}, over the {region}"

        return f"{where} {'on' if self.situation == 'landed' else 'over'} {possessive} {region}"


@dataclass(frozen=True, slots=True)
class ExperimentStatus:
    """One experiment aboard, here: whether it works, and what a result would be worth."""

    name: str
    works: bool
    why_not: str
    worth: float                    # recovered, if run now (0: nothing new)
    held: bool                      # a result for this subject is aboard already


class ScienceSystem:

    # =====================================================
    # Science
    # =====================================================
    #
    # The vessel's experiments (parts with an "experiment":
    # vessel/parts.py) find out about where it is: the body,
    # the situation (landed, flying, in space near or high
    # over), the region below (science/regions.py, from the
    # real maps) or the named site (science/sites.py). R runs
    # them; results ride on the vessel (lost with it) until
    # recovered, landing back on the body it launched from
    # (full value), or transmitted (half). See
    # science/record.py for what they are worth.
    #
    # A survey camera (an experiment that maps) also maps the
    # sunlit ground in its swath as the vessel flies over,
    # every frame, time warp included (the path between
    # frames is sampled): mapped cells unlock the body's
    # sharper elevation data (science/survey.py) and earn
    # points by the percent.

    # The camera's field of view (deg across) and the highest
    # it maps from (body radii up).
    CAMERA_FOV = 40.0
    MAPPING_RADII = 1.0

    # The Sun at least this high (sine of its elevation) for
    # the ground to show.
    DAYLIGHT = 0.05

    # Most swath samples one frame (beyond: sparser).
    MAX_SAMPLES = 2000

    # Within this many site radii, the HUD names the site.
    NEAR_SITE = 3.0

    # Seconds (real) between saves while mapping, and
    # between terrain refreshes over new mapped ground.
    SAVE_INTERVAL = 30.0
    REFRESH_INTERVAL = 5.0

    def __init__(
        self,
        catalog: SiteCatalog,
        record: ScienceRecord,
        path=None
    ):

        self.catalog = catalog
        self.record = record
        self.path = path

        # body id -> (lowlands' top, highlands' bottom).
        self._levels: dict[str, tuple[float, float] | None] = {}

        # Changed since saved; survey masks changed since the
        # terrain was refreshed (body ids).
        self._unsaved = False
        self._last_save = time.monotonic()
        self._remapped: set[str] = set()
        self._last_refresh = time.monotonic()

        # Mapping milestones reached (body id -> tenths).
        self._tenths: dict[str, int] = {}

    # =====================================================
    # Bodies
    # =====================================================

    def register_bodies(
        self,
        scene: Scene,
        home: str | None
    ):
        """
        Give every body in the scene its survey mask (the
        terrain finds them by id). Home's is open: its sharp
        data needs no mapping.
        """

        for _, body in scene.registry.view_with(BodyComponent):

            if not body.profile:
                continue

            mask = self.record.survey(body.profile, gating=body.profile != home)

            survey.register(body.profile, mask)

            self._tenths[body.profile] = int(mask.coverage() * 10.0)

    @staticmethod
    def body_id(
        scene: Scene,
        key: Hashable
    ) -> str:

        if key is None:
            return "sun"

        body = scene.try_get_component(key, BodyComponent)

        return body.profile if body is not None and body.profile else ""

    def _map_of(
        self,
        scene: Scene,
        key: Hashable
    ):

        planet = scene.try_get_component(key, PlanetComponent) if key is not None else None

        if planet is None or not planet.elevation_map:
            return None, planet

        try:
            return load_elevation_map(planet.elevation_map), planet
        except (OSError, ValueError):
            return None, planet

    def region_at(
        self,
        scene: Scene,
        key: Hashable,
        body_id: str,
        direction: np.ndarray
    ) -> Region:
        """The region of a body-frame direction on the body."""

        latitude, _ = survey.latitude_longitude(direction)

        elevation_map, planet = self._map_of(scene, key)

        sea = planet is not None and planet.liquid in ("water", "methane")

        if elevation_map is None:
            return region_of(float(latitude[0]), None, None, sea)

        if body_id not in self._levels:
            self._levels[body_id] = elevation_levels(elevation_map.faces[:, 1:-1:4, 1:-1:4], sea)

        height = float(elevation_map.sample(np.atleast_2d(direction))[0])

        return region_of(float(latitude[0]), height, self._levels[body_id], sea)

    # =====================================================
    # Where the vessel is
    # =====================================================

    def place(
        self,
        scene: Scene,
        vessel_system,
        orbit_system,
        planet_system
    ) -> Place | None:

        found = vessel_system.find(scene)

        if found is None:
            return None

        _, component = found

        state = component.state

        key = component.body

        if state.destroyed or vessel_system.reference_state(orbit_system, key) is None:
            return None

        body_id = self.body_id(scene, key)

        bodies = vessel_system.soi_bodies(scene)

        body = bodies.get(key)

        name = body.name if body is not None else body_id

        environment = vessel_system.environment(scene, key, orbit_system, planet_system)

        distance = float(np.linalg.norm(state.position))

        up = state.position / distance

        pressure, _ = environment.air(distance, up)

        radius = body.radius if body is not None else 1.0

        altitude = distance - radius

        situation = situation_of(state.landed, pressure > 0.0, altitude, radius)

        latitude = longitude = 0.0

        site = None
        nearest = None
        region = GLOBAL

        if key is not None and key in orbit_system.states:

            _, frame = orbit_system.states[key]

            local = frame.T @ up

            lat, lon = survey.latitude_longitude(local)

            latitude, longitude = float(lat[0]), float(lon[0])

            nearest = self.catalog.nearest(body_id, latitude, longitude, radius)

            if situation in REGIONAL:

                site = self.catalog.site_at(body_id, latitude, longitude, radius)

                region = (
                    Region(f"site:{site.id}", site.name, site.value)
                    if site is not None
                    else self.region_at(scene, key, body_id, local)
                )

        return Place(
            body=key,
            body_id=body_id,
            body_name=name,
            situation=situation,
            region=region,
            site=site,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            nearest=nearest
        )

    # =====================================================
    # Experiments
    # =====================================================

    @staticmethod
    def experiments(
        vessel
    ) -> list:
        """The experiments aboard (one of each kind)."""

        found = {}

        for state in vessel.parts:

            experiment = state.part.experiment

            if experiment is not None and experiment.id not in found:
                found[experiment.id] = experiment

        return list(found.values())

    def _result(
        self,
        experiment,
        place: Place
    ) -> ScienceResult:

        region = place.region if place.situation in REGIONAL else GLOBAL

        cap = experiment.value * self.catalog.factor(place.body_id) * region.value

        return ScienceResult(
            subject=subject_key(experiment.id, place.body_id, place.situation, region.id),
            title=f"{experiment.name}, {place.description}",
            cap=cap
        )

    def _works(
        self,
        experiment,
        place: Place,
        has_air: bool
    ) -> str:
        """Why not here ("" when it works)."""

        if place.situation not in experiment.situations:

            words = {"landed": "landed", "flying": "flying in air", "space_low": "in low space", "space_high": "in high space"}

            return "only " + " or ".join(words[item] for item in experiment.situations)

        if experiment.needs_air and not has_air:
            return "no air here"

        return ""

    def status(
        self,
        component,
        place: Place,
        has_air: bool
    ) -> list[ExperimentStatus]:
        """Each experiment aboard: here, now."""

        held = {result.subject for result in component.science}

        found = []

        for experiment in self.experiments(component.vessel):

            why = self._works(experiment, place, has_air)

            result = self._result(experiment, place)

            found.append(ExperimentStatus(
                name=experiment.name,
                works=not why,
                why_not=why,
                worth=self.record.worth(result) if not why else 0.0,
                held=result.subject in held
            ))

        return found

    def run(
        self,
        component,
        place: Place,
        has_air: bool
    ) -> list[str]:
        """Run every experiment that works here; returns notices."""

        notices = []

        held = {result.subject for result in component.science}

        experiments = self.experiments(component.vessel)

        if not experiments:
            return ["No experiments aboard (Science parts in the editor)"]

        for experiment in experiments:

            if self._works(experiment, place, has_air):
                continue

            result = self._result(experiment, place)

            if result.subject in held:
                continue

            worth = self.record.worth(result)

            if worth < 0.05:

                notices.append(f"{experiment.name}: nothing new here")

                continue

            component.science.append(result)

            notices.append(f"{result.title}: ~{worth:,.1f} science")

            Logger.info("[Science] %s (worth %.1f).", result.title, worth)

        if not notices:
            notices.append("Nothing to do here (see the Science panel: K)")

        return notices

    def recover(
        self,
        component
    ) -> float:
        """Bank everything aboard at full value (landed home)."""

        gained = sum(self.record.bank(result) for result in component.science)

        component.science.clear()

        if gained:
            self._changed()

        return gained

    def transmit(
        self,
        component,
        index: int
    ) -> float:
        """Send one result home now, for half its value."""

        if not 0 <= index < len(component.science):
            return 0.0

        result = component.science.pop(index)

        gained = self.record.bank(result, transmitted=True)

        self._changed()

        return gained

    # =====================================================
    # Mapping
    # =====================================================

    def mapping_status(
        self,
        scene: Scene,
        vessel_system,
        orbit_system
    ) -> str:
        """Why the camera is or is not mapping, for the panel."""

        found = vessel_system.find(scene)

        if found is None:
            return ""

        _, component = found

        if not any(experiment.maps for experiment in self.experiments(component.vessel)):
            return "No survey camera aboard"

        bodies = vessel_system.soi_bodies(scene)

        body = bodies.get(component.body)

        if body is None or component.body is None:
            return "Nothing to map round the Sun"

        planet = scene.try_get_component(component.body, PlanetComponent)

        if planet is None or planet.palette == "bands":
            return f"{body.name} has no surface to map"

        if component.state.landed:
            return "Landed: the camera maps from orbit"

        altitude = float(np.linalg.norm(component.state.position)) - body.radius

        if altitude > self.MAPPING_RADII * body.radius:
            return "Too high to map (below one radius up)"

        return f"Mapping the sunlit ground below (swath ~{2.0 * self._swath(altitude, body.radius) * body.radius / 1000.0:,.0f} km)"

    def _swath(
        self,
        altitude: float,
        radius: float
    ) -> float:
        """Half the swath's width, as an angle at the body's center (rad)."""

        half = altitude * math.tan(math.radians(self.CAMERA_FOV * 0.5)) / radius

        horizon = math.acos(min(1.0, radius / (radius + max(altitude, 0.0))))

        return min(half, horizon)

    def map_ground(
        self,
        scene: Scene,
        vessel_system,
        orbit_system,
        span: float
    ) -> list[str]:
        """
        After the vessel moved `span` simulated seconds: mark
        what the camera saw. Returns notices (milestones).
        """

        found = vessel_system.find(scene)

        if found is None or span <= 0.0:
            return []

        _, component = found

        state = component.state

        key = component.body

        if key is None or state.landed or state.destroyed or key not in orbit_system.states:
            return []

        if not any(experiment.maps for experiment in self.experiments(component.vessel)):
            return []

        planet = scene.try_get_component(key, PlanetComponent)

        if planet is None or planet.palette == "bands":
            return []

        bodies = vessel_system.soi_bodies(scene)

        body = bodies.get(key)

        if body is None:
            return []

        radius = body.radius

        distance = float(np.linalg.norm(state.position))

        altitude = distance - radius

        if altitude <= 0.0 or altitude > self.MAPPING_RADII * radius:
            return []

        body_id = self.body_id(scene, key)

        mask = self.record.surveys.get(body_id)

        if mask is None:

            mask = self.record.survey(body_id)

            survey.register(body_id, mask)

        half = self._swath(altitude, radius)

        # Samples over the frame's path, each half a swath
        # apart along the ground.
        speed = float(np.linalg.norm(state.velocity))

        ground = speed * span * radius / distance

        count = int(min(self.MAX_SAMPLES, max(1, math.ceil(ground / max(0.5 * half * radius, 1.0)))))

        times = np.linspace(-span, 0.0, count + 1)[1:] if count > 1 else np.array([0.0])

        positions, _ = kepler.propagate_many(state.position, state.velocity, body.mu, times)

        positions = positions[np.isfinite(positions).all(axis=1)]

        if not len(positions):
            return []

        directions = positions / np.linalg.norm(positions, axis=1, keepdims=True)

        # Into the body's frame then: it turned meanwhile.
        _, frame = orbit_system.states[key]

        spin = orbit_system.spin(scene, key)

        rate = float(np.linalg.norm(spin))

        if rate > 0.0 and len(times) == len(directions):

            axis = spin / rate

            angles = -rate * times

            cos = np.cos(angles)[:, None]
            sin = np.sin(angles)[:, None]

            # Rodrigues: undo the turn since each sample
            # (the body then was turned back by -w t).
            directions = (
                directions * cos
                + np.cross(axis, directions) * sin
                + axis * (directions @ axis)[:, None] * (1.0 - cos)
            )

        local = directions @ frame

        # In daylight only: the Sun's direction from the body.
        sun_inertial = -np.asarray(orbit_system.states[key][0], dtype=np.float64)

        sun_inertial /= max(float(np.linalg.norm(sun_inertial)), 1.0)

        sun = frame.T @ sun_inertial

        lit = local @ sun > self.DAYLIGHT

        if not np.any(lit):
            return []

        latitudes, longitudes = survey.latitude_longitude(local[lit])

        if not mask.mark(latitudes, longitudes, half):
            return []

        self._remapped.add(body_id)

        self._changed(save_now=False)

        notices = []

        gained = self.record.credit_mapping(body_id, self.catalog.factor(body_id))

        tenths = int(mask.coverage() * 10.0)

        if tenths > self._tenths.get(body_id, 0):

            self._tenths[body_id] = tenths

            notices.append(f"Mapped {tenths * 10}% of {body.name}")

        if gained:
            Logger.debug("[Science] Mapping %s: +%.1f.", body_id, gained)

        return notices

    # =====================================================
    # Upkeep
    # =====================================================

    def _changed(
        self,
        save_now: bool = True
    ):

        self._unsaved = True

        if save_now:
            self.save()

    def save(self):

        if self.path is None or not self._unsaved:
            return

        try:
            self.record.save(self.path)
        except OSError as error:
            Logger.error("[Science] Could not save progress: %s", error)
            return

        self._unsaved = False
        self._last_save = time.monotonic()

    def upkeep(
        self,
        planet_system
    ):
        """Each frame: save now and then; rebuild terrain over newly mapped ground."""

        now = time.monotonic()

        if self._unsaved and now - self._last_save > self.SAVE_INTERVAL:
            self.save()

        if self._remapped and now - self._last_refresh > self.REFRESH_INTERVAL:

            for body_id in self._remapped:
                planet_system.refresh_survey(body_id)

            self._remapped.clear()

            self._last_refresh = now
