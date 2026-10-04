import math

from datetime import datetime, timezone

import numpy as np
import pytest

from ecs.components import (
    ClockComponent,
    DirectionalLightComponent,
    NameComponent,
    OrbitComponent,
    PlanetComponent,
    StarComponent,
    TransformComponent
)
from graphics.bodies_block import BodySphere, eclipsing, pack_bodies_block, umbra_glow
from graphics.uniform_blocks import BODIES_BLOCK, MAX_BODIES
from math3d import quaternion
from planet import solar
from planet.bodies import components_for, load_presets, system_members
from planet.orbits import (
    OrbitElements,
    date_of,
    orbit_offset,
    seconds_since_j2000,
    solve_kepler
)
from scene.scene import Scene
from systems.orbit_system import OrbitSystem, body_states


@pytest.fixture(scope="module")
def presets():

    return load_presets()


def system(presets, ids, star=True):

    scene = Scene("orbits")

    entities = {}

    for body_id in ids:

        profile = presets[body_id]

        entity = scene.create_entity()

        scene.add_component(entity, NameComponent(profile.name))
        scene.add_component(entity, TransformComponent())

        for component in components_for(profile).all():
            scene.add_component(entity, component)

        entities[body_id] = entity

    if star:

        sun = scene.create_entity()

        scene.add_component(sun, TransformComponent())
        scene.add_component(sun, DirectionalLightComponent())
        scene.add_component(sun, StarComponent())
        scene.add_component(sun, ClockComponent())

        entities["sun"] = sun

    return scene, entities


def states_at(scene, when):

    bodies = list(scene.registry.view_with(TransformComponent, OrbitComponent))

    return body_states(scene, bodies, seconds_since_j2000(when))


def utc(*args):

    return datetime(*args, tzinfo=timezone.utc)


# =========================================================
# Kepler
# =========================================================

@pytest.mark.parametrize("e", [0.0, 0.2, 0.6, 0.95])
def test_kepler_equation(e):

    for m in np.linspace(-3.0, 3.0, 13):

        E = solve_kepler(m, e)

        assert E - e * math.sin(E) == pytest.approx(m, abs=1e-10)


def test_ellipse():

    elements = OrbitElements(1.0e9, 0.3, 0.2, 0.5, 1.0, 0.0, 1000.0)

    # Periapsis at the start, apoapsis half a period on, back
    # after a whole one.
    assert np.linalg.norm(orbit_offset(elements, 0.0)) == pytest.approx(0.7e9)
    assert np.linalg.norm(orbit_offset(elements, 500.0)) == pytest.approx(1.3e9)

    np.testing.assert_allclose(orbit_offset(elements, 1000.0), orbit_offset(elements, 0.0), atol=1.0)


def test_dates_round_trip():

    moment = utc(2026, 10, 3, 18, 30)

    assert date_of(seconds_since_j2000(moment)) == moment


# =========================================================
# The Real Sky
# =========================================================

def sun_in_body_frame(states, entity):

    position, frame = states[entity]

    return frame.T @ (-position / np.linalg.norm(position))


def test_earths_seasons(presets):

    scene, e = system(presets, ["earth"])

    for when, declination in (
        (utc(2026, 6, 21, 8, 0), 23.44),
        (utc(2026, 12, 21, 20, 0), -23.44),
        (utc(2026, 3, 20, 14, 0), 0.0),
    ):

        sun = sun_in_body_frame(states_at(scene, when), e["earth"])

        assert math.degrees(math.asin(sun[1])) == pytest.approx(declination, abs=0.25)

    # Nearest the Sun in early January.
    january = np.linalg.norm(states_at(scene, utc(2026, 1, 3))[e["earth"]][0])
    july = np.linalg.norm(states_at(scene, utc(2026, 7, 4))[e["earth"]][0])

    assert january < july


def test_clock_time_is_greenwich_solar_time(presets):

    scene, e = system(presets, ["earth"])

    greenwich = np.array([0.0, 0.0, 1.0])

    def solar_noon_offset(when):

        sun = sun_in_body_frame(states_at(scene, when), e["earth"])

        hour, _, _ = solar.solar_time(greenwich, sun)

        return (hour - 12.0) * 60.0

    # The equation of time: the sundial lags ~14 min in
    # February and leads ~16 min in November.
    assert solar_noon_offset(utc(2026, 2, 11, 12)) == pytest.approx(-14.2, abs=1.5)
    assert solar_noon_offset(utc(2026, 11, 3, 12)) == pytest.approx(16.4, abs=1.5)

    # Local time runs forward.
    morning = solar_noon_offset(utc(2026, 5, 1, 9)) / 60.0 + 12.0

    assert morning == pytest.approx(9.0, abs=0.1)


def angle_from_earth(states, e, toward_sun):

    earth = states[e["earth"]][0]
    moon = states[e["moon"]][0]

    sun = -earth / np.linalg.norm(earth)

    direction = (moon - earth) / np.linalg.norm(moon - earth)

    return math.degrees(math.acos(np.clip(direction @ (sun if toward_sun else -sun), -1.0, 1.0)))


def test_real_eclipses(presets):

    scene, e = system(presets, ["earth", "moon"])

    # Total lunar eclipse, 2026-03-03: the Moon in Earth's
    # umbra (~0.7 deg wide at its distance).
    assert angle_from_earth(states_at(scene, utc(2026, 3, 3, 11, 34)), e, toward_sun=False) < 0.6

    # Total solar eclipse over Egypt, 2027-08-02.
    assert angle_from_earth(states_at(scene, utc(2027, 8, 2, 10, 7)), e, toward_sun=True) < 0.3

    # The full moon a month later passes well clear.
    assert angle_from_earth(states_at(scene, utc(2026, 4, 2, 2, 12)), e, toward_sun=False) > 2.0


def test_the_moon_keeps_its_face_to_earth(presets):

    scene, e = system(presets, ["earth", "moon"])

    for day in range(0, 30, 5):

        states = states_at(scene, utc(2026, 5, 1 + day))

        moon, frame = states[e["moon"]]
        earth, _ = states[e["earth"]]

        toward_earth = frame.T @ ((earth - moon) / np.linalg.norm(earth - moon))

        # Its prime meridian (+z) faces Earth.
        assert toward_earth[2] > 0.99


def test_system_members(presets):

    assert system_members(presets, "earth") == ["moon"]
    assert system_members(presets, "moon") == ["earth"]
    assert system_members(presets, "io") == ["jupiter", "europa", "ganymede", "callisto"]
    assert system_members(presets, "mars") == ["phobos", "deimos"]
    assert system_members(presets, "vesta") == []


# =========================================================
# Orbit System
# =========================================================

def place(scene, entity, position):

    scene.get_component(entity, TransformComponent).transform.position = position


def test_the_anchor_stays_and_the_sky_turns(presets):

    scene, e = system(presets, ["earth", "moon"])

    clock = scene.get_component(e["sun"], ClockComponent)
    clock.time = seconds_since_j2000(utc(2026, 5, 1))

    radius = scene.get_component(e["earth"], PlanetComponent).radius

    earth_transform = scene.get_component(e["earth"], TransformComponent).transform
    earth_transform.position = (0.0, -radius, 0.0)

    orbits = OrbitSystem()

    camera = np.array([0.0, 100.0, 0.0])

    orbits.anchor_to(scene, e["earth"])

    orbits.update(scene, 0.0, camera)

    assert orbits.anchor == e["earth"]

    moon_before = np.array(scene.get_component(e["moon"], TransformComponent).transform.position)
    sun_before = scene.get_component(e["sun"], TransformComponent).world_forward.copy()

    # Six hours on at 3600 x real time.
    clock.rate = 3_600.0

    orbits.update(scene, 6.0, camera)

    assert clock.time == pytest.approx(seconds_since_j2000(utc(2026, 5, 1, 6)))

    np.testing.assert_allclose(earth_transform.position, (0.0, -radius, 0.0))

    moon_after = np.array(scene.get_component(e["moon"], TransformComponent).transform.position)

    # The Moon is ~384,000 km away and has wheeled across
    # the sky (Earth turned by ~90 degrees).
    assert 350e6 < np.linalg.norm(moon_after - earth_transform.position) < 410e6
    assert np.linalg.norm(moon_after - moon_before) > 300e6

    light = scene.get_component(e["sun"], DirectionalLightComponent)

    assert light.intensity == pytest.approx(5.0, rel=0.05)

    # The sun has moved through ~90 degrees of sky too.
    sun_after = np.asarray(_light_forward(scene, e["sun"]))

    angle = math.degrees(math.acos(np.clip(np.dot(sun_after, _unit(sun_before)), -1.0, 1.0)))

    assert 70.0 < angle < 110.0


def _light_forward(scene, entity):

    orientation = scene.get_component(entity, TransformComponent).transform.orientation

    return quaternion.rotate_vector(orientation, np.array([0.0, 0.0, -1.0]))


def _unit(v):

    return np.asarray(v) / np.linalg.norm(v)


def test_flying_to_the_moon_moves_the_anchor_seamlessly(presets):

    scene, e = system(presets, ["earth", "moon"])

    orbits = OrbitSystem()

    orbits.anchor_to(scene, e["earth"])

    orbits.update(scene, 0.0, np.array([0.0, 7.0e6, 0.0]))

    assert orbits.anchor == e["earth"]

    moon_transform = scene.get_component(e["moon"], TransformComponent).transform

    before = np.array(moon_transform.position)

    # Next to the Moon: it becomes the anchor, exactly where
    # it was.
    camera = before + np.array([0.0, 2.0e6, 0.0])

    orbits.update(scene, 0.0, camera)

    assert orbits.anchor == e["moon"]

    np.testing.assert_allclose(moon_transform.position, before)

    # Time passes: now the Moon stays and Earth moves.
    clock = scene.get_component(e["sun"], ClockComponent)
    clock.rate = 86_400.0

    orbits.update(scene, 1.0, camera)

    np.testing.assert_allclose(moon_transform.position, before)


# =========================================================
# Eclipses (bodies block)
# =========================================================

def test_eclipsing():

    moon = BodySphere(center=(384_000e3, 0.0, 0.0), radius=1_737e3)

    toward_sun = (1.0, 0.0, 0.0)

    sun = math.radians(0.27)

    # The Moon between Earth and the Sun: its shadow can
    # reach Earth; on the far side it cannot.
    assert eclipsing(moon, (0.0, 0.0, 0.0), 6_471e3, toward_sun, sun)
    assert not eclipsing(moon, (0.0, 0.0, 0.0), 6_471e3, (-1.0, 0.0, 0.0), sun)

    # Far off the line.
    assert not eclipsing(moon, (0.0, 40_000e3, 0.0), 6_471e3, toward_sun, sun)


def test_bodies_block():

    bodies = [
        BodySphere(center=(0.0, -6_371e3, 0.0), radius=6_371e3, glow=umbra_glow(1.0)),
        BodySphere(center=(384_000e3, 0.0, 0.0), radius=1_737e3),
    ]

    raw = pack_bodies_block(bodies, (0.0, 0.0, 0.0), 0.0047, eclipsing_count=1)

    assert len(raw) == BODIES_BLOCK.size

    v = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)

    np.testing.assert_allclose(v[0, :3], (2.0, 0.0047, 1.0), rtol=1e-6)
    np.testing.assert_allclose(v[1], (0.0, -6_371.0, 0.0, 6_371.0))
    np.testing.assert_allclose(v[2], (384_000.0, 0.0, 0.0, 1_737.0))

    # Air reddens its umbra; airless bodies do not.
    glow = v[1 + MAX_BODIES]

    assert glow[0] > glow[1] > glow[2] > 0.0
    assert not v[2 + MAX_BODIES, :3].any()

    assert umbra_glow(0.0) == (0.0, 0.0, 0.0)
