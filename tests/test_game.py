import math

import numpy as np
import pytest

from ecs.components import (
    BodyComponent,
    CameraComponent,
    ClockComponent,
    NameComponent,
    PlanetComponent,
    TransformComponent
)
from game import format
from game.flight import FlightContext, frame_body, go_to_orbit, read_flight
from game.pause_menu import Destination, PauseMenu
from game.quality import QUALITY_NAMES, apply_quality, current_quality
from game.time_warp import WARP_RATES, TimeWarp
from graphics.render_settings import RenderSettings
from math3d.transform import Transform
from scene.scene import Scene


# =========================================================
# Time Warp
# =========================================================

def test_warp_steps_and_clamps():

    warp = TimeWarp()

    assert warp.rate == 1.0 and not warp.warping

    for _ in range(len(WARP_RATES) + 3):
        warp.faster()

    assert warp.rate == WARP_RATES[-1]

    warp.slower()

    assert warp.rate == WARP_RATES[-2]

    warp.stop()

    assert warp.rate == 1.0

    warp.slower()

    assert warp.rate == 1.0


def test_pause_stops_the_clock_and_keeps_the_warp():

    warp = TimeWarp()
    clock = ClockComponent(rate=1.0)

    warp.faster()
    warp.faster()

    warp.paused = True
    warp.apply(clock)

    assert clock.rate == 0.0
    assert warp.label() == "Paused"

    warp.paused = False
    warp.apply(clock)

    assert clock.rate == WARP_RATES[2]
    assert warp.label() == f"{WARP_RATES[2]:,.0f}x"

    warp.apply(None)


# =========================================================
# Quality
# =========================================================

def test_quality_presets_round_trip():

    settings = RenderSettings()

    # The defaults are the high preset.
    assert current_quality(settings) == "High"

    for name in QUALITY_NAMES:

        apply_quality(settings, name)

        assert current_quality(settings) == name

    apply_quality(settings, "Low")

    assert not settings.scatter and settings.atmosphere_samples < RenderSettings().atmosphere_samples

    settings.atmosphere_samples = 7

    assert current_quality(settings) is None


# =========================================================
# Formatting
# =========================================================

@pytest.mark.parametrize("meters, text", [
    (12.4, "12 m"),
    (999.0, "999 m"),
    (2_950.0, "3.0 km"),
    (400_000.0, "400 km"),
    (9_555_000.0, "9,555 km"),
])
def test_distance_formatting(meters, text):

    assert format.distance(meters) == text


def test_pressure_formatting():

    assert format.pressure(92.0) == "92 bar"
    assert format.pressure(1.013) == "1.01 bar"
    assert format.pressure(0.006) == "600 Pa"


def test_body_facts():

    facts = dict(format.body_facts(BodyComponent(surface_gravity=9.81, rotation_hours=-5832.0, surface_pressure_bar=0.0)))

    assert facts["Gravity"].startswith("9.81 m/s2")
    assert facts["Rotation"].endswith("(retrograde)")
    assert facts["Surface pressure"] == "none (vacuum)"


# =========================================================
# Pause Menu
# =========================================================

def test_menu_pages_and_escape():

    menu = PauseMenu({"Planets": [Destination("earth", "Earth")]})

    menu.toggle()

    assert menu.open and menu.page == PauseMenu.MAIN

    menu.page = PauseMenu.TRAVEL

    # Esc: back to the main page first, then closed.
    menu.back()

    assert menu.open and menu.page == PauseMenu.MAIN

    menu.back()

    assert not menu.open


# =========================================================
# Flight Readout
# =========================================================

RADIUS = 1_000_000.0


def flight_scene(camera_position):

    scene = Scene("test")

    def add(name, transform, *components):

        entity = scene.create_entity()

        scene.add_component(entity, NameComponent(name))
        scene.add_component(entity, TransformComponent(transform=transform))

        for component in components:
            scene.add_component(entity, component)

        return entity

    planet = add("Small", Transform(), PlanetComponent(radius=RADIUS, liquid="none"))

    # A larger body much farther away.
    add("Far", Transform(position=(0.0, 0.0, 1.0e9)), PlanetComponent(radius=5.0 * RADIUS))

    camera = add("Camera", Transform(position=camera_position), CameraComponent(primary=True))

    add("Clock", Transform(), ClockComponent(time=1_000.0))

    from systems.transform_system import TransformSystem

    TransformSystem().update(scene)

    return scene, planet, camera


def test_readout_finds_the_nearest_surface():

    latitude = math.radians(30.0)

    position = (RADIUS + 5_000.0) * np.array([0.0, math.sin(latitude), math.cos(latitude)])

    scene, planet, _ = flight_scene(tuple(position))

    context = FlightContext.find(scene)

    assert context.body == planet

    readout = read_flight(context, lambda body, direction: (0.0, 1_200.0))

    assert readout.body == "Small"
    assert readout.altitude == pytest.approx(5_000.0)
    assert readout.above_ground == pytest.approx(3_800.0)
    assert readout.latitude == pytest.approx(30.0)
    assert readout.longitude == pytest.approx(0.0, abs=1e-9)
    assert readout.pressure is None
    assert readout.time == 1_000.0

    # No sun: no local time.
    assert readout.local_hour is None


def test_camera_moves():

    scene, planet, camera = flight_scene((0.0, RADIUS + 100.0, 0.0))

    context = FlightContext.find(scene)

    go_to_orbit(context, 400_000.0)

    transform = scene.get_component(camera, TransformComponent).transform

    assert np.linalg.norm(transform.position) == pytest.approx(RADIUS + 400_000.0)

    # Looking down toward the horizon, not up.
    assert np.dot(transform.forward, np.asarray(transform.position)) < 0.0

    frame_body(context, RADIUS)

    distance = np.linalg.norm(transform.position)

    assert distance > RADIUS / math.sin(math.radians(30.0))

    # Looking at the center.
    np.testing.assert_allclose(transform.forward, -np.asarray(transform.position) / distance, atol=1e-9)


# =========================================================
# Warp Limits, Chase Camera
# =========================================================

def test_warp_is_held_to_the_limit():

    warp = TimeWarp()

    for _ in range(4):
        warp.faster()

    assert warp.limit(5.0)
    assert warp.rate == 5.0 and warp.capped

    # Asking for more stays at the limit...
    warp.faster()

    assert warp.rate == 5.0

    # ...until it lifts.
    assert not warp.limit(float("inf"))

    warp.faster()

    assert warp.rate == 10.0 and not warp.capped


def test_chase_camera_circles_and_looks_at_the_target():

    from game.chase_camera import ChaseCamera

    camera = ChaseCamera(distance=50.0)
    camera.pitch = 0.0

    transform = Transform()

    target = np.array([10.0, 20.0, 30.0])
    up = np.array([0.0, 1.0, 0.0])

    # Spin axis +Z: the ground under +Y moves toward -X,
    # east. The camera stands west, looking east.
    camera.place(transform, target, up, np.array([0.0, 0.0, 1.0]))

    np.testing.assert_allclose(transform.position, target + [50.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(transform.forward, [-1.0, 0.0, 0.0], atol=1e-9)

    # Scrolling in, never past the minimum.
    camera.orbit((0.0, 0.0), 100.0)

    assert camera.distance == ChaseCamera.MIN_DISTANCE

    # A floor lifts it.
    camera.pitch = -60.0

    camera.place(transform, target, up, np.array([0.0, 0.0, 1.0]), floor=-1.0)

    assert transform.position[1] - target[1] == pytest.approx(-1.0)


def test_frustum_faces_point_out():

    from graphics.mesh_factory import MeshFactory

    data = MeshFactory.frustum_data(0.5, 1.25, 2.0, segments=12)

    positions = np.asarray(data.positions, dtype=np.float64).reshape(-1, 3)
    normals = np.asarray(data.normals, dtype=np.float64).reshape(-1, 3)
    indices = np.asarray(data.indices).reshape(-1, 3)

    assert positions[:, 1].min() == 0.0 and positions[:, 1].max() == 2.0

    for a, b, c in indices:

        face = np.cross(positions[b] - positions[a], positions[c] - positions[a])

        # Counter-clockwise seen from where the normals face.
        assert np.dot(face, normals[a] + normals[b] + normals[c]) > 0.0
