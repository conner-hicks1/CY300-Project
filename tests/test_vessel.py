import math

import numpy as np
import pytest

from math3d import quaternion
from vessel import kepler
from vessel.parts import PartError, load_design, load_parts, parse_design
from vessel.physics import Controls, Environment, VesselState, can_coast, coast, land, step, upright
from vessel.vessel import G0, Vessel


EARTH_MU = 3.986004418e14
EARTH_RADIUS = 6_371_000.0


@pytest.fixture(scope="module")
def parts():

    return load_parts()


@pytest.fixture
def pathfinder(parts):

    return Vessel.from_design(load_design("pathfinder", parts))


# =========================================================
# Orbits
# =========================================================

def test_circular_orbit_returns_after_a_period():

    r = EARTH_RADIUS + 400_000.0
    v = kepler.circular_speed(EARTH_MU, r)

    position = np.array([r, 0.0, 0.0])
    velocity = np.array([0.0, 0.0, v])

    shape = kepler.orbit_shape(position, velocity, EARTH_MU)

    assert shape.periapsis == pytest.approx(r, rel=1e-9)
    assert shape.apoapsis == pytest.approx(r, rel=1e-9)
    assert shape.period == pytest.approx(5_544.9, abs=0.5)

    p, w = kepler.propagate(position, velocity, EARTH_MU, shape.period)

    np.testing.assert_allclose(p, position, atol=1e-3)
    np.testing.assert_allclose(w, velocity, atol=1e-6)

    # A quarter of the way round.
    p, _ = kepler.propagate(position, velocity, EARTH_MU, shape.period / 4.0)

    np.testing.assert_allclose(p, [0.0, 0.0, r], atol=1e-3)


@pytest.mark.parametrize("speed_factor", [0.8, 1.2, 1.5])
def test_propagation_matches_integration(speed_factor):

    r = EARTH_RADIUS + 300_000.0

    position = np.array([r, 0.0, 0.0])
    velocity = np.array([0.0, 1000.0, kepler.circular_speed(EARTH_MU, r) * speed_factor])

    # Fine RK4 as the reference.
    def accel(p):
        return -EARTH_MU * p / np.linalg.norm(p) ** 3

    p, v = position.copy(), velocity.copy()

    h = 0.5

    for _ in range(2_000):

        k1v = accel(p); k1p = v
        k2v = accel(p + k1p * h / 2); k2p = v + k1v * h / 2
        k3v = accel(p + k2p * h / 2); k3p = v + k2v * h / 2
        k4v = accel(p + k3p * h); k4p = v + k3v * h

        p = p + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)

    q, w = kepler.propagate(position, velocity, EARTH_MU, 1_000.0)

    np.testing.assert_allclose(q, p, atol=1.0)
    np.testing.assert_allclose(w, v, atol=1e-3)


def test_escape_has_no_apoapsis():

    r = EARTH_RADIUS + 300_000.0

    shape = kepler.orbit_shape([r, 0, 0], [0, 0, 1.5 * math.sqrt(2.0 * EARTH_MU / r)], EARTH_MU)

    assert shape.apoapsis is None and shape.period is None and shape.eccentricity > 1.0


def test_inclination():

    r = EARTH_RADIUS + 300_000.0
    v = kepler.circular_speed(EARTH_MU, r)

    # Equatorial (pole +Y), then over the poles.
    assert kepler.orbit_shape([r, 0, 0], [0, 0, v], EARTH_MU).inclination == pytest.approx(180.0)
    assert kepler.orbit_shape([r, 0, 0], [0, v, 0], EARTH_MU).inclination == pytest.approx(90.0)


# =========================================================
# Parts and Staging
# =========================================================

def test_pathfinder_adds_up(pathfinder):

    assert pathfinder.mass == pytest.approx(159_620.0)
    assert pathfinder.height == pytest.approx(37.2)
    assert pathfinder.radius == pytest.approx(1.25)

    # The engine bottom is the origin; the capsule on top.
    assert pathfinder.parts[-1].bottom == 0.0
    assert pathfinder.parts[0].bottom == pytest.approx(37.2 - 0.5)

    # Enough to reach low orbit (~9.4 km/s with losses).
    assert 10_000.0 < pathfinder.delta_v() < 10_600.0

    # Nothing changed by working that out.
    assert pathfinder.mass == pytest.approx(159_620.0) and pathfinder.next_stage == 0


def test_launch_thrust(pathfinder):

    assert pathfinder.thrust(1.0)[0] == 0.0

    assert pathfinder.stage() == 0

    pathfinder.throttle = 1.0

    sea_level, flow = pathfinder.thrust(1.01325)
    vacuum, _ = pathfinder.thrust(0.0)

    assert sea_level == pytest.approx(2_200_000.0 * 285.0 / 315.0)
    assert vacuum == pytest.approx(2_200_000.0)
    assert flow == pytest.approx(2_200_000.0 / (315.0 * G0))

    # Lifts off on Earth.
    assert sea_level / (pathfinder.mass * 9.81) > 1.2


def test_burn_and_separate(pathfinder):

    pathfinder.stage()
    pathfinder.throttle = 1.0

    upper_tank = pathfinder.parts[3]

    pathfinder.burn(10.0)

    # Only the lower section's tanks feed the booster.
    assert upper_tank.propellant == 25_000.0
    assert pathfinder.parts[6].propellant == pathfinder.parts[8].propellant < 40_000.0

    assert pathfinder.stage() == 1

    assert [state.part.id for state in pathfinder.parts] == [
        "parachute_main", "capsule", "decoupler_2m5", "tank_2m5_medium", "engine_vacuum"
    ]
    assert len(pathfinder.dropped) == 5

    assert pathfinder.parts[-1].ignited
    assert pathfinder.parts[-1].bottom == 0.0

    left, capacity = pathfinder.section_propellant()

    assert left == capacity == 25_000.0

    # The capsule goes home on its own.
    assert pathfinder.stage() == 2

    assert [state.part.id for state in pathfinder.parts] == ["parachute_main", "capsule"]
    assert pathfinder.mass == pytest.approx(2_620.0)

    assert pathfinder.stage() == 3
    assert pathfinder.parts[0].armed

    assert pathfinder.stage() is None
    assert pathfinder.stages_left == 0


def test_runs_dry(pathfinder):

    pathfinder.stage()
    pathfinder.throttle = 1.0

    pathfinder.burn(10_000.0)

    assert pathfinder.burning_engines() == []
    assert pathfinder.thrust()[0] == 0.0
    assert pathfinder.parts[6].propellant == 0.0


def test_bad_designs(parts):

    with pytest.raises(PartError, match="not in the catalog"):
        parse_design({"parts": [{"part": "nothing"}]}, "x", parts)

    with pytest.raises(PartError, match="nothing to fire"):
        parse_design({"parts": [{"part": "capsule"}, {"part": "tank_2m5_short", "stage": 0}]}, "x", parts)

    with pytest.raises(PartError, match="command part"):
        parse_design({"parts": [{"part": "tank_2m5_short"}]}, "x", parts)


# =========================================================
# Physics
# =========================================================

def environment(mu=EARTH_MU, radius=EARTH_RADIUS, spin=0.0, air=None, air_top=0.0):

    return Environment(
        mu=mu,
        spin=np.array([0.0, spin, 0.0]),
        air=air or (lambda r, d: (0.0, 0.0)),
        ground=lambda d: radius,
        air_top=air_top
    )


def on_pad(env, radius=EARTH_RADIUS):

    state = VesselState(
        position=np.array([radius, 0.0, 0.0]),
        velocity=np.zeros(3),
        orientation=upright(np.array([1.0, 0.0, 0.0])),
        angular_velocity=np.zeros(3)
    )

    land(state, env)

    return state


def test_upright_points_the_stack_up():

    for direction in ([1, 0, 0], [0, 1, 0], [0, 0, -1], [0.3, -0.5, 0.8]):

        d = np.array(direction, dtype=float) / np.linalg.norm(direction)

        np.testing.assert_allclose(quaternion.rotate_vector(upright(d), [0, 1, 0]), d, atol=1e-9)


def test_stays_on_the_pad_and_turns_with_the_body(pathfinder):

    spin = 2.0 * math.pi / 86_164.0

    env = environment(spin=spin)

    state = on_pad(env)

    # Throttle up without ignition: nothing.
    pathfinder.throttle = 1.0

    for _ in range(100):
        step(pathfinder, state, env, Controls(), 0.02)

    assert state.landed

    before = state.position.copy()

    # Carried round: 1/4 of a sidereal day in one step
    # (a quarter turn about +Y: x -> -z).
    step(pathfinder, state, env, Controls(), 86_164.0 / 4.0)

    np.testing.assert_allclose(state.position, [before[2], 0.0, -before[0]], atol=1e-3)
    np.testing.assert_allclose(state.up, state.position / EARTH_RADIUS, atol=1e-9)
    assert np.linalg.norm(state.velocity) == pytest.approx(EARTH_RADIUS * spin)


def test_liftoff(pathfinder):

    env = environment()

    state = on_pad(env)

    pathfinder.stage()
    pathfinder.throttle = 1.0

    start = pathfinder.mass

    for _ in range(500):
        step(pathfinder, state, env, Controls(), 0.02)

    assert not state.landed and not state.destroyed

    altitude = np.linalg.norm(state.position) - EARTH_RADIUS

    # 10 s at ~1.3 g of thrust against 1 g.
    assert 100.0 < altitude < 600.0

    assert pathfinder.mass < start


def test_crash_and_soft_landing(pathfinder):

    env = environment()

    hard = VesselState(
        position=np.array([EARTH_RADIUS + 5.0, 0.0, 0.0]),
        velocity=np.array([-50.0, 0.0, 0.0]),
        orientation=upright(np.array([1.0, 0.0, 0.0])),
        angular_velocity=np.zeros(3)
    )

    for _ in range(20):
        step(pathfinder, hard, env, Controls(), 0.02)

    assert hard.destroyed and hard.impact_speed == pytest.approx(50.0, rel=0.05)

    soft = VesselState(
        position=np.array([EARTH_RADIUS + 0.5, 0.0, 0.0]),
        velocity=np.array([-3.0, 0.0, 0.0]),
        orientation=upright(np.array([1.0, 0.0, 0.0])),
        angular_velocity=np.zeros(3)
    )

    for _ in range(20):
        step(pathfinder, soft, env, Controls(), 0.02)

    assert soft.landed and not soft.destroyed


def test_orbit_stays_closed(pathfinder):

    env = environment()

    r = EARTH_RADIUS + 400_000.0

    state = VesselState(
        position=np.array([r, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, kepler.circular_speed(EARTH_MU, r)]),
        orientation=quaternion.identity(),
        angular_velocity=np.zeros(3)
    )

    assert can_coast(pathfinder, state, env)

    for _ in range(int(600 / 0.02)):
        step(pathfinder, state, env, Controls(), 0.02)

    shape = kepler.orbit_shape(state.position, state.velocity, EARTH_MU)

    assert shape.periapsis == pytest.approx(r, abs=2_000.0)
    assert shape.apoapsis == pytest.approx(r, abs=2_000.0)

    # Integration and the exact orbit agree.
    exact = state.copy()

    coast(exact, env, 600.0)

    for _ in range(int(600 / 0.02)):
        step(pathfinder, state, env, Controls(), 0.02)

    assert np.linalg.norm(exact.position - state.position) < 2_000.0


def test_drag_slows_and_moves_with_the_air(pathfinder):

    env = environment(air=lambda r, d: (1.0, 1.2), air_top=EARTH_RADIUS + 100_000.0)

    state = VesselState(
        position=np.array([EARTH_RADIUS + 5_000.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 300.0]),
        orientation=upright(np.array([0.0, 0.0, 1.0])),
        angular_velocity=np.zeros(3)
    )

    assert not can_coast(pathfinder, state, env)

    step(pathfinder, state, env, Controls(), 0.1)

    # 1/2 rho v^2 Cd A / m, over 0.1 s.
    expected = 0.5 * 1.2 * 300.0 ** 2 * 0.45 * math.pi * 1.25 ** 2 / pathfinder.mass * 0.1

    assert 300.0 - state.velocity[2] == pytest.approx(expected, rel=0.01)


def test_steering_and_sas(pathfinder):

    env = environment()

    r = EARTH_RADIUS + 400_000.0

    state = VesselState(
        position=np.array([r, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, kepler.circular_speed(EARTH_MU, r)]),
        orientation=quaternion.identity(),
        angular_velocity=np.zeros(3)
    )

    # (No engines burning: reaction wheels only, slow on
    # a fully fuelled stack.)
    for _ in range(200):
        step(pathfinder, state, env, Controls(pitch=1.0), 0.02)

    assert state.angular_velocity[0] > 0.0

    turned = quaternion.rotate_vector(state.orientation, [0, 1, 0])

    assert turned[1] < 1.0 - 1e-6

    # SAS stops the turn.
    for _ in range(500):
        step(pathfinder, state, env, Controls(sas=True), 0.02)

    assert np.linalg.norm(state.angular_velocity) < 1e-9

    # Without it, the spin goes on.
    state.angular_velocity = np.array([0.01, 0.0, 0.0])

    step(pathfinder, state, env, Controls(sas=False), 0.02)

    assert state.angular_velocity[0] == pytest.approx(0.01)


# =========================================================
# To Orbit
# =========================================================

def test_pathfinder_reaches_low_earth_orbit(pathfinder):
    """
    A gravity turn in Earth's real air (planet/air.py from
    its body profile), steering ideally: Pathfinder makes a
    150 km orbit with propellant to spare.
    """

    from planet.air import air_column_of
    from planet.bodies import components_for, load_presets

    parts = components_for(load_presets()["earth"])

    column = air_column_of(parts.atmosphere, parts.body)

    def air(radius, direction):

        altitude = (radius - EARTH_RADIUS) / 1000.0

        pressure = column.pressure(altitude)

        if pressure < 1e-7:
            return 0.0, 0.0

        return pressure, pressure * 1e5 / (column.gas_constant * column.temperature(altitude))

    spin = 2.0 * math.pi / 86_164.0

    env = Environment(
        mu=EARTH_MU,
        spin=np.array([0.0, spin, 0.0]),
        air=air,
        ground=lambda d: EARTH_RADIUS,
        air_top=EARTH_RADIUS + 1000.0 * column.altitude_of_pressure(1e-7)
    )

    # On the equator at +X; the spin about +Y carries the
    # ground toward -Z: east.
    state = on_pad(env)

    east = np.array([0.0, 0.0, -1.0])

    pathfinder.stage()
    pathfinder.throttle = 1.0

    target = 150_000.0

    dt = 0.05

    for _ in range(int(1_200 / dt)):

        r = float(np.linalg.norm(state.position))

        up = state.position / r

        altitude = r - EARTH_RADIUS

        horizontal = np.cross(np.cross(up, east), up)
        horizontal /= np.linalg.norm(horizontal)

        climb = float(np.dot(state.velocity, up))
        across = float(np.dot(state.velocity, horizontal))

        if pathfinder.next_stage < 2:

            # Booster: a gravity turn, 75 deg over by 45 km.
            pitch = 90.0 - min(max((altitude - 500.0) / 45_000.0, 0.0), 1.0) ** 0.5 * 75.0

        else:

            # Upper stage: climb to the target height, then
            # hold it while speeding up (gravity less the
            # orbital "lift" of the speed across).
            push = pathfinder.thrust(0.0)[0] / pathfinder.mass

            need = EARTH_MU / r ** 2 - across * across / r + ((target - altitude) / 40.0 - climb) / 8.0

            pitch = math.degrees(math.asin(min(max(need / max(push, 1e-6), -0.3), 0.95)))

        heading = math.cos(math.radians(pitch)) * horizontal + math.sin(math.radians(pitch)) * up

        # Ideal steering: point along the heading.
        state.orientation = upright(heading)
        state.angular_velocity = np.zeros(3)

        shape = kepler.orbit_shape(state.position, state.velocity, EARTH_MU)

        if pathfinder.next_stage == 2 and shape.periapsis > EARTH_RADIUS + target - 15_000.0:
            break

        if not pathfinder.burning_engines() and pathfinder.stages_left:
            pathfinder.stage()

        step(pathfinder, state, env, Controls(), dt)

        assert not state.destroyed

    shape = kepler.orbit_shape(state.position, state.velocity, EARTH_MU)

    assert shape.periapsis > EARTH_RADIUS + 130_000.0, shape
    assert shape.apoapsis < EARTH_RADIUS + 200_000.0

    # On the upper stage, propellant to spare.
    assert pathfinder.next_stage == 2
    assert pathfinder.propellant > 1_000.0


# =========================================================
# Coming Home
# =========================================================

def capsule_only(pathfinder):

    for _ in range(4):
        pathfinder.stage()

    pathfinder.dropped.clear()

    return pathfinder


def sea_level_air(radius, direction):

    return 1.0, 1.2


def test_parachute_waits_for_slow_thick_air(pathfinder):

    capsule = capsule_only(pathfinder)

    chute = capsule.parts[0]

    assert chute.armed and chute.deployed == 0.0

    # Too fast: stays packed.
    capsule.open_parachutes(1.0, 600.0, 1.0)

    assert chute.deployed == 0.0

    # Too thin: stays packed.
    capsule.open_parachutes(0.001, 100.0, 1.0)

    assert chute.deployed == 0.0

    # Opens over three seconds, then stays open.
    capsule.open_parachutes(0.5, 200.0, 1.5)

    assert chute.deployed == pytest.approx(0.5)

    capsule.open_parachutes(0.5, 900.0, 5.0)

    assert chute.deployed == 1.0

    assert capsule.drag_area() > 500.0


def test_capsule_lands_under_its_parachute(pathfinder):

    capsule = capsule_only(pathfinder)

    env = Environment(
        mu=EARTH_MU,
        spin=np.zeros(3),
        air=sea_level_air,
        ground=lambda d: EARTH_RADIUS,
        air_top=EARTH_RADIUS + 100_000.0
    )

    state = VesselState(
        position=np.array([EARTH_RADIUS + 3_000.0, 0.0, 0.0]),
        velocity=np.array([-200.0, 0.0, 0.0]),
        orientation=upright(np.array([1.0, 0.0, 0.0])),
        angular_velocity=np.zeros(3)
    )

    for _ in range(int(600 / 0.02)):

        step(capsule, state, env, Controls(), 0.02)

        if state.landed or state.destroyed:
            break

    assert state.landed and not state.destroyed

    # Without it: destroyed.
    bare = capsule_only(Vessel.from_design(load_design("pathfinder", load_parts())))
    bare.parts[0].armed = False

    state = VesselState(
        position=np.array([EARTH_RADIUS + 3_000.0, 0.0, 0.0]),
        velocity=np.array([-200.0, 0.0, 0.0]),
        orientation=upright(np.array([1.0, 0.0, 0.0])),
        angular_velocity=np.zeros(3)
    )

    for _ in range(int(600 / 0.02)):

        step(bare, state, env, Controls(), 0.02)

        if state.landed or state.destroyed:
            break

    assert state.destroyed


def test_sas_points_prograde_and_retrograde(pathfinder):

    env = environment()

    r = EARTH_RADIUS + 400_000.0

    for mode, sign in (("prograde", 1.0), ("retrograde", -1.0)):

        state = VesselState(
            position=np.array([r, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, kepler.circular_speed(EARTH_MU, r)]),
            orientation=upright(np.array([1.0, 0.0, 0.0])),
            angular_velocity=np.zeros(3)
        )

        for _ in range(int(60 / 0.02)):
            step(pathfinder, state, env, Controls(sas_mode=mode), 0.02)

        heading = state.velocity / np.linalg.norm(state.velocity) * sign

        assert np.dot(state.up, heading) > math.cos(math.radians(2.0)), mode

        assert np.linalg.norm(state.angular_velocity) < 0.01


# =========================================================
# Side Boosters
# =========================================================

@pytest.fixture
def heavy(parts):

    return Vessel.from_design(load_design("pathfinder_heavy", parts))


def test_boosters_count_twice_and_stand_off_the_axis(heavy, pathfinder):

    boosters = [state for state in heavy.parts if state.radial and state.part.kind != "decoupler"]

    assert {state.count for state in boosters} == {2}

    # 2 x (40 + 2 x 650 + 900 + 2 x 10,000) + 2 x 60 (separators).
    assert heavy.mass == pytest.approx(pathfinder.mass + 2 * (40 + 1300 + 900 + 20_000) + 2 * 60)

    # Core radius + booster radius, off the axis.
    assert boosters[0].distance == pytest.approx(1.25 + 0.625)

    # Hung on the lowest tank: the booster engine's bottom at its bottom.
    lowest_tank = heavy.core[-2]

    assert boosters[-1].bottom == pytest.approx(lowest_tank.bottom)

    # Symmetric: thrust on the axis, but more drag and inertia.
    assert heavy.drag_area() > pathfinder.drag_area()
    assert heavy.moments_of_inertia[1] > pathfinder.moments_of_inertia[1]


def test_boosters_light_with_the_core_and_burn_their_own_tanks(heavy):

    heavy.stage()
    heavy.throttle = 1.0

    engines = heavy.burning_engines()

    assert len(engines) == 2

    thrust, _ = heavy.thrust(0.0)

    assert thrust == pytest.approx(2_200_000.0 + 2 * 420_000.0)

    core_tank = heavy.core[-2]
    booster_tank = next(state for state in heavy.parts if state.radial and state.part.kind == "tank")

    heavy.burn(10.0)

    # Each copy feeds its own engine: 10 s at 138.1 kg/s
    # from its two tanks.
    flow = 420_000.0 / (310.0 * G0)

    assert booster_tank.propellant == pytest.approx(10_000.0 - flow * 10.0 / 2.0)

    # The core burns only the core's tanks.
    assert core_tank.propellant < 40_000.0


def test_radial_separator_drops_the_boosters_only(heavy):

    heavy.stage()
    heavy.throttle = 1.0

    core_before = len(heavy.core)
    height_before = heavy.height

    assert heavy.stage() == 1

    assert not any(state.radial for state in heavy.parts)
    assert len(heavy.core) == core_before and heavy.height == height_before

    assert len(heavy.dropped) == 5 and all(state.radial for state in heavy.dropped)

    # The core's engine still burns.
    assert [state.part.id for _, state in heavy.burning_engines()] == ["engine_booster"]


def test_dropping_the_core_section_takes_its_boosters(heavy):

    for _ in range(3):
        heavy.stage()

    # Stage 2 dropped the lower core with everything on it.
    assert not any(state.radial for state in heavy.parts)
    assert heavy.parts[-1].part.id == "engine_vacuum"


def test_booster_delta_v_by_stage(heavy, pathfinder):

    stats = heavy.stage_stats(1.01325, 9.80665)

    assert [stage.stage for stage in stats] == [0, 1, 2, 3, 4]

    # Liftoff: core and both boosters, until the boosters run dry.
    assert stats[0].engines == 3
    assert stats[0].burn_time == pytest.approx(20_000.0 / (420_000.0 / (310.0 * G0)), rel=1e-6)

    # The core burns on after they drop.
    assert stats[1].engines == 1 and stats[1].delta_v_vacuum > 0.0

    # Boosters add delta-v and liftoff thrust.
    assert heavy.delta_v() > pathfinder.delta_v() + 300.0
    assert stats[0].twr_surface > pathfinder.stage_stats(1.01325, 9.80665)[0].twr_surface

    assert sum(stage.delta_v_vacuum for stage in stats) == pytest.approx(heavy.delta_v(), rel=1e-9)


def test_bad_radial_groups(parts):

    with pytest.raises(PartError, match="symmetry"):
        parse_design({"parts": [{"part": "capsule", "radial": [{"symmetry": 5, "parts": [{"part": "tank_1m25_short"}]}]}]}, "x", parts)

    with pytest.raises(PartError, match="needs parts"):
        parse_design({"parts": [{"part": "capsule", "radial": [{"symmetry": 2, "parts": []}]}]}, "x", parts)

    with pytest.raises(PartError, match="must be a stage separator"):
        parse_design({"parts": [{"part": "capsule", "radial": [{"symmetry": 2, "decoupler": {"part": "tank_1m25_short"}, "parts": [{"part": "tank_1m25_short"}]}]}]}, "x", parts)

    with pytest.raises(PartError, match="cannot carry radial"):
        parse_design({"parts": [{"part": "capsule", "radial": [{"symmetry": 2, "parts": [
            {"part": "tank_1m25_short", "radial": [{"symmetry": 2, "parts": [{"part": "tank_1m25_short"}]}]}
        ]}]}]}, "x", parts)
