import math

import numpy as np
import pytest

from vessel import heating
from vessel.parts import load_design, load_parts
from vessel.physics import Controls, Environment, VesselState, step, upright
from vessel.vessel import Vessel


EARTH_MU = 3.986004418e14
EARTH_RADIUS = 6_371_000.0


def exponential_air(radius, direction):

    density = 1.225 * math.exp(-(radius - EARTH_RADIUS) / 7_200.0)

    return density / 1.225 * 1.01325, density


EARTH = Environment(
    mu=EARTH_MU,
    spin=np.zeros(3),
    air=exponential_air,
    ground=lambda d: EARTH_RADIUS,
    air_top=EARTH_RADIUS + 140_000.0,
    temperature=lambda radius: 250.0
)


def capsule(shield: bool = True) -> Vessel:

    vessel = Vessel.from_design(load_design("pathfinder", load_parts()))

    # Booster, upper stage, then the capsule on its own.
    for _ in range(3):
        vessel.stage()

    vessel.dropped.clear()

    if not shield:
        vessel.parts = [state for state in vessel.parts if state.part.id != "heat_shield_2m5"]
        vessel._restack()

    return vessel


def come_in(vessel: Vessel, speed: float, angle_deg: float) -> tuple[VesselState, float]:
    """From 120 km, `angle_deg` below the horizon, tail first: (state at the end, hottest skin K)."""

    angle = math.radians(angle_deg)

    velocity = speed * np.array([math.sin(angle), math.cos(angle), 0.0])

    state = VesselState(
        np.array([EARTH_RADIUS + 120_000.0, 0.0, 0.0]),
        velocity,
        upright(-velocity),
        np.zeros(3)
    )

    controls = Controls(sas=True, sas_mode="retrograde")

    peak = 0.0

    for _ in range(int(1_500 / 0.05)):

        step(vessel, state, EARTH, controls, 0.05)

        peak = max(peak, max(part.temperature for part in vessel.parts))

        if state.destroyed or float(np.linalg.norm(state.velocity)) < 250.0:
            break

    return state, peak


def test_flux_numbers():

    # ~1.6 MW/m^2 at 60 km and 7.5 km/s on a 1.25 m capsule.
    flux = heating.stagnation_flux(1.225 * math.exp(-60_000.0 / 7_200.0), 7_500.0, 1.25)

    assert 0.7e6 < flux < 1.6e6

    # v^3, and sqrt(density).
    assert heating.stagnation_flux(0.01, 2_000.0, 1.0) == pytest.approx(8.0 * heating.stagnation_flux(0.01, 1_000.0, 1.0))
    assert heating.stagnation_flux(0.04, 1_000.0, 1.0) == pytest.approx(2.0 * heating.stagnation_flux(0.01, 1_000.0, 1.0))

    assert heating.stagnation_flux(0.0, 7_000.0, 1.0) == 0.0

    # Radiating 1 MW/m^2 back out (emissivity 0.8): ~2,170 K.
    assert heating.equilibrium_temperature(1.0e6) == pytest.approx(2_167.0, abs=5.0)


def test_heat_shield_brings_the_capsule_home_from_orbit():

    vessel = capsule()

    state, peak = come_in(vessel, 7_800.0, -1.5)

    assert not state.destroyed

    # The shield took it, the capsule above stayed cool.
    assert 1_800.0 < peak < 3_300.0
    assert [part.part.id for part in vessel.parts if part.temperature > 1_000.0] in ([], ["heat_shield_2m5"])


def test_and_from_the_moon():

    state, peak = come_in(capsule(), 11_000.0, -6.0)

    assert not state.destroyed and peak > 2_500.0


def test_without_it_the_capsule_burns_up():

    state, _ = come_in(capsule(shield=False), 7_800.0, -1.5)

    assert state.destroyed and state.burned_up == "Crew capsule"

    # High up: it never reaches the ground.
    assert float(np.linalg.norm(state.position)) - EARTH_RADIUS > 20_000.0


def test_nose_first_the_shield_does_not_help():

    vessel = capsule()

    angle = math.radians(-1.5)

    velocity = 7_800.0 * np.array([math.sin(angle), math.cos(angle), 0.0])

    state = VesselState(np.array([EARTH_RADIUS + 120_000.0, 0.0, 0.0]), velocity, upright(velocity), np.zeros(3))

    controls = Controls(sas=True, sas_mode="prograde")

    for _ in range(int(1_500 / 0.05)):

        step(vessel, state, EARTH, controls, 0.05)

        if state.destroyed:
            break

    assert state.destroyed and state.burned_up == "Main parachute, 30 m canopy"


def test_skins_cool_in_space():

    vessel = capsule()

    vessel.parts[-1].temperature = 2_000.0

    heating.heat(vessel, np.array([0.0, 1.0, 0.0]), np.zeros(3), 0.0, heating.SPACE_TEMPERATURE, 600.0)

    assert vessel.parts[-1].temperature == pytest.approx(heating.SPACE_TEMPERATURE)

    # Briefly: partway.
    vessel.parts[-1].temperature = 2_000.0

    heating.heat(vessel, np.array([0.0, 1.0, 0.0]), np.zeros(3), 0.0, heating.SPACE_TEMPERATURE, 1.0)

    assert heating.SPACE_TEMPERATURE < vessel.parts[-1].temperature < 2_000.0

    assert heating.hottest(vessel) is vessel.parts[-1]
