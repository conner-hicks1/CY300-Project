from datetime import datetime

import numpy as np
import pytest

from planet import weather
from planet.orbits import seconds_since_j2000
from vessel import power
from vessel.parts import load_design, load_parts
from vessel.physics import Controls, Environment, VesselState, step, upright
from vessel.records import vessel_from_json, vessel_to_json
from vessel.vessel import Vessel


def at(day: str) -> float:

    return seconds_since_j2000(datetime.fromisoformat(day + "T00:00:00+00:00"))


def test_the_season_follows_the_real_calendar():

    # Mars years 37 and 38 began on these days (Ls 0).
    for day, year in (("2022-12-27", 37), ("2024-11-13", 38)):

        ls = weather.solar_longitude(at(day))

        assert ls < 1.5 or ls > 358.5
        assert weather.mars_year(at(day)) == year

    # The 2018 global storm began near Ls 185 (early June 2018).
    assert weather.solar_longitude(at("2018-06-01")) == pytest.approx(185.0, abs=2.0)
    assert weather.mars_year(at("2018-06-01")) == 34

    assert weather.season(10.0).startswith("northern spring")
    assert weather.season(300.0).endswith("southern summer")


def test_the_air_thickens_in_southern_summer():

    assert weather.pressure_factor(255.0) == pytest.approx(1.12)
    assert weather.pressure_factor(75.0) == pytest.approx(0.88)


def test_storms_come_in_the_dusty_season_and_are_the_same_each_time():

    for year in range(36, 46):

        storms = weather.storms(year)

        assert storms == weather.storms(year)
        assert 3 <= sum(1 for storm in storms if not storm.global_storm) <= 6

        for storm in storms:
            assert weather.DUST_SEASON[0] <= storm.start <= weather.DUST_SEASON[1]

    # Some years have a global storm, most do not.
    globals_ = sum(any(storm.global_storm for storm in weather.storms(year)) for year in range(30, 90))

    assert 8 < globals_ < 35


def test_a_global_storm_darkens_the_whole_planet():

    # Find one, at its height.
    year = next(year for year in range(38, 80) if any(storm.global_storm for storm in weather.storms(year)))

    storm = next(storm for storm in weather.storms(year) if storm.global_storm)

    assert storm.strength(0.5 * storm.length) == 1.0
    assert storm.strength(-1.0) == 0.0 and storm.strength(storm.length + 1.0) == 0.0

    for latitude, longitude in ((60.0, 0.0), (-45.0, 120.0), (0.0, -150.0)):
        assert storm.reach(latitude, longitude) == 1.0

    assert weather.sunlight(storm.tau, 1.0) < 0.15
    assert weather.sunlight(0.4, 1.0) > 0.8


def test_a_regional_storm_fades_away_from_its_center():

    storm = weather.Storm("s", 200.0, 10.0, 0.0, 0.0, 1_000_000.0, 3.0, 20.0)

    assert storm.reach(0.0, 0.0) == 1.0

    # 1,000 km (its radius) away: 1/e; far away: nothing.
    degrees = np.degrees(1_000_000.0 / weather.MARS_RADIUS)

    assert storm.reach(0.0, degrees) == pytest.approx(np.exp(-1.0), rel=1e-3)
    assert storm.reach(0.0, 90.0) < 1e-6


def test_clear_skies_out_of_season():

    # October 2026: Ls ~10, no storms.
    sky = weather.weather_at(at("2026-10-19"), -4.6, 137.4)

    assert sky.tau < 0.6 and sky.storm == "" and sky.light > 0.75
    assert weather.wind_from(sky) == "W"


def test_dust_settles_and_wind_cleans_it():

    calm = weather.Weather(0.0, "", 0.5, 4.0, 4.0, 0.0, 1.0)

    dust = weather.settle_dust(0.0, calm, 100 * weather.SOL)

    assert 0.15 < dust < 0.3

    # Gusts in a fairly clear sky: blown clean.
    gale = weather.Weather(0.0, "", 1.0, 30.0, 30.0, 0.0, 1.0, "Regional storm")

    assert weather.settle_dust(dust, gale, 2 * weather.SOL) < dust - 0.2

    # The same wind in a thick global storm: it only gathers.
    thick = weather.Weather(0.0, "", 7.0, 30.0, 30.0, 0.0, 1.0, "Global dust storm")

    assert weather.settle_dust(dust, thick, 2 * weather.SOL) > dust


def test_dusty_wings_make_less():

    parts = load_parts()

    lander = Vessel.from_design(load_design("mariner", parts))

    power.set_panels(lander, True)
    power.move_panels(lander, power.PANEL_TIME)

    clean = power.solar_power(lander, np.array([0.0, 1.0, 0.0]), 0.43)

    lander.panel_dust = 0.4

    assert power.solar_power(lander, np.array([0.0, 1.0, 0.0]), 0.43) == pytest.approx(0.6 * clean)

    # Kept with the vessel.
    assert vessel_from_json(vessel_to_json(lander), parts).panel_dust == pytest.approx(0.4)


def test_wind_carries_a_falling_vessel():

    parts = load_parts()

    capsule = Vessel.from_design(load_design("pathfinder", parts))

    for _ in range(3):
        capsule.stage()

    radius = 3_389_500.0

    up = np.array([1.0, 0.0, 0.0])

    def environment(wind):

        return Environment(
            mu=4.2828e13,
            spin=np.zeros(3),
            air=lambda r, d: (0.006, 0.015),
            ground=lambda d: radius,
            air_top=radius + 100_000.0,
            wind=(lambda position: wind) if wind is not None else None
        )

    def fall(wind):

        state = VesselState(
            position=up * (radius + 5_000.0),
            velocity=-up * 60.0,
            orientation=upright(up),
            angular_velocity=np.zeros(3)
        )

        for _ in range(500):
            step(capsule, state, environment(wind), Controls(sas=False), 0.02)

        return state

    still = fall(None)
    blown = fall(np.array([0.0, 25.0, 0.0]))

    # Pushed downwind (+Y), a little through such thin air
    # (under a parachute's canopy, much more); not at all
    # without wind.
    assert abs(still.velocity[1]) < 1e-9
    assert blown.velocity[1] > 0.05
