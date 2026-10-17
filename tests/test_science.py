import math

import numpy as np
import pytest

from science import survey
from science.record import ScienceRecord, ScienceResult, situation_of, subject_key
from science.regions import HIGHLANDS, LOWLANDS, NORTH_POLE, PLAINS, SEA, SOUTH_POLE, elevation_levels, region_of
from science.sites import angular_distance, load_sites
from science.survey import SurveyMask, direction_of, latitude_longitude
from vessel.parts import PartError, load_parts, parse_part


MARS_RADIUS = 3_389_500.0


# =========================================================
# Sites
# =========================================================

def test_sites_load_and_are_found():

    catalog = load_sites()

    gale = catalog.site_at("mars", -5.4, 137.8, MARS_RADIUS)

    assert gale is not None and gale.name == "Gale crater"

    # 1 deg off (~59 km): still inside its 77 km.
    assert catalog.site_at("mars", -4.6, 137.8, MARS_RADIUS).id == "gale"

    # Far away: no site, but the nearest is named.
    assert catalog.site_at("mars", 40.0, 0.0, MARS_RADIUS) is None

    site, distance = catalog.nearest("mars", -5.4, 140.0, MARS_RADIUS)

    assert site.id == "gale" and distance == pytest.approx(math.radians(2.2) * MARS_RADIUS * math.cos(math.radians(5.4)), rel=0.01)

    # Overlapping sites: the smallest wins (Hadley inside Imbrium's reach? the pole's).
    assert catalog.site_at("moon", -89.95, 50.0, 1_737_400.0).id == "lunar_south_pole"

    # Harder bodies are worth more.
    assert catalog.factor("earth") < catalog.factor("moon") < catalog.factor("mars") < catalog.factor("pluto")
    assert catalog.factor("nowhere") == catalog.default_factor

    # Every site's body exists, and its numbers are sane.
    from planet.bodies import load_presets

    presets = load_presets()

    for body, sites in catalog.sites.items():

        assert body in presets, body

        for item in sites:
            assert -180.0 <= item.longitude <= 180.0 and item.radius > 0.0 and item.value > 0.0 and item.about


def test_angular_distance():

    assert angular_distance(0, 0, 0, 90) == pytest.approx(math.pi / 2)
    assert angular_distance(90, 0, -90, 0) == pytest.approx(math.pi)
    assert angular_distance(10, 179, 10, -179) == pytest.approx(math.radians(2.0) * math.cos(math.radians(10.0)), rel=1e-3)


# =========================================================
# Regions
# =========================================================

def test_regions_from_heights():

    heights = np.linspace(-5_000.0, 5_000.0, 1001)

    levels = elevation_levels(heights, has_sea=False)

    assert levels[0] == pytest.approx(-2_000.0) and levels[1] == pytest.approx(3_000.0)

    assert region_of(80.0, 0.0, levels) is NORTH_POLE
    assert region_of(-80.0, 0.0, levels) is SOUTH_POLE
    assert region_of(10.0, -4_000.0, levels) is LOWLANDS
    assert region_of(10.0, 0.0, levels) is PLAINS
    assert region_of(10.0, 4_500.0, levels) is HIGHLANDS

    # With a sea, below it is the sea, and the land's own
    # range sets the levels.
    sea_levels = elevation_levels(heights, has_sea=True)

    assert sea_levels[0] > 0.0

    assert region_of(10.0, -100.0, sea_levels, has_sea=True) is SEA

    # No map: the poles and the rest.
    assert region_of(10.0, None, None).id == "surface"


def test_mars_lowlands_and_highlands_from_mola():

    from planet.maps import load_elevation_map

    try:
        mola = load_elevation_map("mola_megdr_16")
    except (OSError, ValueError):
        pytest.skip("MOLA not downloaded")

    if mola is None:
        pytest.skip("MOLA not downloaded")

    levels = elevation_levels(mola.faces[:, 1:-1:4, 1:-1:4], has_sea=False)

    def height(latitude, longitude):
        return float(mola.sample(direction_of(latitude, longitude)[None, :])[0])

    # Hellas's floor is low; Olympus Mons's summit high;
    # the northern plains (Vastitas Borealis) low.
    assert region_of(-42.4, height(-42.4, 70.5), levels) is LOWLANDS
    assert region_of(18.65, height(18.65, -133.8), levels) is HIGHLANDS
    assert region_of(60.0, height(60.0, 30.0), levels) is LOWLANDS


# =========================================================
# Survey
# =========================================================

def test_directions_round_trip():

    for latitude, longitude in ((0.0, 0.0), (45.0, 90.0), (-30.0, -120.0), (89.0, 10.0)):

        lat, lon = latitude_longitude(direction_of(latitude, longitude))

        assert lat[0] == pytest.approx(latitude) and lon[0] == pytest.approx(longitude)


def test_mask_marks_a_swath():

    mask = SurveyMask()

    assert mask.coverage() == 0.0

    # A track along the equator, 2 deg half width.
    longitudes = np.linspace(-180.0, 180.0, 721)

    newly = mask.mark(np.zeros_like(longitudes), longitudes, math.radians(2.0))

    assert newly > 0 and mask.version == 1

    # ~4-5 deg of latitude wide, all the way round: ~4% of
    # the sphere (the band's area is sin(2.5) ~ 4.4%).
    assert 0.03 < mask.coverage() < 0.06

    assert mask.is_mapped(0.0, 100.0) and mask.is_mapped(1.5, -20.0)
    assert not mask.is_mapped(10.0, 100.0)

    # Again: nothing new.
    assert mask.mark([0.0], [0.0], math.radians(2.0)) == 0

    # Over a pole: every longitude round it.
    mask.mark([89.9], [0.0], math.radians(1.0))

    assert mask.is_mapped(89.6, 170.0) and mask.is_mapped(89.6, -90.0)


def test_mask_weights_blend_and_round_trip():

    mask = SurveyMask()

    mask.mark([0.0], [0.0], math.radians(5.0))

    inside = direction_of(0.0, 0.0)
    outside = direction_of(0.0, 40.0)

    weights = mask.weights(np.stack([inside, outside]))

    assert weights[0] == pytest.approx(1.0) and weights[1] == pytest.approx(0.0)

    # Smooth across the edge: some point between is partly in.
    edge = mask.weights(np.stack([direction_of(0.0, longitude) for longitude in np.linspace(3.0, 8.0, 51)]))

    assert np.any((edge > 0.05) & (edge < 0.95))
    assert np.all(np.diff(edge) <= 1e-9)

    # Saved and back.
    again = SurveyMask.from_text(mask.to_text())

    assert np.array_equal(again.cells, mask.cells)

    # A signature changes when the cells round the point do.
    before = mask.signature(np.stack([direction_of(0.0, 20.0)]))

    mask.mark([0.0], [20.0], math.radians(0.5))

    assert mask.signature(np.stack([direction_of(0.0, 20.0)])) != before

    # And not when far ones do.
    here = mask.signature(np.stack([direction_of(0.0, 20.0)]))

    mask.mark([-60.0], [-100.0], math.radians(0.5))

    assert mask.signature(np.stack([direction_of(0.0, 20.0)])) == here


def test_mapping_gates_the_sharp_heights():

    from planet.maps import ElevationMap

    class Detail:
        def sample(self, directions):
            return np.full(len(directions), 100.0)

    dataset = type("Dataset", (), {"detail": "sharp"})()

    elevation = ElevationMap.__new__(ElevationMap)
    elevation.dataset = dataset
    elevation.faces = np.zeros((6, 6, 6), dtype=np.int16)
    elevation.size = 4
    elevation._detail = Detail()

    radius = 1_000_000.0

    directions = np.stack([direction_of(0.0, 0.0), direction_of(0.0, 90.0)])

    fine = elevation.resolution(radius) * 0.1

    # Ungated: the sharp heights everywhere.
    np.testing.assert_allclose(elevation.sample(directions, fine, radius), [100.0, 100.0])

    mask = SurveyMask()
    mask.mark([0.0], [0.0], math.radians(5.0))

    # Gated: only where mapped.
    np.testing.assert_allclose(elevation.sample(directions, fine, radius, mask.weights), [100.0, 0.0])

    # Coarse samples take none either way.
    np.testing.assert_allclose(elevation.sample(directions, elevation.resolution(radius) * 2.0, radius, mask.weights), [0.0, 0.0])


def test_registry():

    survey.clear()

    assert survey.mask_for("mars") is None

    mask = SurveyMask()

    survey.register("mars", mask)

    assert survey.mask_for("mars") is mask

    survey.clear()


# =========================================================
# Record
# =========================================================

def test_results_are_worth_less_each_time():

    record = ScienceRecord()

    result = ScienceResult(subject_key("samples", "mars", "landed", "site:gale"), "Samples", cap=100.0)

    assert record.worth(result) == pytest.approx(75.0)
    assert record.worth(result, transmitted=True) == pytest.approx(37.5)

    assert record.bank(result) == pytest.approx(75.0)

    # 75% of the rest.
    assert record.worth(result) == pytest.approx(18.75)

    record.bank(result, transmitted=True)

    assert record.points == pytest.approx(75.0 + 0.5 * 18.75)

    # Never past the cap.
    for _ in range(50):
        record.bank(result)

    assert record.points <= 100.0 + 1e-9


def test_mapping_credit_and_saving(tmp_path):

    record = ScienceRecord()

    mask = record.survey("moon")

    longitudes = np.linspace(-180.0, 180.0, 721)

    mask.mark(np.zeros_like(longitudes), longitudes, math.radians(2.0))

    gained = record.credit_mapping("moon", 3.0)

    percent = math.floor(mask.coverage() * 100.0)

    assert gained == pytest.approx(percent * 3.0)

    # Once.
    assert record.credit_mapping("moon", 3.0) == 0.0

    path = tmp_path / "progress.json"

    record.subjects["x"] = 0.5

    record.save(path)

    loaded = ScienceRecord.load(path)

    assert loaded.points == pytest.approx(record.points)
    assert loaded.subjects == {"x": 0.5}
    assert np.array_equal(loaded.surveys["moon"].cells, mask.cells)

    # Nothing saved, or broken: a fresh record.
    assert ScienceRecord.load(tmp_path / "missing.json").points == 0.0

    (tmp_path / "broken.json").write_text("{", encoding="utf-8")

    assert ScienceRecord.load(tmp_path / "broken.json").points == 0.0


def test_situations():

    assert situation_of(True, True, 0.0, MARS_RADIUS) == "landed"
    assert situation_of(False, True, 5_000.0, MARS_RADIUS) == "flying"
    assert situation_of(False, False, 300_000.0, MARS_RADIUS) == "space_low"
    assert situation_of(False, False, 2_000_000.0, MARS_RADIUS) == "space_high"


# =========================================================
# Parts
# =========================================================

def test_experiment_parts():

    parts = load_parts()

    camera = parts["survey_camera"].experiment

    assert camera.maps and set(camera.situations) == {"space_low", "space_high"}

    assert parts["sample_kit"].experiment.situations == ("landed",)
    assert parts["air_sampler"].experiment.needs_air
    assert parts["capsule"].experiment.id == "crew_report"

    with pytest.raises(PartError):
        parse_part({
            "id": "x", "kind": "science", "mass": 10,
            "shape": {"radius_top": 0.1, "radius_bottom": 0.1, "height": 0.1},
            "experiment": {"id": "x", "situations": ["underwater"]}
        })


def test_explorer_design_carries_the_instruments():

    from vessel.design import DesignDraft
    from vessel.parts import load_design
    from vessel.vessel import Vessel

    parts = load_parts()

    design = load_design("explorer", parts)

    vessel = Vessel.from_design(design)

    ids = {state.part.experiment.id for state in vessel.parts if state.part.experiment is not None}

    assert ids == {"crew_report", "imaging", "samples", "air"}

    draft = DesignDraft.from_design(design)

    assert draft.problems() == [] and draft.warnings() == []

    # The capsule (and its instruments) is what comes home.
    for _ in range(4):
        vessel.stage()

    assert {state.part.id for state in vessel.parts} >= {"capsule", "survey_camera", "sample_kit", "air_sampler", "heat_shield_2m5"}


# =========================================================
# Running experiments (systems/science_system.py)
# =========================================================

def test_running_recovering_and_transmitting():

    from types import SimpleNamespace

    from science.regions import Region
    from systems.science_system import Place, ScienceSystem
    from vessel.parts import load_design
    from vessel.vessel import Vessel

    catalog = load_sites()

    system = ScienceSystem(catalog, ScienceRecord(), path=None)

    vessel = Vessel.from_design(load_design("explorer", load_parts()))

    component = SimpleNamespace(vessel=vessel, science=[])

    gale = next(site for site in catalog.of("mars") if site.id == "gale")

    landed = Place(
        body=1, body_id="mars", body_name="Mars", situation="landed",
        region=Region("site:gale", gale.name, gale.value), site=gale,
        latitude=gale.latitude, longitude=gale.longitude, altitude=0.0, nearest=(gale, 0.0)
    )

    assert landed.description == "landed at Gale crater, Mars"

    notices = system.run(component, landed, has_air=True)

    # Crew report, samples and air: the camera only works in
    # space.
    assert len(component.science) == 3 and len(notices) == 3

    status = {item.name: item for item in system.status(component, landed, True)}

    assert not status["Surface imaging"].works and status["Surface samples"].held

    # Samples at Gale: 30 x Mars 6 x the site 3, 75% of it.
    samples = next(result for result in component.science if result.subject.startswith("samples"))

    assert system.record.worth(samples) == pytest.approx(30.0 * 6.0 * 3.0 * 0.75)

    # Again here: nothing new to add.
    system.run(component, landed, has_air=True)

    assert len(component.science) == 3

    # Transmit one (half), recover the rest.
    first = system.record.worth(component.science[0], transmitted=True)

    assert system.transmit(component, 0) == pytest.approx(first)

    rest = sum(system.record.worth(result) for result in component.science)

    assert system.recover(component) == pytest.approx(rest)

    assert component.science == [] and system.record.points == pytest.approx(first + rest)

    # No air: the air sampler waits.
    airless = Place(
        body=2, body_id="moon", body_name="Moon", situation="landed",
        region=Region("plains", "Plains"), site=None,
        latitude=0.0, longitude=0.0, altitude=0.0, nearest=None
    )

    assert airless.description == "landed on Moon's plains"

    system.run(component, airless, has_air=False)

    assert not any(result.subject.startswith("air") for result in component.science)

    high = Place(
        body=1, body_id="mars", body_name="Mars", situation="space_high",
        region=Region("global", ""), site=None,
        latitude=0.0, longitude=0.0, altitude=5.0e6, nearest=None
    )

    assert high.description == "in space high over Mars"
