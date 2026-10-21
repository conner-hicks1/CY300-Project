import numpy as np
import pytest

from audio import synth
from audio.mixer import EASE, Mixer, SoundSettings
from audio.output import RATE, NullOutput
from audio.sound import Scene, SoundEngine, levels


@pytest.fixture(scope="module")
def sounds():

    return synth.make_loops(), synth.make_shots()


def test_every_sound_is_made_in_range(sounds):

    loops, shots = sounds

    assert {"engine", "rumble", "wind", "plasma", "rotor", "wheels", "gravel", "ice", "creak", "rcs"} <= set(loops)
    assert {"stage", "clank", "chute", "thud", "crash", "dock", "bump", "click", "chime", "warn", "alarm", "brake", "proximity"} <= set(shots)

    for name, samples in {**loops, **shots}.items():

        assert samples.dtype == np.float32 and samples.ndim == 1, name
        assert 0.1 < float(np.abs(samples).max()) <= 1.0, name


def test_noise_loops_go_round_without_a_click(sounds):

    loops, _ = sounds

    for name in ("engine", "rumble", "wind", "plasma", "gravel", "ice", "creak", "rcs"):

        samples = loops[name]

        # The jump from the end back to the start, no bigger
        # than the loop's own steps.
        steps = np.abs(np.diff(samples))

        assert abs(float(samples[0] - samples[-1])) <= float(np.percentile(steps, 99.9)) + 1e-6, name


def test_loops_ease_to_their_levels(sounds):

    loops, shots = sounds

    mixer = Mixer(loops, shots, SoundSettings(master=1.0))

    assert float(np.abs(mixer.render(512)).max()) == 0.0

    mixer.set_loop("engine", 1.0)

    first = mixer.render(256)

    # Up gradually (no click): EASE is the time constant,
    # there after a few of them.
    assert 0.0 < mixer.level_of("engine") < 1.0
    assert float(np.abs(first[:16]).max()) < float(np.abs(first[-64:]).max())

    for _ in range(int(5 * EASE * RATE / 256)):
        mixer.render(256)

    assert mixer.level_of("engine") == pytest.approx(1.0, abs=0.02)

    mixer.silence()

    for _ in range(int(6 * EASE * RATE / 256)):
        out = mixer.render(256)

    assert mixer.level_of("engine") < 0.01 and float(np.abs(out).max()) < 0.01


def test_a_one_shot_plays_and_ends(sounds):

    loops, shots = sounds

    mixer = Mixer(loops, shots, SoundSettings(master=1.0))

    mixer.play("stage")

    assert float(np.abs(mixer.render(2048)).max()) > 0.1

    for _ in range(int(len(shots["stage"]) / 2048) + 2):
        mixer.render(2048)

    assert float(np.abs(mixer.render(2048)).max()) == 0.0

    # Panned hard left: the right channel silent.
    mixer.play("clank", pan=-1.0)

    out = mixer.render(4096)

    assert float(np.abs(out[:, 1]).max()) < 1e-4 < float(np.abs(out[:, 0]).max())

    # Muted: nothing.
    mixer.settings.muted = True
    mixer.play("crash")

    assert float(np.abs(mixer.render(2048)).max()) == 0.0


def _scene(**changes):

    values = dict(thrust=2.2e6, density=1.2, dynamic_pressure=0.0, heat_flux=0.0, distance=30.0, rotor=0.0, wheels=0.0, rcs=False)

    values.update(changes)

    return Scene(**values)


def test_engines_roar_in_air_and_rumble_in_vacuum():

    air = levels(_scene())
    vacuum = levels(_scene(density=0.0))
    structure = levels(_scene(density=0.0, structure=True))
    far = levels(_scene(density=0.0, distance=500.0, structure=True))
    distant_air = levels(_scene(distance=2_000.0))

    assert air["engine"][0] > 0.4 and vacuum["engine"][0] == 0.0

    # In vacuum: silent; or (a setting) only the rumble
    # through the structure, and only close.
    assert vacuum["rumble"][0] == 0.0
    assert structure["rumble"][0] > 0.3 and far["rumble"][0] < 0.05

    # Far off through air: fainter.
    assert distant_air["engine"][0] < 0.25 * air["engine"][0]

    # A small engine: quieter, higher.
    small = levels(_scene(thrust=6e4))

    assert small["engine"][0] < air["engine"][0] and small["engine"][1] > air["engine"][1]

    # Wind with the dynamic pressure, plasma with the heat.
    assert levels(_scene(dynamic_pressure=25_000.0))["wind"][0] > 0.5
    assert levels(_scene(heat_flux=2e6))["plasma"][0] > 0.4

    # The hull creaks under the load.
    assert levels(_scene(dynamic_pressure=30_000.0))["creak"][0] > 0.2 and air["creak"][0] == 0.0

    # Wheels: gravel, or ice on an icy world.
    rolling = levels(_scene(wheels=0.8, thrust=0.0))
    icy = levels(_scene(wheels=0.8, thrust=0.0, surface="ice"))

    assert rolling["gravel"][0] > 0.2 and rolling["ice"][0] == 0.0
    assert icy["ice"][0] > 0.2 and icy["gravel"][0] == 0.0

    # Paused or in fast warp: quiet, but for the music.
    assert all(level == 0.0 for level, _ in levels(_scene(paused=True)).values())
    assert all(level == 0.0 for name, (level, _) in levels(_scene(warping=True, music=0.6)).items() if name != "music")
    assert levels(_scene(warping=True, music=0.6))["music"][0] == 0.6


def test_the_music_goes_round_by_its_own_level():

    music = synth.make_music()

    assert len(music) == 64 * RATE and abs(float(music[0] - music[-1])) <= float(np.abs(np.diff(music)).max())

    mixer = Mixer({}, {}, SoundSettings(master=1.0, music=1.0, effects=0.0))

    mixer.add_loop("music", music)

    mixer.set_loop("music", 1.0)

    for _ in range(int(5 * EASE * RATE / 1024)):
        out = mixer.render(1024)

    # (Effects off: the music plays on.)
    assert float(np.abs(out).max()) > 0.1

    mixer.settings.music = 0.0

    assert float(np.abs(mixer.render(1024)).max()) == 0.0


def test_without_sound_nothing_is_made(tmp_path):

    engine = SoundEngine(enabled=False, path=tmp_path / "sound.json")

    assert engine.mixer is None and isinstance(engine.output, NullOutput) and not engine.playing

    # Calls are harmless.
    engine.update(_scene())
    engine.play("stage")
    engine.notice("bad")

    engine.settings.master = 0.3
    engine.settings.muted = True

    engine.save()

    again = SoundEngine(enabled=False, path=tmp_path / "sound.json")

    assert again.settings.master == pytest.approx(0.3) and again.settings.muted

    engine.close()
