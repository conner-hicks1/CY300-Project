import numpy as np

from audio.output import RATE


# =========================================================
# Sounds, Made Here
# =========================================================
#
# Every sound the game plays is synthesized when it starts
# (no recordings): shaped noise and tones, numpy.
#
#   loops      seamless (noise shaped in the frequency domain
#              is periodic): an engine's roar in air and its
#              muffled rumble through the structure, the wind,
#              the hull creaking under the air's load, the
#              plasma of coming in, a rotor's buzz, a wheel
#              motor's whine and the ground under the wheels
#              (gravel; crunching ice), the RCS thrusters'
#              hiss; and the music (make_music)
#   one-shots  staging, a separator's clank, a parachute
#              opening, touching down, breaking up, docking,
#              and the interface's click, chimes and alarms
#              (the landing burn's, a debris pass's)
#
# All mono, -1..1 (the mixer pans and mixes them).


def _rng(seed: int):

    return np.random.default_rng(seed)


def shaped_noise(
    seconds: float,
    seed: int,
    low: float,
    high: float,
    slope: float = 1.0,
    peak: float | None = None,
    width: float = 0.5
) -> np.ndarray:
    """A seamless loop of noise between `low` and `high` Hz, falling as 1/f^slope (a bump at `peak`, if given)."""

    count = int(seconds * RATE)

    rng = _rng(seed)

    spectrum = rng.normal(size=count // 2 + 1) + 1j * rng.normal(size=count // 2 + 1)

    frequency = np.fft.rfftfreq(count, 1.0 / RATE)

    shape = np.zeros_like(frequency)

    band = (frequency >= low) & (frequency <= high)

    shape[band] = 1.0 / np.maximum(frequency[band], 1.0) ** (0.5 * slope)

    # Soft band edges.
    shape *= np.clip((frequency - 0.7 * low) / (0.3 * low + 1e-9), 0.0, 1.0)
    shape *= np.clip((1.3 * high - frequency) / (0.3 * high), 0.0, 1.0)

    if peak is not None:
        shape *= 1.0 + 3.0 * np.exp(-((np.log(np.maximum(frequency, 1.0)) - np.log(peak)) / width) ** 2)

    signal = np.fft.irfft(spectrum * shape, count)

    return _normal(signal)


def _normal(
    signal: np.ndarray,
    level: float = 0.9
) -> np.ndarray:

    top = float(np.max(np.abs(signal)))

    return (signal / top * level).astype(np.float32) if top > 0.0 else signal.astype(np.float32)


def _decay(
    seconds: float,
    tau: float,
    attack: float = 0.004
) -> np.ndarray:

    t = np.arange(int(seconds * RATE)) / RATE

    return (np.minimum(t / attack, 1.0) * np.exp(-t / tau)).astype(np.float32)


def _tone(
    seconds: float,
    frequency,
    shape: str = "sine"
) -> np.ndarray:

    t = np.arange(int(seconds * RATE)) / RATE

    phase = 2.0 * np.pi * np.cumsum(np.broadcast_to(frequency, t.shape)) / RATE

    if shape == "square":
        return np.sign(np.sin(phase)).astype(np.float32) * 0.6

    if shape == "saw":
        return (2.0 * ((phase / (2.0 * np.pi)) % 1.0) - 1.0).astype(np.float32)

    return np.sin(phase).astype(np.float32)


def _loop_crackle(
    seconds: float,
    seed: int,
    rate: float
) -> np.ndarray:
    """Sharp random pops (an engine's crackle, gravel), looped."""

    count = int(seconds * RATE)

    rng = _rng(seed)

    pops = np.zeros(count, dtype=np.float32)

    where = rng.integers(0, count, int(rate * seconds))

    pops[where] = rng.uniform(-1.0, 1.0, len(where))

    # Each pop a short ringing burst.
    kernel = _decay(0.006, 0.0015) * rng.normal(size=int(0.006 * RATE)).astype(np.float32)

    filtered = np.real(np.fft.ifft(np.fft.fft(pops) * np.fft.fft(kernel, count)))

    return _normal(filtered)


def _creak(
    seconds: float,
    seed: int
) -> np.ndarray:
    """Metal groaning under load: slow gliding tones, a few a second, looped."""

    count = int(seconds * RATE)

    rng = _rng(seed)

    out = np.zeros(count, dtype=np.float32)

    for _ in range(int(seconds * 1.6)):

        length = rng.uniform(0.15, 0.6)

        size = int(length * RATE)

        start = int(rng.integers(0, count))

        base = rng.uniform(70.0, 220.0)

        glide = np.linspace(base, base * rng.uniform(0.8, 1.25), size)

        # A rough groan: a saw, its harmonics wobbling.
        groan = _tone(length, glide, "saw") * 0.5 + _tone(length, glide * 2.01) * 0.3

        envelope = np.sin(np.linspace(0.0, np.pi, size)) ** 2

        piece = (groan * envelope * rng.uniform(0.4, 1.0)).astype(np.float32)

        # (Round the end into the start: a seamless loop.)
        index = (start + np.arange(size)) % count

        np.add.at(out, index, piece)

    strain = shaped_noise(seconds, seed + 1, 40.0, 700.0, slope=1.5)

    return _normal(out + 0.12 * strain)


def make_loops() -> dict[str, np.ndarray]:

    loops = {}

    # A rocket in air: a deep, broad roar with a crackle.
    roar = shaped_noise(3.0, 1, 18.0, 9_000.0, slope=1.15, peak=90.0, width=0.9)
    loops["engine"] = _normal(roar + 0.25 * _loop_crackle(3.0, 2, 260.0))

    # Through the structure (in vacuum): a muffled rumble.
    loops["rumble"] = shaped_noise(3.0, 3, 15.0, 400.0, slope=1.6, peak=55.0, width=0.6)

    # Wind over the hull, gusting.
    wind = shaped_noise(4.0, 4, 120.0, 2_500.0, slope=1.4, peak=420.0, width=0.7)
    t = np.arange(len(wind)) / RATE
    loops["wind"] = _normal(wind * (0.7 + 0.3 * np.sin(2.0 * np.pi * t / 4.0)))

    # Coming in: a low tearing rumble and a hiss.
    loops["plasma"] = _normal(
        shaped_noise(3.0, 5, 25.0, 900.0, slope=1.3)
        + 0.45 * shaped_noise(3.0, 6, 2_000.0, 7_000.0, slope=0.3)
    )

    # A rotor: two blades at ~2,400 rpm, 80 Hz and its
    # harmonics, chopping air.
    buzz = _tone(2.0, 80.0, "saw") * 0.6 + _tone(2.0, 160.0) * 0.3
    chop = 0.8 + 0.2 * np.sin(2.0 * np.pi * 40.0 * np.arange(len(buzz)) / RATE)
    loops["rotor"] = _normal(buzz * chop + 0.3 * shaped_noise(2.0, 7, 300.0, 4_000.0, slope=0.8))

    # A wheel motor's whine (and its gears).
    loops["wheels"] = _normal(0.6 * _tone(2.0, 310.0) + 0.25 * _tone(2.0, 620.0, "saw") + 0.3 * shaped_noise(2.0, 9, 80.0, 600.0))

    # The ground under the wheels: gravel, stones popping
    # and grinding; ice, a sharp brittle crunch.
    loops["gravel"] = _normal(_loop_crackle(2.0, 8, 900.0) * 0.8 + 0.35 * shaped_noise(2.0, 11, 60.0, 900.0, slope=1.2))
    loops["ice"] = _normal(_loop_crackle(2.0, 12, 1_600.0) * 0.6 + 0.4 * shaped_noise(2.0, 13, 1_800.0, 9_000.0, slope=0.4))

    # The hull creaking under the load: groans gliding low,
    # now and then, over a faint strain.
    loops["creak"] = _creak(6.0, 14)

    # RCS thrusters: a sharp hiss.
    loops["rcs"] = shaped_noise(1.5, 10, 2_500.0, 11_000.0, slope=0.2)

    return loops


def make_shots() -> dict[str, np.ndarray]:

    shots = {}

    # Staging: a thump, a bang, a ringing of metal.
    seconds = 1.2
    ring = sum(_tone(seconds, f) * a for f, a in ((410.0, 0.5), (667.0, 0.35), (1_090.0, 0.25), (1_530.0, 0.15)))
    burst = shaped_noise(seconds, 20, 40.0, 6_000.0, slope=0.9)
    shots["stage"] = _normal(
        _tone(seconds, 58.0) * _decay(seconds, 0.22) * 1.2
        + burst * _decay(seconds, 0.07)
        + ring * _decay(seconds, 0.35) * 0.5
    )

    # A separator: a short clank.
    seconds = 0.6
    clank = sum(_tone(seconds, f) * a for f, a in ((520.0, 0.5), (830.0, 0.4), (1_370.0, 0.3)))
    shots["clank"] = _normal(clank * _decay(seconds, 0.12) + shaped_noise(seconds, 21, 400.0, 6_000.0) * _decay(seconds, 0.02))

    # A parachute: a pop and a rush of cloth.
    seconds = 1.6
    t = np.arange(int(seconds * RATE)) / RATE
    rush = shaped_noise(seconds, 22, 200.0, 3_000.0, slope=0.8) * np.exp(-((t - 0.45) / 0.35) ** 2)
    shots["chute"] = _normal(rush + shaped_noise(seconds, 23, 60.0, 2_000.0) * _decay(seconds, 0.03) * 0.8)

    # Touching down: a heavy thud.
    seconds = 0.7
    shots["thud"] = _normal(_tone(seconds, 42.0) * _decay(seconds, 0.16) + shaped_noise(seconds, 24, 30.0, 1_500.0, slope=1.2) * _decay(seconds, 0.06))

    # Breaking up: a long roar of tearing metal.
    seconds = 2.5
    shots["crash"] = _normal(
        shaped_noise(seconds, 25, 25.0, 8_000.0, slope=1.0) * _decay(seconds, 0.5)
        + _tone(seconds, 35.0) * _decay(seconds, 0.6)
        + _loop_crackle(seconds, 26, 400.0)[: int(seconds * RATE)] * _decay(seconds, 0.4) * 0.5
    )

    # Docking: a latch's clunk.
    seconds = 0.5
    shots["dock"] = _normal(sum(_tone(seconds, f) * a for f, a in ((180.0, 0.6), (300.0, 0.4), (720.0, 0.25))) * _decay(seconds, 0.08))

    # A bump: a dull knock.
    seconds = 0.4
    shots["bump"] = _normal(_tone(seconds, 90.0) * _decay(seconds, 0.07) + shaped_noise(seconds, 27, 60.0, 1_200.0) * _decay(seconds, 0.03) * 0.6)

    # The interface: a click; a chime up (good), down (a
    # warning); an alarm.
    seconds = 0.03
    shots["click"] = _normal(shaped_noise(seconds, 28, 2_000.0, 9_000.0, slope=0.0) * _decay(seconds, 0.006), 0.5)

    def notes(*pitches, length=0.16, tau=0.25):
        parts = [_tone(length + 0.3, pitch) * _decay(length + 0.3, tau) for pitch in pitches]
        out = np.zeros(int((length * (len(pitches) - 1) + length + 0.3) * RATE), dtype=np.float32)
        for index, part in enumerate(parts):
            start = int(index * length * RATE)
            out[start:start + len(part)] += part
        return _normal(out, 0.5)

    shots["chime"] = notes(659.3, 880.0)
    shots["warn"] = notes(880.0, 698.5)

    seconds = 0.9
    beeps = _tone(seconds, 880.0, "square") * ((np.arange(int(seconds * RATE)) / RATE % 0.3) < 0.15)
    shots["alarm"] = _normal(beeps, 0.45)

    # The landing burn: start it now (two quick high beeps).
    seconds = 0.5
    t = np.arange(int(seconds * RATE)) / RATE
    shots["brake"] = _normal(_tone(seconds, 1_320.0, "square") * (((t % 0.18) < 0.09) & (t < 0.36)), 0.45)

    # Debris passing close: a sonar's ping.
    seconds = 1.4
    shots["proximity"] = _normal(
        (_tone(seconds, 1_480.0) + 0.3 * _tone(seconds, 2_960.0)) * _decay(seconds, 0.32, 0.002),
        0.5
    )

    # Power gone: a falling tone.
    seconds = 0.8
    shots["power down"] = _normal(_tone(seconds, np.linspace(620.0, 140.0, int(seconds * RATE))) * _decay(seconds, 0.35), 0.5)

    return shots


# ---------------------------------------------------------
# Music
# ---------------------------------------------------------

# A minor-ish drift, two chords a phrase (Hz of each chord's
# notes).
CHORDS = (
    (110.00, 164.81, 220.00, 261.63, 329.63),     # A minor
    (87.31, 130.81, 174.61, 220.00, 261.63),      # F major
    (130.81, 196.00, 261.63, 329.63, 392.00),     # C major
    (98.00, 146.83, 196.00, 246.94, 293.66),      # G major
    (110.00, 164.81, 220.00, 261.63, 392.00),     # A minor (add 7)
    (73.42, 110.00, 146.83, 174.61, 220.00),      # D minor
    (87.31, 130.81, 174.61, 220.00, 329.63),      # F major (add 7)
    (82.41, 123.47, 164.81, 207.65, 246.94),      # E major
)

CHORD_SECONDS = 8.0


def make_music() -> np.ndarray:
    """A slow ambient loop (64 s): soft pads changing chord every 8 s, a shimmer over them; seamless."""

    seconds = CHORD_SECONDS * len(CHORDS)

    # The pads made at a quarter of the rate (their notes
    # and overtones are all low), then read up to it.
    slow = RATE // 4

    count = int(seconds * slow)

    t = np.arange(count) / slow

    pad = np.zeros(count, dtype=np.float64)

    for index, chord in enumerate(CHORDS):

        # Each chord swells in and out over two of its slots
        # (overlapping the next), wrapping round at the end.
        centre = (index + 0.5) * CHORD_SECONDS

        offset = (t - centre + 0.5 * seconds) % seconds - 0.5 * seconds

        live = np.abs(offset) < CHORD_SECONDS

        envelope = np.cos(np.pi * offset[live] / (2.0 * CHORD_SECONDS)) ** 2

        when = t[live]

        def whole(frequency):
            # (Whole cycles in the loop: it goes round with no
            # seam; a few hundredths of a hertz, unheard.)
            return round(frequency * seconds) / seconds

        for number, note in enumerate(chord):

            # Two detuned voices a note (slowly beating), a
            # soft octave over them.
            voice = sum(np.sin(2.0 * np.pi * whole(note + detune) * when) for detune in (-0.12, 0.12)) + 0.2 * np.sin(2.0 * np.pi * whole(2.0 * note) * when)

            pad[live] += voice * envelope * (0.9 if number == 0 else 0.55)

    full = int(seconds * RATE)

    # (Wrapping round: the end of the loop runs into its start.)
    where = np.arange(full) * (count / full)

    whole = where.astype(np.int64)
    part = where - whole

    out = (pad[whole] * (1.0 - part) + pad[(whole + 1) % count] * part).astype(np.float32)

    # A shimmer: high, airy noise breathing slowly.
    shimmer = shaped_noise(seconds, 40, 2_000.0, 7_000.0, slope=1.0)

    breath = (0.5 + 0.5 * np.sin(2.0 * np.pi * np.arange(full) / RATE / 16.0)).astype(np.float32)

    out += 0.05 * shimmer[:full] * breath * float(np.max(np.abs(out))) / 0.9

    return _normal(out, 0.6)
