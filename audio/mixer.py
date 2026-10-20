import threading

from collections import deque
from dataclasses import dataclass

import numpy as np

from audio.output import RATE


# =========================================================
# The Mixer
# =========================================================
#
# What the output (audio/output.py) plays, block by block
# on its thread:
#
#   loops      each always running, at a level and a pitch the
#              game sets every frame (set_loop): eased toward
#              them over EASE s (no clicks), pitched by
#              playing faster or slower, panned
#   one-shots  played once over the loops (play), each with its
#              level and pan; at most SHOTS at a time
#
# Master, effects and alert levels from the player's
# settings (SoundSettings); a soft limiter at the end (tanh)
# keeps it from clipping.

EASE = 0.12                         # s, a loop's time constant toward a new level
SHOTS = 16


@dataclass(slots=True)
class SoundSettings:

    master: float = 0.8
    effects: float = 1.0
    alerts: float = 0.8
    muted: bool = False


class _Loop:

    __slots__ = ("samples", "position", "level", "target", "pitch", "target_pitch", "pan")

    def __init__(self, samples):

        self.samples = samples
        self.position = 0.0
        self.level = 0.0
        self.target = 0.0
        self.pitch = 1.0
        self.target_pitch = 1.0
        self.pan = 0.0


class Mixer:

    def __init__(
        self,
        loops: dict[str, np.ndarray],
        shots: dict[str, np.ndarray],
        settings: SoundSettings | None = None
    ):

        self.settings = settings or SoundSettings()

        self._loops = {name: _Loop(samples) for name, samples in loops.items()}
        self._shots = shots

        # (samples, position, level, pan, alert) playing.
        self._playing: list[list] = []

        # Asked for from the game's thread.
        self._requests: deque = deque()

        self._lock = threading.Lock()

    # -----------------------------------------------------
    # From the game
    # -----------------------------------------------------

    def set_loop(
        self,
        name: str,
        level: float,
        pitch: float = 1.0,
        pan: float = 0.0
    ):
        """A loop's level (0..1), pitch (a playing rate) and pan (-1 left .. 1 right) from now (eased)."""

        loop = self._loops.get(name)

        if loop is None:
            return

        loop.target = min(max(float(level), 0.0), 1.5)
        loop.target_pitch = min(max(float(pitch), 0.25), 4.0)
        loop.pan = min(max(float(pan), -1.0), 1.0)

    def silence(self):
        """Every loop down (paused, a menu)."""

        for loop in self._loops.values():
            loop.target = 0.0

    def play(
        self,
        name: str,
        level: float = 1.0,
        pan: float = 0.0,
        alert: bool = False
    ):
        """A one-shot (alert: the interface's, by the alerts level)."""

        if name in self._shots:
            self._requests.append((name, float(level), float(pan), alert))

    def level_of(
        self,
        name: str
    ) -> float:
        """A loop's level now (for tests and the developer tools)."""

        loop = self._loops.get(name)

        return loop.level if loop is not None else 0.0

    # -----------------------------------------------------
    # On the audio thread
    # -----------------------------------------------------

    def render(
        self,
        frames: int
    ) -> np.ndarray:
        """The next `frames` of sound: (frames, 2) float32 in -1..1."""

        settings = self.settings

        out = np.zeros((frames, 2), dtype=np.float32)

        with self._lock:

            while self._requests:

                name, level, pan, alert = self._requests.popleft()

                if len(self._playing) >= SHOTS:
                    self._playing.pop(0)

                self._playing.append([self._shots[name], 0, level, pan, alert])

        ease = min(frames / (EASE * RATE), 1.0)

        ramp = np.linspace(0.0, 1.0, frames, dtype=np.float32)

        for loop in self._loops.values():

            start_level = loop.level
            end_level = loop.level + (loop.target - loop.level) * ease

            loop.pitch += (loop.target_pitch - loop.pitch) * ease
            loop.level = end_level

            if start_level < 1e-4 and end_level < 1e-4:
                continue

            samples = loop.samples
            length = len(samples)

            # Pitched: read faster or slower, wrapping round.
            index = (loop.position + np.arange(frames) * loop.pitch) % length

            whole = index.astype(np.int64)
            part = (index - whole).astype(np.float32)

            signal = samples[whole] * (1.0 - part) + samples[(whole + 1) % length] * part

            loop.position = float((loop.position + frames * loop.pitch) % length)

            signal = signal * (start_level + (end_level - start_level) * ramp) * settings.effects

            left, right = _pan(loop.pan)

            out[:, 0] += signal * left
            out[:, 1] += signal * right

        effects = settings.effects
        alerts = settings.alerts

        still = []

        for item in self._playing:

            samples, position, level, pan, alert = item

            chunk = samples[position:position + frames]

            if len(chunk) == 0:
                continue

            left, right = _pan(pan)

            gain = level * (alerts if alert else effects)

            out[: len(chunk), 0] += chunk * gain * left
            out[: len(chunk), 1] += chunk * gain * right

            item[1] = position + frames

            if item[1] < len(samples):
                still.append(item)

        self._playing = still

        if settings.muted:
            return np.zeros((frames, 2), dtype=np.float32)

        # Loops by the effects level, all by the master; a
        # soft limit.
        return np.tanh(out * settings.master * 0.8).astype(np.float32)


def _pan(
    pan: float
) -> tuple[float, float]:
    """Equal-power panning."""

    angle = (pan + 1.0) * 0.25 * np.pi

    return float(np.cos(angle)), float(np.sin(angle))
