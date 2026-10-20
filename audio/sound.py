import json
import math
import threading

from dataclasses import asdict, dataclass
from pathlib import Path

from core.logger import Logger

from audio.mixer import Mixer, SoundSettings
from audio.output import NullOutput, open_output
from audio import synth


# =========================================================
# The Game's Sound
# =========================================================
#
# What the flight sounds like, heard where the camera is:
#
#   engines   a roar in air (louder with the thrust and the
#             air's density, deeper for the big ones), and in
#             vacuum only what carries through the structure:
#             a muffled rumble, and only close to the vessel
#   wind      over the hull: with the dynamic pressure
#   plasma    coming in: with the heating
#   rotors, wheels, RCS: as they work
#
# Far from the camera, through air, sound fades (and arrives
# muffled); in vacuum nothing but the rumble from a vessel
# the camera is on. Events play one-shots: staging, a
# separator, a parachute, touching down, breaking up,
# docking, bumps; notices chime (good), warn, or sound the
# alarm (bad).
#
# The settings (volumes, mute) are kept in
# data/settings/sound.json.

SETTINGS_PATH = Path("data/settings/sound.json")


@dataclass(frozen=True, slots=True)
class Scene:
    """The flight, as the sound needs it (systems/vessel_system.py and the camera give it)."""

    thrust: float                   # N now (engines)
    density: float                  # kg/m^3 outside
    dynamic_pressure: float         # Pa
    heat_flux: float                # W/m^2 on the leading end
    distance: float                 # m, camera to vessel
    rotor: float                    # 0..1 (throttle, with rotors)
    wheels: float                   # 0..1 of top speed driving
    rcs: bool                       # translating
    paused: bool = False
    warping: bool = False           # fast warp: on rails, quiet


def levels(
    scene: Scene
) -> dict[str, tuple[float, float]]:
    """Each loop's (level 0..1, pitch) for a moment of flight."""

    if scene.paused or scene.warping:
        return {name: (0.0, 1.0) for name in ("engine", "rumble", "wind", "plasma", "rotor", "wheels", "rcs")}

    # Air carries sound: thin air, faint (and muffled by the
    # distance); vacuum, nothing outside.
    air = min(scene.density / 1.225, 1.0)

    carry = math.sqrt(air)

    near = 1.0 / (1.0 + scene.distance / 60.0)

    # Through the structure: only on (or very near) it.
    onboard = 1.0 / (1.0 + max(scene.distance - 15.0, 0.0) / 10.0)

    # A big engine: louder and deeper.
    push = min(math.log10(1.0 + scene.thrust / 1_000.0) / 3.5, 1.0)

    engine = push * carry * near * 1.1
    rumble = push * onboard * (1.0 - 0.6 * carry) * 0.8

    deep = 1.1 - 0.35 * push

    wind = min(scene.dynamic_pressure / 30_000.0, 1.0) ** 0.6 * carry * near
    plasma = min(max(math.log10(max(scene.heat_flux, 1.0)) - 5.0, 0.0) / 1.5, 1.0) * near

    return {
        "engine": (engine, deep),
        "rumble": (rumble, 0.9 + 0.2 * push),
        "wind": (wind, 0.8 + 0.5 * min(scene.dynamic_pressure / 30_000.0, 1.0)),
        "plasma": (plasma, 1.0),
        "rotor": (scene.rotor * (0.3 + 0.7 * carry) * near * 0.8, 0.8 + 0.3 * scene.rotor),
        "wheels": (scene.wheels * near * 0.6 * max(carry, 0.3), 0.8 + 0.4 * scene.wheels),
        "rcs": ((0.35 if scene.rcs else 0.0) * max(carry, onboard), 1.0),
    }


# Notice tones (game/hud.py notice_tone) -> a one-shot.
NOTICE_SOUNDS = {"good": "chime", "warn": "warn", "bad": "alarm"}


class SoundEngine:
    """The sounds made (on a thread of their own: a couple of seconds), the output opened, the settings kept."""

    def __init__(
        self,
        enabled: bool = True,
        path: Path = SETTINGS_PATH,
        wait: bool = False
    ):

        self.path = Path(path)

        self.settings = self._load()

        # Until the sounds are made: silent.
        self.mixer: Mixer | None = None
        self.output = NullOutput()

        self._enabled = enabled

        self._ready = threading.Thread(target=self._make, name="sounds", daemon=True)

        # (No sound wanted, headless: nothing made at all.)
        if not enabled:
            return

        self._ready.start()

        if wait:
            self._ready.join()

    def _make(self):

        mixer = Mixer(synth.make_loops(), synth.make_shots(), self.settings)

        self.output = open_output(mixer.render, self._enabled)

        self.mixer = mixer

    @property
    def playing(self) -> bool:

        return self.output.playing

    def update(
        self,
        scene: Scene
    ):

        if self.mixer is None:
            return

        for name, (level, pitch) in levels(scene).items():
            self.mixer.set_loop(name, level, pitch)

    def play(
        self,
        name: str,
        level: float = 1.0,
        pan: float = 0.0,
        alert: bool = False
    ):

        if self.mixer is not None:
            self.mixer.play(name, level, pan, alert)

    def notice(
        self,
        tone: str
    ):
        """A notice's sound by its tone (none for plain ones)."""

        name = NOTICE_SOUNDS.get(tone)

        if name is not None:
            self.play(name, 0.7, 0.0, alert=True)

    def save(self):

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(asdict(self.settings), indent=1), encoding="utf-8")
        except OSError as error:
            Logger.warning("[Audio] Could not save the sound settings: %s", error)

    def _load(self) -> SoundSettings:

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return SoundSettings()

        return SoundSettings(
            master=float(data.get("master", 0.8)),
            effects=float(data.get("effects", 1.0)),
            alerts=float(data.get("alerts", 0.8)),
            muted=bool(data.get("muted", False))
        )

    def close(self):

        if self._ready.is_alive():
            self._ready.join(timeout=5.0)

        self.output.stop()
