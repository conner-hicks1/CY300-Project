import json

from dataclasses import dataclass, field
from pathlib import Path

from core.input import Input
from core.key_codes import Key
from core.logger import Logger


# =========================================================
# Controls: Keys and the Mouse
# =========================================================
#
# The game reads its own keys (Key.Z: full throttle); the
# player may put each on another key (core/input.py
# translates the key pressed into the game's). A key taken
# from another action swaps with it, so every action keeps a
# key and no key does two things. Esc is the menus' and is
# not rebound.
#
# With the mouse's sensitivity, up / down swapped, and the
# interface's size (UI scale), kept in
# data/settings/controls.json.

CONTROLS_PATH = Path("data/settings/controls.json")

# (game key, what it does), by group.
ACTIONS: tuple[tuple[str, tuple[tuple[Key, str], ...]], ...] = (
    ("Flying", (
        (Key.Z, "Full throttle"),
        (Key.X, "Cut the throttle"),
        (Key.LEFT_SHIFT, "Throttle up (free camera: down)"),
        (Key.LEFT_CONTROL, "Throttle down"),
        (Key.SPACE, "Next stage (free camera: up)"),
        (Key.W, "Pitch forward (drive, fly the free camera on)"),
        (Key.S, "Pitch back (reverse, free camera back)"),
        (Key.A, "Yaw left (steer, free camera left)"),
        (Key.D, "Yaw right (steer, free camera right)"),
        (Key.Q, "Roll left"),
        (Key.E, "Roll right"),
        (Key.T, "Stability assist on / off"),
        (Key.F, "Stability assist mode"),
        (Key.G, "Landing legs"),
        (Key.U, "Solar wings"),
        (Key.B, "Dock with the target"),
        (Key.R, "Run the experiments"),
    )),
    ("Thrusters (RCS)", (
        (Key.UP, "Translate forward"),
        (Key.DOWN, "Translate back"),
        (Key.LEFT, "Translate left"),
        (Key.RIGHT, "Translate right"),
        (Key.H, "Translate fore"),
        (Key.N, "Translate aft"),
    )),
    ("Views and panels", (
        (Key.M, "Map"),
        (Key.V, "Free camera"),
        (Key.K, "Science"),
        (Key.C, "Flight computer"),
        (Key.P, "Space programme"),
        (Key.L, "Vessels in flight"),
        (Key.LEFT_BRACKET, "Fly the previous vessel"),
        (Key.RIGHT_BRACKET, "Fly the next vessel"),
    )),
    ("Time and the game", (
        (Key.PERIOD, "Time warp faster"),
        (Key.COMMA, "Time warp slower"),
        (Key.SLASH, "Back to real time"),
        (Key.F1, "Every key on screen"),
        (Key.F2, "Hide the HUD"),
        (Key.F5, "Quicksave"),
        (Key.F9, "Quickload"),
        (Key.F3, "Developer tools"),
    )),
)

BINDABLE: tuple[Key, ...] = tuple(key for _, group in ACTIONS for key, _ in group)

NAMES = {
    Key.SPACE: "Space",
    Key.LEFT_SHIFT: "Left Shift",
    Key.RIGHT_SHIFT: "Right Shift",
    Key.LEFT_CONTROL: "Left Ctrl",
    Key.RIGHT_CONTROL: "Right Ctrl",
    Key.LEFT_ALT: "Left Alt",
    Key.RIGHT_ALT: "Right Alt",
    Key.LEFT_BRACKET: "[",
    Key.RIGHT_BRACKET: "]",
    Key.PERIOD: ".",
    Key.COMMA: ",",
    Key.SLASH: "/",
    Key.SEMICOLON: ";",
    Key.APOSTROPHE: "'",
    Key.MINUS: "-",
    Key.UP: "Up arrow",
    Key.DOWN: "Down arrow",
    Key.LEFT: "Left arrow",
    Key.RIGHT: "Right arrow",
}


def key_name(
    key: Key
) -> str:

    if key in NAMES:
        return NAMES[key]

    name = key.name

    if name.startswith("NUM_"):
        return name[4:]

    if len(name) <= 3:
        return name

    return name.replace("_", " ").title()


def what_does(
    key: Key
) -> str:
    """The action a game key is for."""

    return next((what for _, group in ACTIONS for game_key, what in group if game_key == key), key_name(key))


@dataclass(slots=True)
class ControlSettings:

    # game key -> the key pressed for it (only those moved).
    bindings: dict[Key, Key] = field(default_factory=dict)

    mouse_sensitivity: float = 1.0
    invert_y: bool = False

    ui_scale: float = 1.0

    def key_for(
        self,
        action: Key
    ) -> Key:

        return self.bindings.get(action, action)

    def bind(
        self,
        action: Key,
        key: Key
    ) -> Key | None:
        """Put an action on a key; the action that had it takes this one's old key. Returns that action (or None)."""

        if key == Key.ESCAPE or action not in BINDABLE:
            return None

        old = self.key_for(action)

        if old == key:
            return None

        other = next((game for game in BINDABLE if game != action and self.key_for(game) == key), None)

        self.bindings[action] = key

        if other is not None:
            self.bindings[other] = old

        # (Back where they started: no need to keep them.)
        self.bindings = {game: pressed for game, pressed in self.bindings.items() if game != pressed}

        return other

    def reset_keys(self):

        self.bindings = {}

    def changed(self) -> list[tuple[str, str]]:
        """(what, key) of every action moved off its own key."""

        return [(what_does(game), key_name(pressed)) for game, pressed in self.bindings.items()]

    def apply(self):
        """Into the input: the keys, the mouse."""

        Input.set_bindings(self.bindings)

        Input.mouse_sensitivity = self.mouse_sensitivity
        Input.invert_y = self.invert_y

    # -----------------------------------------------------
    # Kept
    # -----------------------------------------------------

    def save(
        self,
        path: Path | None = CONTROLS_PATH
    ):

        if path is None:
            return

        data = {
            "keys": {game.name: pressed.name for game, pressed in self.bindings.items()},
            "mouse_sensitivity": self.mouse_sensitivity,
            "invert_y": self.invert_y,
            "ui_scale": self.ui_scale,
        }

        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")
        except OSError as error:
            Logger.warning("[Controls] Could not save: %s", error)

    @classmethod
    def load(
        cls,
        path: Path | None = CONTROLS_PATH
    ) -> "ControlSettings":

        if path is None:
            return cls()

        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()

        settings = cls(
            mouse_sensitivity=min(max(float(data.get("mouse_sensitivity", 1.0)), 0.1), 5.0),
            invert_y=bool(data.get("invert_y", False)),
            ui_scale=min(max(float(data.get("ui_scale", 1.0)), 0.75), 1.75)
        )

        for game, pressed in dict(data.get("keys", {})).items():

            try:
                settings.bind(Key[game], Key[pressed])
            except KeyError:
                continue

        return settings
