import math
import random

from pathlib import Path

import numpy as np

from core.input import Input
from core.key_codes import Key
from core.logger import Logger

from ecs.components import NameComponent, VesselComponent

from math3d import quaternion

from vessel import kepler
from vessel.parts import load_design
from vessel.physics import land


# =========================================================
# A Game to Play in Tests
# =========================================================
#
# The real application, headless (a hidden window: no focus,
# no keyboard reaching it; a fixed 1/60 s frame; nothing
# drawn), its saves in a test's own folder. Tests press keys
# and step frames, put vessels where they need them, and
# read the game's state:
#
#   game = Game(tmp_path)
#   game.press(Key.SPACE)
#   game.step(30)
#   game.run_until(lambda: game.vessel.state.landed, 20_000)
#
# One at a time (a GL context each); close() when done (the
# fixture does).

TEST_LOG = "logs/tests.log"

class Game:

    def __init__(
        self,
        saves: Path,
        body: str = "earth",
        sandbox: bool = False,
        design: str | None = None,
        researched: bool = True,
        failures: bool = False,
        terrain: bool = False,
        frame_times: tuple[float, float] | None = None
    ):
        """
        researched: every tech node known (else only the free
        ones: a new career's). failures: parts may fail
        (vessel/reliability.py; off: runs replay the same).
        terrain: the real terrain streamed in and every frame
        drawn, as playing (slow: ~20 frames a second).
        frame_times: each frame between these (s, seeded),
        as an uneven real frame rate; else 1/60 s.
        """

        from application import Application

        # (Its own log: never the game's logs/engine.log.)
        Logger.initialize(log_file=TEST_LOG)

        self.app = Application()

        self.app.initialize(body=body, sandbox=sandbox, headless=True, saves=Path(saves))

        programme = self.app.programme_system

        if researched:
            programme.career.researched = {node.id for node in programme.tree.nodes}

        programme.failures.enabled = failures and not sandbox

        self.frame = 0

        # Played as for real: terrain coming in, frames of
        # uneven length (the game's bugs that only show then).
        self._render = terrain

        if terrain:
            self.app.planet_system.stream = True

        self._frame_times = frame_times

        self._random = random.Random(1)

        self._held: set[Key] = set()

        if design is not None:
            self.launch_design(design)

    # -----------------------------------------------------
    # Time
    # -----------------------------------------------------

    def step(
        self,
        frames: int = 1
    ):

        for _ in range(frames):

            for key in self._held:
                Input.simulate_hold(key)

            if self._frame_times is not None:
                self.app.timer.set_fixed_step(self._random.uniform(*self._frame_times))

            self.app.step(render=self._render)

            self.frame += 1

    def run_until(
        self,
        condition,
        max_frames: int,
        every=None
    ) -> int:
        """Step until condition() holds (every(): each frame, for scripted flying). Returns the frames taken; AssertionError past max_frames."""

        for taken in range(max_frames):

            if condition():
                return taken

            if every is not None:
                every()

            self.step()

        raise AssertionError(f"not done in {max_frames} frames")

    def uneven(
        self,
        frame_times: tuple[float, float],
        seed: int = 1
    ):
        """From now, each frame between these (s, seeded): an uneven real frame rate."""

        self._frame_times = frame_times

        self._random = random.Random(seed)

    @property
    def clock(self) -> float:

        return self.app.orbit_system.time

    def warp(
        self,
        rate: float
    ):
        """Time warp to the fastest step not above `rate` (the vessel may hold it lower)."""

        warp = self.app.time_warp

        warp.index = max(index for index, step in enumerate(warp.rates) if step <= rate)

    # -----------------------------------------------------
    # Keys
    # -----------------------------------------------------

    def press(
        self,
        key: Key
    ):
        """Pressed this coming frame."""

        for held in self._held:
            Input.simulate_hold(held)

        self.app.step(keys=(key,))

        self.frame += 1

        Input.simulate_release(key)

    def hold(
        self,
        key: Key
    ):

        self._held.add(key)

    def let_go(
        self,
        key: Key
    ):

        self._held.discard(key)

        Input.simulate_release(key)

    # -----------------------------------------------------
    # The game's state
    # -----------------------------------------------------

    @property
    def vessel(self) -> VesselComponent:
        """The vessel flown."""

        return self.app.vessel_system.find(self.app.scene)[1]

    @property
    def vessel_entity(self):

        return self.app.vessel_system.find(self.app.scene)[0]

    def vessels(self) -> list[tuple[object, VesselComponent]]:

        return self.app.vessel_system.vessels(self.app.scene)

    def body(
        self,
        name: str
    ):

        return next(entity for entity, label in self.app.scene.registry.view_with(NameComponent) if label.name == name)

    def notices(self) -> list[str]:

        return [text for text, _ in self.app.hud.notices]

    # -----------------------------------------------------
    # Setting things up
    # -----------------------------------------------------

    def launch_design(
        self,
        design: str
    ):
        """A built-in design on the pad, as the part editor's Launch puts it."""

        self.app._launch_design(load_design(design, self.app.parts))

        self.step()

    def stage(self):
        """Space, and the frames it takes to arrive (a probe hears it a frame or a light time later)."""

        self.press(Key.SPACE)

        link = self.app.link

        frames = 2 if link.crewed or not link.connected else 2 + int(link.delay / self.app.HEADLESS_STEP)

        self.step(frames)

    def put_in_orbit(
        self,
        component: VesselComponent | None = None,
        body=None,
        altitude: float = 200_000.0,
        direction=(1.0, 0.0, 0.0),
        pole=(0.0, 1.0, 0.0)
    ):
        """A circular orbit (inertial direction of its position, its pole), nose prograde, engines idle."""

        component = component or self.vessel

        vessels = self.app.vessel_system

        body = body if body is not None else component.body

        mu = vessels.soi_bodies(self.app.scene)[body].mu
        radius = vessels.soi_bodies(self.app.scene)[body].radius + altitude

        out = np.asarray(direction, dtype=np.float64)
        out /= np.linalg.norm(out)

        forward = np.cross(np.asarray(pole, dtype=np.float64), out)
        forward /= np.linalg.norm(forward)

        component.body = body
        component.state.position = out * radius
        component.state.velocity = forward * kepler.circular_speed(mu, radius)
        component.state.landed = False
        component.state.angular_velocity = np.zeros(3)
        component.state.orientation = quaternion.look_rotation(-out, forward)
        component.vessel.throttle = 0.0

        vessels._prediction = None

    def land_at(
        self,
        body,
        latitude: float,
        longitude: float,
        component: VesselComponent | None = None
    ):
        """Standing on the ground at a place on a body."""

        from science.survey import direction_of

        component = component or self.vessel

        vessels = self.app.vessel_system

        _, frame = self.app.orbit_system.states[body]

        up = frame @ direction_of(latitude, longitude)

        component.body = body

        # (Its ground found afresh, exactly: not the last
        # body's.)
        vessels.placed(component.state, height=0.0)

        environment = vessels.environment(self.app.scene, body, self.app.orbit_system, self.app.planet_system, up)

        component.state.position = up * environment.ground(up)
        component.state.orientation = quaternion.look_rotation(np.cross(up, [1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.cross(up, [0.0, 0.0, 1.0]), up)
        component.vessel.throttle = 0.0

        land(component.state, environment)

        vessels._prediction = None

    def close(self):

        self.app.shutdown()


def angle_between(
    a,
    b
) -> float:
    """Degrees."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    return math.degrees(math.acos(float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))))
