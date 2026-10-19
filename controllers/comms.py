import math

from types import SimpleNamespace

import numpy as np

from core.logger import Logger

from game.computer_context import GameFlightContext, hold_sink_rate
from game.computer_panel import ComputerInfo
from game.vessel_controls import read_pilot, release

from systems.comms_system import Link

from vessel.physics import Controls
from vessel.sequence import FlightComputer, save_program

from game import format as readable


# =========================================================
# The Application: Comms
# =========================================================
#
# Piloting through the link home: the pilot's keys (a
# probe's a light time late), commands arriving, the
# flight computer and its panel, staging.
#
# A part of Application (application.py), apart for its
# size: its methods run on the application (self is it).

class CommsController:

    def _pilot(
        self,
        dt: float,
        paused: bool
    ):
        """The pilot's keys (not while paused, typing, or flying the free camera)."""

        found = self.vessel_system.find(self.scene)

        if found is None:
            return

        _, component = found

        busy = (
            paused
            or (self.imgui_layer.wants_keyboard and not self._looking)
            or (self.free_camera and self._looking)
            or component.state.destroyed
        )

        if self.link.crewed:

            if busy:

                release(component.controls)

                return

            if read_pilot(component.vessel, component.controls, dt):
                self.stage_vessel(component)

            return

        # A probe: the keys go home's way to it, a light time
        # late; without a link, nowhere.
        sent = self._sent

        if sent is None or sent[0] != id(component.vessel):
            sent = self._sent = (id(component.vessel), SimpleNamespace(throttle=component.vessel.throttle), Controls(), None)

        _, throttle, controls, last = sent

        def snapshot():
            return (controls.pitch, controls.yaw, controls.roll, controls.sas, controls.sas_mode, throttle.throttle, tuple(controls.translate))

        # Nothing on the way: what is sent is what it has (and
        # what its computer set: not to be sent back over it).
        if last is None or not any(kind == "controls" for _, kind, _ in component.uplink):

            throttle.throttle = component.vessel.throttle

            for name in ("pitch", "yaw", "roll", "sas", "sas_mode"):
                setattr(controls, name, getattr(component.controls, name))

            controls.translate = np.array(component.controls.translate, dtype=np.float64)

            last = snapshot()

        if busy:
            release(controls)
            staged = False
        else:
            staged = read_pilot(throttle, controls, dt)

        now_sent = snapshot()

        if not self.link.connected:

            if now_sent != last or staged:
                self.hud.notify(f"No link ({self.link.reason}): the probe cannot hear you")

            # Forget what could not be sent.
            self._sent = (sent[0], throttle, controls, None)

            return

        if now_sent != last:
            self._uplink(component, "controls", now_sent)

        if staged:
            self._uplink(component, "stage", None)

        self._sent = (sent[0], throttle, controls, now_sent)

    def stage_vessel(
        self,
        component
    ):
        """Fire the next stage (the pilot, the computer), with a notice of what it did."""

        # The first: the launch is paid for (or not, short of
        # funds: then nothing fires).
        if not self.programme_system.pay_launch(component):

            for news in self.programme_system.news:
                self.hud.notify(news)

            self.programme_system.news.clear()

            return

        vessel = component.vessel

        parts = len(vessel.parts)

        stage = vessel.stage()

        if stage is None:
            return

        Logger.info("[Application] Stage %d.", stage)

        # Each engine lighting, each parachute arming, may fail
        # (vessel/reliability.py).
        failures = self.programme_system.staged(component, stage)

        what = []

        if len(vessel.parts) < parts:
            what.append("separated")

        if any(state.ignited and state.stage == stage for state in vessel.parts):
            what.append("engines lit")

        if any(state.armed and state.stage == stage for state in vessel.parts):
            what.append("parachute armed")

        self.hud.notify(f"Stage {stage}: {', '.join(what) or 'fired'}")

        for notice in failures:
            self.hud.notify(notice)

    def _uplink(
        self,
        component,
        kind: str,
        payload
    ):
        """Send a command; it arrives a light time from now (now, crewed)."""

        delay = 0.0 if self.link.crewed else self.link.delay

        component.uplink.append((self.orbit_system.time + delay, kind, payload))

    def _update_comms(
        self,
        paused: bool
    ):
        """Each frame: the link; commands arriving; the computer's turn."""

        found = self.vessel_system.find(self.scene)

        if found is None:

            self.link = Link(False)

            return

        _, component = found

        self.link = self.comms_system.link(self.scene, self.vessel_system, self.orbit_system, self.planet_system)

        now = self.orbit_system.time

        arrived = [item for item in component.uplink if item[0] <= now]

        if arrived:

            component.uplink = [item for item in component.uplink if item[0] > now]

            for _, kind, payload in sorted(arrived, key=lambda item: item[0]):

                # A probe without power: deaf, the command lost.
                if not component.vessel.powered and not self.link.crewed:

                    Logger.info("[Comms] %s had no power: a command was lost.", component.vessel.name)

                    continue

                self._command(component, kind, payload)

        computer = component.computer

        # (The computer needs power: it waits for it.)
        if computer is not None and not computer.done and not component.state.destroyed and not paused and component.vessel.powered:

            line = computer.current

            computer.tick(GameFlightContext(self, component))

            for when, text in computer.log[-3:]:
                if when == now and text:
                    self.hud.notify(f"Computer: {text}")

            if computer.done:
                component.hover = None

            elif line is not computer.current and computer.current is not None:
                Logger.debug("[Computer] Line %d.", computer.current.line)

        hold_sink_rate(self, component)

    def _command(
        self,
        component,
        kind: str,
        payload
    ):
        """A command arriving at the vessel."""

        if kind == "controls":

            pitch, yaw, roll, sas, sas_mode, throttle, translate = payload

            controls = component.controls

            controls.pitch, controls.yaw, controls.roll = pitch, yaw, roll
            controls.translate = np.array(translate, dtype=np.float64)
            controls.sas, controls.sas_mode = sas, sas_mode

            component.vessel.throttle = throttle

        elif kind == "stage":

            self.stage_vessel(component)

        elif kind == "science":

            self.run_science_aboard(component)

        elif kind == "panels":

            self.set_panels(component, payload)

        elif kind == "legs":

            self.set_legs(component, payload)

        elif kind == "program":

            component.computer = FlightComputer(payload)
            component.hover = None

            self.hud.notify("Computer: program received" if not self.link.crewed else "Computer: running")

        elif kind == "stop":

            if component.computer is not None:

                component.computer.stop(GameFlightContext(self, component))

                component.hover = None

    def _computer_warp(self) -> float:
        """Fastest the clock may run not to pass the computer's next deadline."""

        found = self.vessel_system.find(self.scene)

        if found is None:
            return math.inf

        computer = found[1].computer

        if computer is None or computer.done:
            return math.inf

        gap = computer.time_left(GameFlightContext(self, found[1]))

        if gap is None:
            return math.inf

        return max(1.0, gap / (4.0 / 30.0))

    def _draw_computer(self):

        found = self.vessel_system.find(self.scene)

        if not self.computer_panel.open or found is None:
            return

        _, component = found

        link = self.link

        now = self.orbit_system.time

        info = ComputerInfo(
            connected=link.connected,
            link=link.station if link.connected else link.reason,
            delay=link.delay,
            crewed=link.crewed
        )

        names = {"program": "the program", "stop": "Stop", "controls": "controls", "stage": "Stage", "science": "Run experiments", "panels": "the solar wings", "legs": "the landing legs"}

        info.in_flight = [(names.get(kind, kind), when - now) for when, kind, _ in component.uplink if when > now and kind != "controls"]

        computer = component.computer

        if computer is not None:

            info.running = True
            info.done = computer.done
            info.log = computer.log

            current = computer.current

            if current is not None:
                info.line = current.text
                info.line_number = current.line

            info.wait = computer.deadline - now if computer.deadline is not None else None

        for action in self.computer_panel.draw(info):

            if action[0] == "save":

                try:
                    path = save_program(action[1], action[2])
                except OSError as error:
                    self.hud.notify(f"Could not save: {error}")
                else:
                    self.hud.notify(f"Saved {path}")

            elif action[0] in ("send", "stop"):

                if not link.crewed and not link.connected:

                    self.hud.notify(f"No link ({link.reason}): cannot send")

                    continue

                self._uplink(component, "program" if action[0] == "send" else "stop", action[1] if action[0] == "send" else None)

                if not link.crewed and link.delay > 1.0:
                    self.hud.notify(f"Sent: arrives in {readable.countdown(link.delay)}")

    def _link_line(
        self,
        component
    ) -> tuple[str, str] | None:
        """For the HUD: (link text, "good" / "warn" / "bad"), and the computer's line."""

        link = self.link

        if link.connected:
            text = f"Link {link.station}" + (f", {readable.countdown(link.delay)} delay" if link.delay >= 0.5 else "")
            tone = "warn" if link.delay > 5.0 and not link.crewed else "good"
        else:
            text = f"No link: {link.reason}"
            tone = "warn" if link.crewed else "bad"

        computer = component.computer

        if computer is not None and not computer.done and computer.current is not None:
            text += f"  |  Computer line {computer.current.line}"

        return text, tone
