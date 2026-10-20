import ctypes
import sys
import threading
import time

from ctypes import wintypes

import numpy as np

from core.logger import Logger


# =========================================================
# Sound Output
# =========================================================
#
# Plays what the mixer (audio/mixer.py) makes, as it makes
# it: on Windows through the system's own waveOut API
# (winmm.dll, ctypes: nothing to install), a few short
# buffers queued ahead (BUFFERS x BLOCK frames, ~90 ms) and
# refilled by a thread as each finishes. Elsewhere, or when
# no device opens, silent (NullOutput): the game is the same
# without sound.

RATE = 44_100
CHANNELS = 2
BLOCK = 1_024                       # frames a buffer (~23 ms)
BUFFERS = 4

WHDR_DONE = 0x00000001
WAVE_FORMAT_PCM = 1
CALLBACK_NULL = 0
WAVE_MAPPER = 0xFFFFFFFF


class NullOutput:
    """No sound (headless, no device, not Windows)."""

    playing = False

    def start(self):
        pass

    def stop(self):
        pass


class _WaveFormat(ctypes.Structure):

    _fields_ = [
        ("wFormatTag", wintypes.WORD),
        ("nChannels", wintypes.WORD),
        ("nSamplesPerSec", wintypes.DWORD),
        ("nAvgBytesPerSec", wintypes.DWORD),
        ("nBlockAlign", wintypes.WORD),
        ("wBitsPerSample", wintypes.WORD),
        ("cbSize", wintypes.WORD),
    ]


class _WaveHeader(ctypes.Structure):

    _fields_ = [
        ("lpData", ctypes.c_void_p),
        ("dwBufferLength", wintypes.DWORD),
        ("dwBytesRecorded", wintypes.DWORD),
        ("dwUser", ctypes.c_size_t),
        ("dwFlags", wintypes.DWORD),
        ("dwLoops", wintypes.DWORD),
        ("lpNext", ctypes.c_void_p),
        ("reserved", ctypes.c_size_t),
    ]


class WaveOutput:
    """Windows waveOut: `render(frames) -> (frames, 2) float32 in -1..1` called on its own thread."""

    def __init__(
        self,
        render
    ):

        self._render = render

        self._thread: threading.Thread | None = None
        self._running = False

        self._device = wintypes.HANDLE()
        self._headers: list[_WaveHeader] = []
        self._buffers: list = []

        self.playing = False

    def start(self):

        winmm = ctypes.WinDLL("winmm")

        self._winmm = winmm

        form = _WaveFormat(WAVE_FORMAT_PCM, CHANNELS, RATE, RATE * CHANNELS * 2, CHANNELS * 2, 16, 0)

        result = winmm.waveOutOpen(
            ctypes.byref(self._device), ctypes.c_uint(WAVE_MAPPER), ctypes.byref(form),
            ctypes.c_size_t(0), ctypes.c_size_t(0), wintypes.DWORD(CALLBACK_NULL)
        )

        if result != 0:
            raise OSError(f"waveOutOpen failed ({result})")

        size = BLOCK * CHANNELS * 2

        for _ in range(BUFFERS):

            buffer = ctypes.create_string_buffer(size)

            header = _WaveHeader()
            header.lpData = ctypes.cast(buffer, ctypes.c_void_p)
            header.dwBufferLength = size

            winmm.waveOutPrepareHeader(self._device, ctypes.byref(header), ctypes.sizeof(header))

            self._buffers.append(buffer)
            self._headers.append(header)

        for index in range(BUFFERS):
            self._submit(index)

        self._running = True
        self.playing = True

        self._thread = threading.Thread(target=self._loop, name="audio", daemon=True)
        self._thread.start()

    def _submit(
        self,
        index: int
    ):

        samples = np.clip(self._render(BLOCK), -1.0, 1.0)

        data = (samples * 32_000.0).astype("<i2").tobytes()

        ctypes.memmove(self._buffers[index], data, len(data))

        header = self._headers[index]

        header.dwFlags &= ~WHDR_DONE

        self._winmm.waveOutWrite(self._device, ctypes.byref(header), ctypes.sizeof(header))

    def _loop(self):

        while self._running:

            busy = True

            for index, header in enumerate(self._headers):

                if header.dwFlags & WHDR_DONE:

                    busy = False

                    try:
                        self._submit(index)
                    except Exception as error:   # (a bad block: silence, not a dead thread)
                        Logger.error("[Audio] %s", error)

            if busy:
                time.sleep(0.004)

    def stop(self):

        if not self._running:
            return

        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=1.0)

        winmm = self._winmm

        winmm.waveOutReset(self._device)

        for header in self._headers:
            winmm.waveOutUnprepareHeader(self._device, ctypes.byref(header), ctypes.sizeof(header))

        winmm.waveOutClose(self._device)

        self.playing = False


def open_output(
    render,
    enabled: bool = True
):
    """The best output there is (silent when not enabled, not Windows, or no device)."""

    if not enabled or sys.platform != "win32":
        return NullOutput()

    output = WaveOutput(render)

    try:
        output.start()
    except (OSError, AttributeError) as error:

        Logger.warning("[Audio] No sound: %s", error)

        return NullOutput()

    Logger.info("[Audio] Sound on (waveOut, %d Hz, %d x %d frames).", RATE, BUFFERS, BLOCK)

    return output
