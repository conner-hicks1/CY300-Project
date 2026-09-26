import time

from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

from core.assertions import engine_assert
from core.logger import Logger


# =========================================================
# Frame Profiler
# =========================================================
#
# Hierarchical scoped timers:
#
#     profiler.begin_frame()
#
#     with profiler.scope("Render"):
#         with profiler.scope("Shadow pass", gpu=True):
#             ...
#
#     profiler.end_frame()
#
# Scopes nest by call structure; a scope's path is its
# parents' names joined with "/" ("Render/Shadow pass").
# The same path entered several times in one frame is
# summed (e.g. a per-object scope) and its call count
# recorded.
#
# Statistics are rolling over the last `history_size`
# frames. CPU time is wall-clock time spent in the scope
# on this thread; for GL work that is the cost of issuing
# commands, not executing them, which is why scopes can
# also request GPU timing (see graphics/gpu_timer.py).
#
# GPU results arrive a few frames late (the GPU runs
# behind the CPU), so GPU stats lag CPU stats slightly.


class GpuTimerBackend(Protocol):

    def begin(self, path: str) -> object: ...

    def end(self, token: object) -> None: ...

    # (path, milliseconds) for queries that have finished.
    def collect(self) -> list[tuple[str, float]]: ...


@dataclass(slots=True)
class ScopeStats:

    path: str
    name: str
    depth: int

    cpu_average_ms: float
    cpu_max_ms: float

    # None when the scope has no GPU timing (or no
    # results have arrived yet).
    gpu_average_ms: float | None

    # Average number of times entered per frame.
    calls: float


class _ScopeHistory:

    __slots__ = (
        "cpu_ms",
        "gpu_ms",
        "calls",
        "gpu",
    )

    def __init__(
        self,
        history_size: int
    ):

        self.cpu_ms: deque[float] = deque(maxlen=history_size)
        self.gpu_ms: deque[float] = deque(maxlen=history_size)
        self.calls: deque[int] = deque(maxlen=history_size)

        self.gpu = False


class Profiler:

    DEFAULT_HISTORY_SIZE = 240

    # =====================================================
    # Construction
    # =====================================================

    def __init__(
        self,
        history_size: int = DEFAULT_HISTORY_SIZE,
        clock: Callable[[], float] = time.perf_counter
    ):

        engine_assert(
            history_size > 0,
            "Profiler history size must be positive."
        )

        self._history_size = history_size
        self._clock = clock

        self.enabled = True

        # When paused, frames still run but statistics
        # stop updating, so a spike can be inspected.
        self.paused = False

        self._gpu: GpuTimerBackend | None = None

        # Insertion order = order scopes first appeared,
        # which is also a valid depth-first tree order.
        self._scopes: dict[str, _ScopeHistory] = {}

        self._frame_ms: deque[float] = deque(maxlen=history_size)

        # Current frame
        self._in_frame = False
        self._frame_start = 0.0
        self._stack: list[str] = []
        self._frame_cpu: dict[str, float] = {}
        self._frame_calls: dict[str, int] = {}

    # =====================================================
    # GPU Backend
    # =====================================================

    def set_gpu_backend(
        self,
        backend: GpuTimerBackend | None
    ):

        self._gpu = backend

    @property
    def has_gpu_timing(
        self
    ) -> bool:

        return self._gpu is not None

    # =====================================================
    # Frame
    # =====================================================

    def begin_frame(self):

        engine_assert(
            not self._in_frame,
            "Profiler.begin_frame() called twice without end_frame()."
        )

        self._in_frame = True
        self._frame_start = self._clock()

        self._stack.clear()
        self._frame_cpu.clear()
        self._frame_calls.clear()

    def end_frame(self):

        engine_assert(
            self._in_frame,
            "Profiler.end_frame() without begin_frame()."
        )

        engine_assert(
            not self._stack,
            (
                "Profiler.end_frame() with open scopes: "
                f"{'/'.join(self._stack)}"
            )
        )

        frame_ms = (
            self._clock()
            - self._frame_start
        ) * 1000.0

        self._in_frame = False

        # GPU results from earlier frames are collected
        # even when disabled, so queries never pile up.

        gpu_results = self._collect_gpu_results()

        if not self.enabled or self.paused:
            return

        self._frame_ms.append(
            frame_ms
        )

        # Scopes not entered this frame record zero, so
        # averages stay "per frame" rather than "per call".

        for path, history in self._scopes.items():

            history.cpu_ms.append(
                self._frame_cpu.get(path, 0.0)
            )

            history.calls.append(
                self._frame_calls.get(path, 0)
            )

        for path, gpu_ms in gpu_results:

            history = self._scopes.get(
                path
            )

            if history is not None:

                history.gpu_ms.append(
                    gpu_ms
                )

    def _collect_gpu_results(
        self
    ) -> list[tuple[str, float]]:

        if self._gpu is None:
            return []

        try:

            return self._gpu.collect()

        except Exception:

            # A broken profiler must not take the engine
            # down with it: fall back to CPU-only timing.

            Logger.exception(
                "[Profiler] GPU timing failed; disabling it."
            )

            self._gpu = None

            return []

    # =====================================================
    # Scopes
    # =====================================================

    @contextmanager
    def scope(
        self,
        name: str,
        gpu: bool = False
    ) -> Iterator[None]:

        if not self.enabled or not self._in_frame:

            yield
            return

        engine_assert(
            "/" not in name,
            f"Profiler scope names cannot contain '/': {name!r}"
        )

        self._stack.append(
            name
        )

        path = "/".join(
            self._stack
        )

        history = self._scopes.get(
            path
        )

        if history is None:

            history = _ScopeHistory(
                self._history_size
            )

            # Pad so every scope's history lines up with
            # the frame history.

            history.cpu_ms.extend(
                [0.0] * len(self._frame_ms)
            )

            history.calls.extend(
                [0] * len(self._frame_ms)
            )

            self._scopes[path] = history

        gpu_token = None

        if gpu and self._gpu is not None:

            history.gpu = True

            gpu_token = self._gpu.begin(
                path
            )

        start = self._clock()

        try:

            yield

        finally:

            elapsed_ms = (
                self._clock()
                - start
            ) * 1000.0

            if gpu_token is not None:

                self._gpu.end(
                    gpu_token
                )

            self._frame_cpu[path] = (
                self._frame_cpu.get(path, 0.0)
                + elapsed_ms
            )

            self._frame_calls[path] = (
                self._frame_calls.get(path, 0)
                + 1
            )

            self._stack.pop()

    # =====================================================
    # Statistics
    # =====================================================

    @property
    def frame_times_ms(
        self
    ) -> list[float]:

        return list(
            self._frame_ms
        )

    @property
    def frame_average_ms(
        self
    ) -> float:

        return _average(
            self._frame_ms
        )

    @property
    def frame_max_ms(
        self
    ) -> float:

        return max(
            self._frame_ms,
            default=0.0
        )

    def stats(
        self
    ) -> list[ScopeStats]:
        """
        Rolling statistics per scope, in tree order:
        every scope directly follows its parent (siblings
        in first-seen order), even if it first appeared
        after other scopes did (e.g. one that only runs
        every few frames).
        """

        result = []

        for path in self._tree_order():

            history = self._scopes[path]

            result.append(
                ScopeStats(
                    path=path,
                    name=path.rsplit("/", 1)[-1],
                    depth=path.count("/"),
                    cpu_average_ms=_average(history.cpu_ms),
                    cpu_max_ms=max(history.cpu_ms, default=0.0),
                    gpu_average_ms=(
                        _average(history.gpu_ms)
                        if history.gpu and history.gpu_ms
                        else None
                    ),
                    calls=_average(history.calls)
                )
            )

        return result

    def _tree_order(
        self
    ) -> list[str]:

        children: dict[str, list[str]] = {}

        for path in self._scopes:

            parent = (
                path.rsplit("/", 1)[0]
                if "/" in path
                else ""
            )

            children.setdefault(
                parent,
                []
            ).append(
                path
            )

        ordered: list[str] = []

        def visit(parent: str):

            for path in children.get(parent, []):

                ordered.append(path)

                visit(path)

        visit("")

        return ordered

    def reset(self):
        """Discard all collected statistics."""

        self._scopes.clear()
        self._frame_ms.clear()

    # =====================================================
    # Report
    # =====================================================

    def format_report(
        self
    ) -> str:
        """
        Plain-text table of the current statistics (for
        logs and the smoke-test mode).
        """

        lines = [
            f"Frame: avg {self.frame_average_ms:.2f} ms, "
            f"max {self.frame_max_ms:.2f} ms "
            f"over {len(self._frame_ms)} frames",
            f"{'Scope':<34}{'CPU avg':>10}{'CPU max':>10}{'GPU avg':>10}{'calls':>8}",
        ]

        for stat in self.stats():

            gpu = (
                f"{stat.gpu_average_ms:10.3f}"
                if stat.gpu_average_ms is not None
                else f"{'-':>10}"
            )

            lines.append(
                f"{'  ' * stat.depth + stat.name:<34}"
                f"{stat.cpu_average_ms:10.3f}"
                f"{stat.cpu_max_ms:10.3f}"
                f"{gpu}"
                f"{stat.calls:8.1f}"
            )

        return "\n".join(
            lines
        )


def _average(
    values
) -> float:

    return (
        sum(values) / len(values)
        if values
        else 0.0
    )
