import cProfile
import io
import pstats
import time

from dataclasses import dataclass, field
from pathlib import Path

from core.assertions import engine_assert
from core.logger import Logger


# =========================================================
# cProfile Capture
# =========================================================
#
# Function-level Python profile of N whole frames. The
# scoped Profiler says *which stage* is slow; this says
# *which Python functions* inside it are.
#
#     capture.request(frames=120)
#     ... each frame:
#     capture.begin_frame()
#     ...
#     capture.end_frame()     # writes files after N frames
#
# Output (in `directory`):
#
#     capture_<timestamp>.prof   open with snakeviz / pstats
#     capture_<timestamp>.txt    top functions by cumulative
#                                and by own (tottime) time
#
# cProfile adds large overhead to Python calls, so frame
# times measured *during* a capture are inflated; use the
# relative proportions, not the absolute numbers.


@dataclass(slots=True)
class CaptureResult:

    profile_path: Path
    summary_path: Path
    frames: int

    # (function, cumulative ms per frame, own ms per frame)
    top_functions: list[tuple[str, float, float]] = field(
        default_factory=list
    )


class CProfileCapture:

    SUMMARY_LINES = 40

    def __init__(
        self,
        directory="logs/profiles"
    ):

        self._directory = Path(
            directory
        )

        self._profile: cProfile.Profile | None = None

        self._frames_requested = 0
        self._frames_captured = 0

        self._frame_active = False

        self.last_result: CaptureResult | None = None

    # =====================================================
    # Control
    # =====================================================

    def request(
        self,
        frames: int
    ):
        """Start capturing at the next begin_frame()."""

        engine_assert(
            frames > 0,
            "Capture frame count must be positive."
        )

        if self.active:

            Logger.warning(
                "[CProfile] Capture already in progress; ignoring request."
            )

            return

        self._frames_requested = frames
        self._frames_captured = 0

        self._profile = cProfile.Profile()

        Logger.info(
            "[CProfile] Capturing %d frame(s).",
            frames
        )

    @property
    def active(
        self
    ) -> bool:

        return self._profile is not None

    @property
    def progress(
        self
    ) -> tuple[int, int]:

        return (
            self._frames_captured,
            self._frames_requested
        )

    # =====================================================
    # Frame
    # =====================================================

    def begin_frame(self):

        if self._profile is None:
            return

        self._profile.enable()

        self._frame_active = True

    def end_frame(
        self
    ) -> bool:
        """
        Returns True on the frame a capture completes, so
        callers can discard timings inflated by cProfile's
        overhead.
        """

        if (
            self._profile is None
            or not self._frame_active
        ):
            return False

        self._profile.disable()

        self._frame_active = False

        self._frames_captured += 1

        if self._frames_captured >= self._frames_requested:

            self._finish()

            return True

        return False

    def cancel(self):

        if self._profile is not None and self._frame_active:
            self._profile.disable()

        self._profile = None
        self._frame_active = False

    # =====================================================
    # Output
    # =====================================================

    def _finish(self):

        profile = self._profile
        frames = self._frames_captured

        self._profile = None

        self._directory.mkdir(
            parents=True,
            exist_ok=True
        )

        stamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        profile_path = self._directory / f"capture_{stamp}.prof"
        summary_path = self._directory / f"capture_{stamp}.txt"

        profile.dump_stats(
            profile_path
        )

        stats = pstats.Stats(
            profile
        )

        stream = io.StringIO()

        stream.write(
            f"cProfile capture of {frames} frame(s)\n"
            "Times include cProfile overhead; compare "
            "proportions, not absolute values.\n\n"
        )

        for sort_key, title in (
            ("cumulative", "By cumulative time"),
            ("tottime", "By own time (tottime)"),
        ):

            stream.write(
                f"=== {title} ===\n"
            )

            pstats.Stats(
                profile,
                stream=stream
            ).strip_dirs().sort_stats(
                sort_key
            ).print_stats(
                self.SUMMARY_LINES
            )

        summary_path.write_text(
            stream.getvalue(),
            encoding="utf-8"
        )

        self.last_result = CaptureResult(
            profile_path=profile_path,
            summary_path=summary_path,
            frames=frames,
            top_functions=self._top_functions(
                stats,
                frames
            )
        )

        Logger.info(
            "[CProfile] Wrote %s and %s.",
            profile_path,
            summary_path
        )

    @staticmethod
    def _top_functions(
        stats: pstats.Stats,
        frames: int,
        count: int = 15
    ) -> list[tuple[str, float, float]]:

        # stats.stats: (file, line, name) ->
        #     (primitive calls, total calls, tottime, cumtime, callers)
        #
        # Sorted by own time: the functions whose bodies
        # actually burn the frame.

        rows = []

        for (filename, line, name), (_, _, tottime, cumtime, _) in stats.stats.items():

            label = (
                f"{Path(filename).name}:{line}({name})"
                if filename not in ("~", "")
                else name
            )

            rows.append(
                (
                    label,
                    cumtime * 1000.0 / frames,
                    tottime * 1000.0 / frames
                )
            )

        rows.sort(
            key=lambda row: row[2],
            reverse=True
        )

        return rows[:count]
