import time

# Drives the update/render loop
class Timer:

    # Longest frame time reported by delta_time.
    #
    # Window dragging on Windows, breakpoints, or a slow
    # load can stall the loop. Without a clamp, the next
    # frame would move everything by the whole stall.

    DEFAULT_MAX_DELTA_TIME = 0.1

    def __init__(
        self,
        max_delta_time: float = DEFAULT_MAX_DELTA_TIME
    ):

        if max_delta_time <= 0.0:
            raise ValueError(
                "Timer max_delta_time must be positive."
            )

        self._max_delta_time = max_delta_time

        self._start_time = time.perf_counter()
        self._previous_time = self._start_time

        self._delta_time = 0.0
        self._unclamped_delta_time = 0.0
        self._elapsed_time = 0.0

        self._frame_count = 0
        self._fps = 0.0

        self._fps_timer = 0.0
        self._fps_frame_count = 0

    # =====================================================
    # Reset
    # =====================================================

    def reset(self):
        """
        Restart frame timing from now.

        Call right before entering the main loop so the
        first frame does not include loading time.
        """

        self._previous_time = time.perf_counter()

        self._delta_time = 0.0
        self._unclamped_delta_time = 0.0

        self._fps_timer = 0.0
        self._fps_frame_count = 0

    # =====================================================
    # Frame Update
    # =====================================================

    def update(self):

        current_time = time.perf_counter()

        self._unclamped_delta_time = (
            current_time
            - self._previous_time
        )

        self._delta_time = min(
            self._unclamped_delta_time,
            self._max_delta_time
        )

        self._previous_time = current_time

        self._elapsed_time = (
            current_time
            - self._start_time
        )

        self._frame_count += 1

        # ---------------------------------------------
        # FPS calculation
        # ---------------------------------------------

        # FPS uses real time so stalls are still visible.

        self._fps_timer += self._unclamped_delta_time
        self._fps_frame_count += 1

        if self._fps_timer >= 1.0:

            self._fps = (
                self._fps_frame_count
                / self._fps_timer
            )

            self._fps_timer = 0.0
            self._fps_frame_count = 0


    # =====================================================
    # Timing Properties
    # =====================================================

    @property
    def delta_time(self):
        """
        Time in seconds since the previous frame, clamped
        to max_delta_time.
        """
        return self._delta_time

    @property
    def unclamped_delta_time(self):
        """
        Real time in seconds since the previous frame.
        """
        return self._unclamped_delta_time

    @property
    def max_delta_time(self):

        return self._max_delta_time


    @property
    def elapsed_time(self):
        """
        Total time in seconds since the timer was created.
        """
        return self._elapsed_time


    @property
    def fps(self):
        """
        Average FPS over approximately the last second.
        """
        return self._fps


    @property
    def frame_count(self):
        """
        Total number of frames processed.
        """
        return self._frame_count


    @property
    def time_ns(self):
        """
        Current high-resolution timer value in nanoseconds.
        """
        return time.perf_counter_ns()

# Used for profiling CPU-elapsed time between the start of a function call to after the function call.
class Stopwatch:

    def __init__(self):

        self._start = None
        self._elapsed = 0.0


    def start(self):

        self._start = time.perf_counter()


    def stop(self):

        if self._start is None:
            raise RuntimeError(
                "Stopwatch has not been started."
            )

        self._elapsed = (
            time.perf_counter()
            - self._start
        )

        self._start = None

        return self._elapsed


    @property
    def elapsed(self):

        return self._elapsed