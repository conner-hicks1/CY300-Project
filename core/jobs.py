import os
import queue
import threading
import time

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from core.assertions import engine_assert
from core.logger import Logger


class Job:

    # =====================================================
    # Job Handle
    # =====================================================

    __slots__ = (
        "name",
        "_cancelled",
        "_done",
    )

    def __init__(
        self,
        name: str
    ):

        self.name = name

        self._cancelled = threading.Event()
        self._done = threading.Event()

    def cancel(self):
        """
        The completion callback will not run. Work already
        running on a worker still finishes (Python threads
        cannot be interrupted), its result is discarded.
        """

        self._cancelled.set()

    @property
    def cancelled(
        self
    ) -> bool:

        return self._cancelled.is_set()

    @property
    def done(
        self
    ) -> bool:
        """True once the main thread has handled the result."""

        return self._done.is_set()


class JobSystem:

    # =====================================================
    # Background Jobs
    # =====================================================
    #
    # Runs work (mesh generation, file parsing, simulation
    # steps) on worker threads and hands results back to
    # the main thread, which owns the OpenGL context:
    #
    #     jobs.submit(
    #         lambda: build_chunk_mesh(node),          # worker
    #         on_complete=lambda data: upload(data),   # main
    #     )
    #
    #     # once per frame, on the main thread:
    #     jobs.process_completions(budget_seconds=0.002)
    #
    # Completions run only inside process_completions(),
    # and stop once the time budget is used, so a burst of
    # finished jobs (e.g. GPU uploads) is spread over
    # several frames instead of causing a hitch.
    #
    # Python threads share one interpreter lock: pure-Python
    # loops do not run in parallel, but numpy (and file I/O)
    # release the lock for heavy work, which is where these
    # jobs spend their time.

    def __init__(
        self,
        workers: int | None = None
    ):

        if workers is None:

            # Leave a core for the main thread; past ~8,
            # interpreter-lock contention outweighs the gain.
            workers = min(8, max(1, (os.cpu_count() or 2) - 1))

        engine_assert(
            workers >= 1,
            "JobSystem needs at least one worker."
        )

        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="engine-job"
        )

        self._completed: queue.SimpleQueue = queue.SimpleQueue()

        self._pending = 0
        self._lock = threading.Lock()

        self._shutdown = False

        self.workers = workers

        Logger.info(
            "[Jobs] Started %d worker(s).",
            workers
        )

    # =====================================================
    # Submit
    # =====================================================

    def submit(
        self,
        work: Callable[[], Any],
        on_complete: Callable[[Any], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        name: str = "job"
    ) -> Job:
        """
        Run `work()` on a worker. `on_complete(result)` or
        `on_error(exception)` later runs on the main thread
        in process_completions(). Without on_error, errors
        are logged.
        """

        engine_assert(
            not self._shutdown,
            "JobSystem has been shut down."
        )

        job = Job(name)

        with self._lock:
            self._pending += 1

        def run():

            if job.cancelled:

                self._completed.put((job, None, None, on_complete, on_error, True))

                return

            try:

                result = work()

            except BaseException as error:  # noqa: BLE001 - reported on main thread

                self._completed.put((job, None, error, on_complete, on_error, False))

                return

            self._completed.put((job, result, None, on_complete, on_error, False))

        self._executor.submit(run)

        return job

    # =====================================================
    # Completions (main thread)
    # =====================================================

    def process_completions(
        self,
        budget_seconds: float = 0.002
    ) -> int:
        """
        Run completion callbacks for finished jobs until
        `budget_seconds` have been spent (at least one is
        always handled if any are ready). Returns how many
        were handled.
        """

        start = time.perf_counter()

        handled = 0

        while True:

            if handled > 0 and time.perf_counter() - start >= budget_seconds:
                break

            try:
                entry = self._completed.get_nowait()

            except queue.Empty:
                break

            job, result, error, on_complete, on_error, skipped = entry

            with self._lock:
                self._pending -= 1

            handled += 1

            if not skipped and not job.cancelled:
                self._dispatch(job, result, error, on_complete, on_error)

            job._done.set()

        return handled

    @staticmethod
    def _dispatch(
        job: Job,
        result,
        error,
        on_complete,
        on_error
    ):

        try:

            if error is not None:

                if on_error is not None:
                    on_error(error)

                else:

                    Logger.error(
                        "[Jobs] '%s' failed: %s: %s",
                        job.name,
                        type(error).__name__,
                        error
                    )

            elif on_complete is not None:

                on_complete(result)

        except Exception:

            Logger.exception(
                "[Jobs] Completion callback of '%s' failed.",
                job.name
            )

    @property
    def pending(
        self
    ) -> int:
        """Jobs submitted whose completion has not been handled."""

        with self._lock:
            return self._pending

    # =====================================================
    # Shutdown
    # =====================================================

    def shutdown(
        self,
        wait: bool = False
    ):
        """
        Stop accepting work and drop queued jobs. Running
        jobs finish in the background unless `wait`.
        """

        if self._shutdown:
            return

        self._shutdown = True

        self._executor.shutdown(
            wait=wait,
            cancel_futures=True
        )

        Logger.info("[Jobs] Shut down.")
