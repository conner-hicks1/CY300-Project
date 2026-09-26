import ctypes

from collections import deque

import numpy as np

from OpenGL.GL import (
    GL_QUERY_COUNTER_BITS,
    GL_QUERY_RESULT,
    GL_QUERY_RESULT_AVAILABLE,
    GL_TIMESTAMP,
    glDeleteQueries,
    glGenQueries,
    glGetQueryiv,
    glGetQueryObjectiv,
    glGetQueryObjectui64v,
    glQueryCounter
)

from core.logger import Logger


class GpuTimer:

    # =====================================================
    # GPU Timing via Timestamp Queries
    # =====================================================
    #
    # Each timed scope records a GPU timestamp at its start
    # and end (glQueryCounter, core since OpenGL 3.3).
    # Timestamps, unlike GL_TIME_ELAPSED queries, can nest,
    # so parent and child scopes can both be timed.
    #
    # The GPU runs a few frames behind the CPU. Results are
    # read only once GL reports them available, so reading
    # never stalls the pipeline; they are reported a few
    # frames late.
    #
    # Implements core.profiler.GpuTimerBackend.

    # Pending scopes kept at most; if results stop
    # arriving (e.g. a driver that never completes
    # queries) the oldest are dropped instead of growing
    # without bound.
    MAX_PENDING = 512

    def __init__(self):

        self._free: list[int] = []
        self._pending: deque[tuple[str, int, int]] = deque()

        self._result = ctypes.c_uint64()
        self._available = np.zeros(1, dtype=np.int32)

    # =====================================================
    # Support
    # =====================================================

    @staticmethod
    def is_supported() -> bool:
        """
        Some drivers expose timestamp queries with zero
        counter bits, meaning "not really supported".
        """

        bits = np.zeros(1, dtype=np.int32)

        try:

            glGetQueryiv(
                GL_TIMESTAMP,
                GL_QUERY_COUNTER_BITS,
                bits
            )

        except Exception:

            Logger.exception(
                "[GpuTimer] Timestamp query support check failed."
            )

            return False

        return int(bits[0]) > 0

    # =====================================================
    # Timing
    # =====================================================

    def begin(
        self,
        path: str
    ) -> tuple[str, int]:

        query = self._acquire()

        glQueryCounter(
            query,
            GL_TIMESTAMP
        )

        return path, query

    def end(
        self,
        token: tuple[str, int]
    ):

        path, start_query = token

        end_query = self._acquire()

        glQueryCounter(
            end_query,
            GL_TIMESTAMP
        )

        self._pending.append(
            (path, start_query, end_query)
        )

        while len(self._pending) > self.MAX_PENDING:

            _, a, b = self._pending.popleft()

            self._free.extend((a, b))

    def collect(
        self
    ) -> list[tuple[str, float]]:

        results = []

        # Queries complete in submission order, so stop at
        # the first one that is not ready.

        while self._pending:

            path, start_query, end_query = self._pending[0]

            glGetQueryObjectiv(
                end_query,
                GL_QUERY_RESULT_AVAILABLE,
                self._available
            )

            if not self._available[0]:
                break

            self._pending.popleft()

            start_ns = self._read(start_query)
            end_ns = self._read(end_query)

            self._free.extend(
                (start_query, end_query)
            )

            results.append(
                (path, max(0, end_ns - start_ns) / 1_000_000.0)
            )

        return results

    # =====================================================
    # Query Pool
    # =====================================================

    def _acquire(
        self
    ) -> int:

        if not self._free:

            self._free.extend(
                int(query)
                for query in np.atleast_1d(glGenQueries(32))
            )

        return self._free.pop()

    def _read(
        self,
        query: int
    ) -> int:

        # PyOpenGL(-accelerate) has no numpy dtype mapping
        # for this function's 64-bit output (KeyError on
        # GL_UNSIGNED_INT64_AMD), and rejects a bare
        # c_uint64; a ctypes pointer is passed through.

        glGetQueryObjectui64v(
            query,
            GL_QUERY_RESULT,
            ctypes.byref(self._result)
        )

        return int(
            self._result.value
        )

    # =====================================================
    # Shutdown
    # =====================================================

    def delete(self):

        queries = self._free + [
            query
            for _, a, b in self._pending
            for query in (a, b)
        ]

        if queries:

            glDeleteQueries(
                len(queries),
                queries
            )

        self._free.clear()
        self._pending.clear()
