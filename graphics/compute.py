import collections
import ctypes
import threading
import time

from pathlib import Path

import numpy as np

from OpenGL.GL import (
    GL_COMPILE_STATUS,
    GL_COMPUTE_SHADER,
    GL_DYNAMIC_COPY,
    GL_LINK_STATUS,
    GL_SHADER_STORAGE_BARRIER_BIT,
    GL_SHADER_STORAGE_BUFFER,
    GL_BUFFER_UPDATE_BARRIER_BIT,
    glAttachShader,
    glBindBuffer,
    glBindBufferBase,
    glBufferData,
    glBufferSubData,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteBuffers,
    glDeleteProgram,
    glDeleteShader,
    glDispatchCompute,
    glGenBuffers,
    glMapBufferRange,
    glUnmapBuffer,
    GL_MAP_READ_BIT,
    glGetIntegerv,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetUniformLocation,
    glLinkProgram,
    glMemoryBarrier,
    glShaderSource,
    glUniform1f,
    glUniform1i,
    glUniform1ui,
    glUseProgram,
    GL_MAJOR_VERSION,
    GL_MINOR_VERSION
)

from core.exceptions import ShaderError
from core.logger import Logger

from graphics.shader_preprocessor import preprocess_shader
from graphics.uniform_blocks import ENGINE_SHADER_DEFINES


# =========================================================
# GPU Compute
# =========================================================
#
# Compute shaders (OpenGL 4.3+) for the simulations'
# data-parallel cores (graphics/gpu_simulation.py), and the
# plumbing to reach them from worker threads.
#
# OpenGL lives on the main thread. A simulation running on
# a worker hands its GPU work to the GpuQueue and sleeps
# (without the Python lock) until the main thread has run
# it, a little each frame within a time budget. Long
# kernels are generators: each next() does a slice of the
# work (one dispatch), so a big solve spreads over frames
# instead of stalling one.


def compute_supported() -> bool:
    """Compute shaders need OpenGL 4.3 (call with a context current)."""

    try:

        major = int(glGetIntegerv(GL_MAJOR_VERSION))
        minor = int(glGetIntegerv(GL_MINOR_VERSION))

    except Exception:

        return False

    return (major, minor) >= (4, 3)


class ComputeShader:

    def __init__(
        self,
        path,
        defines: dict[str, object] | None = None
    ):

        self.path = Path(path)

        preprocessed = preprocess_shader(self.path, {**ENGINE_SHADER_DEFINES, **(defines or {})})

        shader = glCreateShader(GL_COMPUTE_SHADER)

        glShaderSource(shader, preprocessed.source)
        glCompileShader(shader)

        if not glGetShaderiv(shader, GL_COMPILE_STATUS):

            log = glGetShaderInfoLog(shader)

            glDeleteShader(shader)

            raise ShaderError(
                f"Compute shader compilation failed:\n{self.path}\n\n"
                f"{log.decode(errors='replace') if isinstance(log, bytes) else log}\n"
                f"Source numbers:\n{preprocessed.describe_files()}"
            )

        program = glCreateProgram()

        glAttachShader(program, shader)
        glLinkProgram(program)

        glDeleteShader(shader)

        if not glGetProgramiv(program, GL_LINK_STATUS):

            log = glGetProgramInfoLog(program)

            glDeleteProgram(program)

            raise ShaderError(f"Compute shader link failed: {self.path}\n{log}")

        self.id = int(program)

        self._locations: dict[str, int] = {}

        Logger.info("[Compute] Program created: ID=%d (%s).", self.id, self.path.name)

    def _location(
        self,
        name: str
    ) -> int:

        location = self._locations.get(name)

        if location is None:

            location = int(glGetUniformLocation(self.id, name))

            self._locations[name] = location

        return location

    def bind(self):

        glUseProgram(self.id)

    def set_int(self, name: str, value: int):

        glUniform1i(self._location(name), int(value))

    def set_uint(self, name: str, value: int):

        glUniform1ui(self._location(name), int(value))

    def set_float(self, name: str, value: float):

        glUniform1f(self._location(name), float(value))

    def dispatch(
        self,
        groups_x: int,
        groups_y: int = 1,
        groups_z: int = 1
    ):

        glDispatchCompute(max(int(groups_x), 1), max(int(groups_y), 1), max(int(groups_z), 1))

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT)

    def delete(self):

        if self.id:

            glDeleteProgram(self.id)

            self.id = 0


class StorageBuffer:

    # A shader storage buffer of plain numbers (std430 arrays
    # of float, int or uint).

    def __init__(
        self,
        data: np.ndarray | int
    ):

        self.id = int(glGenBuffers(1))

        self.nbytes = 0

        if isinstance(data, int):
            self.allocate(data)
        else:
            self.upload(data)

    def allocate(
        self,
        nbytes: int
    ):

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.id)
        glBufferData(GL_SHADER_STORAGE_BUFFER, max(int(nbytes), 4), None, GL_DYNAMIC_COPY)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        self.nbytes = max(int(nbytes), 4)

    def upload(
        self,
        array: np.ndarray
    ):

        data = np.ascontiguousarray(array)

        if data.dtype == np.float64:
            data = data.astype(np.float32)
        elif data.dtype == np.int64:
            data = data.astype(np.int32)
        elif data.dtype == np.bool_:
            data = data.astype(np.uint32)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.id)

        if data.nbytes == self.nbytes:
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, data.nbytes, data)
        else:
            glBufferData(GL_SHADER_STORAGE_BUFFER, max(data.nbytes, 4), data if data.nbytes else None, GL_DYNAMIC_COPY)
            self.nbytes = max(data.nbytes, 4)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def bind(
        self,
        binding: int
    ):

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, self.id)

    def read(
        self,
        dtype,
        count: int | None = None
    ) -> np.ndarray:

        dtype = np.dtype(dtype)

        count = self.nbytes // dtype.itemsize if count is None else int(count)

        out = np.empty(count, dtype=dtype)

        if out.nbytes == 0:
            return out

        # Mapped and copied explicitly (PyOpenGL's
        # glGetBufferSubData wrapper is unreliable with
        # caller-provided arrays).
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.id)

        pointer = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, out.nbytes, GL_MAP_READ_BIT)

        try:
            ctypes.memmove(out.ctypes.data, pointer, out.nbytes)
        finally:
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        return out

    def delete(self):

        if self.id:

            glDeleteBuffers(1, [self.id])

            self.id = 0


# =========================================================
# Work for the GL Thread
# =========================================================

class GpuQueueClosed(RuntimeError):
    """The GPU went away (shutdown) before the work ran."""


class _Task:

    __slots__ = ("work", "generator", "result", "error", "done")

    def __init__(self, work):

        self.work = work
        self.generator = None
        self.result = None
        self.error: BaseException | None = None
        self.done = threading.Event()


class GpuQueue:

    def __init__(self):

        self._tasks: collections.deque[_Task] = collections.deque()
        self._lock = threading.Lock()

        self._thread = threading.get_ident()

        self._closed = False

        # Diagnostics.
        self.tasks_run = 0
        self.busy_seconds = 0.0

    @property
    def on_gl_thread(
        self
    ) -> bool:

        return threading.get_ident() == self._thread

    def run(
        self,
        work
    ):
        """
        Run work() on the GL thread and return its result (a
        generator function runs to its end; its return value
        is the result). From the GL thread itself it runs at
        once.
        """

        task = _Task(work)

        if self.on_gl_thread:

            self._advance(task, budget=None)

            return self._finish(task)

        with self._lock:

            if self._closed:
                raise GpuQueueClosed("GPU queue is shut down")

            self._tasks.append(task)

        task.done.wait()

        return self._finish(task)

    @staticmethod
    def _finish(
        task: _Task
    ):

        if task.error is not None:
            raise task.error

        return task.result

    def process(
        self,
        budget_seconds: float
    ):
        """Run queued GPU work for up to the budget (main thread, once per frame)."""

        deadline = time.perf_counter() + budget_seconds

        while True:

            with self._lock:

                if not self._tasks:
                    return

                task = self._tasks[0]

            started = time.perf_counter()

            finished = self._advance(task, budget=deadline)

            self.busy_seconds += time.perf_counter() - started

            if finished:

                with self._lock:
                    self._tasks.popleft()

                self.tasks_run += 1

                task.done.set()

            if time.perf_counter() >= deadline:
                return

    @staticmethod
    def _advance(
        task: _Task,
        budget: float | None
    ) -> bool:
        """Run a task until it ends or the deadline passes; True when it ended."""

        try:

            if task.generator is None:

                outcome = task.work()

                if not hasattr(outcome, "__next__"):

                    task.result = outcome

                    return True

                task.generator = outcome

            while True:

                next(task.generator)

                if budget is not None and time.perf_counter() >= budget:
                    return False

        except StopIteration as stop:

            task.result = stop.value

            return True

        except BaseException as error:

            task.error = error

            return True

    def shutdown(self):
        """Fail every waiting task (their threads must not hang)."""

        with self._lock:

            self._closed = True

            pending = list(self._tasks)

            self._tasks.clear()

        for task in pending:

            task.error = GpuQueueClosed("GPU queue shut down before the work ran")

            task.done.set()
