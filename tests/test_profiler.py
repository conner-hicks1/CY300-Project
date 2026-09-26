import pytest

from core.cprofile_capture import CProfileCapture
from core.profiler import Profiler


class FakeClock:

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, ms):
        self.now += ms / 1000.0


class FakeGpu:
    """Reports each scope's GPU time one frame late, like a real GPU."""

    def __init__(self, gpu_ms):
        self.gpu_ms = gpu_ms
        self.in_flight = []
        self.ready = []

    def begin(self, path):
        return path

    def end(self, token):
        self.in_flight.append((token, self.gpu_ms))

    def collect(self):
        results, self.ready, self.in_flight = self.ready, self.in_flight, []
        return results


def run_frame(profiler, clock, work):

    profiler.begin_frame()
    work()
    profiler.end_frame()


def test_nested_scopes_build_paths_and_times():

    clock = FakeClock()
    profiler = Profiler(clock=clock)

    def work():
        with profiler.scope("Render"):
            clock.advance(1.0)
            with profiler.scope("Shadow"):
                clock.advance(2.0)
            with profiler.scope("Scene"):
                clock.advance(3.0)
        with profiler.scope("Swap"):
            clock.advance(4.0)

    for _ in range(3):
        run_frame(profiler, clock, work)

    stats = {s.path: s for s in profiler.stats()}

    assert list(stats) == ["Render", "Render/Shadow", "Render/Scene", "Swap"]
    assert stats["Render"].cpu_average_ms == pytest.approx(6.0)
    assert stats["Render/Shadow"].cpu_average_ms == pytest.approx(2.0)
    assert stats["Render/Scene"].depth == 1
    assert stats["Swap"].cpu_average_ms == pytest.approx(4.0)
    assert profiler.frame_average_ms == pytest.approx(10.0)


def test_late_child_is_listed_under_its_parent():

    clock = FakeClock()
    profiler = Profiler(clock=clock)

    def frame(with_poll):
        def work():
            with profiler.scope("Update"):
                if with_poll:
                    with profiler.scope("Shader poll"):
                        pass
                with profiler.scope("Camera"):
                    pass
            with profiler.scope("Swap"):
                pass
        return work

    run_frame(profiler, clock, frame(False))
    run_frame(profiler, clock, frame(True))

    assert [s.path for s in profiler.stats()] == [
        "Update",
        "Update/Camera",
        "Update/Shader poll",
        "Swap",
    ]


def test_repeated_scope_in_one_frame_is_summed():

    clock = FakeClock()
    profiler = Profiler(clock=clock)

    def work():
        for _ in range(4):
            with profiler.scope("Draw"):
                clock.advance(0.5)

    run_frame(profiler, clock, work)

    (stat,) = profiler.stats()

    assert stat.cpu_average_ms == pytest.approx(2.0)
    assert stat.calls == pytest.approx(4.0)


def test_skipped_scope_averages_per_frame():

    clock = FakeClock()
    profiler = Profiler(clock=clock)

    for frame in range(4):

        def work():
            if frame % 2 == 0:
                with profiler.scope("Shader poll"):
                    clock.advance(8.0)

        run_frame(profiler, clock, work)

    (stat,) = profiler.stats()

    # 8 ms every other frame = 4 ms per frame on average.
    assert stat.cpu_average_ms == pytest.approx(4.0)
    assert stat.cpu_max_ms == pytest.approx(8.0)


def test_scope_first_seen_later_is_aligned():

    clock = FakeClock()
    profiler = Profiler(clock=clock)

    run_frame(profiler, clock, lambda: clock.advance(1.0))

    def work():
        with profiler.scope("Late"):
            clock.advance(2.0)

    run_frame(profiler, clock, work)

    (stat,) = profiler.stats()

    # Padded with a zero for the frame before it existed.
    assert stat.cpu_average_ms == pytest.approx(1.0)


def test_history_is_rolling():

    clock = FakeClock()
    profiler = Profiler(history_size=2, clock=clock)

    for ms in (100.0, 1.0, 1.0):
        run_frame(profiler, clock, lambda: clock.advance(ms))

    assert profiler.frame_times_ms == pytest.approx([1.0, 1.0])


def test_gpu_results_attach_to_scopes():

    clock = FakeClock()
    profiler = Profiler(clock=clock)
    profiler.set_gpu_backend(FakeGpu(gpu_ms=3.0))

    def work():
        with profiler.scope("Scene", gpu=True):
            clock.advance(1.0)
        with profiler.scope("CPU only"):
            clock.advance(1.0)

    run_frame(profiler, clock, work)

    stats = {s.path: s for s in profiler.stats()}

    # Nothing yet: GPU results lag a frame.
    assert stats["Scene"].gpu_average_ms is None

    run_frame(profiler, clock, work)

    stats = {s.path: s for s in profiler.stats()}

    assert stats["Scene"].gpu_average_ms == pytest.approx(3.0)
    assert stats["CPU only"].gpu_average_ms is None


def test_failing_gpu_backend_falls_back_to_cpu_only():

    class BrokenGpu(FakeGpu):
        def collect(self):
            raise KeyError("GL_UNSIGNED_INT64_AMD")

    clock = FakeClock()
    profiler = Profiler(clock=clock)
    profiler.set_gpu_backend(BrokenGpu(gpu_ms=1.0))

    def work():
        with profiler.scope("Scene", gpu=True):
            clock.advance(1.0)

    run_frame(profiler, clock, work)
    run_frame(profiler, clock, work)

    assert not profiler.has_gpu_timing
    assert profiler.stats()[0].cpu_average_ms == pytest.approx(1.0)


def test_disabled_and_paused_record_nothing():

    clock = FakeClock()
    profiler = Profiler(clock=clock)

    def work():
        with profiler.scope("A"):
            clock.advance(1.0)

    profiler.enabled = False
    run_frame(profiler, clock, work)

    assert profiler.stats() == []
    assert profiler.frame_times_ms == []

    profiler.enabled = True
    run_frame(profiler, clock, work)

    profiler.paused = True
    run_frame(profiler, clock, work)

    assert len(profiler.frame_times_ms) == 1


def test_unbalanced_frames_are_rejected():

    profiler = Profiler(clock=FakeClock())

    with pytest.raises(Exception):
        profiler.end_frame()

    profiler.begin_frame()

    with pytest.raises(Exception):
        profiler.begin_frame()


def test_scope_outside_frame_is_a_no_op():

    profiler = Profiler(clock=FakeClock())

    with profiler.scope("Loading"):
        pass

    assert profiler.stats() == []


def test_scope_closes_on_exception():

    clock = FakeClock()
    profiler = Profiler(clock=clock)

    profiler.begin_frame()

    with pytest.raises(RuntimeError):
        with profiler.scope("Boom"):
            raise RuntimeError("boom")

    # Stack is balanced again, so the frame can end.
    profiler.end_frame()


def test_report_lists_every_scope():

    clock = FakeClock()
    profiler = Profiler(clock=clock)

    def work():
        with profiler.scope("Render"):
            with profiler.scope("Post"):
                clock.advance(1.0)

    run_frame(profiler, clock, work)

    report = profiler.format_report()

    assert "Render" in report
    assert "  Post" in report


def test_cprofile_capture_writes_files(tmp_path):

    capture = CProfileCapture(tmp_path)

    capture.request(3)

    def busy():
        return sum(i * i for i in range(2000))

    finished = []

    for _ in range(3):
        capture.begin_frame()
        busy()
        finished.append(capture.end_frame())

    # Completion is reported exactly once, on the last frame.
    assert finished == [False, False, True]
    assert not capture.active

    result = capture.last_result

    assert result.frames == 3
    assert result.profile_path.is_file()
    assert "By cumulative time" in result.summary_path.read_text()
    assert result.top_functions


def test_cprofile_capture_inactive_frames_are_ignored(tmp_path):

    capture = CProfileCapture(tmp_path)

    capture.begin_frame()
    capture.end_frame()

    assert capture.last_result is None
    assert list(tmp_path.iterdir()) == []
