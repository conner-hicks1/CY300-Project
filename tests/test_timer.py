import time

import pytest

from core import timer as timer_module
from core.timer import Timer


class FakeClock:

    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):

    fake = FakeClock()

    monkeypatch.setattr(timer_module.time, "perf_counter", fake)

    return fake


def test_delta_time_is_clamped(clock):

    timer = Timer(max_delta_time=0.1)

    clock.now += 5.0
    timer.update()

    assert timer.delta_time == pytest.approx(0.1)
    assert timer.unclamped_delta_time == pytest.approx(5.0)


def test_reset_excludes_loading_time(clock):

    timer = Timer()

    # Slow "loading" between construction and the loop.
    clock.now += 3.0
    timer.reset()

    clock.now += 0.016
    timer.update()

    assert timer.delta_time == pytest.approx(0.016)


def test_rejects_non_positive_clamp():

    with pytest.raises(ValueError):
        Timer(max_delta_time=0.0)


def test_real_clock_smoke():

    timer = Timer()

    time.sleep(0.01)
    timer.update()

    assert 0.0 < timer.delta_time <= timer.max_delta_time
