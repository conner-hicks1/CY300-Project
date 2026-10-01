import threading
import time

import pytest

from core.jobs import JobSystem


@pytest.fixture
def jobs():

    system = JobSystem(workers=2)

    yield system

    system.shutdown(wait=True)


def drain(jobs, timeout=5.0):

    deadline = time.monotonic() + timeout

    while jobs.pending and time.monotonic() < deadline:
        jobs.process_completions(budget_seconds=1.0)
        time.sleep(0.001)


def test_work_runs_off_thread_and_completes_on_caller(jobs):

    main = threading.get_ident()

    seen = {}

    def work():
        seen["worker"] = threading.get_ident()
        return 21

    def done(result):
        seen["callback"] = threading.get_ident()
        seen["result"] = result * 2

    jobs.submit(work, on_complete=done)

    drain(jobs)

    assert seen["worker"] != main
    assert seen["callback"] == main
    assert seen["result"] == 42


def test_callbacks_only_run_inside_process_completions(jobs):

    results = []

    job = jobs.submit(lambda: 1, on_complete=results.append)

    time.sleep(0.05)

    assert results == []
    assert not job.done

    drain(jobs)

    assert results == [1]
    assert job.done


def test_errors_go_to_on_error(jobs):

    errors = []

    def fail():
        raise ValueError("bad input")

    jobs.submit(fail, on_error=errors.append)

    drain(jobs)

    assert isinstance(errors[0], ValueError)


def test_unhandled_error_is_logged_not_raised(jobs):

    jobs.submit(lambda: 1 / 0)

    drain(jobs)

    assert jobs.pending == 0


def test_cancelled_job_skips_callback(jobs):

    release = threading.Event()
    results = []

    blocker = jobs.submit(release.wait)
    other = jobs.submit(release.wait)
    queued = jobs.submit(lambda: "late", on_complete=results.append)

    queued.cancel()
    release.set()

    drain(jobs)

    assert results == []
    assert queued.done and blocker.done and other.done


def test_budget_spreads_completions_over_frames(jobs):

    handled = []

    def slow_callback(_):
        time.sleep(0.01)
        handled.append(1)

    for _ in range(5):
        jobs.submit(lambda: None, on_complete=slow_callback)

    time.sleep(0.1)

    # At least one per call, then stop once over budget.
    first = jobs.process_completions(budget_seconds=0.005)

    assert first == 1

    drain(jobs)

    assert len(handled) == 5


def test_submit_after_shutdown_is_rejected():

    system = JobSystem(workers=1)
    system.shutdown(wait=True)

    with pytest.raises(Exception, match="shut down"):
        system.submit(lambda: None)
