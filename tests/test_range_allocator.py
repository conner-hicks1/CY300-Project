import random

import pytest

from graphics.range_allocator import RangeAllocator


def test_allocates_first_fit_in_order():

    allocator = RangeAllocator(100)

    assert allocator.allocate(10) == 0
    assert allocator.allocate(20) == 10
    assert allocator.used == 30
    assert allocator.free_ranges == [(30, 70)]


def test_returns_none_when_full():

    allocator = RangeAllocator(10)

    assert allocator.allocate(10) == 0
    assert allocator.allocate(1) is None


def test_free_merges_neighbours():

    allocator = RangeAllocator(30)

    a = allocator.allocate(10)
    b = allocator.allocate(10)
    c = allocator.allocate(10)

    allocator.free(a, 10)
    allocator.free(c, 10)

    assert allocator.free_ranges == [(0, 10), (20, 10)]

    allocator.free(b, 10)

    assert allocator.free_ranges == [(0, 30)]
    assert allocator.used == 0


def test_freed_space_is_reused():

    allocator = RangeAllocator(30)

    allocator.allocate(10)
    middle = allocator.allocate(10)
    allocator.allocate(10)

    allocator.free(middle, 10)

    assert allocator.allocate(5) == middle
    assert allocator.allocate(5) == middle + 5
    assert allocator.allocate(1) is None


def test_double_free_is_rejected():

    allocator = RangeAllocator(20)

    start = allocator.allocate(10)
    allocator.free(start, 10)

    with pytest.raises(Exception, match="twice"):
        allocator.free(start, 10)


def test_grow_extends_trailing_free_space():

    allocator = RangeAllocator(10)

    allocator.allocate(4)
    allocator.grow(20)

    assert allocator.free_ranges == [(4, 16)]
    assert allocator.allocate(16) == 4


def test_grow_when_full():

    allocator = RangeAllocator(10)

    allocator.allocate(10)
    allocator.grow(15)

    assert allocator.allocate(5) == 10


def test_random_churn_never_overlaps():

    rng = random.Random(7)

    allocator = RangeAllocator(1000)

    live: dict[int, int] = {}

    for _ in range(2000):

        if live and rng.random() < 0.45:

            start = rng.choice(list(live))
            allocator.free(start, live.pop(start))

        else:

            size = rng.randint(1, 40)
            start = allocator.allocate(size)

            if start is not None:
                live[start] = size

        # Invariants: live ranges disjoint, accounting exact.
        spans = sorted(live.items())

        for (a, sa), (b, _) in zip(spans, spans[1:]):
            assert a + sa <= b

        assert allocator.used == sum(live.values())

    for start, size in list(live.items()):
        allocator.free(start, size)

    assert allocator.free_ranges == [(0, 1000)]
