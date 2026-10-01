import bisect

from core.assertions import engine_assert


class RangeAllocator:

    # =====================================================
    # Range Allocator
    # =====================================================
    #
    # Hands out [start, start + size) ranges of a linear
    # space (vertex or index slots in a GPU buffer), and
    # takes them back. First fit over a sorted free list;
    # freed ranges merge with free neighbours, so streaming
    # chunks in and out does not fragment the buffer
    # forever.
    #
    # Pure Python; graphics/geometry_pool.py maps it onto
    # GL buffers and grows them when allocate() fails.

    def __init__(
        self,
        capacity: int
    ):

        engine_assert(
            capacity >= 0,
            "RangeAllocator capacity cannot be negative."
        )

        self._capacity = capacity

        # Sorted, non-overlapping, non-adjacent (start, size).
        self._free: list[tuple[int, int]] = (
            [(0, capacity)]
            if capacity > 0
            else []
        )

        self._used = 0

    # =====================================================
    # Queries
    # =====================================================

    @property
    def capacity(
        self
    ) -> int:

        return self._capacity

    @property
    def used(
        self
    ) -> int:

        return self._used

    @property
    def free_ranges(
        self
    ) -> list[tuple[int, int]]:

        return list(self._free)

    @property
    def largest_free(
        self
    ) -> int:

        return max(
            (size for _, size in self._free),
            default=0
        )

    # =====================================================
    # Allocate / Free
    # =====================================================

    def allocate(
        self,
        size: int
    ) -> int | None:
        """Start of a free range of `size`, or None if none fits."""

        engine_assert(
            size > 0,
            "Allocation size must be positive."
        )

        for index, (start, free_size) in enumerate(self._free):

            if free_size < size:
                continue

            if free_size == size:
                del self._free[index]
            else:
                self._free[index] = (start + size, free_size - size)

            self._used += size

            return start

        return None

    def free(
        self,
        start: int,
        size: int
    ):

        engine_assert(
            size > 0 and 0 <= start and start + size <= self._capacity,
            f"Freed range [{start}, {start + size}) is outside the allocator."
        )

        index = bisect.bisect_left(
            self._free,
            (start, 0)
        )

        # Must not overlap a free neighbour (double free).

        if index > 0:

            previous_start, previous_size = self._free[index - 1]

            engine_assert(
                previous_start + previous_size <= start,
                f"Range [{start}, {start + size}) freed twice."
            )

        if index < len(self._free):

            engine_assert(
                start + size <= self._free[index][0],
                f"Range [{start}, {start + size}) freed twice."
            )

        self._free.insert(
            index,
            (start, size)
        )

        self._used -= size

        # Merge with the next, then the previous range.

        if index + 1 < len(self._free):

            next_start, next_size = self._free[index + 1]

            if start + size == next_start:

                self._free[index] = (start, size + next_size)

                del self._free[index + 1]

        if index > 0:

            previous_start, previous_size = self._free[index - 1]

            current_start, current_size = self._free[index]

            if previous_start + previous_size == current_start:

                self._free[index - 1] = (previous_start, previous_size + current_size)

                del self._free[index]

    def grow(
        self,
        new_capacity: int
    ):
        """Extend the space (the caller grows the GL buffer to match)."""

        engine_assert(
            new_capacity >= self._capacity,
            "RangeAllocator can only grow."
        )

        added = new_capacity - self._capacity

        if added == 0:
            return

        old_capacity = self._capacity

        self._capacity = new_capacity

        # Append the new space, merging with a trailing free range.

        if self._free and sum(self._free[-1]) == old_capacity:

            start, size = self._free[-1]

            self._free[-1] = (start, size + added)

        else:

            self._free.append(
                (old_capacity, added)
            )
