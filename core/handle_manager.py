from typing import (
    Generic,
    TypeVar,
    cast
)

from core.assertions import (
    engine_assert,
    engine_fail
)
from core.handle import Handle
from core.logger import Logger


T = TypeVar("T")


class _Slot(
    Generic[T]
):

    __slots__ = (
        "value",
        "generation",
        "occupied"
    )

    def __init__(self):

        self.value: T | None = None
        self.generation = 0
        self.occupied = False


class HandleManager(
    Generic[T]
):

    _next_manager_id = 1

    # =====================================================
    # Construction
    # =====================================================

    def __init__(
        self,
        name="HandleManager"
    ):

        self._name = name

        self._manager_id = (
            HandleManager._next_manager_id
        )

        HandleManager._next_manager_id += 1

        self._slots: list[_Slot[T]] = []

        self._free_indices: list[int] = []

        self._active_count = 0

        Logger.debug(
            "[%s] Created with manager ID %d.",
            self._name,
            self._manager_id
        )

    # =====================================================
    # Creation
    # =====================================================

    def create(
        self,
        value: T
    ) -> Handle:

        engine_assert(
            value is not None,
            f"{self._name} cannot store None."
        )

        # -------------------------------------------------
        # Reuse Free Slot
        # -------------------------------------------------

        if self._free_indices:

            index = self._free_indices.pop()

            slot = self._slots[
                index
            ]

        # -------------------------------------------------
        # New Slot
        # -------------------------------------------------

        else:

            index = len(
                self._slots
            )

            slot = _Slot[T]()

            self._slots.append(
                slot
            )

        engine_assert(
            not slot.occupied,
            (
                f"{self._name} attempted to allocate "
                f"an occupied slot."
            )
        )

        slot.value = value
        slot.occupied = True

        self._active_count += 1

        return Handle(
            manager_id=self._manager_id,
            index=index,
            generation=slot.generation
        )

    # =====================================================
    # Validation
    # =====================================================

    def is_valid(
        self,
        handle: Handle
    ) -> bool:

        if not isinstance(
            handle,
            Handle
        ):

            return False

        if not handle.is_valid:

            return False

        if (
            handle.manager_id
            != self._manager_id
        ):

            return False

        if not (
            0
            <= handle.index
            < len(self._slots)
        ):

            return False

        slot = self._slots[
            handle.index
        ]

        return (
            slot.occupied
            and
            slot.generation
            == handle.generation
        )

    # =====================================================
    # Access
    # =====================================================

    def get(
        self,
        handle: Handle
    ) -> T:

        # Per-frame hot path: format the message only on
        # failure.

        if not self.is_valid(handle):

            engine_fail(
                f"{self._name} received invalid "
                f"or stale handle: {handle}"
            )

        value = self._slots[
            handle.index
        ].value

        return cast(
            T,
            value
        )

    def try_get(
        self,
        handle: Handle
    ) -> T | None:

        if not self.is_valid(
            handle
        ):

            return None

        return self._slots[
            handle.index
        ].value

    # =====================================================
    # Replacement
    # =====================================================

    def set(
        self,
        handle: Handle,
        value: T
    ):

        engine_assert(
            self.is_valid(handle),
            (
                f"{self._name} received invalid "
                f"or stale handle: {handle}"
            )
        )

        engine_assert(
            value is not None,
            f"{self._name} cannot store None."
        )

        self._slots[
            handle.index
        ].value = value

    # =====================================================
    # Destruction
    # =====================================================

    def destroy(
        self,
        handle: Handle
    ):

        engine_assert(
            self.is_valid(handle),
            (
                f"{self._name} cannot destroy "
                f"invalid or stale handle: {handle}"
            )
        )

        slot = self._slots[
            handle.index
        ]

        slot.value = None
        slot.occupied = False

        # Every destruction changes the generation.
        # All previous handles to this slot are now stale.

        slot.generation += 1

        self._free_indices.append(
            handle.index
        )

        self._active_count -= 1

    # =====================================================
    # Iteration
    # =====================================================

    def active_handles(self):

        for index, slot in enumerate(
            self._slots
        ):

            if not slot.occupied:
                continue

            yield Handle(
                manager_id=self._manager_id,
                index=index,
                generation=slot.generation
            )

    # =====================================================
    # Clear
    # =====================================================

    def clear(self):

        handles = list(
            self.active_handles()
        )

        for handle in handles:

            self.destroy(
                handle
            )

    # =====================================================
    # Information
    # =====================================================

    @property
    def manager_id(self) -> int:

        return self._manager_id

    @property
    def active_count(self) -> int:

        return self._active_count

    @property
    def capacity(self) -> int:

        return len(
            self._slots
        )

    @property
    def free_count(self) -> int:

        return len(
            self._free_indices
        )

    def __len__(self):

        return self._active_count