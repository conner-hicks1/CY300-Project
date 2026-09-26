from dataclasses import dataclass


@dataclass(
    frozen=True,
    slots=True
)
class Handle:

    manager_id: int
    index: int
    generation: int

    INVALID_MANAGER = 0
    INVALID_INDEX = -1

    # =====================================================
    # Factory
    # =====================================================

    @classmethod
    def invalid(cls):

        return cls(
            manager_id=cls.INVALID_MANAGER,
            index=cls.INVALID_INDEX,
            generation=0
        )

    # =====================================================
    # State
    # =====================================================

    @property
    def is_valid(self) -> bool:

        return (
            self.manager_id != self.INVALID_MANAGER
            and
            self.index != self.INVALID_INDEX
        )

    # =====================================================
    # Representation
    # =====================================================

    def __repr__(self):

        if not self.is_valid:
            return "Handle(INVALID)"

        return (
            f"Handle("
            f"manager={self.manager_id}, "
            f"index={self.index}, "
            f"generation={self.generation}"
            f")"
        )