from dataclasses import dataclass


@dataclass(
    frozen=True,
    slots=True
)
class Entity:

    index: int
    generation: int

    INVALID_INDEX = -1

    @classmethod
    def invalid(cls):

        return cls(
            index=cls.INVALID_INDEX,
            generation=0
        )

    @property
    def is_valid(self) -> bool:

        return (
            self.index
            != self.INVALID_INDEX
        )

    def __repr__(self):

        if not self.is_valid:

            return "Entity(INVALID)"

        return (
            f"Entity("
            f"index={self.index}, "
            f"generation={self.generation}"
            f")"
        )