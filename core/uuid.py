import uuid as _uuid

from core.assertions import engine_assert


class UUID:
    """
    Persistent 64-bit identifier.

    UUIDs are intended for serialized/persistent identity,
    such as entities, assets, scenes, and resources.
    """

    __slots__ = ("_value",)

    INVALID_VALUE = 0

    # =====================================================
    # Construction
    # =====================================================

    def __init__(
        self,
        value: int | None = None
    ):

        if value is None:

            value = self._generate()

        engine_assert(
            isinstance(value, int),
            "UUID value must be an integer."
        )

        engine_assert(
            0 <= value <= 0xFFFFFFFFFFFFFFFF,
            "UUID value must fit in 64 bits."
        )

        self._value = value

    # =====================================================
    # Generation
    # =====================================================

    @staticmethod
    def _generate() -> int:

        # uuid4() supplies 128 random bits.
        # Keep the lower 64 bits for a compact engine ID.

        value = (
            _uuid.uuid4().int
            & 0xFFFFFFFFFFFFFFFF
        )

        # Reserve zero as INVALID.

        while value == UUID.INVALID_VALUE:

            value = (
                _uuid.uuid4().int
                & 0xFFFFFFFFFFFFFFFF
            )

        return value

    # =====================================================
    # Factories
    # =====================================================

    @classmethod
    def invalid(cls):

        return cls(
            cls.INVALID_VALUE
        )

    @classmethod
    def from_int(
        cls,
        value: int
    ):

        return cls(
            value
        )

    @classmethod
    def from_hex(
        cls,
        value: str
    ):

        engine_assert(
            isinstance(value, str),
            "UUID hex value must be a string."
        )

        try:

            integer = int(
                value,
                16
            )

        except ValueError as error:

            raise ValueError(
                f"Invalid UUID hexadecimal string: "
                f"'{value}'"
            ) from error

        return cls(
            integer
        )

    # =====================================================
    # State
    # =====================================================

    @property
    def value(self) -> int:

        return self._value

    @property
    def is_valid(self) -> bool:

        return (
            self._value
            != self.INVALID_VALUE
        )

    # =====================================================
    # Conversion
    # =====================================================

    def to_hex(self) -> str:

        return f"{self._value:016x}"

    def __int__(self):

        return self._value

    def __str__(self):

        return self.to_hex()

    def __repr__(self):

        return (
            f"UUID(0x{self._value:016X})"
        )

    # =====================================================
    # Comparison / Hashing
    # =====================================================

    def __eq__(
        self,
        other
    ):

        if isinstance(
            other,
            UUID
        ):

            return (
                self._value
                == other._value
            )

        return NotImplemented

    def __hash__(self):

        return hash(
            self._value
        )