from core.logger import Logger


class EngineAssertionError(AssertionError):
    """Raised when an internal engine invariant is violated."""
    pass


def engine_assert(
    condition: bool,
    message: str = "Engine assertion failed."
):

    if condition:
        return

    Logger.critical(
        "[Assertion] %s",
        message
    )

    raise EngineAssertionError(
        message
    )


def engine_assert_not_none(
    value,
    message: str = "Expected value to not be None."
):

    engine_assert(
        value is not None,
        message
    )

    return value