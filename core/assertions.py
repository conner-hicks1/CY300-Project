from core.logger import Logger


class EngineAssertionError(AssertionError):
    """Raised when an internal engine invariant is violated."""
    pass


def engine_assert(
    condition: bool,
    message: str = "Engine assertion failed."
):
    """
    Note: `message` is built by the caller before this
    runs, even when the condition holds. In per-frame hot
    paths, prefer

        if not condition:
            engine_fail(f"... {expensive} ...")

    so the f-string is only formatted on failure.
    """

    if condition:
        return

    engine_fail(
        message
    )


def engine_fail(
    message: str = "Engine assertion failed."
):

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