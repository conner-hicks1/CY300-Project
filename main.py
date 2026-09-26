import argparse

from application import Application

from core.assertions import (
    EngineAssertionError
)
from core.exceptions import (
    EngineError
)
from core.logger import Logger


def parse_arguments(argv=None):

    parser = argparse.ArgumentParser(
        description="OpenGL engine demo."
    )

    parser.add_argument(
        "--exit-after",
        type=float,
        metavar="SECONDS",
        help="Quit automatically after this many seconds (smoke test)."
    )

    parser.add_argument(
        "--screenshot",
        metavar="PATH",
        help="With --exit-after, save the final frame to PATH."
    )

    arguments = parser.parse_args(argv)

    if (
        arguments.screenshot is not None
        and arguments.exit_after is None
    ):

        parser.error(
            "--screenshot requires --exit-after."
        )

    return arguments


def main():

    arguments = parse_arguments()

    # =====================================================
    # Logging
    # =====================================================

    Logger.initialize()

    Logger.info(
        "[Main] Starting engine."
    )

    # Application construction itself acquires no external
    # resources.

    application = Application()

    try:

        # -------------------------------------------------
        # Initialize
        # -------------------------------------------------

        application.initialize()

        # -------------------------------------------------
        # Run
        # -------------------------------------------------

        application.run(
            exit_after=arguments.exit_after,
            screenshot_path=arguments.screenshot
        )

    except EngineAssertionError:

        Logger.exception(
            "[Main] Engine assertion failed."
        )

        raise

    except EngineError:

        Logger.exception(
            "[Main] Fatal engine error."
        )

        raise

    except Exception:

        Logger.exception(
            "[Main] Unexpected exception."
        )

        raise

    finally:

        # -------------------------------------------------
        # Application Shutdown
        # -------------------------------------------------

        try:

            application.shutdown()

        except Exception:

            Logger.exception(
                "[Main] Exception during application shutdown."
            )

        # -------------------------------------------------
        # Logging Shutdown
        # -------------------------------------------------

        Logger.info(
            "[Main] Engine shutdown."
        )

        Logger.shutdown()


if __name__ == "__main__":

    main()