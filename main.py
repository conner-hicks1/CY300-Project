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
        description="CY300 spaceflight game."
    )

    parser.add_argument(
        "--body",
        metavar="ID",
        help=(
            "Body to start at: a profile in assets/bodies/ "
            "(earth, moon, mars, titan...; default earth)."
        )
    )

    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Sandbox: launch anything free, travel anywhere, no missions (its own saves)."
    )

    parser.add_argument(
        "--dev",
        action="store_true",
        help="Start with the developer tools open (F3 toggles)."
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

    parser.add_argument(
        "--hide-ui",
        action="store_true",
        help="Start with the HUD hidden (F2 toggles)."
    )

    parser.add_argument(
        "--no-vsync",
        action="store_true",
        help="Disable VSync to measure uncapped frame time."
    )

    parser.add_argument(
        "--cprofile",
        type=int,
        metavar="FRAMES",
        help=(
            "Capture a cProfile of the first FRAMES frames into "
            "logs/profiles/."
        )
    )

    arguments = parser.parse_args(argv)

    if (
        arguments.cprofile is not None
        and arguments.cprofile <= 0
    ):

        parser.error(
            "--cprofile FRAMES must be positive."
        )

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
        "[Main] Starting."
    )

    # Application construction itself acquires no external
    # resources.

    application = Application()

    try:

        # -------------------------------------------------
        # Initialize
        # -------------------------------------------------

        application.initialize(
            body=arguments.body,
            dev=arguments.dev,
            sandbox=arguments.sandbox
        )

        if arguments.no_vsync:

            application.window.set_vsync(
                False
            )

        if arguments.cprofile is not None:

            application.cprofile_capture.request(
                arguments.cprofile
            )

        # -------------------------------------------------
        # Run
        # -------------------------------------------------

        if arguments.hide_ui:
            application.set_ui_visible(False)

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