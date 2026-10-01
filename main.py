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
        "--scene",
        metavar="PATH",
        help=(
            "Scene file to open (default: "
            "assets/scenes/planet.scene.json, or the built-in "
            "planet demo if that is missing)."
        )
    )

    parser.add_argument(
        "--play",
        action="store_true",
        help="Start in Play mode (simulation running)."
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
        help="Start with the editor and debug UI hidden (F1 toggles)."
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
        "[Main] Starting engine."
    )

    # Application construction itself acquires no external
    # resources.

    application = Application()

    try:

        # -------------------------------------------------
        # Initialize
        # -------------------------------------------------

        application.initialize(
            scene_path=arguments.scene
        )

        if arguments.play:
            application.editor.play()

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