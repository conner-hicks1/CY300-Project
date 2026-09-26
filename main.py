from application import Application

from core.assertions import (
    EngineAssertionError
)
from core.exceptions import (
    EngineError
)
from core.logger import Logger


def main():

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

        application.run()

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