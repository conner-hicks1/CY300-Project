import logging
import sys

from pathlib import Path


class Logger:

    _logger = None
    _initialized = False

    # =====================================================
    # Initialization
    # =====================================================

    @classmethod
    def initialize(
        cls,
        name="Engine",
        level=logging.DEBUG,
        log_to_file=True,
        log_file="logs/engine.log"
    ):

        if cls._initialized:
            return

        logger = logging.getLogger(
            name
        )

        logger.setLevel(
            level
        )

        logger.propagate = False

        # Remove stale handlers.
        for handler in logger.handlers[:]:

            handler.close()

            logger.removeHandler(
                handler
            )

        formatter = logging.Formatter(
            fmt=(
                "[%(asctime)s.%(msecs)03d] "
                "[%(levelname)-8s] "
                "%(message)s"
            ),
            datefmt="%H:%M:%S"
        )

        # -------------------------------------------------
        # Console
        # -------------------------------------------------

        console_handler = logging.StreamHandler(
            sys.stdout
        )

        console_handler.setLevel(
            level
        )

        console_handler.setFormatter(
            formatter
        )

        logger.addHandler(
            console_handler
        )

        # -------------------------------------------------
        # File
        # -------------------------------------------------

        if log_to_file:

            path = Path(
                log_file
            )

            path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            file_handler = logging.FileHandler(
                path,
                mode="w",
                encoding="utf-8"
            )

            file_handler.setLevel(
                level
            )

            file_handler.setFormatter(
                formatter
            )

            logger.addHandler(
                file_handler
            )

        cls._logger = logger
        cls._initialized = True

        cls.info(
            "[Core] Logger initialized."
        )

    # =====================================================
    # Internal
    # =====================================================

    @classmethod
    def _get_logger(cls):

        if not cls._initialized:
            cls.initialize()

        return cls._logger

    # =====================================================
    # Logging
    # =====================================================

    @classmethod
    def debug(
        cls,
        message,
        *args
    ):

        cls._get_logger().debug(
            message,
            *args
        )

    @classmethod
    def info(
        cls,
        message,
        *args
    ):

        cls._get_logger().info(
            message,
            *args
        )

    @classmethod
    def warning(
        cls,
        message,
        *args
    ):

        cls._get_logger().warning(
            message,
            *args
        )

    @classmethod
    def error(
        cls,
        message,
        *args
    ):

        cls._get_logger().error(
            message,
            *args
        )

    @classmethod
    def critical(
        cls,
        message,
        *args
    ):

        cls._get_logger().critical(
            message,
            *args
        )

    @classmethod
    def exception(
        cls,
        message,
        *args
    ):

        cls._get_logger().exception(
            message,
            *args
        )

    # =====================================================
    # Configuration
    # =====================================================

    @classmethod
    def set_level(
        cls,
        level
    ):

        logger = cls._get_logger()

        logger.setLevel(
            level
        )

        for handler in logger.handlers:

            handler.setLevel(
                level
            )

    # =====================================================
    # Shutdown
    # =====================================================

    @classmethod
    def shutdown(cls):

        if not cls._initialized:
            return

        cls.info(
            "[Core] Logger shutting down."
        )

        for handler in cls._logger.handlers[:]:

            handler.flush()
            handler.close()

            cls._logger.removeHandler(
                handler
            )

        cls._logger = None
        cls._initialized = False