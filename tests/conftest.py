from core.logger import Logger


def pytest_configure(config):
    """The tests log to their own file (logs/tests.log), never over the game's logs/engine.log."""

    Logger.initialize(log_file="logs/tests.log")
