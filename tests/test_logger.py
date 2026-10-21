from core import logger


def test_earlier_logs_are_kept(tmp_path):

    path = tmp_path / "engine.log"

    for run in range(5):

        logger._keep_earlier(path)

        path.write_text(f"run {run}", encoding="utf-8")

    assert path.read_text(encoding="utf-8") == "run 4"

    assert [tmp_path.joinpath(f"engine.{index}.log").read_text(encoding="utf-8") for index in range(1, logger.KEEP + 1)] == ["run 3", "run 2", "run 1"]

    assert not tmp_path.joinpath(f"engine.{logger.KEEP + 1}.log").exists()

    # An empty log is not worth keeping.
    path.write_text("", encoding="utf-8")

    logger._keep_earlier(path)

    assert tmp_path.joinpath("engine.1.log").read_text(encoding="utf-8") == "run 3"
