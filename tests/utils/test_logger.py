import logging
from src.utils.logger import get_logger


def test_get_logger_returns_logger():
    logger = get_logger("test_task")
    assert isinstance(logger, logging.Logger)


def test_get_logger_name():
    logger = get_logger("my_task")
    assert logger.name == "pipeline.my_task"


def test_get_logger_same_instance():
    logger1 = get_logger("task_a")
    logger2 = get_logger("task_a")
    assert logger1 is logger2
