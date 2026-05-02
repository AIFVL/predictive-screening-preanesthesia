import logging


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"api.{name}")
