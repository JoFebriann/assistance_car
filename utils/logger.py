import logging
from pathlib import Path
from config.settings import LOG_DIR

LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str):

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(LOG_DIR / "system.log")
    file_handler.setFormatter(formatter)

    # Console handler (IMPORTANT for debugging)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # new line for better readability in logs
    logger.info(f"")
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger