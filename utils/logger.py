import os
import logging

def get_logger(name: str, log_file: str) -> logging.Logger:
    """
    Returns a logger with the specified name and log file.
    Ensures that loggers are only configured once.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        # Add file handler only once
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger