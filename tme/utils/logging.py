"""
Logging utilities.

Copyright (c) 2025 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import os
import sys
import logging
from typing import Optional

__all__ = ["get_logger", "setup_logging", "debug_enabled", "LevelFormatter"]

LOGGER_NAME = "tme"

# Prevent No handler warnings when library is used without logging setup
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())


class LevelFormatter(logging.Formatter):
    """Format records with a per-level layout."""

    FORMATS = {
        logging.DEBUG: "[debug] %(name)s: %(message)s",
        logging.INFO: "%(message)s",
        logging.WARNING: "warning: %(message)s",
        logging.ERROR: "error: %(message)s",
        logging.CRITICAL: "error: %(message)s",
    }

    def format(self, record: logging.LogRecord) -> str:
        fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        return logging.Formatter(fmt).format(record)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a pytme logger.

    Parameters
    ----------
    name : str, optional
        Logger name. If None, returns the root tme logger.
        If provided, returns a child logger (e.g., "tme.matching").

    Returns
    -------
    logging.Logger
        The requested logger instance.
    """
    if name is None:
        return logging.getLogger(LOGGER_NAME)
    if not name.startswith(LOGGER_NAME):
        name = f"{LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def setup_logging(
    debug: bool = False,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure logger.

    Parameters
    ----------
    debug : bool, optional
        Enable debug output. Default is False (INFO level).
    log_file : str, optional
        Path to write log output to file.

    Returns
    -------
    logging.Logger
        Configured root logger for tme.

    Examples
    --------
    Basic setup with default INFO level:

    >>> from tme.utils.logging import setup_logging, get_logger
    >>> setup_logging()
    >>> logger = get_logger()
    >>> logger.info("Starting processing...")

    Enable debug output:

    >>> setup_logging(debug=True)

    Log to file for batch jobs:

    >>> setup_logging(log_file="job.log")
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(LevelFormatter())
    logger.addHandler(handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    return logger


def debug_enabled() -> bool:
    """Check whether PYTME_DEBUG is set to some value other than '' or '0'."""
    return os.environ.get("PYTME_DEBUG", "").strip() not in ("", "0")
