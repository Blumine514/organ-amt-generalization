"""Logging utilities."""

import logging


def get_logger(name="amt"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger(name)
