"""Logging configuration shared by scripts, notebooks and the dashboard."""

from __future__ import annotations

import logging

from futbol.config import get_settings


def setup_logging() -> None:
    """Configure the root logger once. Modules then use logging.getLogger(__name__)."""
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
