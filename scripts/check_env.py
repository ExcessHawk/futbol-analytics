"""Sanity check for the development environment.

Usage:  python scripts/check_env.py
Verifies the Python version, that every dependency imports, and that the DB answers.
"""

from __future__ import annotations

import importlib
import logging
import sys

from futbol.config import get_settings
from futbol.db import ping
from futbol.logging_setup import setup_logging

log = logging.getLogger("check_env")

# import name -> what it is used for (import names differ from pip names sometimes)
REQUIRED_MODULES = {
    "pandas": "data wrangling",
    "numpy": "numerics",
    "statsbombpy": "StatsBomb open data",
    "sqlalchemy": "database toolkit",
    "psycopg": "PostgreSQL driver",
    "scipy": "statistics / optimisation",
    "sklearn": "metrics and calibration",
    "matplotlib": "plotting",
    "mplsoccer": "pitch plots",
    "streamlit": "dashboard",
    "cv2": "image preprocessing",
    "pytesseract": "OCR wrapper",
    "rapidfuzz": "fuzzy team names",
}


def check_imports() -> bool:
    ok = True
    for module, purpose in REQUIRED_MODULES.items():
        try:
            mod = importlib.import_module(module)
            log.info("OK   %-12s %-10s (%s)", module, getattr(mod, "__version__", "?"), purpose)
        except ImportError as exc:
            log.error("FAIL %-12s %s", module, exc)
            ok = False
    return ok


def check_db() -> bool:
    settings = get_settings()
    try:
        version = ping()
        log.info("OK   database at %s:%s -> %s", settings.db_host, settings.db_port, version[:40])
        return True
    except Exception as exc:  # broad on purpose: any failure means "not reachable"
        log.error("FAIL database at %s:%s -> %s", settings.db_host, settings.db_port, exc)
        return False


def main() -> int:
    setup_logging()
    log.info("Python %s", sys.version.split()[0])
    results = [check_imports(), check_db()]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
