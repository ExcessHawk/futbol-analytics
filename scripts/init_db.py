"""Create (or update) the database schema.

Usage:  python scripts/init_db.py
"""

import logging

from futbol.db import init_schema
from futbol.logging_setup import setup_logging

if __name__ == "__main__":
    setup_logging()
    init_schema()
    logging.getLogger("init_db").info("Schema ready")
