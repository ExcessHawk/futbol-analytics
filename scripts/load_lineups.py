"""Fill the lineup tables from the raw API-Football payloads already on disk (no requests).

Needed once on a database created before lineups were stored; new downloads store them
automatically. Also fetches next fixtures and injury reports (2 requests per league).

Usage:  python scripts/load_lineups.py
"""

import logging

from futbol.db import init_schema
from futbol.ingest.api_fixtures import load_lineups_from_disk
from futbol.ingest.api_football import has_key
from futbol.ingest.api_matchday import update_matchday
from futbol.leagues import ALL_LEAGUES
from futbol.logging_setup import setup_logging

log = logging.getLogger("load_lineups")

if __name__ == "__main__":
    setup_logging()
    init_schema()
    log.info("Lineups loaded for %d fixtures", load_lineups_from_disk())
    if has_key():
        log.info("Matchday: %d requests", update_matchday(ALL_LEAGUES))
