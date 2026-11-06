"""Download finished fixtures with team and player statistics from API-Football.

Usage:
  python scripts/download_api_football.py                         # all leagues, 2015 -> current
  python scripts/download_api_football.py --seasons 2026          # only the season in progress
  python scripts/download_api_football.py --leagues SP1 --seasons 2024 --budget 30

Resumable: fixtures already stored are skipped. Paced by futbol.ingest.api_football.RateLimiter.
"""

import argparse
import logging

from futbol.db import init_schema, read_sql
from futbol.ingest.api_fixtures import download
from futbol.ingest.api_football import LIMITER, account_status
from futbol.leagues import ALL_LEAGUES
from futbol.logging_setup import setup_logging

log = logging.getLogger("download_api_football")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--leagues", nargs="+", default=ALL_LEAGUES)
    parser.add_argument("--seasons", nargs="+", type=int, default=list(range(2015, 2027)))
    parser.add_argument("--budget", type=int, default=3000, help="max requests this run")
    args = parser.parse_args()
    setup_logging()
    init_schema()

    status = account_status()  # free call: plan and usage
    log.info(
        "Plan %s, %s/%s requests used today",
        status["plan"],
        status["requests_today"],
        status["limit_day"],
    )
    used = download(args.leagues, args.seasons, args.budget)
    log.info("Done: %d requests this run, %s left today", used, LIMITER.remaining_today)
    summary = read_sql(
        "SELECT f.league_code, f.season, count(DISTINCT f.fixture_id) AS fixtures, "
        "count(DISTINCT p.player_id) AS players, round(avg(t.xg)::numeric, 2) AS avg_xg, "
        "count(t.xg) AS xg_rows FROM api_fixture f "
        "LEFT JOIN api_player_match p USING (fixture_id) LEFT JOIN api_team_match t USING (fixture_id) "
        "GROUP BY 1, 2 ORDER BY 1, 2"
    )
    log.info("Stored:\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
