"""Bring everything that can be current up to today, in ~1 minute.

  1. football-data: re-download the season in progress (results, stats, odds) of the
     Big-5 and of Liga MX / Argentina / Brasil,
     the upcoming fixtures file and the full season schedule (fixturedownload.com)
  2. API-Football: new finished fixtures (stats, lineups), next fixtures, injury reports
  3. models: walk-forward predictions for the current season, predictions for the
     upcoming fixtures and today's team ratings, with the stored tuned parameters

StatsBomb event data is not touched: there is no newer free event data to fetch.

Usage:  python scripts/update.py
"""

import logging

import requests

from futbol.db import init_schema, read_sql
from futbol.experiments import load_params, update_current_season
from futbol.ingest.api_fixtures import download
from futbol.ingest.api_football import ApiFootballError, has_key
from futbol.ingest.api_matchday import update_matchday
from futbol.ingest.football_data import (
    current_season_start,
    drop_stale_fixtures,
    ingest_current_season,
    ingest_extra_leagues,
    ingest_fixtures,
)
from futbol.ingest.schedule import ingest_schedule
from futbol.leagues import ALL_LEAGUES, BIG5, EXTRA
from futbol.logging_setup import setup_logging

log = logging.getLogger("update")


def update_api_football() -> None:
    """New finished fixtures of the season in progress, with team and player stats.

    Calendar-year leagues (Brazil, Argentina) may already be in next year's season, so both
    start years are tried; a season without fixtures costs one request. If the paid plan has
    lapsed the API refuses the current season: that is logged and the rest of the update stands.
    """
    if not has_key():
        log.info("API-Football: no APIFOOTBALL_KEY, skipped")
        return
    year = current_season_start()
    try:
        used = download(ALL_LEAGUES, [year, year + 1], budget=400)
        log.info("API-Football: %d requests", used)
    except (ApiFootballError, requests.RequestException) as exc:
        log.warning("API-Football update skipped: %s", exc)
    try:  # next fixtures and injury reports: 2 requests per league
        log.info("API-Football matchday: %d requests", update_matchday(ALL_LEAGUES))
    except (ApiFootballError, requests.RequestException) as exc:
        log.warning("API-Football matchday skipped: %s", exc)


def main() -> None:
    setup_logging()
    init_schema()
    ingest_current_season(BIG5)
    ingest_fixtures(BIG5)
    ingest_schedule(BIG5)
    ingest_extra_leagues(EXTRA)  # Liga MX, Argentina, Brasil: results + odds
    drop_stale_fixtures()
    update_current_season(load_params(), ALL_LEAGUES)
    update_api_football()

    latest = read_sql(
        "SELECT league_code, max(match_date) AS last_played, count(*) AS played "
        "FROM fd_matches WHERE home_goals IS NOT NULL "
        "AND match_date > CURRENT_DATE - 60 GROUP BY 1 ORDER BY 1"
    )
    log.info("Up to date:\n%s", latest.to_string(index=False))


if __name__ == "__main__":
    main()
