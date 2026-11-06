"""Full-season schedule from fixturedownload.com (rounds, kickoff times, venues).

football-data.co.uk only publishes the next round a few days ahead; this feed has the
whole season. Its team names differ ('Man Utd' vs 'Man United'), so names are mapped the
same way as StatsBomb's (futbol.ingest.xref): played matches with the same date and score
vote for which name is which.
"""

from __future__ import annotations

import logging

import pandas as pd
import requests

from futbol.db import get_engine, read_sql, upsert_rows
from futbol.ingest.football_data import USER_AGENT, current_season_start, season_label
from futbol.ingest.xref import vote_mapping

log = logging.getLogger(__name__)

FEED_URL = "https://fixturedownload.com/feed/json/{slug}-{year}"
SLUGS = {"E0": "epl", "SP1": "la-liga", "D1": "bundesliga", "I1": "serie-a", "F1": "ligue-1"}
MIN_VOTES = 3  # a team that has played 3+ matches is identified unambiguously


def fetch_schedule(league_code: str, year: int) -> pd.DataFrame:
    url = FEED_URL.format(slug=SLUGS[league_code], year=year)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def map_team_names(raw: pd.DataFrame, fd: pd.DataFrame) -> dict[str, str]:
    """feed name -> football-data name, voted by played fixtures (date + score)."""
    played = raw.dropna(subset=["HomeTeamScore", "AwayTeamScore"])
    feed = pd.DataFrame(
        {
            "match_date": pd.to_datetime(played.DateUtc).dt.date,
            "home_team_id": played.HomeTeam,  # vote_mapping only needs hashable ids
            "away_team_id": played.AwayTeam,
            "home_score": played.HomeTeamScore.astype(int),
            "away_score": played.AwayTeamScore.astype(int),
        }
    )
    fd = fd.assign(
        match_date=pd.to_datetime(fd.match_date).dt.date,
        home_goals=fd.home_goals.astype(int),
        away_goals=fd.away_goals.astype(int),
    )
    mapping = vote_mapping(feed, fd, min_votes=MIN_VOTES)  # fd_team -> feed name
    return dict(zip(mapping.team_id, mapping.fd_team, strict=True))


def ingest_schedule(leagues: list[str], year: int | None = None) -> int:
    year = year or current_season_start()
    season = season_label(year)
    total = 0
    for league in leagues:
        raw = fetch_schedule(league, year)
        fd = read_sql(
            "SELECT match_date, home_team, away_team, home_goals, away_goals FROM fd_matches "
            "WHERE league_code = :l AND season = :s AND home_goals IS NOT NULL",
            l=league,
            s=season,
        )
        names = map_team_names(raw, fd)
        unmapped = sorted(set(raw.HomeTeam) - set(names))
        if unmapped:
            log.warning(
                "%s: no football-data name for %s (too few played matches?)", league, unmapped
            )
        rows = pd.DataFrame(
            {
                "league_code": league,
                "season": season,
                "round": raw.RoundNumber.astype(int),
                "kickoff_utc": pd.to_datetime(raw.DateUtc, utc=True),
                "home_team": raw.HomeTeam.map(names),
                "away_team": raw.AwayTeam.map(names),
                "venue": raw.Location,
                "updated_at": pd.Timestamp.now(tz="UTC"),
            }
        ).dropna(subset=["home_team", "away_team"])
        # Keep the feed's names as aliases: the bet-slip OCR matches them too ("Man Utd").
        aliases = pd.DataFrame(
            {"league_code": league, "alias": list(names), "team": list(names.values()),
             "source": "fixturedownload"}
        )  # fmt: skip
        with get_engine().begin() as conn:
            upsert_rows(conn, "team_alias", aliases, key=["league_code", "alias"])
            total += upsert_rows(
                conn, "schedule", rows, key=["league_code", "season", "home_team", "away_team"]
            )
        log.info(
            "%s %s schedule: %d matches, %d teams mapped", league, season, len(rows), len(names)
        )
    return total
