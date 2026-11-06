"""Compute derived tables from raw events (Phase 3).

Builds, for every StatsBomb competition-season in the database:
  team_match_stats     one row per team per match (xG, xGA, PPDA, progressive passes...)
  player_season_stats  season totals per player (per-90 is computed at read time)
  match_xg             one row per match with both teams' xG, keyed by football-data names
                       (input for the xG prediction model)
and the football-data <-> StatsBomb team cross-reference.

Usage:  python scripts/build_metrics.py
"""

import logging

import pandas as pd

from futbol.db import get_engine, read_sql
from futbol.ingest.xref import build_team_xref
from futbol.logging_setup import setup_logging
from futbol.metrics import load_events, player_season_stats, team_match_stats

log = logging.getLogger("build_metrics")


def main() -> None:
    setup_logging()
    build_team_xref()

    seasons = read_sql("SELECT DISTINCT competition_id, season_id FROM matches ORDER BY 1, 2")
    team_frames, player_frames = [], []
    for c, s in seasons.itertuples(index=False):
        matches = read_sql(
            "SELECT match_id, home_team_id, away_team_id, home_score, away_score "
            "FROM matches WHERE competition_id = :c AND season_id = :s",
            c=c, s=s,
        )  # fmt: skip
        lineups = read_sql(
            "SELECT l.* FROM lineups l JOIN matches m USING (match_id) "
            "WHERE m.competition_id = :c AND m.season_id = :s",
            c=c, s=s,
        )  # fmt: skip
        events = load_events(c, s)
        loaded = events.match_id.unique()
        matches = matches[matches.match_id.isin(loaded)]
        log.info("comp %s season %s: %d matches, %d events", c, s, len(matches), len(events))

        team_frames.append(team_match_stats(events, matches))
        player_frames.append(
            player_season_stats(events, lineups).assign(competition_id=c, season_id=s)
        )

    engine = get_engine()
    # Derived tables are fully rebuilt each run: simpler and safe, they're cheap to recompute.
    # The view depends on team_match_stats, so it must go before the table is replaced.
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP VIEW IF EXISTS match_xg")
    pd.concat(team_frames).to_sql("team_match_stats", engine, if_exists="replace", index=False)
    pd.concat(player_frames).to_sql("player_season_stats", engine, if_exists="replace", index=False)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE VIEW match_xg AS
            SELECT x.league_code, c.season_name AS season, m.match_date,
                   hx.fd_team AS home_team, ax.fd_team AS away_team,
                   m.home_score AS home_goals, m.away_score AS away_goals,
                   h.xg AS home_xg, a.xg AS away_xg, h.npxg AS home_npxg, a.npxg AS away_npxg
            FROM matches m
            JOIN competitions c USING (competition_id, season_id)
            JOIN competition_xref x USING (competition_id)
            JOIN team_match_stats h ON h.match_id = m.match_id AND h.team_id = m.home_team_id
            JOIN team_match_stats a ON a.match_id = m.match_id AND a.team_id = m.away_team_id
            JOIN team_xref hx ON hx.league_code = x.league_code AND hx.team_id = m.home_team_id
            JOIN team_xref ax ON ax.league_code = x.league_code AND ax.team_id = m.away_team_id
            """
        )
    log.info("Derived tables rebuilt")


if __name__ == "__main__":
    main()
