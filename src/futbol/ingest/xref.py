"""Map football-data team names to StatsBomb team ids.

Names differ between sources ('Man United' vs 'Manchester United'), so instead of fuzzy
string matching we match *fixtures*: two rows with the same league, date and score are
almost surely the same match, so their home teams are the same team (and their away
teams too). Every such pair is a vote; each name maps to the id with most votes.
A real team collects ~30+ votes per season; a coincidental pair gets 1 or 2.
"""

from __future__ import annotations

import logging

import pandas as pd

from futbol.db import get_engine, read_sql, upsert_rows

log = logging.getLogger(__name__)

MIN_VOTES = 10


def vote_mapping(sb: pd.DataFrame, fd: pd.DataFrame, min_votes: int = MIN_VOTES) -> pd.DataFrame:
    """Return fd_team -> team_id with vote counts.

    sb columns: match_date, home_team_id, away_team_id, home_score, away_score
    fd columns: match_date, home_team, away_team, home_goals, away_goals
    """
    joined = sb.merge(
        fd,
        left_on=["match_date", "home_score", "away_score"],
        right_on=["match_date", "home_goals", "away_goals"],
    )
    votes = (
        pd.concat(
            [
                joined[["home_team", "home_team_id"]].set_axis(["fd_team", "team_id"], axis=1),
                joined[["away_team", "away_team_id"]].set_axis(["fd_team", "team_id"], axis=1),
            ]
        )
        .value_counts()
        .rename("match_votes")
        .reset_index()
    )

    best = votes.sort_values("match_votes", ascending=False).drop_duplicates("fd_team")
    best = best[best["match_votes"] >= min_votes]
    dup = best[best.duplicated("team_id", keep=False)]
    if not dup.empty:
        raise ValueError(f"Ambiguous mapping, several names -> same team:\n{dup}")
    return best.reset_index(drop=True)


def build_team_xref() -> pd.DataFrame:
    """Build team_xref for every StatsBomb competition that has a football-data league."""
    sb = read_sql(
        """
        SELECT x.league_code, c.season_name AS season, m.match_date,
               m.home_team_id, m.away_team_id, m.home_score, m.away_score
        FROM matches m
        JOIN competitions c USING (competition_id, season_id)
        JOIN competition_xref x USING (competition_id)
        """
    )
    fd = read_sql(
        "SELECT league_code, season, match_date, home_team, away_team, home_goals, away_goals "
        "FROM fd_matches WHERE home_goals IS NOT NULL"
    )
    results = []
    for (league, season), sb_part in sb.groupby(["league_code", "season"]):
        fd_part = fd[(fd.league_code == league) & (fd.season == season)].drop(
            columns=["league_code", "season"]
        )
        mapping = vote_mapping(sb_part.drop(columns=["league_code", "season"]), fd_part)
        mapping.insert(0, "league_code", league)
        n_teams = sb_part.home_team_id.nunique()
        log.info("%s %s: mapped %d/%d teams (min votes %d)", league, season, len(mapping), n_teams,
                 mapping.match_votes.min())  # fmt: skip
        results.append(mapping)
    xref = pd.concat(results, ignore_index=True)
    with get_engine().begin() as conn:
        upsert_rows(conn, "team_xref", xref, key=["league_code", "fd_team"])
    return xref
