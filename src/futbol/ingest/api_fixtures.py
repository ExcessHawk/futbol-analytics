"""Bulk download of finished fixtures with per-team and per-player statistics.

/fixtures?ids= returns up to 20 fixtures per request, each with events, lineups, team
statistics and player statistics: a 380-match season costs 19 requests (+1 for the list).

Raw responses are stored gzipped under data/raw/api_football/{league}/{season}/{id}.json.gz
before parsing, so the data outlives the paid plan and can be re-parsed for new fields.
Re-runs skip fixtures already in api_fixture (resumable after a quota stop).
"""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Iterable

import pandas as pd
from sqlalchemy import text

from futbol.config import RAW_DIR
from futbol.db import copy_rows, get_engine, read_sql
from futbol.ingest.api_football import ApiFootballError, _get
from futbol.leagues import LEAGUES

log = logging.getLogger(__name__)

RAW_API = RAW_DIR / "api_football"
FINISHED = "FT-AET-PEN"
IDS_PER_CALL = 20

# Team statistic "type" -> api_team_match column
TEAM_STATS = {
    "Shots on Goal": "shots_on", "Shots off Goal": "shots_off", "Total Shots": "shots_total",
    "Blocked Shots": "shots_blocked", "Shots insidebox": "shots_inside", "Corner Kicks": "corners",
    "Fouls": "fouls", "Offsides": "offsides", "Yellow Cards": "yellow", "Red Cards": "red",
    "Goalkeeper Saves": "saves", "Total passes": "passes", "Ball Possession": "possession",
    "expected_goals": "xg",
}  # fmt: skip


# --------------------------------------------------------------------------- parsing (pure)


def _num(value) -> float | None:
    """'38%' -> 38.0, '3.12' -> 3.12, None -> None."""
    if value is None:
        return None
    try:
        return float(str(value).rstrip("%"))
    except ValueError:
        return None


def parse_fixture(item: dict, league_code: str) -> dict:
    fx, teams, goals = item["fixture"], item["teams"], item["goals"]
    return {
        "fixture_id": fx["id"],
        "league_code": league_code,
        "season": item["league"]["season"],
        "kickoff_utc": pd.Timestamp(fx["date"]),
        "round": item["league"].get("round"),
        "status": fx["status"]["short"],
        "home_id": teams["home"]["id"],
        "home_name": teams["home"]["name"],
        "away_id": teams["away"]["id"],
        "away_name": teams["away"]["name"],
        "home_goals": goals.get("home"),
        "away_goals": goals.get("away"),
        "referee": fx.get("referee"),
    }


def parse_team_stats(item: dict) -> list[dict]:
    home_id = item["teams"]["home"]["id"]
    rows = []
    for block in item.get("statistics") or []:
        row = {"fixture_id": item["fixture"]["id"], "team_id": block["team"]["id"],
               "is_home": block["team"]["id"] == home_id}  # fmt: skip
        for stat in block.get("statistics") or []:
            col = TEAM_STATS.get(stat["type"])
            if col:
                row[col] = _num(stat["value"])
        rows.append(row)
    return rows


def parse_player_stats(item: dict) -> list[dict]:
    starters = {
        p["player"]["id"]
        for lineup in item.get("lineups") or []
        for p in lineup.get("startXI") or []
        if p.get("player", {}).get("id")
    }
    rows = []
    for team in item.get("players") or []:
        for entry in team.get("players") or []:
            pid = entry["player"]["id"]
            st = (entry.get("statistics") or [{}])[0]
            g, sh, gl = st.get("games") or {}, st.get("shots") or {}, st.get("goals") or {}
            cards, fouls, passes = (
                st.get("cards") or {},
                st.get("fouls") or {},
                st.get("passes") or {},
            )
            if pid is None:
                continue
            rows.append(
                {
                    "fixture_id": item["fixture"]["id"], "player_id": pid, "team_id": team["team"]["id"],
                    "player_name": entry["player"]["name"], "position": g.get("position"),
                    "minutes": g.get("minutes"), "substitute": g.get("substitute"),
                    "started": pid in starters if starters else None, "rating": _num(g.get("rating")),
                    "shots_total": sh.get("total"), "shots_on": sh.get("on"), "goals": gl.get("total"),
                    "assists": gl.get("assists"), "key_passes": passes.get("key"),
                    "yellow": cards.get("yellow"), "red": cards.get("red"),
                    "fouls_committed": fouls.get("committed"), "fouls_drawn": fouls.get("drawn"),
                }
            )  # fmt: skip
    return rows


# --------------------------------------------------------------------------- download


def _chunks(ids: list[int], n: int) -> Iterable[list[int]]:
    for i in range(0, len(ids), n):
        yield ids[i : i + n]


def finished_fixture_ids(league_code: str, season: int) -> list[int]:
    league_id = LEAGUES[league_code].api_football_id
    data = _get("fixtures", {"league": league_id, "season": season, "status": FINISHED})
    return [f["fixture"]["id"] for f in data.get("response", [])]


PLAYER_INT_COLUMNS = ["minutes", "shots_total", "shots_on", "goals", "assists", "key_passes",
                      "yellow", "red", "fouls_committed", "fouls_drawn"]  # fmt: skip


def _ints(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """pandas turns int columns with gaps into floats ('8.0'), which COPY rejects for smallint."""
    for c in columns:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")
    return df


def _store(items: list[dict], league_code: str, season: int) -> None:
    folder = RAW_API / league_code / str(season)
    folder.mkdir(parents=True, exist_ok=True)
    for item in items:
        path = folder / f"{item['fixture']['id']}.json.gz"
        path.write_bytes(gzip.compress(json.dumps(item, ensure_ascii=False).encode("utf-8")))
    fixtures = _ints(
        pd.DataFrame([parse_fixture(i, league_code) for i in items]), ["home_goals", "away_goals"]
    )
    teams = pd.DataFrame([r for i in items for r in parse_team_stats(i)])
    teams = _ints(teams, [c for c in TEAM_STATS.values() if c not in ("possession", "xg")])
    players = pd.DataFrame([r for i in items for r in parse_player_stats(i)])
    players = _ints(players, PLAYER_INT_COLUMNS)
    ids = fixtures.fixture_id.tolist()
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM api_fixture WHERE fixture_id = ANY(:ids)"), {"ids": ids})
        copy_rows(conn, "api_fixture", fixtures)
        if not teams.empty:
            copy_rows(
                conn, "api_team_match", teams.reindex(columns=_columns(conn, "api_team_match"))
            )
        if not players.empty:
            players = players.drop_duplicates(["fixture_id", "player_id"])
            copy_rows(conn, "api_player_match", players)


def _columns(conn, table: str) -> list[str]:
    rows = conn.execute(
        text(
            "SELECT column_name FROM information_schema.columns WHERE table_name = :t ORDER BY ordinal_position"
        ),
        {"t": table},
    )
    return [r[0] for r in rows]


def download_season(league_code: str, season: int, max_requests: int | None = None) -> int:
    """Fetch every finished fixture of a league-season not stored yet. Returns requests used."""
    stored = set(
        read_sql(
            "SELECT fixture_id FROM api_fixture WHERE league_code = :l AND season = :s",
            l=league_code,
            s=season,
        ).fixture_id  # fmt: skip
    )
    ids = [i for i in finished_fixture_ids(league_code, season) if i not in stored]
    used = 1
    for chunk in _chunks(ids, IDS_PER_CALL):
        if max_requests is not None and used >= max_requests:
            break
        items = _get("fixtures", {"ids": "-".join(map(str, chunk))}).get("response", [])
        used += 1
        if items:
            _store(items, league_code, season)
    log.info("%s %s: %d new fixtures (%d were stored), %d requests", league_code, season,
             len(ids), len(stored), used)  # fmt: skip
    return used


def download(leagues: list[str], seasons: list[int], budget: int) -> int:
    """Seasons newest first, so a budget stop still leaves the most useful data."""
    used = 0
    for season in sorted(seasons, reverse=True):
        for league in leagues:
            if used >= budget:
                log.warning("Budget of %d requests reached", budget)
                return used
            try:
                used += download_season(league, season, max_requests=budget - used)
            except ApiFootballError as exc:
                log.warning("%s %s: %s", league, season, exc)
                if "Reserva diaria" in str(exc):
                    return used
    return used
