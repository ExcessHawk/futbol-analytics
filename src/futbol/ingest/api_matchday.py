"""What the next matchday looks like: upcoming fixtures, reported injuries and player photos.

Two requests per league and day (/fixtures?next and /injuries). Injury coverage depends on
the league: API-Football reports them for the Big-5 and Brazil, not for Liga MX or Argentina.

Photos and logos (media.api-sports.io) don't count against the daily quota, but API-Football
asks clients not to hotlink them: each photo is downloaded once and served from disk.
"""

from __future__ import annotations

import html
import logging

import pandas as pd
import requests
from sqlalchemy import text

from futbol.config import RAW_DIR
from futbol.db import copy_rows, get_engine, read_sql, upsert_rows
from futbol.ingest.api_football import _get
from futbol.leagues import LEAGUES

log = logging.getLogger(__name__)

NEXT_FIXTURES = 20  # per league: covers the next matchday (or two) of a 20-team league
MEDIA_DIR = RAW_DIR / "api_football_media"
PHOTO_URL = "https://media.api-sports.io/football/players/{id}.png"


# --------------------------------------------------------------------------- parsing (pure)


def parse_upcoming(item: dict, league_code: str) -> dict:
    fx, teams = item["fixture"], item["teams"]
    return {
        "fixture_id": fx["id"],
        "league_code": league_code,
        "season": item["league"]["season"],
        "kickoff_utc": pd.Timestamp(fx["date"]),
        "round": item["league"].get("round"),
        "home_id": teams["home"]["id"],
        "home_name": teams["home"]["name"],
        "away_id": teams["away"]["id"],
        "away_name": teams["away"]["name"],
    }


def parse_injury(item: dict, league_code: str) -> dict | None:
    player, team, fx = item.get("player") or {}, item.get("team") or {}, item.get("fixture") or {}
    if player.get("id") is None or team.get("id") is None or fx.get("id") is None:
        return None
    return {
        "fixture_id": fx["id"],
        "league_code": league_code,
        "team_id": team["id"],
        "player_id": player["id"],
        "player_name": html.unescape(player.get("name") or "?"),
        "kind": player.get("type") or "Missing Fixture",
        "reason": player.get("reason"),
        "kickoff_utc": pd.Timestamp(fx["date"]) if fx.get("date") else None,
    }


# --------------------------------------------------------------------------- download


def fetch_upcoming(league_code: str, season: int) -> int:
    """Replace the league's not-yet-played fixtures. Returns how many were stored."""
    league_id = LEAGUES[league_code].api_football_id
    items = _get("fixtures", {"league": league_id, "season": season, "next": NEXT_FIXTURES})
    rows = pd.DataFrame([parse_upcoming(i, league_code) for i in items.get("response", [])])
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM api_upcoming WHERE league_code = :l"), {"l": league_code})
        if not rows.empty:
            copy_rows(conn, "api_upcoming", rows)
    return len(rows)


def fetch_injuries(league_code: str, season: int) -> int:
    """Upsert every injury report of the league-season (past fixtures included)."""
    league_id = LEAGUES[league_code].api_football_id
    items = _get("injuries", {"league": league_id, "season": season}).get("response", [])
    rows = pd.DataFrame([r for i in items if (r := parse_injury(i, league_code))])
    if rows.empty:
        return 0
    rows = rows.drop_duplicates(["fixture_id", "player_id"], keep="last")
    rows["fetched_at"] = pd.Timestamp.now(tz="UTC")
    with get_engine().begin() as conn:
        upsert_rows(conn, "api_injury", rows, key=["fixture_id", "player_id"])
    return len(rows)


def update_matchday(leagues: list[str]) -> int:
    """Upcoming fixtures and injuries of each league's latest season. Returns requests used."""
    seasons = read_sql("SELECT league_code, max(season) AS season FROM api_fixture GROUP BY 1")
    latest = dict(zip(seasons.league_code, seasons.season, strict=True))
    used = 0
    for league in leagues:
        season = latest.get(league)
        if season is None:
            continue
        n = fetch_upcoming(league, int(season))
        used += 1
        if n == 0:  # calendar-year league between seasons: the next one may have started
            n = fetch_upcoming(league, int(season) + 1)
            used += 1
        injuries = fetch_injuries(league, int(season))
        used += 1
        log.info("%s: %d upcoming fixtures, %d injury reports", league, n, injuries)
    return used


# --------------------------------------------------------------------------- photos


def player_photo(player_id: int) -> bytes | None:
    """The player's photo, downloaded once and cached on disk. None if unavailable."""
    path = MEDIA_DIR / "players" / f"{int(player_id)}.png"
    if path.exists():
        return path.read_bytes() or None  # an empty file marks "the API has no photo"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(PHOTO_URL.format(id=int(player_id)), timeout=10)
    except requests.RequestException as exc:
        log.warning("photo %s: %s", player_id, exc)
        return None  # not cached: retried next time
    ok = resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/")
    path.write_bytes(resp.content if ok else b"")
    return resp.content if ok else None
