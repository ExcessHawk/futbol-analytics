"""Current-season player statistics from API-Football (api-sports.io).

Why this source: FBref lost its Opta licence in January 2026, Understat blocks scripts and
WhoScored/Sofascore forbid scraping. API-Football is an official, documented API with
per-player season totals (appearances, minutes, shots, shots on target, goals, cards).

Plans: Free = 100 requests/day, 10/minute, seasons 2022-2024 only (the API answers
"Free plans do not have access to this season" otherwise). Pro = 7,500/day, 300/minute,
every season. Exceeding the per-minute rate can get the account blocked by their firewall
without notice, so every call goes through RateLimiter, which paces requests well below
the limit and reads the quota headers the API returns.

Setup: create a free key at https://dashboard.api-football.com and set APIFOOTBALL_KEY in .env.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import unicodedata
from dataclasses import dataclass, field

import pandas as pd
import requests
from sqlalchemy import text

from futbol.config import get_settings
from futbol.db import get_engine, read_sql
from futbol.leagues import LEAGUES

log = logging.getLogger(__name__)

API_BASE = "https://v3.football.api-sports.io"
LEAGUE_IDS = {code: lg.api_football_id for code, lg in LEAGUES.items() if lg.api_football_id}
CACHE_HOURS = 24

PLAYER_COLUMNS = [
    "player",
    "team",
    "appearances",
    "minutes",
    "shots",
    "sot",
    "goals",
    "assists",
    "yellow",
]


class ApiFootballError(RuntimeError):
    """The API answered with an error (missing key, plan limits, quota...)."""


def has_key() -> bool:
    return bool(get_settings().apifootball_key)


@dataclass
class RateLimiter:
    """Paces calls below the per-minute limit and stops before the daily quota runs out.

    The limits come from the response headers (X-RateLimit-Limit per minute,
    x-ratelimit-requests-remaining per day), so the same code is safe on Free and Pro.
    """

    safety: float = 0.4  # use at most 40% of the per-minute allowance
    daily_reserve: int = 50  # never spend the last N requests of the day
    per_minute: int = 10  # assumed until the first response says otherwise
    remaining_today: int | None = None
    _last: float = field(default=0.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def wait(self) -> None:
        if self.remaining_today is not None and self.remaining_today <= self.daily_reserve:
            raise ApiFootballError(
                f"Reserva diaria alcanzada ({self.remaining_today} peticiones restantes): "
                "se reanuda tras las 00:00 UTC."
            )
        with self._lock:
            interval = 60.0 / max(self.per_minute * self.safety, 1.0)
            delay = self._last + interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()

    def update(self, headers) -> None:
        if (v := headers.get("X-RateLimit-Limit")) and v.isdigit():
            self.per_minute = int(v)
        if (v := headers.get("x-ratelimit-requests-remaining")) and v.isdigit():
            self.remaining_today = int(v)
        if (v := headers.get("X-RateLimit-Remaining")) and v.isdigit() and int(v) <= 2:
            time.sleep(10)  # close to the per-minute ceiling: let the window roll over


LIMITER = RateLimiter()


def _get(path: str, params: dict) -> dict:
    key = get_settings().apifootball_key
    if not key:
        raise ApiFootballError("Falta APIFOOTBALL_KEY en el archivo .env")
    LIMITER.wait()
    resp = requests.get(
        f"{API_BASE}/{path}", params=params, headers={"x-apisports-key": key}, timeout=30
    )
    LIMITER.update(resp.headers)
    resp.raise_for_status()
    data = resp.json()
    errors = data.get("errors")
    if errors:  # HTTP 200 with an error payload: a dict {field: message} or a list
        items = errors.items() if isinstance(errors, dict) else enumerate(errors)
        raise ApiFootballError("; ".join(str(v) for _, v in items))
    return data


def account_status() -> dict:
    """Plan name and today's request usage."""
    r = _get("status", {}).get("response", {})
    return {
        "plan": r.get("subscription", {}).get("plan"),
        "requests_today": r.get("requests", {}).get("current"),
        "limit_day": r.get("requests", {}).get("limit_day"),
    }


def search_term(name: str) -> str:
    """API search wants >= 4 ASCII letters; the surname is the most specific part."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    parts = [p for p in ascii_name.replace(".", " ").split() if len(p) >= 4]
    return (parts[-1] if parts else ascii_name).lower()


def _parse(response: list[dict], league_id: int) -> pd.DataFrame:
    rows = []
    for item in response:
        for st in item.get("statistics", []):
            if st.get("league", {}).get("id") != league_id:
                continue
            games, shots, goals, cards = (
                st.get(k) or {} for k in ("games", "shots", "goals", "cards")
            )
            rows.append(
                {
                    "player": item["player"]["name"],
                    "team": st["team"]["name"],
                    "appearances": games.get("appearences") or 0,  # (sic) the API's spelling
                    "minutes": games.get("minutes") or 0,
                    "shots": shots.get("total") or 0,
                    "sot": shots.get("on") or 0,
                    "goals": goals.get("total") or 0,
                    "assists": goals.get("assists") or 0,
                    "yellow": cards.get("yellow") or 0,
                }
            )
    return pd.DataFrame(rows, columns=PLAYER_COLUMNS)


def player_season_stats(name: str, league_code: str, season_year: int) -> pd.DataFrame:
    """Season totals for players matching `name` in a league (cached 24 h in the database)."""
    league_id = LEAGUE_IDS[league_code]
    query = search_term(name)
    cached = read_sql(
        "SELECT payload FROM api_player_cache WHERE league_code = :l AND season = :s AND query = :q "
        "AND fetched_at > now() - make_interval(hours => :h)",
        l=league_code, s=season_year, q=query, h=CACHE_HOURS,
    )  # fmt: skip
    if not cached.empty:
        payload = cached.payload.iloc[0]
        return _parse(payload if isinstance(payload, list) else json.loads(payload), league_id)

    data = _get("players", {"search": query, "league": league_id, "season": season_year})
    response = data.get("response", [])
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "INSERT INTO api_player_cache (league_code, season, query, payload, fetched_at) "
                "VALUES (:l, :s, :q, CAST(:p AS jsonb), now()) "
                "ON CONFLICT (league_code, season, query) DO UPDATE "
                "SET payload = EXCLUDED.payload, fetched_at = EXCLUDED.fetched_at"
            ),
            {"l": league_code, "s": season_year, "q": query, "p": json.dumps(response)},
        )
    log.info(
        "API-Football: %s in %s %s -> %d players", query, league_code, season_year, len(response)
    )
    return _parse(response, league_id)


STAT_COLUMN = {
    "player_sot": "sot",
    "player_shots": "shots",
    "player_goals": "goals",
    "player_cards": "yellow",
}


def per_match_average(row: pd.Series, market: str) -> float | None:
    """Season total / appearances: the player's usual output in a match he plays in."""
    if not row.appearances:
        return None
    return float(row[STAT_COLUMN[market]]) / float(row.appearances)
