"""football-data.co.uk ingestion: results, shots and bookmaker odds as CSV.

Raw CSVs are cached in data/raw/football_data/ so re-runs don't hit the site again
(only the current season and the fixtures file are re-downloaded).
"""

from __future__ import annotations

import io
import logging
from datetime import date

import pandas as pd
import requests
from sqlalchemy import text

from futbol.config import RAW_DIR
from futbol.db import get_engine, upsert_rows
from futbol.leagues import BIG5, EXTRA  # noqa: F401  (re-exported for older imports)

log = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk"
CACHE_DIR = RAW_DIR / "football_data"

USER_AGENT = "futbol-analytics/0.1 (portfolio project; python-requests)"
MATCH_KEY = ["league_code", "season", "home_team", "away_team", "match_date"]

# Odds columns in order of preference, as (home, draw, away, label).
# Closing odds are the sharpest public estimate of the true probability; Pinnacle's
# are the market reference. Pinnacle columns are missing from 2026/27 onwards.
ODDS_PREFERENCE = [
    ("PSCH", "PSCD", "PSCA", "pinnacle_closing"),
    ("AvgCH", "AvgCD", "AvgCA", "market_avg_closing"),
    ("PSH", "PSD", "PSA", "pinnacle_opening"),
    ("AvgH", "AvgD", "AvgA", "market_avg"),
    ("BbAvH", "BbAvD", "BbAvA", "betbrain_avg"),
    ("B365H", "B365D", "B365A", "bet365"),
]


def season_label(start_year: int) -> str:
    """2015 -> '2015/2016' (same format StatsBomb uses)."""
    return f"{start_year}/{start_year + 1}"


def season_code(start_year: int) -> str:
    """2015 -> '1516' (football-data URL format)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season_start(today: date | None = None) -> int:
    """European seasons start in July/August."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def _download(url: str) -> bytes:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.content


def _read_csv(content: bytes) -> pd.DataFrame:
    """Files mix encodings and sometimes have trailing empty columns/rows."""
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(content), encoding=encoding, on_bad_lines="skip")
            break
        except UnicodeDecodeError:
            continue
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.rename(columns=lambda c: str(c).strip())
    teams = [
        c for c in ("HomeTeam", "AwayTeam", "Home", "Away") if c in df.columns
    ]  # main / extra files
    return df.dropna(subset=teams)


def fetch_csv(path: str, cache: bool) -> pd.DataFrame:
    """Fetch BASE_URL/path, using the on-disk cache when allowed."""
    local = CACHE_DIR / path.replace("/", "_")
    if cache and local.exists():
        return _read_csv(local.read_bytes())
    content = _download(f"{BASE_URL}/{path}")
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(content)
    return _read_csv(content)


# Total goals over/under 2.5, same idea: closing prices first.
OU_PREFERENCE = [
    ("PC>2.5", "PC<2.5", "pinnacle_closing"),
    ("AvgC>2.5", "AvgC<2.5", "market_avg_closing"),
    ("P>2.5", "P<2.5", "pinnacle_opening"),
    ("Avg>2.5", "Avg<2.5", "market_avg"),
    ("BbAv>2.5", "BbAv<2.5", "betbrain_avg"),
    ("B365>2.5", "B365<2.5", "bet365"),
]

# football-data column -> fd_matches column, for per-match counts.
STAT_COLUMNS = {
    "FTHG": "home_goals", "FTAG": "away_goals",
    "HS": "home_shots", "AS": "away_shots",
    "HST": "home_sot", "AST": "away_sot",
    "HC": "home_corners", "AC": "away_corners",
    "HY": "home_yellow", "AY": "away_yellow",
    "HR": "home_red", "AR": "away_red",
    "HF": "home_fouls", "AF": "away_fouls",
}  # fmt: skip


def pick_prices(
    df: pd.DataFrame, preference: list[tuple], out_cols: list[str], source_col: str
) -> pd.DataFrame:
    """Per row, take the first group of odds columns (by preference) that is fully present."""
    out = pd.DataFrame(index=df.index, columns=[*out_cols, source_col])
    for *cols, label in preference:
        if not set(cols) <= set(df.columns):
            continue
        prices = df[cols].apply(pd.to_numeric, errors="coerce")
        usable = out[source_col].isna() & prices.notna().all(axis=1) & (prices > 1).all(axis=1)
        out.loc[usable, out_cols] = prices[usable].to_numpy()
        out.loc[usable, source_col] = label
    return out


def pick_odds(df: pd.DataFrame) -> pd.DataFrame:
    """1X2 odds: first fully present home/draw/away triple by preference."""
    return pick_prices(df, ODDS_PREFERENCE, ["odds_home", "odds_draw", "odds_away"], "odds_source")


def pick_over_under(df: pd.DataFrame) -> pd.DataFrame:
    return pick_prices(df, OU_PREFERENCE, ["odds_over25", "odds_under25"], "ou_odds_source")


def transform(df: pd.DataFrame, league_code: str, season: str) -> pd.DataFrame:
    """Raw football-data rows -> fd_matches rows."""
    out = pd.DataFrame(
        {
            "league_code": league_code,
            "season": season,
            # Dates appear as dd/mm/yy or dd/mm/yyyy depending on the season.
            "match_date": pd.to_datetime(df["Date"], dayfirst=True, format="mixed").dt.date,
            "home_team": df["HomeTeam"].str.strip(),
            "away_team": df["AwayTeam"].str.strip(),
        }
    )
    for src, dst in STAT_COLUMNS.items():
        out[dst] = pd.to_numeric(df[src], errors="coerce").astype("Int64") if src in df else pd.NA
    referee = df["Referee"].astype("string").str.strip() if "Referee" in df else pd.NA
    out["referee"] = referee.replace("", pd.NA) if "Referee" in df else pd.NA
    out = out.join(pick_odds(df)).join(pick_over_under(df))
    out["updated_at"] = pd.Timestamp.now(tz="UTC")
    return out.drop_duplicates(["home_team", "away_team", "match_date"], keep="last")


PLAYED_COLUMNS = list(STAT_COLUMNS.values())


def ingest_current_season(leagues: list[str]) -> None:
    """Refresh only the season in progress (the daily update path)."""
    year = current_season_start()
    ingest_league_seasons(leagues, year, year)


def ingest_league_seasons(leagues: list[str], first_season: int, last_season: int) -> None:
    """Load played matches for every league x season. Past seasons come from cache."""
    current = current_season_start()
    for league in leagues:
        for year in range(first_season, last_season + 1):
            path = f"mmz4281/{season_code(year)}/{league}.csv"
            raw = fetch_csv(path, cache=year < current)
            rows = transform(raw, league, season_label(year))
            with get_engine().begin() as conn:
                upsert_rows(
                    conn,
                    "fd_matches",
                    rows,
                    key=MATCH_KEY,
                )
            log.info("%s %s: %d matches", league, season_label(year), len(rows))


def ingest_fixtures(leagues: list[str]) -> int:
    """Upcoming fixtures (next ~week) with pre-match odds. Goals stay NULL until played."""
    raw = fetch_csv("fixtures.csv", cache=False)
    raw = raw[raw["Div"].isin(leagues)]
    season = season_label(current_season_start())
    n = 0
    with get_engine().begin() as conn:
        for league, part in raw.groupby("Div"):
            # Drop the result columns so a fixture row can never blank out a played match.
            rows = transform(part, league, season).drop(columns=PLAYED_COLUMNS)
            n += upsert_rows(conn, "fd_matches", rows, key=MATCH_KEY)
    log.info("Upcoming fixtures: %d", n)
    return n


# --------------------------------------------------------------------------- extra leagues

# The "new leagues" files (Mexico, Argentina, Brazil...) carry results and closing 1X2 odds
# only - no shots, corners or cards - in one CSV covering every season.
EXTRA_ODDS_PREFERENCE = [
    ("PSCH", "PSCD", "PSCA", "pinnacle_closing"),
    ("AvgCH", "AvgCD", "AvgCA", "market_avg_closing"),
    ("B365CH", "B365CD", "B365CA", "bet365_closing"),
]
FIRST_EXTRA_YEAR = 2012


def season_start_year(season: str) -> int:
    """'2026/2027' -> 2026, '2026' -> 2026 (Argentina switched formats in 2014)."""
    return int(str(season)[:4])


def transform_extra(df: pd.DataFrame, league_code: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "league_code": league_code,
            "season": df["Season"].astype(str).str.strip(),
            "match_date": pd.to_datetime(df["Date"], dayfirst=True, format="mixed").dt.date,
            "home_team": df["Home"].str.strip(),
            "away_team": df["Away"].str.strip(),
            "home_goals": pd.to_numeric(df["HG"], errors="coerce").astype("Int64"),
            "away_goals": pd.to_numeric(df["AG"], errors="coerce").astype("Int64"),
        }
    )
    out = out.join(
        pick_prices(
            df, EXTRA_ODDS_PREFERENCE, ["odds_home", "odds_draw", "odds_away"], "odds_source"
        )
    )
    out["updated_at"] = pd.Timestamp.now(tz="UTC")
    out = out[out.season.map(season_start_year) >= FIRST_EXTRA_YEAR]
    # Same pairing twice in a season is normal here (Apertura/Clausura, play-offs): the date
    # is part of the identity, so only exact duplicates are dropped.
    return out.drop_duplicates(["home_team", "away_team", "match_date"], keep="last")


def ingest_extra_leagues(leagues: list[str]) -> None:
    """Every season of each extra league (one file each, re-downloaded: it holds the current season)."""
    for league in leagues:
        raw = fetch_csv(f"new/{league}.csv", cache=False).rename(columns=lambda c: c.strip())
        raw = raw.dropna(subset=["Home", "Away"])
        rows = transform_extra(raw, league)
        with get_engine().begin() as conn:
            upsert_rows(conn, "fd_matches", rows, key=MATCH_KEY)
        log.info(
            "%s: %d matches, seasons %s..%s",
            league,
            len(rows),
            rows.season.min(),
            rows.season.max(),
        )


def drop_stale_fixtures() -> int:
    """Unplayed rows dated in the past are postponed fixtures whose new date arrived as a new
    row (the date is part of the key): remove them so they don't linger as 'upcoming'."""
    with get_engine().begin() as conn:
        n = conn.execute(
            text(
                "DELETE FROM fd_matches WHERE home_goals IS NULL AND match_date < CURRENT_DATE - 2"
            )
        ).rowcount
    if n:
        log.info("Removed %d stale (postponed) fixture rows", n)
    return n
