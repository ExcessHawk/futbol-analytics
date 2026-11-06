"""StatsBomb Open Data ingestion.

Split in two layers on purpose:
  * transform_* functions: pure pandas, no network, no DB -> unit-testable
  * ingest_season: orchestration (download in parallel, write per match in a transaction)
"""

from __future__ import annotations

import logging
import socket
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from sqlalchemy import text

from futbol.db import copy_rows, get_engine, upsert_rows

# statsbombpy warns on every call that no API credentials were given. Open data needs none.
warnings.filterwarnings("ignore", module="statsbombpy")
from statsbombpy import sb  # noqa: E402

log = logging.getLogger(__name__)

# statsbombpy calls requests without a timeout, so a stalled connection blocks a worker
# thread forever (seen in practice: all 8 workers hung). requests/urllib3 fall back to
# the socket default timeout, so setting it here bounds every StatsBomb HTTP call.
HTTP_TIMEOUT_S = 60
FETCH_RETRIES = 4
socket.setdefaulttimeout(HTTP_TIMEOUT_S)

SKIPPED_EVENT_TYPES = {"Ball Receipt*"}

# Event columns that are mutually exclusive by event type, collapsed into one column each.
SUB_TYPE_COLS = ["pass_type", "shot_type", "duel_type", "goalkeeper_type"]
OUTCOME_COLS = [
    "pass_outcome",
    "shot_outcome",
    "dribble_outcome",
    "duel_outcome",
    "interception_outcome",
    "goalkeeper_outcome",
]
END_LOCATION_COLS = [
    "pass_end_location",
    "carry_end_location",
    "shot_end_location",
    "goalkeeper_end_location",
]
BOOL_COLS = [
    "under_pressure",
    "counterpress",
    "pass_cross",
    "pass_switch",
    "pass_through_ball",
    "pass_shot_assist",
    "pass_goal_assist",
]
EVENT_COLUMNS = [
    "event_id", "match_id", "idx", "period", "minute", "second", "type", "sub_type",
    "outcome", "team_id", "player_id", "position", "possession", "possession_team_id",
    "play_pattern", "x", "y", "end_x", "end_y", "under_pressure", "counterpress",
    "body_part", "pass_length", "pass_height", "pass_recipient_id", "pass_cross",
    "pass_switch", "pass_through_ball", "pass_shot_assist", "pass_goal_assist",
    "shot_xg", "shot_key_pass_id",
]  # fmt: skip


# --------------------------------------------------------------------------- transforms


def _coalesce(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """First non-null value across columns (like SQL COALESCE). Missing columns are ignored."""
    present = [c for c in cols if c in df.columns]
    out = pd.Series(None, index=df.index, dtype=object)
    for c in present:
        out = out.where(out.notna(), df[c])
    return out


def _split_xy(points: pd.Series) -> tuple[pd.Series, pd.Series]:
    """[x, y] or [x, y, z] lists -> two float Series (NaN where missing)."""
    x = points.map(lambda p: p[0] if isinstance(p, list) else None)
    y = points.map(lambda p: p[1] if isinstance(p, list) else None)
    return x.astype(float), y.astype(float)


def transform_events(raw: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """Flattened statsbombpy events -> rows for the `events` table."""
    df = raw[~raw["type"].isin(SKIPPED_EVENT_TYPES)].copy()
    out = pd.DataFrame(index=df.index)
    out["event_id"] = df["id"]
    out["match_id"] = match_id
    out["idx"] = df["index"]
    for col in ["period", "minute", "second", "type", "team_id", "possession",
                "possession_team_id", "play_pattern", "position"]:  # fmt: skip
        out[col] = df.get(col)
    out["player_id"] = df.get("player_id")
    out["sub_type"] = _coalesce(df, SUB_TYPE_COLS)
    out["outcome"] = _coalesce(df, OUTCOME_COLS)
    out["body_part"] = _coalesce(df, ["pass_body_part", "shot_body_part"])
    out["x"], out["y"] = _split_xy(df["location"])
    out["end_x"], out["end_y"] = _split_xy(_coalesce(df, END_LOCATION_COLS))
    for col in BOOL_COLS:
        out[col] = df[col].fillna(False).astype(bool) if col in df.columns else False
    out["pass_length"] = df.get("pass_length")
    out["pass_height"] = df.get("pass_height")
    out["pass_recipient_id"] = df.get("pass_recipient_id")
    out["shot_xg"] = df.get("shot_statsbomb_xg")
    out["shot_key_pass_id"] = df.get("shot_key_pass_id")

    int_cols = ["player_id", "pass_recipient_id", "possession_team_id", "possession"]
    for col in int_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out[EVENT_COLUMNS].reset_index(drop=True)


def _clock_to_minutes(clock: str | None) -> float | None:
    """'68:38' -> 68.63. StatsBomb lineup times use the match clock."""
    if not clock:
        return None
    mm, ss = clock.split(":")
    return int(mm) + int(ss) / 60


def match_end_minute(events: pd.DataFrame) -> float:
    """Match clock at the final whistle, from the last event of the last period."""
    last = events[events["period"] == events["period"].max()]
    return float((last["minute"] + last["second"] / 60).max())


def transform_lineups(
    raw: dict[str, pd.DataFrame],
    match_id: int,
    team_ids: dict[str, int],
    end_minute: float,
) -> pd.DataFrame:
    """statsbombpy lineups -> one row per player who played.

    Minutes = sum over the player's position spells of (to - from); an open spell
    ('to' is None) ends at the final whistle. Approximation: first-half stoppage
    time is not counted separately (StatsBomb's clock restarts at 45:00).
    """
    rows = []
    for team_name, players in raw.items():
        for p in players.itertuples(index=False):
            spells = p.positions or []
            if not spells:
                continue  # unused substitute
            minutes_by_position: dict[str, float] = {}
            for s in spells:
                start = _clock_to_minutes(s["from"]) or 0.0
                end = _clock_to_minutes(s["to"]) or end_minute
                dur = max(end - start, 0.0)
                minutes_by_position[s["position"]] = minutes_by_position.get(s["position"], 0) + dur
            rows.append(
                {
                    "match_id": match_id,
                    "player_id": int(p.player_id),
                    "team_id": team_ids[team_name],
                    "jersey_number": p.jersey_number,
                    "is_starter": any(s["start_reason"] == "Starting XI" for s in spells),
                    "minutes_played": round(sum(minutes_by_position.values()), 2),
                    "main_position": max(minutes_by_position, key=minutes_by_position.get),
                    "_player_name": p.player_name,
                    "_nickname": p.player_nickname,
                }
            )
    return pd.DataFrame(rows)


def transform_matches(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """statsbombpy matches -> (teams rows, matches rows)."""
    teams = pd.concat(
        [
            raw[["home_team_id", "home_team"]].set_axis(["team_id", "team_name"], axis=1),
            raw[["away_team_id", "away_team"]].set_axis(["team_id", "team_name"], axis=1),
        ]
    ).drop_duplicates("team_id")
    matches = raw[
        ["match_id", "competition_id", "season_id", "match_date", "kick_off", "match_week",
         "home_team_id", "away_team_id", "home_score", "away_score"]
    ].copy()  # fmt: skip
    matches["match_date"] = pd.to_datetime(matches["match_date"]).dt.date
    matches["kick_off"] = matches["kick_off"].where(matches["kick_off"].notna(), None)
    return teams, matches


# --------------------------------------------------------------------------- orchestration


def _fetch_match(match_id: int) -> tuple[int, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Network-bound part, run in worker threads. Retries with exponential backoff."""
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            events = sb.events(match_id=match_id)
            lineups = sb.lineups(match_id=match_id)
            return match_id, events, lineups
        except Exception as exc:  # timeouts, 429/5xx surfaced by requests, bad JSON
            if attempt == FETCH_RETRIES:
                raise
            wait = 2**attempt
            log.warning(
                "match %d attempt %d failed (%s), retrying in %ds", match_id, attempt, exc, wait
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


def _write_match(
    match_id: int,
    raw_events: pd.DataFrame,
    raw_lineups: dict[str, pd.DataFrame],
) -> int:
    """Replace one match's events + lineups atomically (delete-then-insert = idempotent)."""
    # Lineups are keyed by team *name*, and names are not consistent across StatsBomb
    # files ('Marseille' in lineups vs 'Olympique de Marseille' in matches). Events carry
    # both name and id, so the mapping comes from the same match's events.
    team_ids = dict(zip(raw_events["team"], raw_events["team_id"], strict=True))
    events = transform_events(raw_events, match_id)
    lineups = transform_lineups(raw_lineups, match_id, team_ids, match_end_minute(events))

    # Every player referenced anywhere must exist in `players` (FK).
    players = lineups[["player_id", "_player_name", "_nickname"]].set_axis(
        ["player_id", "player_name", "nickname"], axis=1
    )
    with get_engine().begin() as conn:
        upsert_rows(conn, "players", players, key=["player_id"])
        conn.execute(text("DELETE FROM events WHERE match_id = :m"), {"m": match_id})
        conn.execute(text("DELETE FROM lineups WHERE match_id = :m"), {"m": match_id})
        copy_rows(conn, "lineups", lineups.drop(columns=["_player_name", "_nickname"]))
        n = copy_rows(conn, "events", events)
    return n


def ingest_season(
    competition_id: int,
    season_id: int,
    workers: int = 8,
    force: bool = False,
    limit: int | None = None,
) -> None:
    """Download one competition-season and load it. Re-runnable: skips loaded matches."""
    comps = sb.competitions()
    comp = comps[(comps.competition_id == competition_id) & (comps.season_id == season_id)]
    if comp.empty:
        raise ValueError(f"Competition {competition_id}/{season_id} not in StatsBomb open data")
    c = comp.iloc[0]
    log.info("Ingesting %s %s", c.competition_name, c.season_name)

    raw_matches = sb.matches(competition_id=competition_id, season_id=season_id)
    teams, matches = transform_matches(raw_matches)
    with get_engine().begin() as conn:
        comp_cols = [
            "competition_id",
            "season_id",
            "competition_name",
            "season_name",
            "country_name",
        ]
        upsert_rows(conn, "competitions", pd.DataFrame([c[comp_cols]]), key=comp_cols[:2])
        upsert_rows(conn, "teams", teams, key=["team_id"])
        upsert_rows(conn, "matches", matches, key=["match_id"])
        loaded = set(
            conn.execute(
                text(
                    "SELECT DISTINCT e.match_id FROM events e JOIN matches m USING (match_id) "
                    "WHERE m.competition_id = :c AND m.season_id = :s"
                ),
                {"c": competition_id, "s": season_id},
            ).scalars()
        )

    todo = [int(m) for m in matches.match_id if force or int(m) not in loaded]
    if limit:
        todo = todo[:limit]
    log.info(
        "%d matches total, %d already loaded, %d to fetch", len(matches), len(loaded), len(todo)
    )

    # Threads, not processes: the bottleneck is waiting on HTTP, which releases the GIL.
    # DB writes happen here in the main thread, one transaction per match.
    done = 0
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(_fetch_match, m) for m in todo]
        for fut in as_completed(futures):
            match_id, raw_events, raw_lineups = fut.result()
            n = _write_match(match_id, raw_events, raw_lineups)
            done += 1
            if done % 25 == 0 or done == len(todo):
                log.info("  %d/%d matches (last: %d, %d events)", done, len(todo), match_id, n)
    finally:
        # Without cancel_futures, an error here would wait silently for every queued
        # download to finish before surfacing - it looks exactly like a hang.
        pool.shutdown(wait=True, cancel_futures=True)
