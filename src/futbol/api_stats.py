"""Season tables from the API-Football tables (api_fixture, api_team_match, api_player_match).

Pure pandas: the dashboard queries the rows, these functions shape them, tests cover them.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

TEAM_STAT_COLS = [
    "xg",
    "shots_total",
    "shots_on",
    "corners",
    "yellow",
    "red",
    "fouls",
    "possession",
]
PLAYER_COUNT_COLS = ["goals", "assists", "shots_total", "shots_on", "key_passes", "yellow", "red",
                     "fouls_committed", "fouls_drawn"]  # fmt: skip


# --------------------------------------------------------------------------- teams


def team_match_view(rows: pd.DataFrame) -> pd.DataFrame:
    """api_team_match joined with api_fixture -> one row per team per match, with the
    opponent's numbers alongside ('_against') so concessions are one column away.

    rows: fixture_id, kickoff_utc, home_id, home_name, away_id, away_name, home_goals,
          away_goals, team_id, is_home and the TEAM_STAT_COLS.
    """
    r = rows.copy()
    r["team"] = np.where(r.is_home, r.home_name, r.away_name)
    r["opponent"] = np.where(r.is_home, r.away_name, r.home_name)
    r["goals_for"] = np.where(r.is_home, r.home_goals, r.away_goals)
    r["goals_against"] = np.where(r.is_home, r.away_goals, r.home_goals)
    opp = r[["fixture_id", "team_id", *TEAM_STAT_COLS]].rename(
        columns={c: f"{c}_against" for c in TEAM_STAT_COLS} | {"team_id": "opp_id"}
    )
    r["opp_id"] = np.where(r.is_home, r.away_id, r.home_id)
    out = r.merge(opp, on=["fixture_id", "opp_id"], how="left")
    out["points"] = np.select(
        [out.goals_for > out.goals_against, out.goals_for == out.goals_against], [3, 1], 0
    )
    return out.sort_values("kickoff_utc").reset_index(drop=True)


def team_table(view: pd.DataFrame) -> pd.DataFrame:
    """Per-match averages per team, ordered by points (a quick table, not the official one)."""
    g = view.groupby(["team_id", "team"])
    t = g.agg(
        played=("fixture_id", "nunique"),
        points=("points", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
        xg=("xg", "sum"),
        xga=("xg_against", "sum"),
        xg_matches=("xg", "count"),
    )
    for col in ["shots_total", "shots_on", "corners", "yellow", "possession"]:
        t[f"{col}_pm"] = g[col].mean()
        if col != "possession":
            t[f"{col}_against_pm"] = g[f"{col}_against"].mean()
    t["xg_pm"] = t.xg / t.xg_matches.replace(0, np.nan)
    t["xga_pm"] = t.xga / t.xg_matches.replace(0, np.nan)
    t["goals_minus_xg"] = t.goals_for - t.xg.where(t.xg_matches > 0)
    t["goal_diff"] = t.goals_for - t.goals_against
    t = t.reset_index().sort_values(
        ["points", "goal_diff", "goals_for"], ascending=False, ignore_index=True
    )
    t.insert(0, "position", range(1, len(t) + 1))
    return t


# --------------------------------------------------------------------------- players


def player_table(rows: pd.DataFrame, min_minutes: int = 0) -> pd.DataFrame:
    """Season totals and per-90 rates per player. Only appearances (minutes > 0) count.

    rows: api_player_match joined with api_fixture (needs kickoff_utc for the latest team).
    """
    played = rows[rows.minutes.fillna(0) > 0]
    latest = played.sort_values("kickoff_utc").groupby("player_id").tail(1).set_index("player_id")
    g = played.groupby("player_id")
    t = g[PLAYER_COUNT_COLS].sum(min_count=1).fillna(0)
    t["matches"] = g.size()
    t["starts"] = g.started.sum()
    t["minutes"] = g.minutes.sum()
    # Minutes-weighted rating: a 10-minute cameo shouldn't weigh like a full match.
    t["rating"] = (played.rating * played.minutes).groupby(played.player_id).sum() / (
        played.minutes.where(played.rating.notna()).groupby(played.player_id).sum()
    ).replace(0, np.nan)
    t["player"] = latest.player_name
    t["team_id"] = latest.team_id
    t["position"] = g.position.agg(lambda s: s.mode().iat[0] if s.notna().any() else None)
    per90 = 90 / t.minutes.replace(0, np.nan)
    for col in ["goals", "assists", "shots_total", "shots_on", "key_passes", "yellow"]:
        t[f"{col}_p90"] = t[col] * per90
    t["sot_per_match"] = t.shots_on / t.matches
    t = t.reset_index()
    return t[t.minutes >= min_minutes].sort_values("minutes", ascending=False, ignore_index=True)


def player_log(rows: pd.DataFrame, player_id: int) -> pd.DataFrame:
    """Match-by-match record of one player, newest first, with the opponent's name."""
    p = rows[rows.player_id == player_id].copy()
    counts = [c for c in PLAYER_COUNT_COLS if c in p.columns]
    p[counts] = p[counts].fillna(0)  # the API sends null, not 0, when a player had none
    p["opponent"] = np.where(p.team_id == p.home_id, p.away_name, p.home_name)
    p["venue"] = np.where(p.team_id == p.home_id, "Local", "Visitante")
    return p.sort_values("kickoff_utc", ascending=False).reset_index(drop=True)


def hit_rates(
    log: pd.DataFrame, column: str, thresholds: list[int], only_starts: bool = False
) -> pd.DataFrame:
    """How often the player reached each threshold ('3 o más tiros a puerta') this season.

    Counted over matches he actually played (minutes > 0), or over his starts only.
    """
    played = log[log.minutes.fillna(0) > 0]
    if only_starts:
        played = played[played.started.fillna(False).astype(bool)]
    values = played[column].fillna(0)
    n = len(values)
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "hits": [int((values >= k).sum()) for k in thresholds],
            "matches": n,
            "rate": [float((values >= k).mean()) if n else np.nan for k in thresholds],
        }
    )


# --------------------------------------------------------------------------- bet-slip support

# Player-prop market (futbol.betslip) -> per-match column in api_player_match.
PROP_STAT = {
    "player_sot": "shots_on",
    "player_shots": "shots_total",
    "player_goals": "goals",
    "player_cards": "yellow",
}


def _plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", " ", text).strip()


def find_player(
    rows: pd.DataFrame, name: str, team_names: list[str]
) -> tuple[int, str, str | None] | None:
    """(player_id, player name, which of `team_names` he plays for) for a name read off a slip.

    Matches on the surname ('K. Mbappé' -> 'Kylian Mbappé'), prefers players of the two
    teams in the match, then the one with most minutes. None when nobody fits.
    """
    tokens = [t for t in _plain(name).split() if len(t) >= 3]
    if not tokens:
        return None
    surname = tokens[-1]
    p = rows.assign(
        team_name=np.where(rows.team_id == rows.home_id, rows.home_name, rows.away_name)
    )
    players = p.groupby("player_id").agg(player=("player_name", "last"), team=("team_name", "last"),
                                         minutes=("minutes", "sum"))  # fmt: skip
    plain = players.player.map(_plain)
    hit = plain.str.contains(rf"\b{re.escape(surname)}\b") | (
        plain.map(lambda n: fuzz.token_set_ratio(n, _plain(name))) >= 90
    )
    cands = players[hit].copy()
    if cands.empty:
        return None

    def side(team: str) -> str | None:
        scored = [(fuzz.WRatio(_plain(team), _plain(t)), t) for t in team_names]
        best = max(scored) if scored else (0, None)
        return best[1] if best[0] >= 80 else None

    cands["side"] = cands.team.map(side)
    cands = cands.sort_values(["side", "minutes"], ascending=[True, False], na_position="last")
    best = cands.iloc[0]
    return int(best.name), str(best.player), best.side


def prop_record(rows: pd.DataFrame, player_id: int, market: str, line: float) -> dict:
    """Season average and how often the player cleared this line (e.g. 3+ = line 2.5)."""
    stat = PROP_STAT[market]
    log = player_log(rows, player_id)
    played = log[log.minutes.fillna(0) > 0]
    need = int(np.floor(line)) + 1
    return {
        "avg": float(played[stat].mean()) if len(played) else np.nan,
        "hits": int((played[stat] >= need).sum()),
        "matches": len(played),
        "need": need,
    }
