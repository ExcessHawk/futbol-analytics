"""Probable starting XI from a team's recent lineups and its injury list.

The rule a fan would use, made explicit:
  1. formation: the one the coach used last match;
  2. each slot of that formation goes to whoever has started there most in recent matches
     (recency-weighted), skipping players reported out;
  3. a slot left empty (its regular is injured) goes to the most used available player of
     the same position (G/D/M/F).

Lineups are only official ~30 min before kick-off; this is the guess before that. How good the
guess is gets measured by `backtest_probable_xi` on past matches (correct starters out of 11).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RECENT_MATCHES = 6
DECAY = 0.5  # weight of the match before the last one, relative to the last (tuned)
SAME_POSITION_CREDIT = 0.3  # starting in another slot of the same line still counts a bit

# history columns: fixture_id, kickoff_utc, formation, player_id, player_name, pos, grid, started


@dataclass
class ProbableXI:
    formation: str | None
    players: pd.DataFrame  # player_id, player_name, pos, grid, number, confidence, doubtful
    based_on: int  # how many recent matches were used


def _recent(history: pd.DataFrame, k: int) -> tuple[pd.DataFrame, dict[int, float]]:
    fixtures = (
        history.drop_duplicates("fixture_id")
        .sort_values("kickoff_utc", ascending=False)
        .fixture_id.head(k)
        .tolist()
    )
    weights = {f: DECAY**i for i, f in enumerate(fixtures)}
    return history[history.fixture_id.isin(fixtures)], weights


def _template(recent: pd.DataFrame, weights: dict[int, float]) -> tuple[str | None, list[tuple]]:
    """Formation and (grid, pos) slots of the most recent match with a complete XI."""
    for fixture in sorted(weights, key=weights.get, reverse=True):
        xi = recent[(recent.fixture_id == fixture) & recent.started & recent.grid.notna()]
        if len(xi) == 11:
            return xi.formation.iloc[0], list(xi[["grid", "pos"]].itertuples(index=False))
    return None, []


def probable_xi(
    history: pd.DataFrame,
    unavailable: set[int] = frozenset(),
    doubtful: set[int] = frozenset(),
    k: int = RECENT_MATCHES,
) -> ProbableXI:
    recent, weights = _recent(history, k)
    formation, slots = _template(recent, weights)
    empty = pd.DataFrame(columns=["player_id", "player_name", "pos", "grid", "number",
                                  "confidence", "doubtful"])  # fmt: skip
    if not slots:
        return ProbableXI(None, empty, len(weights))
    total = sum(weights.values())
    recent = recent.assign(w=recent.fixture_id.map(weights))
    starts = recent[recent.started]

    # Share of recent (weighted) matches each player started: the shown confidence.
    start_share = starts.groupby("player_id").w.sum() / total
    info = recent.sort_values("kickoff_utc").groupby("player_id").last()

    # score[player, slot] = weighted starts in that exact slot (same formation) plus partial
    # credit for starting anywhere in the same line.
    candidates = []
    same_formation = starts[starts.formation == formation]
    for grid, pos in slots:
        exact = same_formation[same_formation.grid == grid].groupby("player_id").w.sum()
        line = starts[starts.pos == pos].groupby("player_id").w.sum() * SAME_POSITION_CREDIT
        score = exact.add(line, fill_value=0)
        candidates += [(s, pid, grid) for pid, s in score.items() if pid not in unavailable]

    chosen: dict[str, int] = {}
    used: set[int] = set()
    for _, pid, grid in sorted(candidates, key=lambda c: -c[0]):
        if grid not in chosen and pid not in used:
            chosen[grid] = pid
            used.add(pid)

    # Empty slot: the most used available player of that line (bench players included).
    usage = recent.groupby("player_id").w.sum()
    for grid, pos in slots:
        if grid in chosen:
            continue
        pool = info[(info.pos == pos) & ~info.index.isin(used | set(unavailable))]
        if pool.empty:
            pool = info[~info.index.isin(used | set(unavailable))]
        if pool.empty:
            continue
        pid = usage.reindex(pool.index).fillna(0).idxmax()
        chosen[grid] = pid
        used.add(pid)

    rows = []
    for grid, pos in slots:
        if grid not in chosen:
            continue
        pid = chosen[grid]
        rows.append({
            "player_id": int(pid), "player_name": info.at[pid, "player_name"], "pos": pos,
            "grid": grid, "number": info.at[pid, "number"] if "number" in info else None,
            "confidence": float(start_share.get(pid, 0.0)), "doubtful": pid in doubtful,
        })  # fmt: skip
    return ProbableXI(formation, pd.DataFrame(rows, columns=empty.columns), len(weights))


def last_xi(history: pd.DataFrame) -> set[int]:
    """Baseline guess: exactly the XI of the previous match."""
    recent, weights = _recent(history, 1)
    return set(recent[recent.started].player_id)


def backtest_probable_xi(
    lineups: pd.DataFrame,
    injuries: pd.DataFrame | None = None,
    since: str = "2023-07-01",
    k: int = RECENT_MATCHES,
) -> pd.DataFrame:
    """For every team-match since `since`, guess the XI from the k previous league matches.

    lineups: all lineup rows (with team_id) of one or more leagues.
    injuries: optional api_injury rows; players listed 'Missing Fixture' for that fixture
              are excluded (the information a fan had before kick-off).
    Returns one row per team-match: correct starters of the model and of the baseline.
    """
    out = []
    out_inj = (
        injuries[injuries.kind == "Missing Fixture"].groupby("fixture_id").player_id.agg(set)
        if injuries is not None and not injuries.empty
        else pd.Series(dtype=object)
    )
    for team, g in lineups.groupby("team_id"):
        order = g.drop_duplicates("fixture_id").sort_values("kickoff_utc")
        order = order[order.formation.notna()]
        fixtures, dates = order.fixture_id.to_numpy(), order.kickoff_utc.to_numpy()
        for i in range(k, len(fixtures)):
            if dates[i] < np.datetime64(pd.Timestamp(since).tz_localize(None)):
                continue
            actual = set(g[(g.fixture_id == fixtures[i]) & g.started].player_id)
            if len(actual) != 11:
                continue
            past = g[g.fixture_id.isin(fixtures[i - k : i])]
            unavailable = out_inj.get(fixtures[i], set())
            guess = set(probable_xi(past, unavailable, k=k).players.player_id)
            out.append({"team_id": team, "fixture_id": fixtures[i],
                        "kickoff_utc": dates[i], "model": len(guess & actual),
                        "last_xi": len(last_xi(past) & actual)})  # fmt: skip
    return pd.DataFrame(out)
