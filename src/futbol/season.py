"""Current-season tables from football-data results (no event data needed).

Market expected points (xPts): before every match the closing odds imply P(win), P(draw),
P(loss). A team's expected points for that match are 3*P(win) + P(draw); summed over the
season they say how many points the market thought the team would collect *given the
fixtures it actually played*. Points minus xPts mixes luck (finishing, refereeing,
bounces) with anything the market missed - it says "over/under-performing expectations",
not "lucky".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from futbol.models import odds_to_probs

FORM_MATCHES = 5
RESULT_LETTER = {3: "G", 1: "E", 0: "P"}  # ganado / empatado / perdido


def team_match_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """fd_matches rows (played only) -> one row per team per match, from its perspective."""
    played = matches.dropna(subset=["home_goals", "away_goals"]).reset_index(drop=True)
    probs = odds_to_probs(played)

    def side(is_home: bool) -> pd.DataFrame:
        us, them = ("home", "away") if is_home else ("away", "home")
        p_win, p_loss = probs[f"p_{us}"], probs[f"p_{them}"]
        out = pd.DataFrame(
            {
                "match_date": pd.to_datetime(played.match_date),
                "team": played[f"{us}_team"],
                "opponent": played[f"{them}_team"],
                "is_home": is_home,
                "goals_for": played[f"{us}_goals"],
                "goals_against": played[f"{them}_goals"],
                "shots_for": played[f"{us}_shots"],
                "shots_against": played[f"{them}_shots"],
                "sot_for": played[f"{us}_sot"],
                "sot_against": played[f"{them}_sot"],
                "p_win": p_win,
                "xpts": 3 * p_win + probs["p_draw"],
                "p_loss": p_loss,
            }
        )
        out["points"] = np.select(
            [out.goals_for > out.goals_against, out.goals_for == out.goals_against], [3, 1], 0
        )
        return out

    return pd.concat([side(True), side(False)], ignore_index=True).sort_values("match_date")


def standings(matches: pd.DataFrame) -> pd.DataFrame:
    """League table with shots, conversion, form and market expected points.

    Ordered by points, goal difference, goals scored. (Some leagues break ties by
    head-to-head first, so mid-table order can differ from the official one on ties.)
    """
    rows = team_match_rows(matches)
    g = rows.groupby("team")
    table = g.agg(
        played=("points", "size"),
        won=("points", lambda s: int((s == 3).sum())),
        drawn=("points", lambda s: int((s == 1).sum())),
        lost=("points", lambda s: int((s == 0).sum())),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
        points=("points", "sum"),
        xpts=("xpts", "sum"),
        shots_for=("shots_for", "sum"),
        shots_against=("shots_against", "sum"),
        sot_for=("sot_for", "sum"),
        sot_against=("sot_against", "sum"),
    )
    # Counts are integers even when goals arrive as floats (a column with a not-yet-played
    # match is float in pandas): the dashboard formats them with "{:+d}".
    for col in ("goals_for", "goals_against", "points"):
        table[col] = table[col].round().astype(int)
    table["goal_diff"] = table.goals_for - table.goals_against
    table["pts_minus_xpts"] = table.points - table.xpts
    table["shots_pm"] = table.shots_for / table.played
    table["sot_pm"] = table.sot_for / table.played
    table["shots_against_pm"] = table.shots_against / table.played
    # Goals per shot on target: finishing + opposition keeping, very noisy over few matches.
    table["conversion"] = table.goals_for / table.sot_for.replace(0, np.nan)
    table["form"] = g["points"].apply(
        lambda s: " ".join(RESULT_LETTER[int(p)] for p in s.tail(FORM_MATCHES))
    )
    table = table.reset_index().sort_values(
        ["points", "goal_diff", "goals_for"], ascending=False, ignore_index=True
    )
    table.insert(0, "position", range(1, len(table) + 1))
    return table


def regular_phase(matches: pd.DataFrame) -> pd.Series:
    """Mask of regular-phase matches in a single-round-robin tournament (Liga MX).

    Each team plays every other team once (n - 1 matches); anything after a team has played
    them all - play-in, liguilla - is a play-off and doesn't count for the table.
    """
    played = matches.dropna(subset=["home_goals", "away_goals"]).sort_values("match_date")
    teams = set(played.home_team) | set(played.away_team)
    rounds = len(teams) - 1
    count: dict[str, int] = {t: 0 for t in teams}
    regular = pd.Series(False, index=matches.index)
    for idx, row in played.iterrows():
        if count[row.home_team] < rounds and count[row.away_team] < rounds:
            regular[idx] = True
            count[row.home_team] += 1
            count[row.away_team] += 1
    return regular
