"""Football metrics computed from StatsBomb events.

Coordinates: StatsBomb pitch is 120 x 80 yards and every event is recorded from the
acting team's perspective, attacking towards x = 120. The opponent goal centre is
(120, 40); the penalty area is x >= 102 and 18 <= y <= 62.

Definitions (there is no single standard, these are documented choices):

* npxG: xG excluding penalties. Penalties are worth ~0.76 xG each and depend on
  referee decisions, so they inflate xG without saying much about chance creation.

* Progressive pass: completed open-play pass that starts outside the team's own
  defensive 40% (x >= 48) and either reduces the distance to the goal centre by at
  least 25% or ends inside the penalty area (having started outside it).

* PPDA (passes per defensive action): opponent open-play passes in *their* own 60% of
  the pitch divided by our defensive actions (tackles, interceptions, fouls committed,
  failed challenges) in *our* attacking 60%. Lower = more intense pressing.
  Season PPDA = sum of passes / sum of actions, not the mean of per-match ratios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from futbol.db import read_sql

PITCH_LENGTH = 120.0
GOAL = np.array([120.0, 40.0])
SET_PIECE_PASSES = {"Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in"}
DEFENSIVE_ACTIONS = {"Interception", "Foul Committed", "Dribbled Past"}
PPDA_PASS_ZONE_MAX_X = 0.6 * PITCH_LENGTH  # 72: opponent's own 60%
PPDA_ACTION_ZONE_MIN_X = 0.4 * PITCH_LENGTH  # 48: our attacking 60%


# --------------------------------------------------------------------------- loading


def load_events(competition_id: int, season_id: int) -> pd.DataFrame:
    """Columns needed for metrics, for one competition-season."""
    return read_sql(
        """
        SELECT e.event_id, e.match_id, e.type, e.sub_type, e.outcome, e.team_id, e.player_id,
               e.x, e.y, e.end_x, e.end_y, e.shot_xg, e.shot_key_pass_id,
               e.pass_shot_assist, e.pass_goal_assist, e.minute, e.period, e.play_pattern
        FROM events e JOIN matches m USING (match_id)
        WHERE m.competition_id = :c AND m.season_id = :s
          AND e.type IN ('Pass','Shot','Carry','Duel','Interception','Foul Committed',
                         'Dribbled Past','Dribble','Pressure','Ball Recovery')
        """,
        c=competition_id,
        s=season_id,
    )


# --------------------------------------------------------------------------- event flags


def _dist_to_goal(x: pd.Series, y: pd.Series) -> np.ndarray:
    return np.hypot(GOAL[0] - x.to_numpy(float), GOAL[1] - y.to_numpy(float))


def in_penalty_area(x: pd.Series, y: pd.Series) -> pd.Series:
    return (x >= 102) & (y >= 18) & (y <= 62)


def is_progressive(df: pd.DataFrame) -> pd.Series:
    """Progressive movement (pass or carry) per the module definition. Ignores type/outcome."""
    start = _dist_to_goal(df["x"], df["y"])
    end = _dist_to_goal(df["end_x"], df["end_y"])
    closer = end <= 0.75 * start
    into_box = in_penalty_area(df["end_x"], df["end_y"]) & ~in_penalty_area(df["x"], df["y"])
    return (df["x"] >= PPDA_ACTION_ZONE_MIN_X) & (closer | into_box)


def is_open_play_pass(df: pd.DataFrame) -> pd.Series:
    return (df["type"] == "Pass") & ~df["sub_type"].isin(SET_PIECE_PASSES)


def is_progressive_pass(df: pd.DataFrame) -> pd.Series:
    completed = df["outcome"].isna()  # StatsBomb: NULL pass outcome = completed
    return is_open_play_pass(df) & completed & is_progressive(df)


def is_defensive_action(df: pd.DataFrame) -> pd.Series:
    tackle = (df["type"] == "Duel") & (df["sub_type"] == "Tackle")
    return tackle | df["type"].isin(DEFENSIVE_ACTIONS)


def is_shot(df: pd.DataFrame) -> pd.Series:
    return df["type"] == "Shot"


def is_penalty(df: pd.DataFrame) -> pd.Series:
    return is_shot(df) & (df["sub_type"] == "Penalty")


# --------------------------------------------------------------------------- team metrics


def team_match_stats(events: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, team) with attacking and defensive metrics.

    matches columns: match_id, home_team_id, away_team_id, home_score, away_score
    """
    ev = events.assign(
        shot=is_shot(events),
        np_xg=events["shot_xg"].where(is_shot(events) & ~is_penalty(events), 0.0),
        xg=events["shot_xg"].where(is_shot(events), 0.0),
        open_pass=is_open_play_pass(events),
        prog_pass=is_progressive_pass(events),
        completed_pass=(events["type"] == "Pass") & events["outcome"].isna(),
        all_pass=events["type"] == "Pass",
        ppda_pass=is_open_play_pass(events) & (events["x"] < PPDA_PASS_ZONE_MAX_X),
        ppda_action=is_defensive_action(events) & (events["x"] >= PPDA_ACTION_ZONE_MIN_X),
    )
    per_team = (
        ev.groupby(["match_id", "team_id"])
        .agg(
            shots=("shot", "sum"),
            xg=("xg", "sum"),
            npxg=("np_xg", "sum"),
            passes=("all_pass", "sum"),
            passes_completed=("completed_pass", "sum"),
            progressive_passes=("prog_pass", "sum"),
            ppda_passes_allowed_by_opp=(
                "ppda_pass",
                "sum",
            ),  # passes this team made, seen by the opponent
            defensive_actions_high=("ppda_action", "sum"),
        )
        .reset_index()
    )

    # One row per team per match, home and away, with goals from the official score.
    sides = pd.concat(
        [
            matches.assign(
                team_id=matches.home_team_id,
                opponent_id=matches.away_team_id,
                is_home=True,
                goals=matches.home_score,
                goals_against=matches.away_score,
            ),
            matches.assign(
                team_id=matches.away_team_id,
                opponent_id=matches.home_team_id,
                is_home=False,
                goals=matches.away_score,
                goals_against=matches.home_score,
            ),
        ]  # fmt: skip
    )[["match_id", "team_id", "opponent_id", "is_home", "goals", "goals_against"]]

    out = sides.merge(per_team, on=["match_id", "team_id"], how="left")
    opp = per_team.rename(columns=lambda c: c if c == "match_id" else f"opp_{c}").rename(
        columns={"opp_team_id": "opponent_id"}
    )
    out = out.merge(opp, on=["match_id", "opponent_id"], how="left")

    out["xga"] = out["opp_xg"]
    out["npxga"] = out["opp_npxg"]
    out["shots_against"] = out["opp_shots"]
    # PPDA for this team: opponent passes in their build-up zone / our high defensive actions.
    out["ppda_passes"] = out["opp_ppda_passes_allowed_by_opp"]
    out["ppda_actions"] = out["defensive_actions_high"]
    out["ppda"] = out["ppda_passes"] / out["ppda_actions"].replace(0, np.nan)
    out["points"] = np.select(
        [out.goals > out.goals_against, out.goals == out.goals_against], [3, 1], default=0
    )
    keep = ["match_id", "team_id", "opponent_id", "is_home", "goals", "goals_against", "points",
            "shots", "shots_against", "xg", "xga", "npxg", "npxga", "passes", "passes_completed",
            "progressive_passes", "ppda_passes", "ppda_actions", "ppda"]  # fmt: skip
    return out[keep]


def team_season_table(tms: pd.DataFrame) -> pd.DataFrame:
    """Aggregate team_match_stats into a league table with per-match rates."""
    g = tms.groupby("team_id")
    table = g.agg(
        played=("match_id", "nunique"),
        points=("points", "sum"),
        goals=("goals", "sum"),
        goals_against=("goals_against", "sum"),
        xg=("xg", "sum"),
        xga=("xga", "sum"),
        npxg=("npxg", "sum"),
        npxga=("npxga", "sum"),
        shots=("shots", "sum"),
        progressive_passes=("progressive_passes", "sum"),
        ppda_passes=("ppda_passes", "sum"),
        ppda_actions=("ppda_actions", "sum"),
    )
    table["goals_minus_xg"] = table.goals - table.xg
    table["xgd"] = table.xg - table.xga
    table["ppda"] = table.ppda_passes / table.ppda_actions
    table["prog_passes_pm"] = table.progressive_passes / table.played
    return table.drop(columns=["ppda_passes", "ppda_actions"]).reset_index()


# --------------------------------------------------------------------------- player metrics


def position_group(position: str | None) -> str | None:
    """Collapse StatsBomb's 24 positions into GK / DF / MF / FW."""
    if not position:
        return None
    if position == "Goalkeeper":
        return "GK"
    if "Back" in position:
        return "DF"
    if "Forward" in position or "Wing" in position or "Striker" in position:
        return "FW"
    return "MF"


def player_season_stats(events: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    """Season totals per player (raw counts; per-90 is derived at read time).

    lineups columns: match_id, player_id, team_id, minutes_played, main_position, is_starter
    """
    ev = events
    shots = ev[is_shot(ev)]
    np_shots = shots[shots.sub_type != "Penalty"]
    passes = ev[ev.type == "Pass"]
    carries = ev[ev.type == "Carry"]

    # xA: xG of the shots each player's pass set up (via the shot's key_pass_id).
    key_pass_xg = shots[["shot_key_pass_id", "shot_xg"]].dropna()
    xa = (
        passes[["event_id", "player_id"]]
        .merge(key_pass_xg, left_on="event_id", right_on="shot_key_pass_id")
        .groupby("player_id")["shot_xg"]
        .sum()
    )

    by_player = pd.DataFrame(
        {
            "np_goals": np_shots[np_shots.outcome == "Goal"].groupby("player_id").size(),
            "npxg": np_shots.groupby("player_id")["shot_xg"].sum(),
            "shots": shots.groupby("player_id").size(),
            "xa": xa,
            "assists": passes[passes.pass_goal_assist].groupby("player_id").size(),
            "key_passes": passes[passes.pass_shot_assist | passes.pass_goal_assist]
            .groupby("player_id")
            .size(),
            "passes": passes.groupby("player_id").size(),
            "passes_completed": passes[passes.outcome.isna()].groupby("player_id").size(),
            "progressive_passes": passes[is_progressive_pass(passes)].groupby("player_id").size(),
            "progressive_carries": carries[is_progressive(carries)].groupby("player_id").size(),
            "dribbles_completed": ev[(ev.type == "Dribble") & (ev.outcome == "Complete")]
            .groupby("player_id")
            .size(),
            "pressures": ev[ev.type == "Pressure"].groupby("player_id").size(),
            "tackles_interceptions": ev[
                ((ev.type == "Duel") & (ev.sub_type == "Tackle")) | (ev.type == "Interception")
            ]
            .groupby("player_id")
            .size(),
            "ball_recoveries": ev[ev.type == "Ball Recovery"].groupby("player_id").size(),
        }
    ).fillna(0)

    lu = lineups.assign(group=lineups.main_position.map(position_group))
    minutes = lu.groupby("player_id").agg(
        minutes=("minutes_played", "sum"),
        matches=("match_id", "nunique"),
        starts=("is_starter", "sum"),
    )
    # Team and position = where the player spent most minutes (handles January transfers).
    main_team = lu.groupby(["player_id", "team_id"])["minutes_played"].sum().reset_index()
    main_team = main_team.sort_values("minutes_played").drop_duplicates("player_id", keep="last")
    main_group = lu.groupby(["player_id", "group"])["minutes_played"].sum().reset_index()
    main_group = main_group.sort_values("minutes_played").drop_duplicates("player_id", keep="last")

    out = (
        minutes.join(main_team.set_index("player_id")["team_id"])
        .join(main_group.set_index("player_id")["group"].rename("position_group"))
        .join(by_player)
        .fillna({c: 0 for c in by_player.columns})
        .reset_index()
    )
    int_cols = [c for c in by_player.columns if c not in ("npxg", "xa")]
    out[int_cols] = out[int_cols].astype(int)
    return out


PER90_METRICS = {
    "npxg": "npxG",
    "np_goals": "Goles (sin pen.)",
    "shots": "Tiros",
    "xa": "xA",
    "key_passes": "Pases clave",
    "progressive_passes": "Pases progresivos",
    "progressive_carries": "Conducciones prog.",
    "dribbles_completed": "Regates completados",
    "pressures": "Presiones",
    "tackles_interceptions": "Entradas + intercep.",
    "ball_recoveries": "Recuperaciones",
}


def per90(stats: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Divide counting stats by minutes/90. Only meaningful with a minutes threshold."""
    cols = cols or list(PER90_METRICS)
    out = stats.copy()
    factor = 90.0 / out["minutes"].replace(0, np.nan)
    for c in cols:
        out[c] = out[c] * factor
    out["pass_pct"] = out["passes_completed"] / out["passes"].replace(0, np.nan)
    return out
