"""Metric definitions on tiny hand-built event sets."""

import pandas as pd
import pytest

from futbol.ingest.xref import vote_mapping
from futbol.metrics import is_progressive_pass, position_group, team_match_stats


def ev(**kw):
    base = dict(event_id="e", match_id=1, type="Pass", sub_type=None, outcome=None, team_id=1,
                player_id=10, x=60.0, y=40.0, end_x=60.0, end_y=40.0, shot_xg=None)  # fmt: skip
    base.update(kw)
    return base


def test_progressive_pass_rules():
    df = pd.DataFrame([
        ev(x=60, end_x=90),                       # 60->30 yards to goal: progressive
        ev(x=60, end_x=65),                       # small gain: not progressive
        ev(x=30, end_x=80),                       # starts in own defensive 40%: excluded
        ev(x=60, end_x=90, outcome="Incomplete"), # not completed
        ev(x=60, end_x=90, sub_type="Free Kick"), # set piece
        ev(x=100, y=10, end_x=104, end_y=30),     # small move but enters the box
    ])  # fmt: skip
    assert is_progressive_pass(df).tolist() == [True, False, False, False, False, True]


def test_ppda_counts_opponent_build_up_passes_over_our_high_actions():
    events = pd.DataFrame(
        [ev(team_id=2, x=30) for _ in range(12)]          # opponent passes in their own 60%
        + [ev(team_id=2, x=100)]                          # opponent pass high up: not counted
        + [ev(team_id=1, type="Interception", x=70), ev(team_id=1, type="Duel", sub_type="Tackle", x=50),
           ev(team_id=1, type="Foul Committed", x=90),
           ev(team_id=1, type="Interception", x=20)]      # our deep action: not counted
    )  # fmt: skip
    matches = pd.DataFrame({"match_id": [1], "home_team_id": [1], "away_team_id": [2],
                            "home_score": [1], "away_score": [0]})  # fmt: skip
    stats = team_match_stats(events, matches).set_index("team_id")
    assert stats.loc[1, "ppda"] == pytest.approx(12 / 3)
    assert stats.loc[1, "points"] == 3 and stats.loc[2, "points"] == 0


def test_position_groups():
    assert position_group("Goalkeeper") == "GK"
    assert position_group("Left Wing Back") == "DF"
    assert position_group("Right Wing") == "FW"
    assert position_group("Center Defensive Midfield") == "MF"


def test_fixture_voting_maps_names():
    dates = pd.date_range("2015-08-01", periods=12, freq="7D").date
    sb = pd.DataFrame({"match_date": dates, "home_team_id": [1, 2] * 6, "away_team_id": [2, 1] * 6,
                       "home_score": [1, 2, 0, 3, 1, 1, 2, 0, 1, 4, 2, 1], "away_score": [0] * 12})  # fmt: skip
    fd = sb.rename(columns={"home_score": "home_goals", "away_score": "away_goals"}).assign(
        home_team=lambda d: d.home_team_id.map({1: "Man United", 2: "Man City"}),
        away_team=lambda d: d.away_team_id.map({1: "Man United", 2: "Man City"}),
    )[["match_date", "home_team", "away_team", "home_goals", "away_goals"]]
    m = vote_mapping(sb, fd, min_votes=5).set_index("fd_team").team_id
    assert m["Man United"] == 1 and m["Man City"] == 2
