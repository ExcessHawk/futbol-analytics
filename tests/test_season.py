"""Current-season standings and market expected points."""

import pandas as pd
import pytest

from futbol.season import standings


def fd_rows(rows):
    cols = [
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "odds_home",
        "odds_draw",
        "odds_away",
    ]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["home_shots", "away_shots", "home_sot", "away_sot"]:
        df[c] = 5
    return df


def test_points_form_and_unplayed_fixtures():
    df = fd_rows([
        ("2026-08-10", "A", "B", 2, 0, 2.0, 3.5, 4.0),
        ("2026-08-17", "B", "A", 1, 1, 2.5, 3.2, 3.0),
        ("2026-08-24", "A", "C", 0, 1, 1.5, 4.0, 7.0),
        ("2026-08-31", "C", "A", None, None, 3.0, 3.3, 2.4),  # not played yet: ignored
    ])  # fmt: skip
    t = standings(df).set_index("team")
    assert t.loc["A", "played"] == 3
    assert t.loc["A", "points"] == 4
    assert t.loc["A", "form"] == "G E P"
    assert t.loc["C", "points"] == 3
    assert t.index[0] == "A"  # 4 pts beats C's 3


def test_expected_points_match_implied_probabilities():
    # Fair odds (no margin): P(home)=0.5, P(draw)=0.25, P(away)=0.25.
    df = fd_rows([("2026-08-10", "A", "B", 0, 0, 2.0, 4.0, 4.0)])
    t = standings(df).set_index("team")
    assert t.loc["A", "xpts"] == pytest.approx(3 * 0.5 + 0.25)
    assert t.loc["B", "xpts"] == pytest.approx(3 * 0.25 + 0.25)
    assert t.loc["A", "pts_minus_xpts"] == pytest.approx(1 - 1.75)
