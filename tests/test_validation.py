"""Scoring rules and the no-leakage guarantee of the backtest."""

import numpy as np
import pandas as pd
import pytest

from futbol.validation import backtest, per_match_scores, score_table


def test_uniform_forecast_reference_values():
    probs = np.full((3, 3), 1 / 3)
    s = per_match_scores(probs, np.array([0, 1, 2]))
    assert s.brier.mean() == pytest.approx(2 / 3)
    assert s.log_loss.mean() == pytest.approx(np.log(3))


def test_perfect_forecast_scores_zero():
    s = per_match_scores(np.eye(3), np.array([0, 1, 2]))
    assert s.brier.sum() == 0 and s.rps.sum() == 0


def test_rps_prefers_near_misses():
    # Outcome is a home win. Putting the mass on draw is "less wrong" than on away.
    near = per_match_scores(np.array([[0.1, 0.8, 0.1]]), np.array([0])).rps[0]
    far = per_match_scores(np.array([[0.1, 0.1, 0.8]]), np.array([0])).rps[0]
    assert near < far


def _league(n_weeks=10):
    rows = []
    start = pd.Timestamp("2021-08-02")  # a Monday
    for w in range(n_weeks):
        for d, (h, a) in enumerate([("A", "B"), ("C", "D"), ("B", "C"), ("D", "A")]):
            rows.append((start + pd.Timedelta(days=7 * w + d), "2021/2022", h, a, 1, 0))
    return pd.DataFrame(
        rows, columns=["match_date", "season", "home_team", "away_team", "home_goals", "away_goals"]
    )


def test_backtest_never_trains_on_same_week_or_future():
    data = _league()
    seen = []

    def spy(train, test, block):
        seen.append((train.match_date.max(), test.match_date.min(), block))
        return {"m": pd.DataFrame({"p_home": 0.5, "p_draw": 0.3, "p_away": 0.2}, index=test.index)}

    preds = backtest(
        data, spy, test_mask=data.match_date >= "2021-09-01", window_days=None, min_train_matches=1
    )
    assert len(preds) > 0
    for train_max, test_min, block in seen:
        assert train_max < block <= test_min


def test_score_table_only_compares_common_matches():
    base = pd.DataFrame({"league_code": "E0", "season": "s", "home_goals": [1, 0], "away_goals": [0, 0],
                         "home_team": ["A", "B"], "away_team": ["B", "A"],
                         "match_date": pd.to_datetime(["2026-08-10", "2026-08-17"])})  # fmt: skip
    m1 = base.assign(model="m1", p_home=0.5, p_draw=0.3, p_away=0.2)
    m2 = base.iloc[:1].assign(
        model="m2", p_home=0.6, p_draw=0.2, p_away=0.2
    )  # predicted only 1 match
    table = score_table(pd.concat([m1, m2]))
    assert (table.matches == 1).all()


def test_same_pairing_twice_in_a_season_counts_as_two_matches():
    """Liga MX: Pumas-Cruz Azul in the Clausura and again in the liguilla, same season label."""
    rows = pd.DataFrame(
        {"league_code": "MEX", "season": "2025/2026", "home_team": "Pumas", "away_team": "Cruz Azul",
         "match_date": pd.to_datetime(["2026-03-15", "2026-05-25"]), "home_goals": [2, 1],
         "away_goals": [2, 2], "model": "m", "p_home": 0.4, "p_draw": 0.3, "p_away": 0.3}
    )  # fmt: skip
    assert score_table(rows).matches.iloc[0] == 2
