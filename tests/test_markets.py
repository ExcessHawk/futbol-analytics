"""Count markets: totals maths, referee shrinkage, forecaster outputs."""

import numpy as np
import pandas as pd
import pytest

from futbol.markets import (
    MARKETS,
    MatchForecaster,
    main_line,
    p_total_over,
    referee_factors,
)


def test_total_over_matches_simulation():
    rng = np.random.default_rng(0)
    lam, mu, line = 5.6, 4.4, 9.5
    sims = rng.poisson(lam, 200_000) + rng.poisson(mu, 200_000)
    assert p_total_over(np.array([lam]), np.array([mu]), line)[0] == pytest.approx(
        (sims > line).mean(), abs=0.005
    )


def test_main_line_is_half_point_near_mean():
    assert main_line(pd.Series([9, 10, 11, 12])) == 10.5  # mean 10.5 -> floor 10 + 0.5


def test_referee_factor_shrinks_small_samples_to_one():
    df = pd.DataFrame(
        {
            "referee": ["Strict"] * 2 + ["Average"] * 40,
            "home_yellow": [5, 5] + [2] * 40,
            "away_yellow": [5, 5] + [2] * 40,
        }
    )
    f = referee_factors(df, MARKETS["yellow"])
    raw_ratio = 10 / df.eval("home_yellow + away_yellow").mean()
    assert 1 < f["Strict"] < raw_ratio  # pulled towards the league average with only 2 games


def league(seed=0, n_rounds=4):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(8)]
    rows, day = [], 0
    for _ in range(n_rounds):
        for h in teams:
            for a in teams:
                if h != a:
                    rows.append({
                        "match_date": pd.Timestamp("2025-08-01") + pd.Timedelta(days=day),
                        "home_team": h, "away_team": a,
                        "home_goals": rng.poisson(1.5), "away_goals": rng.poisson(1.1),
                        "home_corners": rng.poisson(5.5), "away_corners": rng.poisson(4.5),
                        "home_yellow": rng.poisson(1.8), "away_yellow": rng.poisson(2.0),
                        "home_shots": rng.poisson(13), "away_shots": rng.poisson(11),
                        "referee": None,
                    })  # fmt: skip
                    day += 1
    return pd.DataFrame(rows)


def test_forecast_outputs_are_valid_probabilities():
    df = league()
    fc = MatchForecaster({"xi": 0.001, "ridge": 5.0}).fit(
        df, df.match_date.max() + pd.Timedelta(days=1)
    )
    out = fc.forecast(pd.DataFrame({"home_team": ["T0", "T1"], "away_team": ["T2", "T3"]}))
    np.testing.assert_allclose(out[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0)
    for col in ["p_over_2_5", "p_btts", "p_over_corners", "p_over_yellow", "p_over_shots"]:
        assert out[col].between(0, 1).all(), col
    assert out.exp_total_corners.between(7, 13).all()
    m = fc.score_matrix("T0", "T2")
    assert m.sum() == pytest.approx(1.0)
