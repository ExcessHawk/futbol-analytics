"""Model tests: maths first (gradient, probabilities), then parameter recovery."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from futbol.models import (
    PoissonModel,
    dc_tau,
    odds_to_probs,
    outcome_probs,
    score_matrices,
)


def simulate_league(n_teams=12, rounds=6, seed=0):
    """Double round-robin repeated `rounds` times with known true parameters."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    attack = rng.normal(0, 0.3, n_teams)
    attack -= attack.mean()
    defence = rng.normal(0, 0.3, n_teams)
    defence -= defence.mean()
    mu0, home_adv = 0.1, 0.25
    rows, day = [], 0
    for _ in range(rounds):
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                lam = np.exp(mu0 + home_adv + attack[i] + defence[j])
                mu = np.exp(mu0 + attack[j] + defence[i])
                rows.append((pd.Timestamp("2020-01-01") + pd.Timedelta(days=day), teams[i], teams[j],
                             rng.poisson(lam), rng.poisson(mu)))  # fmt: skip
                day += 1
    df = pd.DataFrame(
        rows, columns=["match_date", "home_team", "away_team", "home_goals", "away_goals"]
    )
    return df, dict(zip(teams, attack, strict=True)), home_adv


def test_score_matrix_sums_to_one_and_probs_too():
    lam, mu = np.array([0.5, 1.4, 3.0]), np.array([0.7, 1.1, 0.4])
    for rho in (0.0, -0.1):
        m = score_matrices(lam, mu, rho)
        np.testing.assert_allclose(m.sum(axis=(1, 2)), 1.0)
        np.testing.assert_allclose(outcome_probs(m).sum(axis=1), 1.0)


def test_dc_tau_only_touches_low_scores():
    lam, mu = np.full(4, 1.3), np.full(4, 1.1)
    x, y = np.array([0, 1, 2, 3]), np.array([0, 1, 0, 3])
    tau = dc_tau(x, y, lam, mu, -0.1)
    assert tau[0] > 1 and tau[1] > 1  # negative rho raises 0-0 and 1-1
    np.testing.assert_array_equal(tau[2:], 1.0)


def test_negative_rho_increases_draw_probability():
    lam, mu = np.array([1.3]), np.array([1.1])
    p_plain = outcome_probs(score_matrices(lam, mu, 0.0))[0, 1]
    p_dc = outcome_probs(score_matrices(lam, mu, -0.1))[0, 1]
    assert p_dc > p_plain


def test_analytic_gradient_matches_numerical():
    df, _, _ = simulate_league(n_teams=6, rounds=1)
    model = PoissonModel(ridge=3.0)
    teams = sorted(set(df.home_team))
    idx = {t: i for i, t in enumerate(teams)}
    h, a = df.home_team.map(idx).to_numpy(), df.away_team.map(idx).to_numpy()
    yh, ya = df.home_goals.to_numpy(float), df.away_goals.to_numpy(float)
    w = np.linspace(0.5, 1, len(df))
    n = len(teams)
    t = np.full(n, 0.1)

    captured = {}

    def fake_minimize(fun, x0, jac, method):  # capture the closure the model builds
        captured["f"] = fun

        class R:
            x = x0

        return R

    import futbol.models as mod

    original = mod.minimize
    mod.minimize = fake_minimize
    try:
        model._optimise(h, a, yh, ya, w, n, t, t)
    finally:
        mod.minimize = original
    f = captured["f"]
    theta = np.random.default_rng(1).normal(0, 0.2, 2 + 2 * n)
    err = check_grad(lambda th: f(th)[0], lambda th: f(th)[1], theta)
    assert err < 1e-4


def test_recovers_home_advantage_and_team_order():
    df, true_attack, true_home = simulate_league(n_teams=12, rounds=8)
    model = PoissonModel(xi=0.0, ridge=0.5, dixon_coles=False).fit(
        df, df.match_date.max() + pd.Timedelta(days=1)
    )
    assert model.home_adv_ == pytest.approx(true_home, abs=0.08)
    fitted = dict(zip(model.teams_, model.attack_, strict=True))
    corr = np.corrcoef([true_attack[t] for t in model.teams_], [fitted[t] for t in model.teams_])[
        0, 1
    ]
    assert corr > 0.85


def test_unseen_team_gets_newcomer_prior_not_average():
    df, _, _ = simulate_league(n_teams=10, rounds=2)
    model = PoissonModel().fit(df, df.match_date.max() + pd.Timedelta(days=1))
    new = pd.DataFrame({"home_team": ["PROMOTED"], "away_team": ["T00"]})
    att, _ = model._team_params(new.home_team)
    assert att[0] == pytest.approx(model.newcomer_prior_[0])
    assert model.newcomer_prior_[0] < 0  # weaker than the average team


def test_fits_on_fractional_xg_targets():
    df, _, _ = simulate_league(n_teams=8, rounds=2)
    df["home_xg"] = df.home_goals * 0.9 + 0.15
    df["away_xg"] = df.away_goals * 0.9 + 0.1
    model = PoissonModel().fit(df, df.match_date.max() + pd.Timedelta(days=1), "home_xg", "away_xg")
    preds = model.predict(df.head(5))
    np.testing.assert_allclose(preds[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0, atol=1e-9)


def test_odds_to_probs_removes_margin():
    odds = pd.DataFrame({"odds_home": [2.0], "odds_draw": [3.4], "odds_away": [3.8]})
    probs = odds_to_probs(odds)
    assert probs.sum(axis=1).iloc[0] == pytest.approx(1.0)
    assert probs.p_home.iloc[0] > probs.p_away.iloc[0]
