"""Walk-forward backtesting and probabilistic scoring.

Leakage rule: a prediction for a match played in week W may only use matches played
strictly before the Monday that starts week W. The model is refitted every week.
Matches later in the same week therefore use slightly stale info - conservative, never
optimistic. `backtest` asserts the rule on every block.

Scores (lower is better for all three):
  * Brier: sum over the 3 outcomes of (p - o)^2. 0 = perfect, uniform 1/3 guess = 0.667.
  * Log loss: -log(p of what happened). Punishes confident mistakes hard. Uniform = 1.099.
  * RPS (ranked probability score): like Brier on cumulative probs; respects that
    a draw is "closer" to a home win than an away win is. Standard in football papers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

log = logging.getLogger(__name__)

PROB_COLS = ["p_home", "p_draw", "p_away"]
OUTCOMES = ["home", "draw", "away"]


# --------------------------------------------------------------------------- scoring


def outcome_index(home_goals: pd.Series, away_goals: pd.Series) -> np.ndarray:
    """0 = home win, 1 = draw, 2 = away win."""
    return np.select([home_goals > away_goals, home_goals == away_goals], [0, 1], default=2)


def per_match_scores(probs: np.ndarray, outcome: np.ndarray) -> pd.DataFrame:
    onehot = np.eye(3)[outcome]
    brier = ((probs - onehot) ** 2).sum(axis=1)
    log_loss = -np.log(np.clip(probs[np.arange(len(outcome)), outcome], 1e-15, None))
    cum_p, cum_o = np.cumsum(probs, axis=1)[:, :2], np.cumsum(onehot, axis=1)[:, :2]
    rps = ((cum_p - cum_o) ** 2).sum(axis=1) / 2
    return pd.DataFrame({"brier": brier, "log_loss": log_loss, "rps": rps})


def score_table(preds: pd.DataFrame, reference: str | None = None) -> pd.DataFrame:
    """Mean scores per model over the matches *all* models predicted (fair comparison).

    With `reference`, adds the paired log-loss difference vs that model and its 95%
    interval (mean +/- 1.96 standard errors of the per-match differences).
    """
    key = ["league_code", "season", "match_date", "home_team", "away_team"]
    common = preds.groupby(key)["model"].nunique()
    common = common[common == preds.model.nunique()].index
    p = preds.set_index(key).loc[common].reset_index()

    outcome = outcome_index(p.home_goals, p.away_goals)
    p = pd.concat(
        [p.reset_index(drop=True), per_match_scores(p[PROB_COLS].to_numpy(), outcome)], axis=1
    )
    p["correct"] = p[PROB_COLS].to_numpy().argmax(axis=1) == outcome

    table = p.groupby("model").agg(
        matches=("brier", "size"),
        brier=("brier", "mean"),
        log_loss=("log_loss", "mean"),
        rps=("rps", "mean"),
        accuracy=("correct", "mean"),
    )
    if reference and reference in table.index:
        ref = p[p.model == reference].set_index(key)["log_loss"]
        diffs = {}
        for model, part in p.groupby("model"):
            d = part.set_index(key)["log_loss"] - ref
            se = d.std(ddof=1) / np.sqrt(len(d))
            diffs[model] = (d.mean(), d.mean() - 1.96 * se, d.mean() + 1.96 * se)
        table[["ll_vs_ref", "ll_vs_ref_lo", "ll_vs_ref_hi"]] = pd.DataFrame(
            diffs, index=["d", "lo", "hi"]
        ).T
    return table.sort_values("log_loss")


def calibration_data(preds: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed frequency per outcome, quantile bins (equal match counts)."""
    outcome = outcome_index(preds.home_goals, preds.away_goals)
    rows = []
    for i, (col, name) in enumerate(zip(PROB_COLS, OUTCOMES, strict=True)):
        observed, predicted = calibration_curve(
            (outcome == i).astype(int), preds[col], n_bins=n_bins, strategy="quantile"
        )
        rows.append(pd.DataFrame({"outcome": name, "predicted": predicted, "observed": observed}))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- backtest

# A predictor receives (train, test) and returns a DataFrame indexed like `test` with
# PROB_COLS (+ optional exp_home/exp_away). It may return several models at once as a
# dict, so expensive fits are shared (Poisson and Dixon-Coles use the same fit).
Predictor = Callable[[pd.DataFrame, pd.DataFrame, pd.Timestamp], dict[str, pd.DataFrame]]


def week_start(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    return (d - pd.to_timedelta(d.dt.day_of_week, unit="D")).dt.normalize()


def backtest(
    data: pd.DataFrame,
    predictor: Predictor,
    test_mask: pd.Series,
    window_days: int | None = 3 * 365,
    same_season_only: bool = False,
    min_train_matches: int = 30,
) -> pd.DataFrame:
    """Walk forward through test matches week by week, refitting before each week.

    data: one league's matches with match_date, season, home_team, away_team, goals...
    test_mask: which rows to predict. Training may use any row (test or not) before the block.
    """
    data = data.assign(match_date=pd.to_datetime(data.match_date)).sort_values("match_date")
    data["block"] = week_start(data.match_date)
    test = data[test_mask.reindex(data.index).fillna(False).astype(bool)]
    out = []
    for block, block_matches in test.groupby("block"):
        train = data[data.match_date < block]
        if window_days:
            train = train[train.match_date >= block - pd.Timedelta(days=window_days)]
        if same_season_only:
            train = train[train.season.isin(block_matches.season.unique())]
        train = train.dropna(subset=["home_goals", "away_goals"])
        if len(train) < min_train_matches:
            continue
        # The leakage guard: nothing from this week or later in the training set.
        assert train.match_date.max() < block_matches.match_date.min()
        for model, pred in predictor(train, block_matches, block).items():
            # inner join: a model may skip matches (e.g. the market where odds are missing)
            out.append(block_matches.join(pred, how="inner").assign(model=model))
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)
