"""Match forecasts beyond 1X2: goals, corners, yellow cards and shots.

Every count market uses the same attack/defence Poisson model as goals (futbol.models),
fitted on that count instead:
    home corners ~ Poisson(exp(mu0 + home_adv + generate[home] + concede[away]))
For cards, "attack" is a team's tendency to be booked and "defence" how much it gets
its opponents booked. Totals of two independent Poissons are Poisson(lambda + mu), which
gives P(total > line) in closed form.

Honest limits:
  * Corners are mildly over-dispersed (variance ~1.2x the mean league-wide), so Poisson
    tails are a bit too thin; a negative binomial would fix it. The backtest calibration
    shows how much it matters.
  * Cards depend heavily on the referee and on game state (a tight derby vs a 3-0).
    Only the Premier League data names the referee, so only there a referee factor is used.
  * Goals markets (over/under, both teams to score, scorelines) come from the Dixon-Coles
    score matrix, so they inherit its assumptions (see futbol.models).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import poisson

from futbol.models import MAX_GOALS, PoissonModel, outcome_probs, score_matrices


@dataclass(frozen=True)
class Market:
    key: str
    home_col: str
    away_col: str
    label: str


MARKETS = {
    m.key: m
    for m in [
        Market("goals", "home_goals", "away_goals", "Goles"),
        Market("corners", "home_corners", "away_corners", "Córners"),
        Market("yellow", "home_yellow", "away_yellow", "Tarjetas amarillas"),
        Market("shots", "home_shots", "away_shots", "Tiros"),
        Market("sot", "home_sot", "away_sot", "Tiros a puerta"),
    ]
}
REFEREE_SHRINKAGE = 10  # matches of "league average" blended into each referee's record


def main_line(totals: pd.Series) -> float:
    """The x.5 line closest to the typical total, e.g. mean 10.3 corners -> 9.5 / 10.5."""
    return float(np.floor(totals.mean()) + 0.5)


def p_total_over(lam: np.ndarray, mu: np.ndarray, line: float) -> np.ndarray:
    """P(home + away > line) when both are independent Poisson counts."""
    return poisson.sf(np.floor(line), np.asarray(lam) + np.asarray(mu))


def referee_factors(train: pd.DataFrame, market: Market) -> dict[str, float]:
    """Multiplier per referee on the total count, shrunk towards 1 (= league average).

    factor = (referee total + k * league mean) / (n matches + k) / league mean
    """
    df = train.dropna(subset=["referee", market.home_col, market.away_col])
    if df.empty:
        return {}
    total = df[market.home_col] + df[market.away_col]
    league_mean = total.mean()
    g = total.groupby(df.referee).agg(["sum", "count"])
    k = REFEREE_SHRINKAGE
    return ((g["sum"] + k * league_mean) / (g["count"] + k) / league_mean).to_dict()


@dataclass
class MatchForecaster:
    """Fits one model per market on the same training window and forecasts fixtures."""

    params: dict
    models: dict[str, PoissonModel] = field(default_factory=dict, init=False)
    lines: dict[str, float] = field(default_factory=dict, init=False)
    referee_factor: dict[str, float] = field(default_factory=dict, init=False)

    def fit(self, train: pd.DataFrame, ref_date: pd.Timestamp) -> MatchForecaster:
        for key, m in MARKETS.items():
            if not {m.home_col, m.away_col} <= set(train.columns):
                continue  # this data source has no such statistic
            data = train.dropna(subset=[m.home_col, m.away_col, "home_goals", "away_goals"])
            if len(data) < 30:
                continue
            model = PoissonModel(**self.params, dixon_coles=(key == "goals"))
            self.models[key] = model.fit(data, ref_date, m.home_col, m.away_col)
            self.lines[key] = main_line(data[m.home_col] + data[m.away_col])
        if "referee" in train and train.referee.notna().any():
            self.referee_factor = referee_factors(train, MARKETS["yellow"])
        return self

    def counts(
        self,
        key: str,
        fixtures: pd.DataFrame,
        line: float | None = None,
        use_referee: bool = True,
    ) -> pd.DataFrame:
        """Expected home/away counts and P(total > line) for one market."""
        lam, mu = self.models[key].expected_goals(fixtures.home_team, fixtures.away_team)
        if key == "yellow" and use_referee and self.referee_factor and "referee" in fixtures:
            f = fixtures.referee.map(self.referee_factor).fillna(1.0).to_numpy(float)
            lam, mu = lam * f, mu * f
        line = self.lines[key] if line is None else line
        return pd.DataFrame(
            {"exp_home": lam, "exp_away": mu, "line": line, "p_over": p_total_over(lam, mu, line)},
            index=fixtures.index,
        )

    def goals_detail(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """1X2, over/under 2.5, both teams to score and most likely score from the matrix."""
        model = self.models["goals"]
        lam, mu = model.expected_goals(fixtures.home_team, fixtures.away_team)
        m = score_matrices(lam, mu, model.rho_)
        probs = outcome_probs(m)
        i, j = np.meshgrid(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1), indexing="ij")
        flat = m.reshape(len(m), -1)
        best = flat.argmax(axis=1)
        return pd.DataFrame(
            {
                "p_home": probs[:, 0],
                "p_draw": probs[:, 1],
                "p_away": probs[:, 2],
                "exp_home_goals": lam,
                "exp_away_goals": mu,
                "p_over_2_5": (m * (i + j > 2.5)).sum(axis=(1, 2)),
                "p_btts": (m * ((i > 0) & (j > 0))).sum(axis=(1, 2)),
                "top_score": [f"{b // (MAX_GOALS + 1)}-{b % (MAX_GOALS + 1)}" for b in best],
                "p_top_score": flat[np.arange(len(m)), best],
            },
            index=fixtures.index,
        )

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        model = self.models["goals"]
        lam, mu = model.expected_goals(pd.Series([home]), pd.Series([away]))
        return score_matrices(lam, mu, model.rho_)[0]

    def forecast(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Everything for a set of fixtures, one row each (the dashboard table)."""
        out = self.goals_detail(fixtures)
        for key in ("corners", "yellow", "shots"):
            if key not in self.models:
                continue
            c = self.counts(key, fixtures)
            out[f"exp_home_{key}"] = c.exp_home
            out[f"exp_away_{key}"] = c.exp_away
            out[f"exp_total_{key}"] = c.exp_home + c.exp_away
            out[f"line_{key}"] = c.line
            out[f"p_over_{key}"] = c.p_over
        return out


def binary_scores(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Log loss and Brier score for a yes/no forecast."""
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, float)
    return {
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
    }


def _per_match_log_loss(p: pd.Series, y: pd.Series) -> pd.Series:
    p = p.clip(1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def market_score_table(preds: pd.DataFrame, baseline: str = "league_average") -> pd.DataFrame:
    """Scores per market and model, plus the paired log-loss gap to the baseline.

    Fairness: within each (market, league) only matches that every model available *in
    that league* predicted are kept (the referee model exists only for the Premier League,
    so it must not shrink the other leagues' sample).
    preds: market_predictions rows joined with actual counts (actual_home, actual_away).
    """
    key = ["league_code", "season", "match_date", "home_team", "away_team"]
    df = preds.dropna(subset=["actual_home", "actual_away"]).copy()
    parts = []
    for _, part in df.groupby(["market", "league_code"]):
        common = part.groupby(key)["model"].nunique()
        common = common[common == part.model.nunique()].index
        parts.append(part.set_index(key).loc[common].reset_index())
    df = pd.concat(parts, ignore_index=True)
    df["total"] = df.actual_home + df.actual_away
    df["y"] = (df.total > df.line).astype(int)
    df["ll"] = _per_match_log_loss(df.p_over, df.y)

    rows = []
    for (market, model), g in df.groupby(["market", "model"]):
        exp_total = g.exp_home + g.exp_away
        row = {
            "market": market,
            "model": model,
            "matches": len(g),
            **binary_scores(g.p_over, g.y),
            "mae_total": float((exp_total - g.total).abs().mean())
            if exp_total.notna().all()
            else np.nan,
            "over_rate": float(g.y.mean()),
        }
        base = df[(df.market == market) & (df.model == baseline)].set_index(key)["ll"]
        d = (g.set_index(key)["ll"] - base.reindex(g.set_index(key).index)).dropna()
        if len(d) > 1:
            se = d.std(ddof=1) / np.sqrt(len(d))
            row |= {"ll_vs_base": d.mean(), "ll_vs_base_lo": d.mean() - 1.96 * se,
                    "ll_vs_base_hi": d.mean() + 1.96 * se}  # fmt: skip
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["market", "log_loss"], ignore_index=True)


def opponent_factor(fc: MatchForecaster, key: str, team: str, home: str, away: str) -> float:
    """How much more (or less) of `key` a team produces in this match than in a typical one.

    A player's season average already reflects his own team's strength, so only the
    opponent and the venue need adjusting: this match's expected team count divided by
    the team's mean expected count against every other team, half at home, half away.
    """
    model = fc.models[key]
    lam_home, lam_away = model.expected_goals(pd.Series([home]), pd.Series([away]))
    this_match = float(lam_home[0] if team == home else lam_away[0])
    others = pd.Series([t for t in model.teams_ if t != team])
    mine = pd.Series([team] * len(others))
    at_home, _ = model.expected_goals(mine, others)
    _, away_games = model.expected_goals(others, mine)
    typical = (at_home.mean() + away_games.mean()) / 2
    return this_match / typical


def player_over_probability(
    fc: MatchForecaster, key: str, team: str, home: str, away: str, avg: float, line: float
) -> tuple[float, float]:
    """P(player count > line) with count ~ Poisson(avg * opponent factor). Returns (p, factor).

    Not validated: there is no public current-season player data to backtest it on.
    Player counts are also over-dispersed (a substitute appearance, a red card), so the
    Poisson tails are optimistic for high lines.
    """
    factor = opponent_factor(fc, key, team, home, away)
    lam = max(avg, 0.0) * factor
    return float(poisson.sf(np.floor(line), lam)), factor
