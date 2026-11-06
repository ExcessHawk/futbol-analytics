"""Match outcome models: Poisson attack/defence model with optional Dixon-Coles correction.

Model (Maher 1982; Dixon & Coles 1997):
    home goals ~ Poisson(lambda),  log lambda = mu0 + home_adv + attack[home] + defence[away]
    away goals ~ Poisson(mu),      log mu     = mu0 +            attack[away] + defence[home]
Here `defence` is "goals conceded" strength: higher = worse defence.

Assumptions (and where they break):
  1. Goals are Poisson: constant scoring rate within a match. Ignores game state
     (a team leading 1-0 sits back), red cards, etc.
  2. Home and away goals are independent given lambda, mu. Real data has more 0-0 and
     1-1 than that predicts; Dixon-Coles' rho term corrects the four low scores only.
  3. Team strength is constant within the training window, smoothed by time decay
     (weight = exp(-xi * days_ago)). Transfers, injuries and managers are invisible.
  4. One global home advantage for all teams.

Fitting with xG: the Poisson likelihood k*log(l) - l is still a valid (quasi-)likelihood
for non-integer k, so attack/defence can be fitted on xG instead of goals. rho is then
fitted in a second stage on actual integer scores, since tau() is only defined for them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

MAX_GOALS = 10  # P(a team scores > 10) is ~0 for any realistic lambda


def dc_tau(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """Dixon-Coles low-score adjustment factor. 1 for every score except 0-0, 0-1, 1-0, 1-1."""
    tau = np.ones_like(lam, dtype=float)
    tau = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, tau)
    tau = np.where((x == 0) & (y == 1), 1 + lam * rho, tau)
    tau = np.where((x == 1) & (y == 0), 1 + mu * rho, tau)
    tau = np.where((x == 1) & (y == 1), 1 - rho, tau)
    return tau


def score_matrices(lam: np.ndarray, mu: np.ndarray, rho: float = 0.0) -> np.ndarray:
    """P(home=i, away=j) for each match: shape (n, MAX_GOALS+1, MAX_GOALS+1)."""
    k = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(k[None, :], lam[:, None])
    pa = poisson.pmf(k[None, :], mu[:, None])
    m = ph[:, :, None] * pa[:, None, :]
    if rho:
        m[:, 0, 0] *= 1 - lam * mu * rho
        m[:, 0, 1] *= 1 + lam * rho
        m[:, 1, 0] *= 1 + mu * rho
        m[:, 1, 1] *= 1 - rho
    return m / m.sum(axis=(1, 2), keepdims=True)  # renormalise the truncated tail


def outcome_probs(matrices: np.ndarray) -> np.ndarray:
    """Score matrices -> (n, 3) array of P(home win), P(draw), P(away win)."""
    home = np.tril(np.ones(matrices.shape[1:]), k=-1)  # i > j
    p_home = (matrices * home).sum(axis=(1, 2))
    p_draw = np.trace(matrices, axis1=1, axis2=2)
    return np.column_stack([p_home, p_draw, 1 - p_home - p_draw])


@dataclass
class PoissonModel:
    """Attack/defence Poisson model. Set dixon_coles=False for the plain independent model.

    xi:     time-decay rate per day. 0.0019 -> weight halves every ~365 days.
    ridge:  L2 penalty pulling team parameters to their prior (0 = average team).
    newcomer_min_matches: teams with fewer training matches get the "newcomer" prior
            (mean of the 3 weakest established teams) instead of "average team".
    """

    xi: float = 0.0019
    ridge: float = 5.0
    dixon_coles: bool = True
    newcomer_min_matches: int = 10

    teams_: list[str] = field(default_factory=list, init=False)
    mu0_: float = field(default=0.0, init=False)
    home_adv_: float = field(default=0.0, init=False)
    attack_: np.ndarray = field(default_factory=lambda: np.zeros(0), init=False)
    defence_: np.ndarray = field(default_factory=lambda: np.zeros(0), init=False)
    rho_: float = field(default=0.0, init=False)
    newcomer_prior_: tuple[float, float] = field(default=(0.0, 0.0), init=False)

    # ------------------------------------------------------------------ fitting

    def fit(
        self,
        df: pd.DataFrame,
        ref_date: pd.Timestamp,
        home_col: str = "home_goals",
        away_col: str = "away_goals",
    ) -> PoissonModel:
        """Fit on matches strictly before ref_date (caller guarantees no leakage).

        df needs: match_date, home_team, away_team, home_col, away_col, and
        home_goals/away_goals (for rho).
        """
        self.teams_ = sorted(set(df.home_team) | set(df.away_team))
        index = {t: i for i, t in enumerate(self.teams_)}
        h = df.home_team.map(index).to_numpy()
        a = df.away_team.map(index).to_numpy()
        yh = df[home_col].to_numpy(float)
        ya = df[away_col].to_numpy(float)
        days = (pd.Timestamp(ref_date) - pd.to_datetime(df.match_date)).dt.days.to_numpy()
        w = np.exp(-self.xi * days)
        n = len(self.teams_)

        # Pass 1: everyone shrinks to "average team".
        zeros = np.zeros(n)
        theta = self._optimise(h, a, yh, ya, w, n, zeros, zeros, init_mean=np.mean(np.r_[yh, ya]))

        # Newcomer prior = mean of the 3 weakest established teams. Always computed: it is
        # also what predict() uses for teams absent from training (promoted on matchday 1).
        counts = np.bincount(h, minlength=n) + np.bincount(a, minlength=n)
        newcomer = counts < self.newcomer_min_matches
        est = np.flatnonzero(~newcomer)
        if len(est) >= 3:
            att, dfn = theta[2 : 2 + n], theta[2 + n :]
            weakest = est[np.argsort((att - dfn)[est])[:3]]
            self.newcomer_prior_ = (float(att[weakest].mean()), float(dfn[weakest].mean()))

        # Pass 2: teams with little data shrink to the newcomer prior instead of average.
        if newcomer.any() and len(est) >= 3:
            t_att = np.where(newcomer, self.newcomer_prior_[0], 0.0)
            t_def = np.where(newcomer, self.newcomer_prior_[1], 0.0)
            theta = self._optimise(h, a, yh, ya, w, n, t_att, t_def, init=theta)

        self.mu0_, self.home_adv_ = float(theta[0]), float(theta[1])
        self.attack_, self.defence_ = theta[2 : 2 + n], theta[2 + n :]

        if self.dixon_coles:
            played = df[["home_goals", "away_goals"]].notna().all(axis=1).to_numpy()
            lam, mu = self._rates(h[played], a[played])
            self.rho_ = self._fit_rho(
                df.home_goals.to_numpy()[played].astype(int),
                df.away_goals.to_numpy()[played].astype(int),
                lam, mu, w[played],
            )  # fmt: skip
        return self

    def _optimise(self, h, a, yh, ya, w, n, t_att, t_def, init=None, init_mean=1.4):
        def loss_and_grad(theta):
            mu0, home, att, dfn = theta[0], theta[1], theta[2 : 2 + n], theta[2 + n :]
            eta_h = np.clip(mu0 + home + att[h] + dfn[a], -10, 5)
            eta_a = np.clip(mu0 + att[a] + dfn[h], -10, 5)
            lam, mu = np.exp(eta_h), np.exp(eta_a)
            # Negative weighted Poisson log-likelihood (dropping the constant log k! term).
            nll = -np.sum(w * (yh * eta_h - lam + ya * eta_a - mu))
            pen = self.ridge * (np.sum((att - t_att) ** 2) + np.sum((dfn - t_def) ** 2))
            gh, ga = -w * (yh - lam), -w * (ya - mu)  # d nll / d eta
            grad = np.empty_like(theta)
            grad[0] = gh.sum() + ga.sum()
            grad[1] = gh.sum()
            grad[2 : 2 + n] = (np.bincount(h, gh, n) + np.bincount(a, ga, n)
                               + 2 * self.ridge * (att - t_att))  # fmt: skip
            grad[2 + n :] = (np.bincount(a, gh, n) + np.bincount(h, ga, n)
                             + 2 * self.ridge * (dfn - t_def))  # fmt: skip
            return nll + pen, grad

        if init is None:
            init = np.zeros(2 + 2 * n)
            init[0] = np.log(max(init_mean, 0.1))
            init[1] = 0.2
        res = minimize(loss_and_grad, init, jac=True, method="L-BFGS-B")
        return res.x

    @staticmethod
    def _fit_rho(x, y, lam, mu, w) -> float:
        def nll(rho):
            return -np.sum(w * np.log(np.clip(dc_tau(x, y, lam, mu, rho), 1e-10, None)))

        return float(minimize_scalar(nll, bounds=(-0.25, 0.25), method="bounded").x)

    # ------------------------------------------------------------------ prediction

    def _team_params(self, teams: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Attack/defence for each team; unseen teams (e.g. promoted) get the newcomer prior."""
        index = {t: i for i, t in enumerate(self.teams_)}
        att = np.array(
            [self.attack_[index[t]] if t in index else self.newcomer_prior_[0] for t in teams]
        )
        dfn = np.array(
            [self.defence_[index[t]] if t in index else self.newcomer_prior_[1] for t in teams]
        )
        return att, dfn

    def _rates(self, h_idx, a_idx):
        lam = np.exp(self.mu0_ + self.home_adv_ + self.attack_[h_idx] + self.defence_[a_idx])
        mu = np.exp(self.mu0_ + self.attack_[a_idx] + self.defence_[h_idx])
        return lam, mu

    def expected_goals(self, home: pd.Series, away: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        ha, hd = self._team_params(home)
        aa, ad = self._team_params(away)
        lam = np.exp(self.mu0_ + self.home_adv_ + ha + ad)
        mu = np.exp(self.mu0_ + aa + hd)
        return lam, mu

    def predict(self, df: pd.DataFrame, use_rho: bool | None = None) -> pd.DataFrame:
        """Outcome probabilities and expected goals for df[home_team, away_team]."""
        use_rho = self.dixon_coles if use_rho is None else use_rho
        lam, mu = self.expected_goals(df.home_team, df.away_team)
        probs = outcome_probs(score_matrices(lam, mu, self.rho_ if use_rho else 0.0))
        return pd.DataFrame(
            {"p_home": probs[:, 0], "p_draw": probs[:, 1], "p_away": probs[:, 2],
             "exp_home": lam, "exp_away": mu},
            index=df.index,
        )  # fmt: skip

    def ratings(self) -> pd.DataFrame:
        """Team parameters on the goals scale, for display."""
        return pd.DataFrame(
            {
                "team": self.teams_,
                "attack": self.attack_,
                "defence": self.defence_,
                "goals_for_vs_avg": np.exp(self.mu0_ + self.attack_),
                "goals_against_vs_avg": np.exp(self.mu0_ + self.defence_),
            }
        ).sort_values("attack", ascending=False)


def base_rate_probs(train: pd.DataFrame) -> np.ndarray:
    """Naive baseline: historical home/draw/away frequencies from the training window."""
    hg, ag = train.home_goals, train.away_goals
    freq = np.array([(hg > ag).mean(), (hg == ag).mean(), (hg < ag).mean()])
    return freq / freq.sum()


def odds_to_probs(odds: pd.DataFrame) -> pd.DataFrame:
    """Decimal odds -> probabilities by proportional normalisation.

    1/odds sums to ~1.02-1.08 because of the bookmaker margin; dividing by the sum
    removes it. (Shin's method or power methods handle favourite-longshot bias better.)
    """
    inv = 1 / odds[["odds_home", "odds_draw", "odds_away"]].astype(float)
    probs = inv.div(inv.sum(axis=1), axis=0)
    probs.columns = ["p_home", "p_draw", "p_away"]
    return probs
