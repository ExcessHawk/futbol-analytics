"""Model experiments: walk-forward backtests, upcoming predictions and ratings (Phases 5-6).

Experiments (all stored in the `predictions` table):

  multi_season   football-data goals, every league in futbol.leagues, scored from 2015 on.
                 Models: base rates, Poisson, Dixon-Coles (trained on goals, 3-year
                 window with time decay) vs the betting market (closing odds).
                 Hyperparameters (xi, ridge) are tuned on 2013/14-2014/15 only, which
                 are never scored: tuning on the test set would be leakage too.

  xg_vs_goals    StatsBomb 2015/16 Big-5, the only seasons with xG. In-season data only
                 (no previous seasons exist), so every model sees the same information:
                 Dixon-Coles on goals vs on xG vs on (xG + goals)/2, plus the market and
                 the multi-season goals model as references.

  upcoming       Dixon-Coles fitted on everything up to today, applied to unplayed fixtures.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
from sqlalchemy import text

from futbol.db import copy_rows, get_engine, read_sql, upsert_rows
from futbol.leagues import ALL_LEAGUES, LEAGUES
from futbol.markets import MARKETS, MatchForecaster
from futbol.models import PoissonModel, base_rate_probs, odds_to_probs
from futbol.validation import PROB_COLS, backtest, score_table

log = logging.getLogger(__name__)

TUNING_SEASONS = ["2013/2014", "2014/2015"]
XI_GRID = [0.001, 0.0019, 0.003]
RIDGE_GRID = [2.0, 5.0, 10.0]
DEFAULT_PARAMS = {"xi": 0.0019, "ridge": 5.0}  # half-life ~1 year (Dixon & Coles 1997)
IN_SEASON_WARMUP_DAYS = 35  # ~5 rounds before in-season models start predicting
PRED_COLUMNS = ["experiment", "model", "league_code", "season", "match_date", "home_team",
                "away_team", "p_home", "p_draw", "p_away", "exp_home", "exp_away"]  # fmt: skip


# --------------------------------------------------------------------------- predictors


def poisson_predictor(params: dict, home_col="home_goals", away_col="away_goals", suffix="goals"):
    """One fit serves two models: plain Poisson (rho=0) and Dixon-Coles."""

    def predict(train, test, block):
        model = PoissonModel(**params).fit(train, block, home_col, away_col)
        return {
            f"poisson_{suffix}": model.predict(test, use_rho=False),
            f"dixon_coles_{suffix}": model.predict(test, use_rho=True),
        }

    return predict


def baseline_predictor(train, test, block):
    probs = base_rate_probs(train)
    base = pd.DataFrame([probs] * len(test), columns=PROB_COLS, index=test.index)
    market = odds_to_probs(test).dropna()  # closing odds of the match itself: the benchmark
    return {"base_rates": base, "market": market}


def combine(*predictors):
    def predict(train, test, block):
        out = {}
        for p in predictors:
            out.update(p(train, test, block))
        return out

    return predict


# --------------------------------------------------------------------------- experiments


Scope = list[tuple[str, str | None]]  # (league, season) to replace; season None = all seasons


def latest_season(d: pd.DataFrame) -> str:
    """Season label of a league's most recent match ('2026/2027' or '2026')."""
    return str(d.loc[pd.to_datetime(d.match_date).idxmax(), "season"])


def test_mask(d: pd.DataFrame, league: str, current_only: bool) -> pd.Series:
    if current_only:
        return d.season == latest_season(d)
    return pd.to_datetime(d.match_date) >= pd.Timestamp(LEAGUES[league].first_test_date())


def load_fd(league: str) -> pd.DataFrame:
    return read_sql(
        "SELECT * FROM fd_matches WHERE league_code = :l AND home_goals IS NOT NULL ORDER BY match_date",
        l=league,
    )


def tune(leagues: list[str]) -> dict:
    """Grid-search xi and ridge on the tuning seasons by Dixon-Coles log loss."""
    data = {lg: load_fd(lg) for lg in leagues}
    results = []
    for xi, ridge in itertools.product(XI_GRID, RIDGE_GRID):
        params = {"xi": xi, "ridge": ridge}
        preds = pd.concat(
            backtest(d, poisson_predictor(params), test_mask=d.season.isin(TUNING_SEASONS))
            for d in data.values()
        )
        ll = score_table(preds[preds.model == "dixon_coles_goals"]).log_loss.iloc[0]
        results.append((xi, ridge, ll))
        log.info("  tuning xi=%.4f ridge=%4.1f -> log loss %.4f", xi, ridge, ll)
    xi, ridge, _ = min(results, key=lambda r: r[2])
    log.info("Chosen: xi=%s ridge=%s", xi, ridge)
    params = {"xi": xi, "ridge": ridge}
    save_params(params)
    return params


def save_params(params: dict) -> None:
    rows = pd.DataFrame({"name": list(params), "value": list(params.values())})
    rows["tuned_at"] = pd.Timestamp.now(tz="UTC")
    with get_engine().begin() as conn:
        upsert_rows(conn, "model_params", rows, key=["name"])


def load_params() -> dict:
    """Last tuned hyperparameters; falls back to the literature default if never tuned."""
    rows = read_sql("SELECT name, value FROM model_params")
    params = dict(zip(rows.name, rows.value, strict=True))
    if not {"xi", "ridge"} <= params.keys():
        log.warning("No tuned parameters stored yet; using xi=0.0019, ridge=5")
        return dict(DEFAULT_PARAMS)
    return {"xi": params["xi"], "ridge": params["ridge"]}


def multi_season(leagues: list[str], params: dict, current_only: bool = False) -> pd.DataFrame:
    """Walk-forward predictions for every test season, or only each league's latest season."""
    out = []
    for league in leagues:
        d = load_fd(league)
        mask = test_mask(d, league, current_only)
        preds = backtest(d, combine(poisson_predictor(params), baseline_predictor), test_mask=mask)
        log.info("multi_season %s: %d predictions", league, len(preds))
        out.append(preds)
    return pd.concat(out).assign(experiment="multi_season")


def xg_vs_goals(params: dict, history: pd.DataFrame) -> pd.DataFrame:
    xg = read_sql(
        """
        SELECT x.*, f.odds_home, f.odds_draw, f.odds_away
        FROM match_xg x
        JOIN fd_matches f USING (league_code, season, home_team, away_team)
        """
    )
    if xg.empty:
        log.warning("match_xg is empty: run ingest_statsbomb.py and build_metrics.py first")
        return pd.DataFrame()
    xg["home_mix"] = (xg.home_xg + xg.home_goals) / 2
    xg["away_mix"] = (xg.away_xg + xg.away_goals) / 2
    predictor = combine(
        poisson_predictor(params, "home_goals", "away_goals", "goals"),
        poisson_predictor(params, "home_xg", "away_xg", "xg"),
        poisson_predictor(params, "home_mix", "away_mix", "mix"),
        lambda train, test, block: {"market": odds_to_probs(test).dropna()},
    )
    out = []
    for (league, season), d in xg.groupby(["league_code", "season"]):
        d = d.reset_index(drop=True)
        start = pd.to_datetime(d.match_date).min() + pd.Timedelta(days=IN_SEASON_WARMUP_DAYS)
        preds = backtest(
            d, predictor, test_mask=pd.to_datetime(d.match_date) >= start,
            window_days=None, same_season_only=True,
        )  # fmt: skip
        preds = preds[~preds.model.str.startswith("poisson")]  # keep the comparison focused
        out.append(preds)
        log.info("xg_vs_goals %s %s: %d predictions", league, season, len(preds))
    preds = pd.concat(out)

    # Reference: the multi-season goals model on the very same matches (it had 3 years of history).
    ref = history[history.model == "dixon_coles_goals"].merge(
        preds[["league_code", "season", "home_team", "away_team"]].drop_duplicates(),
        on=["league_code", "season", "home_team", "away_team"],
    )
    ref = ref.assign(model="dixon_coles_goals_3y_history")
    return pd.concat([preds, ref]).assign(experiment="xg_vs_goals")


def upcoming(leagues: list[str], params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    today = pd.Timestamp.today().normalize()
    preds, ratings = [], []
    for league in leagues:
        all_rows = read_sql("SELECT * FROM fd_matches WHERE league_code = :l", l=league)
        played = all_rows.dropna(subset=["home_goals", "away_goals"])
        played = played[pd.to_datetime(played.match_date) >= today - pd.Timedelta(days=3 * 365)]
        model = PoissonModel(**params).fit(played, today)
        fixtures = all_rows[
            all_rows.home_goals.isna() & (pd.to_datetime(all_rows.match_date) >= today)
        ]
        if not fixtures.empty:
            preds.append(fixtures.join(model.predict(fixtures)).assign(model="dixon_coles_goals"))
        # Only teams in the current season's table are relevant for display.
        current = set(all_rows[all_rows.season == latest_season(all_rows)].home_team)
        r = model.ratings()
        ratings.append(r[r.team.isin(current)].assign(league_code=league, as_of=today.date(),
                                                      home_adv=model.home_adv_, rho=model.rho_))  # fmt: skip
    p = pd.concat(preds).assign(experiment="upcoming") if preds else pd.DataFrame()
    log.info("upcoming: %d fixtures", len(p))
    return p, pd.concat(ratings)


def _delete_scope(conn, table: str, scope: Scope | None, experiment: str | None = None) -> None:
    """Delete what is about to be rewritten: whole table/experiment, or per league (and season)."""
    exp = "experiment = :e AND " if experiment else ""
    if scope is None:
        conn.execute(text(f"DELETE FROM {table} WHERE {exp}TRUE"), {"e": experiment})
        return
    for league, season in scope:
        season_sql = " AND season = :s" if season else ""
        conn.execute(
            text(f"DELETE FROM {table} WHERE {exp}league_code = :l{season_sql}"),
            {"e": experiment, "l": league, "s": season},
        )


def save(preds: pd.DataFrame, experiment: str, scope: Scope | None = None) -> None:
    """Replace an experiment's predictions: all of them, or only the given (league, season)s."""
    with get_engine().begin() as conn:
        _delete_scope(conn, "predictions", scope, experiment)
        if not preds.empty:
            rows = preds.reindex(columns=PRED_COLUMNS)
            rows["match_date"] = pd.to_datetime(rows.match_date).dt.date
            copy_rows(conn, "predictions", rows)


def report(preds: pd.DataFrame, title: str, reference: str = "market") -> None:
    table = score_table(preds, reference=reference)
    log.info("%s\n%s", title, table.round(4).to_string())


# --------------------------------------------------------------------------- count markets

MARKET_COLUMNS = ["model", "market", "league_code", "season", "match_date", "home_team",
                  "away_team", "exp_home", "exp_away", "line", "p_over"]  # fmt: skip


def markets_predictor(params: dict):
    """Per block: Poisson per market vs the league-average baseline (+ market odds for goals)."""

    def predict(train, test, block):
        fc = MatchForecaster(params).fit(train, block)
        out = {}
        for key, m in MARKETS.items():
            if key not in fc.models:
                continue
            tr = train.dropna(subset=[m.home_col, m.away_col])
            if key == "goals":
                # Standard 2.5 line, from the Dixon-Coles score matrix, so it compares to the odds.
                line = 2.5
                g = fc.goals_detail(test)
                out["poisson|goals"] = pd.DataFrame(
                    {"exp_home": g.exp_home_goals, "exp_away": g.exp_away_goals,
                     "line": line, "p_over": g.p_over_2_5}
                )  # fmt: skip
                ou = test[["odds_over25", "odds_under25"]].astype(float)
                inv = 1 / ou
                out["market|goals"] = pd.DataFrame(
                    {"exp_home": np.nan, "exp_away": np.nan, "line": line,
                     "p_over": inv.odds_over25 / inv.sum(axis=1)}
                ).dropna(subset=["p_over"])  # fmt: skip
            else:
                line = fc.lines[key]
                out[f"poisson|{key}"] = fc.counts(key, test, line)
                if key == "yellow" and fc.referee_factor:
                    out["poisson_sin_arbitro|yellow"] = fc.counts(
                        key, test, line, use_referee=False
                    )
            totals = tr[m.home_col] + tr[m.away_col]
            out[f"league_average|{key}"] = pd.DataFrame(
                {"exp_home": tr[m.home_col].mean(), "exp_away": tr[m.away_col].mean(),
                 "line": line, "p_over": (totals > line).mean()},
                index=test.index,
            )  # fmt: skip
        return out

    return predict


def markets_backtest(leagues: list[str], params: dict, current_only: bool = False) -> pd.DataFrame:
    out = []
    for league in leagues:
        d = load_fd(league)
        mask = test_mask(d, league, current_only)
        preds = backtest(d, markets_predictor(params), test_mask=mask)
        if preds.empty:
            continue
        preds[["model", "market"]] = preds.model.str.split("|", expand=True)
        out.append(preds)
        log.info("markets %s: %d predictions", league, len(preds))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=MARKET_COLUMNS)


def save_markets(preds: pd.DataFrame, scope: Scope | None = None) -> None:
    with get_engine().begin() as conn:
        _delete_scope(conn, "market_predictions", scope)
        rows = preds.reindex(columns=MARKET_COLUMNS)
        rows["match_date"] = pd.to_datetime(rows.match_date).dt.date
        copy_rows(conn, "market_predictions", rows)


def load_market_predictions() -> pd.DataFrame:
    """Stored count-market predictions joined with what actually happened."""
    preds = read_sql("SELECT * FROM market_predictions")
    actual = read_sql(
        "SELECT league_code, season, match_date, home_team, away_team, home_goals, away_goals, "
        "home_corners, away_corners, home_yellow, away_yellow, home_shots, away_shots, home_sot, away_sot "
        "FROM fd_matches WHERE home_goals IS NOT NULL"
    )
    df = preds.merge(actual, on=["league_code", "season", "match_date", "home_team", "away_team"])
    df["actual_home"], df["actual_away"] = np.nan, np.nan
    for key, m in MARKETS.items():
        sel = df.market == key
        df.loc[sel, "actual_home"] = df.loc[sel, m.home_col]
        df.loc[sel, "actual_away"] = df.loc[sel, m.away_col]
    return df


def current_scope(leagues: list[str]) -> Scope:
    """(league, latest season) for each league: what a daily update rewrites."""
    return [(lg, latest_season(load_fd(lg))) for lg in leagues]


def update_current_season(params: dict, leagues: list[str] = ALL_LEAGUES) -> None:
    """Daily path (~1 min): recompute only each league's season in progress, fixtures, ratings."""
    scope = current_scope(leagues)
    ms = multi_season(leagues, params, current_only=True)
    save(ms, "multi_season", scope=scope)
    save_markets(markets_backtest(leagues, params, current_only=True), scope=scope)
    up, ratings = upcoming(leagues, params)
    save(up, "upcoming")
    ratings.to_sql("team_ratings", get_engine(), if_exists="replace", index=False)
    log.info(
        "Current seasons %s: %d backtest predictions, %d upcoming", dict(scope), len(ms), len(up)
    )
