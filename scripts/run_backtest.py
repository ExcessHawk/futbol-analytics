"""Fit the models walk-forward and store their predictions. See futbol/experiments.py.

Usage:
  python scripts/run_backtest.py                       # full: tuning + every experiment, all leagues
  python scripts/run_backtest.py --skip-tuning         # full, reusing the stored params
  python scripts/run_backtest.py --leagues MEX ARG BRA # only these leagues (others untouched)
  python scripts/run_backtest.py --current-only        # seasons in progress + fixtures (~1 min)
  python scripts/run_backtest.py --markets-only        # goals/corners/cards/shots markets
"""

import argparse

from futbol.db import get_engine, init_schema
from futbol.experiments import (
    load_market_predictions,
    load_params,
    markets_backtest,
    multi_season,
    report,
    save,
    save_markets,
    tune,
    upcoming,
    update_current_season,
    xg_vs_goals,
)
from futbol.leagues import ALL_LEAGUES, BIG5
from futbol.logging_setup import setup_logging
from futbol.markets import market_score_table


def run_markets(params: dict, leagues: list[str]) -> None:
    save_markets(markets_backtest(leagues, params), scope=[(lg, None) for lg in leagues])
    preds = load_market_predictions()
    table = market_score_table(preds[preds.league_code.isin(leagues)])
    print(f"== count markets ({', '.join(leagues)}; common matches)")
    print(table.round(4).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tuning", action="store_true", help="reuse the last tuned params")
    parser.add_argument(
        "--leagues", nargs="+", default=ALL_LEAGUES, help="league codes (futbol.leagues)"
    )
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="only the season in progress (implies --skip-tuning)",
    )
    parser.add_argument(
        "--markets-only", action="store_true", help="only the count markets (stored params)"
    )
    args = parser.parse_args()
    setup_logging()
    init_schema()
    leagues = args.leagues

    if args.current_only:
        update_current_season(load_params(), leagues)
        return
    if args.markets_only:
        run_markets(load_params(), leagues)
        return

    # Tuning always uses the Big-5 tuning seasons, so every league shares one set of params.
    params = load_params() if args.skip_tuning else tune(BIG5)

    ms = multi_season(leagues, params)
    save(ms, "multi_season", scope=[(lg, None) for lg in leagues])
    report(ms, f"== multi_season ({', '.join(leagues)}; common matches)")
    for league, part in ms.groupby("league_code"):
        report(part, f"== multi_season {league}")

    if set(BIG5) & set(leagues):  # StatsBomb xG only exists for European 2015/16
        xvg = xg_vs_goals(params, ms)
        save(xvg, "xg_vs_goals")
        if not xvg.empty:
            report(xvg, "== xg_vs_goals (Big-5 2015/16, in-season)")

    run_markets(params, leagues)

    up, ratings = upcoming(ALL_LEAGUES, params)
    save(up, "upcoming")
    ratings.to_sql("team_ratings", get_engine(), if_exists="replace", index=False)


if __name__ == "__main__":
    main()
