"""Load football-data.co.uk results, odds and upcoming fixtures.

Usage:
  python scripts/ingest_football_data.py                    # Big-5, 2012/13 -> current
  python scripts/ingest_football_data.py --leagues E0 --from 2020
"""

import argparse

from futbol.db import init_schema
from futbol.ingest.football_data import (
    BIG5,
    current_season_start,
    ingest_fixtures,
    ingest_league_seasons,
)
from futbol.logging_setup import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--leagues", nargs="+", default=BIG5)
    parser.add_argument(
        "--from", dest="first", type=int, default=2012, help="first season start year"
    )
    parser.add_argument("--to", dest="last", type=int, default=current_season_start())
    args = parser.parse_args()

    setup_logging()
    init_schema()
    ingest_league_seasons(args.leagues, args.first, args.last)
    ingest_fixtures(args.leagues)


if __name__ == "__main__":
    main()
