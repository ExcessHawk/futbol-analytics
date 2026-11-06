"""Load StatsBomb open data competition-seasons into the database.

Usage:
  python scripts/ingest_statsbomb.py                       # complete 2015/16 leagues (default)
  python scripts/ingest_statsbomb.py --only 2:27           # Premier League 2015/16 only
  python scripts/ingest_statsbomb.py --only 2:27 --limit 5 # quick smoke test
"""

import argparse

from futbol.db import init_schema
from futbol.ingest.statsbomb import ingest_season
from futbol.logging_setup import setup_logging

# competition_id:season_id. The only complete league seasons in the open data:
# Premier League, La Liga, Serie A, Ligue 1 2015/16. (Bundesliga 2015/16, 9:27, looks
# like it belongs here but holds only 34 matches - a single team's season.)
COMPLETE_2015_16 = ["2:27", "11:27", "12:27", "7:27"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--only", nargs="+", default=COMPLETE_2015_16, help="competition:season pairs"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="max matches per season (testing)")
    parser.add_argument("--force", action="store_true", help="reload matches already loaded")
    args = parser.parse_args()

    setup_logging()
    init_schema()
    for pair in args.only:
        comp, season = (int(v) for v in pair.split(":"))
        ingest_season(comp, season, workers=args.workers, force=args.force, limit=args.limit)


if __name__ == "__main__":
    main()
