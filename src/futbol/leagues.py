"""Registry of supported leagues: where their data comes from and how their seasons work.

Everything league-specific lives here, so adding a league is one entry, not a hunt
through the code for "Big-5" assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class League:
    code: str  # football-data code, also our league_code
    name: str
    country: str
    source: str  # "main": mmz4281/{yyzz}/{code}.csv (rich stats) | "extra": new/{code}.csv
    calendar_season: bool = False  # season = calendar year ("2026"), not "2026/2027"
    has_stats: bool = True  # shots / corners / cards per match available
    schedule_slug: str | None = None  # fixturedownload.com feed, if any
    tournaments: bool = False  # Apertura / Clausura with play-offs (Liga MX)
    api_football_id: int | None = None  # league id in API-Football (api-sports.io)

    def first_test_date(self) -> date:
        """First date scored in the multi-season backtest (earlier data only trains)."""
        return date(2015, 1, 1) if self.calendar_season else date(2015, 7, 1)


LEAGUES: dict[str, League] = {
    lg.code: lg
    for lg in [
        League(
            "E0", "Premier League", "Inglaterra", "main", schedule_slug="epl", api_football_id=39
        ),
        League("SP1", "LaLiga", "España", "main", schedule_slug="la-liga", api_football_id=140),
        League(
            "D1", "Bundesliga", "Alemania", "main", schedule_slug="bundesliga", api_football_id=78
        ),
        League("I1", "Serie A", "Italia", "main", schedule_slug="serie-a", api_football_id=135),
        League("F1", "Ligue 1", "Francia", "main", schedule_slug="ligue-1", api_football_id=61),
        League(
            "MEX",
            "Liga MX",
            "México",
            "extra",
            has_stats=False,
            tournaments=True,
            api_football_id=262,
        ),
        League(
            "ARG",
            "Liga Profesional",
            "Argentina",
            "extra",
            calendar_season=True,
            has_stats=False,
            api_football_id=128,
        ),
        League(
            "BRA",
            "Brasileirão",
            "Brasil",
            "extra",
            calendar_season=True,
            has_stats=False,
            api_football_id=71,
        ),
    ]
}

BIG5 = ["E0", "SP1", "D1", "I1", "F1"]
EXTRA = ["MEX", "ARG", "BRA"]
ALL_LEAGUES = BIG5 + EXTRA
LEAGUE_NAMES = {code: lg.name for code, lg in LEAGUES.items()}


def tournament_of(match_date: date) -> str:
    """Liga MX: Apertura runs July-December, Clausura January-June."""
    return f"Apertura {match_date.year}" if match_date.month >= 7 else f"Clausura {match_date.year}"
