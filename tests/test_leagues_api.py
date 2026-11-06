"""Extra leagues (Liga MX / Argentina / Brasil) and the API-Football pipeline."""

import pandas as pd
import pytest

from futbol.ingest.api_fixtures import parse_player_stats, parse_team_stats
from futbol.ingest.api_football import ApiFootballError, RateLimiter
from futbol.ingest.football_data import season_start_year, transform_extra
from futbol.leagues import LEAGUES, tournament_of
from futbol.season import regular_phase


def test_extra_csv_keeps_repeated_pairings():
    """Pumas-Cruz Azul twice in the same season label (Clausura + liguilla) are two matches."""
    raw = pd.DataFrame(
        {"Season": ["2025/2026"] * 2, "Date": ["15/03/2026", "25/05/2026"], "Home": ["UNAM Pumas"] * 2,
         "Away": ["Cruz Azul"] * 2, "HG": [2, 1], "AG": [2, 2], "PSCH": [2.4, 2.6], "PSCD": [3.3, 3.2],
         "PSCA": [3.0, 2.8]}
    )  # fmt: skip
    out = transform_extra(raw, "MEX")
    assert len(out) == 2
    assert out.odds_source.tolist() == ["pinnacle_closing"] * 2


def test_season_formats():
    assert season_start_year("2026/2027") == 2026 and season_start_year("2026") == 2026
    assert LEAGUES["BRA"].calendar_season and not LEAGUES["MEX"].calendar_season
    assert tournament_of(pd.Timestamp("2026-08-10").date()) == "Apertura 2026"
    assert tournament_of(pd.Timestamp("2027-02-10").date()) == "Clausura 2027"


def test_regular_phase_excludes_playoffs():
    teams = ["A", "B", "C", "D"]  # 3 rounds of a single round robin, then a play-off
    rows = [("2026-07-01", "A", "B"), ("2026-07-01", "C", "D"), ("2026-07-08", "A", "C"),
            ("2026-07-08", "B", "D"), ("2026-07-15", "A", "D"), ("2026-07-15", "B", "C"),
            ("2026-07-22", "A", "B")]  # fmt: skip
    m = pd.DataFrame(rows, columns=["match_date", "home_team", "away_team"]).assign(
        home_goals=1, away_goals=0
    )
    mask = regular_phase(m)
    assert mask.tolist() == [True] * 6 + [False]
    assert set(teams) == set(m.home_team) | set(m.away_team)


ITEM = {
    "fixture": {"id": 7, "date": "2026-11-01T20:00:00+00:00", "status": {"short": "FT"}, "referee": "X"},
    "league": {"season": 2026, "round": "Regular Season - 11"},
    "teams": {"home": {"id": 1, "name": "H"}, "away": {"id": 2, "name": "A"}},
    "goals": {"home": 2, "away": 1},
    "statistics": [
        {"team": {"id": 1}, "statistics": [{"type": "Corner Kicks", "value": 5},
                                            {"type": "Ball Possession", "value": "38%"},
                                            {"type": "expected_goals", "value": "3.12"}]},
    ],
    "lineups": [{"startXI": [{"player": {"id": 10}}]}],
    "players": [{"team": {"id": 1}, "players": [
        {"player": {"id": 10, "name": "Starter"}, "statistics": [{"games": {"minutes": 90, "substitute": False},
                                                                   "shots": {"total": 4, "on": 3}}]},
        {"player": {"id": 11, "name": "Sub"}, "statistics": [{"games": {"minutes": 12, "substitute": True},
                                                               "shots": {"total": None, "on": None}}]},
    ]}],
}  # fmt: skip


def test_api_parsers():
    team = parse_team_stats(ITEM)[0]
    assert team["corners"] == 5 and team["possession"] == 38.0 and team["xg"] == pytest.approx(3.12)
    assert team["is_home"] is True
    players = {p["player_id"]: p for p in parse_player_stats(ITEM)}
    assert players[10]["shots_on"] == 3 and players[10]["started"] is True
    assert players[11]["started"] is False and players[11]["minutes"] == 12


def test_rate_limiter_reads_headers_and_keeps_a_reserve():
    limiter = RateLimiter(daily_reserve=50)
    limiter.update({"X-RateLimit-Limit": "300", "x-ratelimit-requests-remaining": "51"})
    assert limiter.per_minute == 300
    limiter.wait()  # 51 left: still allowed
    limiter.update({"x-ratelimit-requests-remaining": "50"})
    with pytest.raises(ApiFootballError):
        limiter.wait()


def test_ambiguous_names_resolved_by_league():
    from futbol.betslip import TeamIndex, parse_slip

    teams = pd.DataFrame({"league_code": ["BRA", "BRA", "MEX", "MEX"],
                          "team": ["Santos", "Flamengo RJ", "Santos Laguna", "Necaxa"]})  # fmt: skip
    idx = TeamIndex.build(teams, pd.DataFrame(columns=["league_code", "alias", "team"]))
    lines = ["Hoy (19:00)", "GANA PA 90' Necaxa 2.10x", "Santos", "Necaxa",
             "Mañana (17:00)", "GANA PA 90' Flamengo 1.70x", "Santos", "Flamengo"]  # fmt: skip
    picks = parse_slip(lines, idx)
    assert picks[["home_team", "league_code"]].values.tolist() == [
        ["Santos Laguna", "MEX"],
        ["Santos", "BRA"],
    ]


def test_team_and_player_tables():
    from futbol.api_stats import hit_rates, player_log, player_table, team_match_view, team_table

    fx = pd.DataFrame(
        {"fixture_id": [1, 1, 2, 2], "kickoff_utc": pd.to_datetime(["2026-08-01"] * 2 + ["2026-08-08"] * 2, utc=True),
         "home_id": [10, 10, 20, 20], "home_name": ["A", "A", "B", "B"], "away_id": [20, 20, 10, 10],
         "away_name": ["B", "B", "A", "A"], "home_goals": [2, 2, 0, 0], "away_goals": [1, 1, 0, 0],
         "team_id": [10, 20, 20, 10], "is_home": [True, False, True, False], "xg": [1.8, 0.9, 0.7, 1.1],
         "shots_total": [12, 8, 9, 10], "shots_on": [5, 3, 2, 4], "corners": [6, 2, 4, 5], "yellow": [1, 3, 2, 0],
         "red": 0, "fouls": 10, "possession": [55, 45, 48, 52]}
    )  # fmt: skip
    t = team_table(team_match_view(fx)).set_index("team")
    assert t.loc["A", "points"] == 4 and t.loc["B", "points"] == 1
    assert t.loc["A", "corners_against_pm"] == pytest.approx((2 + 4) / 2)
    assert t.loc["A", "xg_pm"] == pytest.approx((1.8 + 1.1) / 2)

    pl = pd.DataFrame(
        {"fixture_id": [1, 2, 3], "player_id": 9, "team_id": 10, "player_name": "X", "position": "F",
         "minutes": [90, 90, 0], "started": [True, True, False], "rating": [7.0, 8.0, None],
         "shots_total": [4, 2, None], "shots_on": [3, None, None], "goals": [1, None, None], "assists": 0,
         "key_passes": 1, "yellow": 0, "red": 0, "fouls_committed": 1, "fouls_drawn": 1,
         "kickoff_utc": pd.to_datetime(["2026-08-01", "2026-08-08", "2026-08-15"], utc=True),
         "home_id": [10, 20, 10], "home_name": ["A", "B", "A"], "away_name": ["B", "A", "C"]}
    )  # fmt: skip
    row = player_table(pl).iloc[0]
    assert row.matches == 2 and row.shots_on == 3  # unused sub (0 minutes) doesn't count
    log = player_log(pl, 9)
    assert log.shots_on.tolist() == [0, 0, 3]  # nulls from the API are zeros, newest first
    rates = hit_rates(log, "shots_on", [1, 3]).set_index("threshold")
    assert rates.loc[3, "hits"] == 1 and rates.loc[3, "matches"] == 2
