import sys

import pandas as pd

from futbol.config import PROJECT_ROOT
from futbol.ingest.api_fixtures import parse_lineups
from futbol.ingest.api_matchday import parse_injury
from futbol.lineups import backtest_probable_xi, last_xi, probable_xi

sys.path.insert(0, str(PROJECT_ROOT / "app"))
from pitch import place  # noqa: E402

GRID_442 = ["1:1", "2:4", "2:3", "2:2", "2:1", "3:4", "3:3", "3:2", "3:1", "4:2", "4:1"]
POS_442 = ["G"] + ["D"] * 4 + ["M"] * 4 + ["F"] * 2


def match(fixture_id: int, days_ago: int, xi: list[int], bench: list[int] = (), formation="4-4-2"):
    """Lineup rows of one match: player k of `xi` plays slot k of a 4-4-2."""
    when = pd.Timestamp("2026-11-01") - pd.Timedelta(days=days_ago)
    rows = [
        {"fixture_id": fixture_id, "team_id": 1, "kickoff_utc": when, "formation": formation,
         "player_id": pid, "player_name": f"P{pid}", "number": pid, "pos": pos, "grid": grid,
         "started": True}
        for pid, grid, pos in zip(xi, GRID_442, POS_442, strict=True)
    ] + [
        {"fixture_id": fixture_id, "team_id": 1, "kickoff_utc": when, "formation": formation,
         "player_id": pid, "player_name": f"P{pid}", "number": pid, "pos": "F", "grid": None,
         "started": False}
        for pid in bench
    ]  # fmt: skip
    return pd.DataFrame(rows)


REGULARS = list(range(1, 12))


def test_parse_lineups_keeps_grid_for_starters_only_and_unescapes_names():
    item = {
        "fixture": {"id": 7},
        "lineups": [
            {
                "team": {"id": 1},
                "formation": "4-3-3",
                "coach": {"name": "Coach"},
                "startXI": [
                    {
                        "player": {
                            "id": 10,
                            "name": "O&apos;Brien",
                            "number": 5,
                            "pos": "D",
                            "grid": "2:1",
                        }
                    }
                ],  # fmt: skip
                "substitutes": [
                    {"player": {"id": 11, "name": "Sub", "number": 20, "pos": "F", "grid": None}}
                ],  # fmt: skip
            }
        ],
    }
    teams, players = parse_lineups(item)
    assert teams == [{"fixture_id": 7, "team_id": 1, "formation": "4-3-3", "coach": "Coach"}]
    assert players[0]["player_name"] == "O'Brien" and players[0]["grid"] == "2:1"
    assert players[1]["started"] is False and players[1]["grid"] is None


def test_parse_injury():
    row = parse_injury(
        {"player": {"id": 5, "name": "X", "type": "Questionable", "reason": "Knock"},
         "team": {"id": 2}, "fixture": {"id": 9, "date": "2026-11-06T20:00:00+00:00"}},
        "E0",
    )  # fmt: skip
    assert row["kind"] == "Questionable" and row["fixture_id"] == 9 and row["team_id"] == 2
    assert parse_injury({"player": {}, "team": {"id": 2}, "fixture": {"id": 9}}, "E0") is None


def test_probable_xi_repeats_a_stable_team():
    history = pd.concat(match(f, 7 * f, REGULARS) for f in range(1, 6))
    guess = probable_xi(history)
    assert guess.formation == "4-4-2"
    assert set(guess.players.player_id) == set(REGULARS)
    assert (guess.players.confidence == 1).all()
    # Same player keeps the same slot.
    assert dict(zip(guess.players.grid, guess.players.player_id, strict=True))["1:1"] == 1


def test_injured_regular_is_replaced_by_a_player_of_the_same_line():
    # 11 (a forward) is out; 12 is a forward who usually comes off the bench.
    history = pd.concat(match(f, 7 * f, REGULARS, bench=[12, 13]) for f in range(1, 6))
    guess = probable_xi(history, unavailable={11})
    ids = set(guess.players.player_id)
    assert 11 not in ids and len(ids) == 11
    replacement = guess.players[guess.players.grid == GRID_442[10]].iloc[0]
    assert replacement.pos == "F" and replacement.player_id in (12, 13)


def test_doubtful_players_are_flagged_not_dropped():
    history = pd.concat(match(f, 7 * f, REGULARS) for f in range(1, 4))
    guess = probable_xi(history, doubtful={9})
    assert guess.players.set_index("player_id").at[9, "doubtful"]


def test_backtest_counts_correct_starters():
    # Six identical matches, then one with two changes: the guess keeps 9 of the 11.
    games = [match(f, 70 - 7 * f, REGULARS) for f in range(1, 7)]
    games.append(match(7, 0, [*REGULARS[:9], 20, 21]))
    lineups = pd.concat(games)
    bt = backtest_probable_xi(lineups, since="2020-01-01")
    assert len(bt) == 1 and bt.model.iloc[0] == 9 and bt.last_xi.iloc[0] == 9
    assert last_xi(pd.concat(games[:6])) == set(REGULARS)


def test_pitch_orientation():
    """Home attacks up with its left flank on the left; away mirrors both."""
    p = pd.DataFrame({
        "grid": ["1:1", "2:1", "2:4", "1:1", "2:1", "2:4"], "formation": "4-4-2",
        "side": ["home", "home", "home", "away", "away", "away"],
    })  # fmt: skip
    placed = place(p)
    home_gk, home_lb, home_rb, away_gk, away_lb, away_rb = placed.itertuples()
    assert home_gk.y < home_lb.y < 52.5 < away_lb.y < away_gk.y
    assert home_lb.x < home_rb.x  # col 1 = left flank, on the left when attacking up
    assert away_lb.x > away_rb.x  # attacking down, the away left flank is on the screen's right
