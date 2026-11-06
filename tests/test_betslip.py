"""Bet-slip analyser: odds parsing, layout-agnostic slip parsing, OCR end to end, evaluation."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from futbol.betslip import (
    TeamIndex,
    decode_image,
    evaluate,
    ocr_lines,
    odds_consistency,
    parse_odds,
    parse_slip,
    parse_slip_variants,
    parse_total_odds,
    preprocess_variants,
    summarize,
)
from futbol.config import get_settings
from futbol.markets import MatchForecaster
from futbol.sample_ticket import as_phone_photo, render_ticket, sample_total_odds, to_png_bytes

TEAMS = pd.DataFrame(
    {
        "league_code": ["E0"] * 6 + ["SP1"] * 2,
        "team": [
            "Arsenal",
            "Hull",
            "Man City",
            "Nott'm Forest",
            "Leeds",
            "Tottenham",
            "Ath Madrid",
            "Betis",
        ],
    }
)
ALIASES = pd.DataFrame(
    {"league_code": ["E0"], "alias": ["Man Utd"], "team": ["Man United"]}
)  # alias of a team not in TEAMS: must be ignored, not crash
IDX = TeamIndex.build(TEAMS, ALIASES)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1.85", 1.85),
        ("1,85", 1.85),
        ("+150", 2.5),
        ("-200", 1.5),
        ("5/2", 3.5),
        ("$100.00", None),
        ("2.5", None),
    ],
)
def test_odds_formats(text, expected):
    assert parse_odds(text) == expected


def test_team_names_fuzzy_and_aliases():
    assert IDX.match("Manchester City")[0] == "Man City"
    assert IDX.match("Nottingham Forest")[0] == "Nott'm Forest"
    assert IDX.match("Atlético de Madrid")[0] == "Ath Madrid"
    assert IDX.match("Arsenai")[0] == "Arsenal"  # OCR typo
    assert IDX.match("Manchester United") is None  # not a team in this index


DETAILS_BELOW = [
    "Mi boleto",
    "Arsenal vs Hull City", "Resultado final", "Arsenal 1.40",
    "Nottingham Forest - Manchester City", "Ambos equipos anotan", "Sí 1.75",
    "Leeds United vs Tottenham Hotspur", "Total de córners", "Más de 9.5 1.90",
    "Cuota total 4.65", "Apuesta $100.00",
]  # fmt: skip
DETAILS_ABOVE = [
    "Mi boleto",
    "Arsenal 1.40", "Resultado final", "Arsenal vs Hull City",
    "Sí 1.75", "Ambos equipos anotan", "Nottingham Forest - Manchester City",
    "Más de 9.5 1.90", "Total de córners", "Leeds United vs Tottenham Hotspur",
    "Cuota total 4.65", "Apuesta $100.00",
]  # fmt: skip


@pytest.mark.parametrize("lines", [DETAILS_BELOW, DETAILS_ABOVE], ids=["below", "above"])
def test_parser_adapts_to_layout(lines):
    picks = parse_slip(lines, IDX)
    assert picks[["home_team", "away_team"]].values.tolist() == [
        ["Arsenal", "Hull"], ["Nott'm Forest", "Man City"], ["Leeds", "Tottenham"]
    ]  # fmt: skip
    assert picks[["market", "selection"]].values.tolist() == [
        ["1x2", "home"],
        ["btts", "yes"],
        ["corners", "over"],
    ]
    assert picks.line.tolist()[2] == 9.5
    assert picks.odds.tolist() == [1.40, 1.75, 1.90]  # the footer's 4.65 / $100 never leak in


def test_total_odds_check_catches_a_misread_digit():
    assert parse_total_odds(["Cuota total 1 8.08"]) == 18.08  # OCR split the number
    good = pd.DataFrame({"odds": [1.40, 1.85, 1.75, 2.10, 1.90]})
    bad = pd.DataFrame({"odds": [1.40, 1.85, 1.75, 2.10, 4.90]})
    assert odds_consistency(good, 18.08) is None
    assert "revísalas" in odds_consistency(bad, 18.08)


tesseract_missing = (
    shutil.which(get_settings().tesseract_cmd) is None
    and not Path(get_settings().tesseract_cmd).exists()
)


@pytest.mark.skipif(tesseract_missing, reason="Tesseract not installed")
@pytest.mark.parametrize("variant", ["clean", "dark", "photo"])
def test_ocr_end_to_end_on_sample_slip(variant):
    img = render_ticket(dark=variant == "dark")
    if variant == "photo":
        img = as_phone_photo(img)
    # The app's pipeline: every preprocessing variant is read, then picks and odds are voted.
    variants = preprocess_variants(decode_image(to_png_bytes(img)))
    variant_lines = [[ln.text for ln in ocr_lines(im)] for im in variants.values()]
    lines = variant_lines[1]
    teams = pd.DataFrame(
        {"league_code": "E0", "team": ["Arsenal", "Hull", "Fulham", "Newcastle", "Nott'm Forest", "Man City",
                                       "Sunderland", "Chelsea", "Leeds", "Tottenham", "Valencia", "Real Madrid"]}
    )  # fmt: skip
    picks = parse_slip_variants(
        variant_lines,
        TeamIndex.build(teams, pd.DataFrame(columns=["league_code", "alias", "team"])),
    )
    assert len(picks) == 6
    assert picks.market.tolist() == ["1x2", "goals", "btts", "1x2", "corners", "player_sot"]
    assert picks.selection.tolist() == ["home", "over", "yes", "away", "over", "over"]
    assert picks.player.iloc[5] == "Kylian Mbappé" and picks.line.iloc[5] == 2.5
    assert picks.odds.notna().all()
    if variant != "photo":  # a phone photo may misread a digit: that is what the total check is for
        assert picks.odds.tolist() == [1.40, 1.85, 1.75, 2.10, 1.90, 2.40]
        assert parse_total_odds(lines) == pytest.approx(sample_total_odds(), abs=0.01)


def test_evaluation_and_parlay_summary():
    rng = np.random.default_rng(0)
    teams = ["Arsenal", "Hull", "Leeds", "Tottenham", "Fulham", "Chelsea"]
    rows = [
        {"match_date": pd.Timestamp("2025-08-01") + pd.Timedelta(days=d), "home_team": h, "away_team": a,
         "home_goals": rng.poisson(1.5), "away_goals": rng.poisson(1.1), "home_corners": rng.poisson(5.5),
         "away_corners": rng.poisson(4.5), "home_yellow": rng.poisson(2), "away_yellow": rng.poisson(2),
         "home_shots": rng.poisson(13), "away_shots": rng.poisson(11), "referee": None}
        for d, (h, a) in enumerate((h, a) for _ in range(4) for h in teams for a in teams if h != a)
    ]  # fmt: skip
    df = pd.DataFrame(rows)
    fc = MatchForecaster({"xi": 0.001, "ridge": 5.0}).fit(
        df, df.match_date.max() + pd.Timedelta(days=1)
    )
    picks = pd.DataFrame(
        [
            {"home_team": "Arsenal", "away_team": "Hull", "league_code": "E0", "market": "1x2", "selection": "home", "line": None, "odds": 1.8},
            {"home_team": "Leeds", "away_team": "Tottenham", "league_code": "E0", "market": "goals", "selection": "over", "line": 2.5, "odds": 1.9},
            {"home_team": "Leeds", "away_team": "Tottenham", "league_code": "E0", "market": "corners", "selection": "under", "line": 9.5, "odds": 1.9},
            {"home_team": "Arsenal", "away_team": "Hull", "league_code": "E0", "market": "1x2", "selection": "home", "line": None, "odds": None},
        ]
    )  # fmt: skip
    out = evaluate(picks, lambda league: fc)
    assert out.p_model.iloc[:3].between(0, 1).all()
    assert out.problem.iloc[3] == "falta la cuota"
    assert out.ev.iloc[0] == pytest.approx(out.p_model.iloc[0] * 1.8 - 1)
    s = summarize(out)
    assert s.n_picks == 3
    assert s.p_win == pytest.approx(out.p_model.iloc[:3].prod())
    assert s.same_match_picks  # two picks on Leeds vs Tottenham


def test_suspect_odds_points_at_the_misread_pick():
    from futbol.betslip import suspect_odds

    bad = pd.DataFrame({"odds": [1.40, 1.85, 1.75, 2.10, 4.90]})
    assert suspect_odds(bad, 18.08) == (4, 1.90)
    assert (
        suspect_odds(pd.DataFrame({"odds": [1.40, 1.85]}), 99.0) is None
    )  # no one-digit fix explains it


@pytest.mark.parametrize(
    ("text", "line"),
    [("Tiros a puerta: 3 o más", 2.5), ("Tiros a puerta: 3 0 más", 2.5), ("Tiros a puerta: 30 más", 2.5),
     ("Tiros a puerta: 10 o más", 9.5), ("3+ tiros a puerta", 2.5), ("Más de 1.5 tiros a puerta", 1.5)],
)  # fmt: skip
def test_player_prop_thresholds_survive_ocr_noise(text, line):
    from futbol.betslip import parse_player_prop

    market, value, name = parse_player_prop(["Kylian Mbappé 2.40", text], IDX)
    assert (market, value, name) == ("player_sot", line, "Kylian Mbappé")


@pytest.mark.parametrize(
    "lines",
    [
        ["Sí 1.75", "Ambos equipos anotan"],
        ["Total de tiros a puerta", "Más de 8.5"],
        ["sáb 7 nov · 14:00"],
    ],
)
def test_not_player_props(lines):
    from futbol.betslip import parse_player_prop

    assert parse_player_prop(lines, IDX) is None


def test_player_prop_evaluation_needs_team_and_average():
    rng = np.random.default_rng(1)
    teams = ["Leeds", "Tottenham", "Fulham", "Chelsea"]
    rows = [
        {"match_date": pd.Timestamp("2025-08-01") + pd.Timedelta(days=d), "home_team": h, "away_team": a,
         "home_goals": rng.poisson(1.4), "away_goals": rng.poisson(1.1),
         "home_shots": rng.poisson(14 if a == "Fulham" else 11), "away_shots": rng.poisson(10)}
        for d, (h, a) in enumerate((h, a) for _ in range(6) for h in teams for a in teams if h != a)
    ]  # fmt: skip
    df = pd.DataFrame(rows)
    fc = MatchForecaster({"xi": 0.001, "ridge": 5.0}).fit(
        df, df.match_date.max() + pd.Timedelta(days=1)
    )
    base = {"bet_type": "Jugador", "home_team": "Leeds", "away_team": "Fulham", "league_code": "E0",
            "market": "player_shots", "selection": "over", "line": 2.5, "odds": 2.2, "player": "X"}  # fmt: skip
    picks = pd.DataFrame(
        [
            base | {"player_team": "Leeds", "player_avg": 2.0},
            base | {"player_team": "Leeds", "player_avg": None},
            base | {"player_team": "Chelsea", "player_avg": 2.0},  # not in this match
        ]
    )
    out = evaluate(picks, lambda league: fc)
    assert 0 < out.p_model.iloc[0] < 1
    assert out.opponent_factor.iloc[0] > 1  # Fulham concede more shots than average
    assert "promedio" in out.problem.iloc[1]
    assert "equipo del jugador" in out.problem.iloc[2]
