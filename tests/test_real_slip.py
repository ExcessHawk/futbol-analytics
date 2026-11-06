"""Regression test on a real bookmaker slip (OCR text only, from three dark-mode screenshots).

Layout traits this covers: one team per line with crest debris before the name, the pick
as "GANA PA 90' <team> 1.89x" above the teams, a kick-off line opening every block, a
player prop with abbreviations ("VAL vs RMA", "K. Mbappé RMA | 3.0+"), and a header with
stake, boost and boosted combined odds. Two OCR variants disagree on two odds; voting
must recover both.
"""

import pandas as pd
import pytest

from futbol.betslip import TeamIndex, parse_meta, parse_slip_variants

GRAY = [
    "$ 103,348.00 Ganancia pote o", "NN", "Entrada $50.00 Boost +80%: 2,066.96x444",
    "Hoy (14:00)", "GANA 'PA (90' Sevilla 1.89x", "W Sevilla", "Alaves",
    "Manana (11:30)", "GANA 'PA 90' Manchester Ci... 1.68x", "Nottingham Forest", "(6) Manchester City",
    "Mañana (14:00) - VAL vs RMA", "TIROS AL ARCO Bt 90'", "3 ; 2.09x", "awa K. Mbappé RMA | 3.0+",
    "08/nov (07:00)", "GANA 'PA '90' Real Betis 2.5x", "Elche", "~@ Real Betis",
    "08/nov (08:00)", "GANA 'PA (90' Chelsea 1.95x", "i Sunderland", "ÚÓ Chelsea",
    "08/nov (08:00)", "GANA 'PA '90' Brighton 1.91x", "6 Brighton", "Brentford",
    "08/nov (08:00)", "GANA 'PA '90' Liverpool 1.56x", "RS Crystal Palace", "© Liverpool",
    "08/nov (08:00)", "GANA (PA '90' Bournemouth 1.92x", "lo] Ipswich Town", "€ Bournemouth",
    "08/nov (09:15)", "GANA 'PA '90' Real Sociedad 1.67x", "ES Real Sociedad", "@ Rayo Vallecano",
    "08/nov (10:30)", "GANA 'PA '90' Aston Villa 3.10x", "© Manchester United", "B Aston Villa",
    "08/nov (11:30)", "GANA (PA) (90') Villarreal 1.42x", "Villarreal", "Getafe",
]  # fmt: skip
CLAHE = [
    "A $ 103,348.00 Ganancia pote: oo", "Entrada $ 50.00 Boost +80%: 2,066.96x-1,4€",
    "Hoy (14:00)", "GANA PA 90' Sevilla 1.89x", "y Sevilla", "Alavés",
    "Mañana (11:30)", "GANA PA 90' Manchester Ci... 1.68x", "a Nottingham Forest", "@ Manchester City",
    "Mañana (14:00) *- VAL vs RMA", "TIROS ALARCO B+ 90'", "; 2.09x", "MA K. Mbappé RMA 3.0+",
    "08/nov (07:00)", "GANA PA 90' Real Betis 2.11x", "@ Elche", "Y Real Betis",
    "08/nov (08:00)", "GANA PA 90' Chelsea 1.95x", "$ Sunderland", "ep Chelsea",
    "08/nov (08:00)", "GANA PA 90' Brighton 1.91x", "0 Brighton", "@ Brentford",
    "08/nov (08:00)", "GANA PA 90’ Liverpool 1.56x", "X Crystal Palace", "$ Liverpool",
    "08/nov (08:00)", "GANA PA 90' Bournemouth 1.92x", "y Ipswich Town", "$ Bournemouth",
    "08/nov (09:15)", "GANA PA 90' Real Sociedad 167x", "B Real Sociedad", "Y Rayo Vallecano",
    "08/nov (10:30)", "GANA PA 90' Aston Villa 3.10x", "O Manchester United", "| Aston Villa",
    "08/nov (11:30)", "GANA PA 90' Villarreal 1.42x", "Villarreal", "Getafe",
]  # fmt: skip

TEAMS = pd.DataFrame(
    {
        "league_code": ["SP1"] * 11 + ["E0"] * 13,
        "team": ["Sevilla", "Alaves", "Valencia", "Real Madrid", "Elche", "Betis", "Sociedad", "Vallecano",
                 "Villarreal", "Getafe", "Ath Madrid",
                 "Nott'm Forest", "Man City", "Sunderland", "Chelsea", "Brighton", "Brentford", "Crystal Palace",
                 "Liverpool", "Ipswich", "Bournemouth", "Man United", "Aston Villa", "Arsenal"],
    }
)  # fmt: skip
NO_ALIASES = pd.DataFrame(columns=["league_code", "alias", "team"])
# Both readings of "VAL vs RMA" are real fixtures; the slip means the sooner one.
FIXTURES = pd.DataFrame(
    {
        "league_code": ["SP1", "SP1"],
        "home_team": ["Vallecano", "Valencia"],
        "away_team": ["Real Madrid", "Real Madrid"],
        "kickoff_utc": pd.to_datetime(["2027-03-01", "2026-11-07"], utc=True),
    }
)

EXPECTED = [
    ("Sevilla", "Alaves", "1x2", "home", 1.89),
    ("Nott'm Forest", "Man City", "1x2", "away", 1.68),
    ("Valencia", "Real Madrid", "player_sot", "over", 2.09),
    ("Elche", "Betis", "1x2", "away", 2.11),
    ("Sunderland", "Chelsea", "1x2", "away", 1.95),
    ("Brighton", "Brentford", "1x2", "home", 1.91),
    ("Crystal Palace", "Liverpool", "1x2", "away", 1.56),
    ("Ipswich", "Bournemouth", "1x2", "away", 1.92),
    ("Sociedad", "Vallecano", "1x2", "home", 1.67),
    ("Man United", "Aston Villa", "1x2", "away", 3.10),
    ("Villarreal", "Getafe", "1x2", "home", 1.42),
]


@pytest.fixture(scope="module")
def picks():
    idx = TeamIndex.build(TEAMS, NO_ALIASES)
    return parse_slip_variants([GRAY, CLAHE], idx, FIXTURES)


def test_every_pick_is_read(picks):
    got = list(
        picks[["home_team", "away_team", "market", "selection", "odds"]].itertuples(
            index=False, name=None
        )
    )
    assert got == EXPECTED


def test_player_prop_details(picks):
    mbappe = picks.iloc[2]
    assert mbappe.player == "K. Mbappé"
    assert mbappe.player_team == "Real Madrid"  # from "RMA" next to the name
    assert mbappe.line == 2.5  # "3.0+" = three or more


def test_header_and_odds_check(picks):
    meta = parse_meta(CLAHE)
    assert (meta.stake, meta.boost_pct, meta.printed_total) == (50.0, 80.0, 2066.96)
    # The eleven odds multiply to the slip's pre-boost total: nothing misread, nothing missing.
    assert picks.odds.prod() == pytest.approx(meta.base_total, rel=1e-3)
