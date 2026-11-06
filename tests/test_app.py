"""Smoke tests: every dashboard view runs end-to-end against the real database.

Runs through the real entry point (Inicio.py + st.navigation), switching pages the way a
user does. Skipped automatically when the database isn't reachable (e.g. CI without Postgres).
"""

import pytest
from streamlit.testing.v1 import AppTest

from futbol.config import PROJECT_ROOT
from futbol.db import ping

ENTRY = str(PROJECT_ROOT / "app" / "Inicio.py")

try:
    ping()
    DB_UP = True
except Exception:
    DB_UP = False

pytestmark = pytest.mark.skipif(not DB_UP, reason="database not reachable")

VIEWS = [
    "views/inicio.py",
    "views/temporada.py",
    "views/predicciones.py",
    "views/equipos.py",
    "views/jugadores.py",
]


def open_view(view: str) -> AppTest:
    at = AppTest.from_file(ENTRY, default_timeout=120).run()
    at.switch_page(view).run()
    return at


@pytest.mark.parametrize("view", VIEWS)
def test_view_renders_without_exceptions(view):
    at = open_view(view)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("league", ["E0", "SP1", "D1", "I1", "F1", "MEX", "ARG", "BRA"])
def test_current_season_view_for_every_league(league):
    at = open_view("views/temporada.py")
    at.session_state["league"] = league
    at.run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("league", ["E0", "SP1", "MEX", "ARG", "BRA"])
def test_teams_and_players_views_for_league(league):
    for view, key in [
        ("views/equipos.py", "teams_league"),
        ("views/jugadores.py", "players_league"),
    ]:
        at = open_view(view)
        at.session_state[key] = league
        at.run()
        assert not at.exception, (view, [e.value for e in at.exception])


def test_player_hit_rates_for_every_stat():
    at = open_view("views/jugadores.py")
    for stat in ["shots_on", "shots_total", "goals", "yellow"]:
        at.session_state["player_stat"] = stat
        at.run()
        assert not at.exception, (stat, [e.value for e in at.exception])


@pytest.mark.parametrize("league", ["E0", "SP1", "D1", "I1", "F1", "MEX", "ARG", "BRA"])
def test_match_simulator_for_every_league(league):
    at = open_view("views/predicciones.py")
    at.session_state["sim_league"] = league
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Goles esperados" == m.label for m in at.metric)


@pytest.mark.parametrize(
    ("league", "season"), [("E0", "2026/2027"), ("SP1", "2020/2021"), ("I1", "2012/2013")]
)
def test_match_history_card(league, season):
    """Played-match card renders for the current season, a past one and one with no predictions."""
    at = open_view("views/predicciones.py")
    at.session_state["hist_league"] = league
    at.run()
    at.session_state[f"hist_season_{league}"] = season
    at.run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("league", ["E0", "SP1", "D1", "I1", "F1", "MEX", "ARG", "BRA"])
def test_upcoming_round_for_every_league(league):
    at = open_view("views/predicciones.py")
    at.session_state["pred_league"] = league
    at.run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("variant", ["Captura", "Captura en modo oscuro", "Foto con el celular"])
def test_bet_slip_analyser_with_sample(variant):
    at = open_view("views/apuesta.py")
    at.session_state["slip_source"] = "Usar el boleto de ejemplo"
    at.session_state["slip_variant"] = variant
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.label == "Probabilidad de acertar todo" for m in at.metric)


@pytest.mark.parametrize("league", ["E0", "SP1", "D1", "I1", "F1", "MEX", "ARG", "BRA"])
@pytest.mark.parametrize("mode", ["next", "played"])
def test_lineups_view_for_every_league(league, mode):
    at = open_view("views/alineaciones.py")
    at.session_state["xi_league"] = league
    at.session_state["xi_mode"] = mode
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if not at.info:  # data present: a pitch and a player card were drawn
        assert at.get("arrow_vega_lite_chart") or at.get("vega_lite_chart")
        assert any(sb.label.startswith("Jugador") for sb in at.selectbox)
