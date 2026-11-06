"""Shared data access and chart helpers for the Streamlit pages.

All queries are wrapped in st.cache_data: Streamlit re-runs the whole script on every
widget interaction, so without caching each click would hit the database again.
"""

from __future__ import annotations

from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.calibration import calibration_curve

from futbol.betslip import TeamIndex, decode_image, ocr_lines, preprocess_variants
from futbol.db import read_sql
from futbol.experiments import load_market_predictions, load_params
from futbol.ingest.api_football import ApiFootballError, player_season_stats
from futbol.leagues import LEAGUES
from futbol.markets import MatchForecaster, market_score_table
from futbol.validation import calibration_data, score_table

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK_MUTED = "#52514e"

# Every supported league (Big-5 + Liga MX, Argentina, Brasil): see futbol.leagues.
LEAGUE_NAMES = {code: f"{lg.name}" for code, lg in LEAGUES.items()}
MODEL_NAMES = {
    "market": "Mercado (cuotas de cierre)",
    "dixon_coles_goals": "Dixon-Coles (goles)",
    "poisson_goals": "Poisson (goles)",
    "base_rates": "Frecuencias históricas",
    "dixon_coles_xg": "Dixon-Coles (xG)",
    "dixon_coles_mix": "Dixon-Coles (xG + goles)",
    "dixon_coles_goals_3y_history": "Dixon-Coles (goles, 3 temporadas)",
}
CACHE_TTL = 3600


POSITIVE, NEGATIVE = BLUE, "#e34948"  # diverging pair (blue <-> red), validated palette


# --------------------------------------------------------------------------- freshness


@st.cache_data(ttl=CACHE_TTL)
def freshness() -> pd.Series:
    """Season in progress, last played match date and last refresh time."""
    return read_sql(
        """
        SELECT max(season) AS season,
               max(match_date) FILTER (WHERE home_goals IS NOT NULL) AS last_played,
               max(updated_at) AS updated_at
        FROM fd_matches
        WHERE season = (SELECT max(season) FROM fd_matches)
        """
    ).iloc[0]


def short_season(season: str) -> str:
    """'2026/2027' -> '2026/27'; calendar-year seasons ('2026') stay as they are."""
    if "/" not in str(season):
        return str(season)
    start, end = str(season).split("/")
    return f"{start}/{end[-2:]}"


@st.cache_data(ttl=CACHE_TTL)
def current_season(league_code: str) -> str:
    """Season label of the league's most recent match: '2026/2027' in Europe, '2026' in Brazil."""
    return read_sql(
        "SELECT season FROM fd_matches WHERE league_code = :l AND home_goals IS NOT NULL "
        "ORDER BY match_date DESC LIMIT 1",
        l=league_code,
    ).season.iloc[0]


def sidebar_status() -> None:
    """Data date always visible, plus a button to drop cached queries after an update."""
    f = freshness()
    # Shown in the server's local time zone: a bare UTC time read as "the wrong hour".
    updated = (
        pd.Timestamp(f.updated_at).tz_convert(datetime.now().astimezone().tzinfo)
        if pd.notna(f.updated_at)
        else None
    )
    st.sidebar.caption(
        f"**Temporada {short_season(f.season)}** · último partido jugado: "
        f"{pd.Timestamp(f.last_played):%d/%m/%Y}"
        + (f" · datos actualizados {updated:%d/%m/%Y %H:%M}" if updated is not None else "")
    )
    if st.sidebar.button("Recargar datos", help="Tras ejecutar `python scripts/update.py`"):
        st.cache_data.clear()
        st.rerun()


# --------------------------------------------------------------------------- current season


@st.cache_data(ttl=CACHE_TTL)
def season_matches(league_code: str, season: str) -> pd.DataFrame:
    return read_sql(
        "SELECT * FROM fd_matches WHERE league_code = :l AND season = :s ORDER BY match_date",
        l=league_code,
        s=season,
    )


@st.cache_data(ttl=CACHE_TTL)
def season_predictions(league_code: str, season: str) -> pd.DataFrame:
    """Walk-forward Dixon-Coles predictions (made before each match) for one season."""
    return read_sql(
        """
        SELECT p.match_date, p.home_team, p.away_team, p.p_home, p.p_draw, p.p_away,
               f.home_goals, f.away_goals, f.odds_home, f.odds_draw, f.odds_away
        FROM predictions p
        JOIN fd_matches f USING (league_code, season, home_team, away_team)
        WHERE p.experiment = 'multi_season' AND p.model = 'dixon_coles_goals'
          AND p.league_code = :l AND p.season = :s
        ORDER BY p.match_date DESC
        """,
        l=league_code,
        s=season,
    )


@st.cache_data(ttl=CACHE_TTL)
def upcoming_predictions(league_code: str | None = None) -> pd.DataFrame:
    df = read_sql(
        """
        SELECT p.*, f.odds_home, f.odds_draw, f.odds_away
        FROM predictions p
        JOIN fd_matches f USING (league_code, season, home_team, away_team)
        WHERE p.experiment = 'upcoming' AND (CAST(:l AS text) IS NULL OR p.league_code = :l)
        ORDER BY p.match_date
        """,
        l=league_code,
    )
    df["league"] = df.league_code.map(LEAGUE_NAMES)
    return df


# --------------------------------------------------------------------------- predictions


@st.cache_data(ttl=CACHE_TTL)
def predictions(experiment: str) -> pd.DataFrame:
    df = read_sql(
        """
        SELECT p.*, f.home_goals, f.away_goals
        FROM predictions p
        LEFT JOIN fd_matches f USING (league_code, season, home_team, away_team)
        WHERE p.experiment = :e
        """,
        e=experiment,
    )
    df["league"] = df.league_code.map(LEAGUE_NAMES)
    df["model_name"] = df.model.map(MODEL_NAMES).fillna(df.model)
    return df


@st.cache_data(ttl=CACHE_TTL)
def scores(experiment: str, by_season: bool = False) -> pd.DataFrame:
    p = predictions(experiment).dropna(subset=["home_goals"])
    if not by_season:
        return score_table(p, reference="market").reset_index()
    parts = [score_table(g).reset_index().assign(season=s) for s, g in p.groupby("season")]
    return pd.concat(parts, ignore_index=True)


@st.cache_data(ttl=CACHE_TTL)
def calibration(experiment: str, model: str) -> pd.DataFrame:
    p = predictions(experiment)
    return calibration_data(p[(p.model == model)].dropna(subset=["home_goals"]))


@st.cache_data(ttl=CACHE_TTL)
def team_ratings() -> pd.DataFrame:
    try:
        return read_sql("SELECT * FROM team_ratings")
    except Exception:  # table appears after the first run_backtest.py
        return pd.DataFrame()


# --------------------------------------------------------------------------- charts


def color_scale(domain: list[str]) -> alt.Scale:
    """Fixed categorical order: colour follows the series, never its rank."""
    palette = [BLUE, ORANGE, AQUA, YELLOW]
    return alt.Scale(domain=domain, range=palette[: len(domain)])


# --------------------------------------------------------------------------- match markets

MARKET_LABELS = {
    "goals": "Goles (+2.5)",
    "corners": "Córners",
    "yellow": "Tarjetas amarillas",
    "shots": "Tiros",
}
MARKET_MODEL_NAMES = {
    "poisson": "Modelo (Poisson)",
    "poisson_sin_arbitro": "Modelo sin árbitro",
    "league_average": "Media de la liga",
    "market": "Mercado (cuotas)",
}


@st.cache_resource(ttl=CACHE_TTL, show_spinner="Ajustando el modelo con los datos de hoy…")
def forecaster(league_code: str) -> MatchForecaster:
    """All market models for one league, fitted on the last 3 seasons up to today."""
    today = pd.Timestamp.today().normalize()
    played = read_sql(
        "SELECT * FROM fd_matches WHERE league_code = :l AND home_goals IS NOT NULL "
        "AND match_date >= :since",
        l=league_code,
        since=(today - pd.Timedelta(days=3 * 365)).date(),
    )
    return MatchForecaster(load_params()).fit(played, today)


@st.cache_data(ttl=CACHE_TTL)
def fixtures(league_code: str) -> pd.DataFrame:
    """Unplayed fixtures from today on, with the market's over/under 2.5 odds when known."""
    return read_sql(
        "SELECT match_date, home_team, away_team, referee, odds_home, odds_draw, odds_away, "
        "odds_over25, odds_under25 FROM fd_matches "
        "WHERE league_code = :l AND home_goals IS NULL AND match_date >= CURRENT_DATE "
        "ORDER BY match_date",
        l=league_code,
    )


@st.cache_data(ttl=CACHE_TTL)
def season_teams(league_code: str, season: str) -> list[str]:
    df = read_sql(
        "SELECT DISTINCT home_team FROM fd_matches WHERE league_code = :l AND season = :s",
        l=league_code,
        s=season,
    )
    return sorted(df.home_team)


@st.cache_data(ttl=CACHE_TTL)
def market_scores(league_code: str | None = None) -> pd.DataFrame:
    df = load_market_predictions()
    if league_code:
        df = df[df.league_code == league_code]
    return market_score_table(df)


@st.cache_data(ttl=CACHE_TTL)
def market_calibration(market: str, n_bins: int = 10) -> pd.DataFrame:
    """Predicted P(over) vs observed frequency for the Poisson model, quantile bins."""
    df = load_market_predictions()
    df = df[(df.market == market) & (df.model == "poisson")].dropna(subset=["actual_home"])
    y = (df.actual_home + df.actual_away > df.line).astype(int)
    observed, predicted = calibration_curve(y, df.p_over, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"predicted": predicted, "observed": observed})


def market_verdict(scores: pd.DataFrame, market: str) -> tuple[str, str]:
    """One-line reliability verdict per market from the backtest (icon, text)."""
    row = scores[(scores.market == market) & (scores.model == "poisson")]
    if row.empty or pd.isna(row.ll_vs_base_hi.iloc[0]):
        return ":material/help:", "sin validación todavía"
    r = row.iloc[0]
    if r.ll_vs_base_hi < 0:
        return ":material/check_circle:", (
            f"mejor que la media de la liga (Δ log loss {r.ll_vs_base:+.4f}, IC95 "
            f"[{r.ll_vs_base_lo:+.4f}, {r.ll_vs_base_hi:+.4f}], {int(r.matches):,} partidos)"
        )
    return ":material/warning:", (
        f"no se distingue de usar la media de la liga (Δ log loss {r.ll_vs_base:+.4f}, IC95 "
        f"[{r.ll_vs_base_lo:+.4f}, {r.ll_vs_base_hi:+.4f}]): tómalo como orientativo"
    )


# --------------------------------------------------------------------------- schedule & history

LOCAL_TZ = datetime.now().astimezone().tzinfo


@st.cache_data(ttl=CACHE_TTL)
def upcoming_schedule(league_code: str) -> pd.DataFrame:
    """Unplayed matches of the season in progress, with odds/referee when football-data has them."""
    df = read_sql(
        """
        SELECT s.round, s.kickoff_utc, s.home_team, s.away_team, s.venue,
               f.referee, f.odds_home, f.odds_draw, f.odds_away, f.odds_over25, f.odds_under25
        FROM schedule s
        LEFT JOIN fd_matches f USING (league_code, season, home_team, away_team)
        WHERE s.league_code = :l AND s.season = (SELECT max(season) FROM schedule)
          AND f.home_goals IS NULL
          AND s.kickoff_utc >= now() - interval '3 hours'
        ORDER BY s.kickoff_utc
        """,
        l=league_code,
    )
    df["kickoff"] = pd.to_datetime(df.kickoff_utc, utc=True).dt.tz_convert(LOCAL_TZ)
    return df


@st.cache_data(ttl=CACHE_TTL)
def league_seasons(league_code: str) -> list[str]:
    df = read_sql(
        "SELECT DISTINCT season FROM fd_matches WHERE league_code = :l AND home_goals IS NOT NULL "
        "ORDER BY season DESC",
        l=league_code,
    )
    return df.season.tolist()


@st.cache_data(ttl=CACHE_TTL)
def played_matches(league_code: str, season: str) -> pd.DataFrame:
    """Played matches with their statistics (and the round, when the schedule knows it)."""
    return read_sql(
        """
        SELECT f.*, s.round
        FROM fd_matches f
        LEFT JOIN schedule s USING (league_code, season, home_team, away_team)
        WHERE f.league_code = :l AND f.season = :s AND f.home_goals IS NOT NULL
        ORDER BY f.match_date DESC, f.home_team
        """,
        l=league_code,
        s=season,
    )


@st.cache_data(ttl=CACHE_TTL)
def pre_match_predictions(league_code: str, season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """What the models said before each match: 1X2 (model + market) and count markets."""
    one_x_two = read_sql(
        "SELECT model, home_team, away_team, p_home, p_draw, p_away, exp_home, exp_away FROM predictions "
        "WHERE experiment = 'multi_season' AND league_code = :l AND season = :s "
        "AND model IN ('dixon_coles_goals', 'market')",
        l=league_code,
        s=season,
    )
    counts = read_sql(
        "SELECT market, home_team, away_team, exp_home, exp_away, line, p_over FROM market_predictions "
        "WHERE model = 'poisson' AND league_code = :l AND season = :s",
        l=league_code,
        s=season,
    )
    return one_x_two, counts


DAYS_ES = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
MONTHS_ES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def fecha_es(ts: pd.Timestamp, with_time: bool = True) -> str:
    """'sáb 7 nov, 09:00' - Spanish regardless of the machine's locale."""
    text = f"{DAYS_ES[ts.weekday()]} {ts.day} {MONTHS_ES[ts.month - 1]}"
    return f"{text}, {ts:%H:%M}" if with_time else text


# --------------------------------------------------------------------------- bet slips


@st.cache_resource(ttl=CACHE_TTL)
def team_index() -> TeamIndex:
    """Every known name of every current team, for matching OCR text."""
    teams = read_sql(
        "SELECT DISTINCT league_code, home_team AS team FROM fd_matches "
        "WHERE match_date > CURRENT_DATE - 400"  # current teams of every league, whatever its calendar
    )
    aliases = read_sql("SELECT league_code, alias, team FROM team_alias")
    return TeamIndex.build(teams, aliases)


@st.cache_data(show_spinner="Leyendo el boleto con OpenCV + Tesseract…")
def read_slip(image_bytes: bytes) -> tuple[np.ndarray, dict[str, list[str]], float]:
    """OCR one image with every preprocessing variant. Cached by image content.

    Returns (image shown in the page, {variant: lines}, mean confidence).
    """
    variants = preprocess_variants(decode_image(image_bytes))
    lines, confidences = {}, []
    for name, image in variants.items():
        read = ocr_lines(image)
        lines[name] = [ln.text for ln in read]
        confidences += [ln.confidence for ln in read]
    return variants["clahe_otsu"], lines, float(np.mean(confidences or [0]))


@st.cache_data(ttl=CACHE_TTL)
def slip_fixtures() -> pd.DataFrame:
    """Upcoming schedule: resolves abbreviations like 'VAL vs RMA' to the soonest real fixture."""
    return read_sql(
        "SELECT league_code, home_team, away_team, kickoff_utc FROM schedule "
        "WHERE kickoff_utc > now() - interval '1 day'"
    )


@st.cache_data(ttl=CACHE_TTL, show_spinner="Buscando estadísticas del jugador en API-Football…")
def player_lookup(name: str, league_code: str, season_year: int) -> tuple[pd.DataFrame, str | None]:
    """(candidates, error). Errors are returned, not raised, so they are cached too and a
    plan/quota error doesn't trigger a new API request on every rerun."""
    try:
        return player_season_stats(name, league_code, season_year), None
    except (ApiFootballError, requests.RequestException) as exc:
        return pd.DataFrame(), str(exc)


# --------------------------------------------------------------------------- API-Football season data


@st.cache_data(ttl=CACHE_TTL)
def api_season(league_code: str) -> int | None:
    """Latest API-Football season stored for a league (start year)."""
    df = read_sql("SELECT max(season) AS s FROM api_fixture WHERE league_code = :l", l=league_code)
    return None if df.s.isna().iloc[0] else int(df.s.iloc[0])


@st.cache_data(ttl=CACHE_TTL)
def api_team_rows(league_code: str, season: int) -> pd.DataFrame:
    return read_sql(
        """
        SELECT f.fixture_id, f.kickoff_utc, f.round, f.home_id, f.home_name, f.away_id, f.away_name,
               f.home_goals, f.away_goals, t.team_id, t.is_home, t.xg, t.shots_total, t.shots_on,
               t.corners, t.yellow, t.red, t.fouls, t.possession
        FROM api_team_match t JOIN api_fixture f USING (fixture_id)
        WHERE f.league_code = :l AND f.season = :s
        """,
        l=league_code,
        s=season,
    )


@st.cache_data(ttl=CACHE_TTL)
def api_player_rows(league_code: str, season: int) -> pd.DataFrame:
    return read_sql(
        """
        SELECT p.*, f.kickoff_utc, f.home_id, f.home_name, f.away_id, f.away_name,
               f.home_goals, f.away_goals
        FROM api_player_match p JOIN api_fixture f USING (fixture_id)
        WHERE f.league_code = :l AND f.season = :s
        """,
        l=league_code,
        s=season,
    )


def api_season_label(league_code: str, season: int) -> str:
    """API seasons are start years: 2026 -> '2026/27' in Europe and Mexico, '2026' in Brazil."""
    return (
        str(season) if LEAGUES[league_code].calendar_season else f"{season}/{str(season + 1)[-2:]}"
    )


def api_freshness_note(rows: pd.DataFrame) -> str:
    last = pd.to_datetime(rows.kickoff_utc).max()
    return (
        f"{rows.fixture_id.nunique()} partidos · último: {last:%d/%m/%Y} · fuente: API-Football. "
        "Se actualiza con `python scripts/update.py`."
    )
