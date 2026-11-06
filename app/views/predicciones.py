"""Predictions: upcoming fixtures (all markets), simulator, per-market reliability,
1X2 backtest validation, xG experiment and match browser."""

import altair as alt
import match_card
import numpy as np
import pandas as pd
import streamlit as st
from common import (
    BLUE,
    INK_MUTED,
    LEAGUE_NAMES,
    MARKET_LABELS,
    MARKET_MODEL_NAMES,
    MODEL_NAMES,
    calibration,
    color_scale,
    current_season,
    fecha_es,
    forecaster,
    freshness,
    league_seasons,
    market_calibration,
    market_scores,
    market_verdict,
    played_matches,
    pre_match_predictions,
    scores,
    season_teams,
    team_ratings,
    upcoming_schedule,
)
from styles import apply_columns, diverging, sequential

from futbol.leagues import LEAGUES


def diverging_inverse(col: pd.Series) -> list[str]:
    """For score gaps lower is better: flip the sign so 'better' renders blue."""
    return diverging(-col)


st.title("Predicciones")

PROB_FORMAT = {c: st.column_config.ProgressColumn(t, format="percent", min_value=0, max_value=1)
               for c, t in [("p_home", "Local"), ("p_draw", "Empate"), ("p_away", "Visitante")]}  # fmt: skip
season = freshness().season

tab_next, tab_hist, tab_sim, tab_rel, tab_val, tab_xg = st.tabs(
    ["Próximos", "Historial", "Simulador", "Fiabilidad", "Validación 1X2", "¿Mejora el xG?"]
)  # fmt: skip
market_score_all = market_scores()

# ---------------------------------------------------------------- upcoming fixtures
with tab_next:
    league = (
        st.segmented_control(
            "Liga",
            list(LEAGUE_NAMES),
            format_func=LEAGUE_NAMES.get,
            default="E0",
            key="pred_league",
        )
        or "E0"
    )
    sched = upcoming_schedule(league)
    if sched.empty and not LEAGUES[league].schedule_slug:
        st.info(
            f"Todavía no tenemos el calendario de próximos partidos de {LEAGUE_NAMES[league]}. "
            "Mientras tanto, la pestaña **Simulador** predice cualquier enfrentamiento y **Historial** "
            "tiene todos los partidos jugados.",
            icon=":material/event_upcoming:",
        )
    elif sched.empty:
        st.info(
            "No quedan partidos por jugar en el calendario de esta temporada.",
            icon=":material/event_busy:",
        )
    else:
        # "Next round" = round of the next match to be played, not the lowest pending round:
        # postponed games from earlier rounds would otherwise hijack the default.
        rounds = sorted(sched["round"].unique())
        next_round = int(sched["round"].iloc[0])
        rnd = st.selectbox(
            "Jornada", rounds, index=rounds.index(next_round), key=f"round_{league}",
            format_func=lambda r: f"Jornada {r}" + (" (próxima)" if r == next_round else ""),
        )  # fmt: skip
        fx = sched[sched["round"] == rnd].reset_index(drop=True)
        fc = forecaster(league)
        table = fx.join(fc.forecast(fx))
        table["Partido"] = table.home_team + "  vs  " + table.away_team
        table["fecha"] = table.kickoff.map(fecha_es)
        table["goles"] = (
            table.exp_home_goals.map("{:.1f}".format)
            + " – "
            + table.exp_away_goals.map("{:.1f}".format)
        )
        st.caption(
            "Haz clic en un partido para ver su predicción completa. Horas en tu zona horaria."
        )
        event = st.dataframe(
            table[["fecha", "Partido", "venue", "p_home", "p_draw", "p_away", "goles", "p_over_2_5",
                   "exp_total_corners", "exp_total_yellow"]],
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key=f"upcoming_{league}_{rnd}",
            column_config={
                "fecha": "Fecha y hora",
                "venue": "Estadio", **PROB_FORMAT,
                "goles": "Goles esp.",
                "p_over_2_5": st.column_config.NumberColumn("+2.5 goles", format="percent"),
                "exp_total_corners": st.column_config.NumberColumn("Córners esp.", format="%.1f"),
                "exp_total_yellow": st.column_config.NumberColumn("Amarillas esp.", format="%.1f"),
            },
        )  # fmt: skip
        picked = event.selection.rows[0] if event.selection.rows else 0
        row = table.iloc[picked]
        if not event.selection.rows:
            st.caption(f"Mostrando el primer partido de la jornada: {row.Partido}.")
        with st.container(border=True):
            st.caption(f"{fecha_es(row.kickoff)} ({row.kickoff:%d/%m/%Y}) · {row.venue}")
            match_card.render(
                fc, row.home_team, row.away_team, market_score_all,
                referee=row.referee if isinstance(row.referee, str) else None,
                odds_over25=row.odds_over25 if pd.notna(row.odds_over25) else None,
                odds_under25=row.odds_under25 if pd.notna(row.odds_under25) else None,
            )  # fmt: skip

    ratings = team_ratings()
    if not ratings.empty and league in set(ratings.league_code):
        st.subheader(f"Fuerza actual · {LEAGUE_NAMES[league]}")
        r = ratings[ratings.league_code == league].sort_values("attack", ascending=False)
        home_mult = float(np.exp(r.home_adv.iloc[0]))
        st.caption(
            f"Goles que marcaría y recibiría cada equipo contra un rival medio en campo neutral. "
            f"Jugar en casa multiplica los goles esperados por {home_mult:.2f}; "
            f"ρ de Dixon-Coles: {r.rho.iloc[0]:+.3f}. Azul más oscuro = mejor."
        )
        ratings_view = (
            r[["team", "goals_for_vs_avg", "goals_against_vs_avg"]]
            .style.pipe(apply_columns, sequential, ["goals_for_vs_avg"])
            .pipe(apply_columns, sequential, ["goals_against_vs_avg"], reverse=True)
            .format({"goals_for_vs_avg": "{:.2f}", "goals_against_vs_avg": "{:.2f}"})
        )
        st.dataframe(
            ratings_view,
            hide_index=True,
            use_container_width=True,
            column_config={
                "team": "Equipo",
                "goals_for_vs_avg": "Goles a favor esperados",
                "goals_against_vs_avg": "Goles en contra esperados",
            },
        )

# ---------------------------------------------------------------- history
with tab_hist:
    c1, c2, c3 = st.columns([2, 1, 2])
    h_league = c1.selectbox(
        "Liga", list(LEAGUE_NAMES), format_func=LEAGUE_NAMES.get, key="hist_league"
    )
    h_season = c2.selectbox("Temporada", league_seasons(h_league), key=f"hist_season_{h_league}")
    games = played_matches(h_league, h_season)
    h_team = c3.selectbox(
        "Equipo", ["(todos)", *sorted(set(games.home_team) | set(games.away_team))],
        key=f"hist_team_{h_league}_{h_season}",
    )  # fmt: skip
    if h_team != "(todos)":
        games = games[(games.home_team == h_team) | (games.away_team == h_team)]
    games = games.reset_index(drop=True)
    games["Partido"] = (
        games.home_team + "  " + games.home_goals.astype(int).astype(str) + " – "
        + games.away_goals.astype(int).astype(str) + "  " + games.away_team
    )  # fmt: skip
    games["Tiros"] = (
        games.home_shots.astype("Int64").astype(str)
        + " – "
        + games.away_shots.astype("Int64").astype(str)
    )
    games["Córners"] = (
        games.home_corners.astype("Int64").astype(str)
        + " – "
        + games.away_corners.astype("Int64").astype(str)
    )
    games["Amarillas"] = (
        games.home_yellow.astype("Int64").astype(str)
        + " – "
        + games.away_yellow.astype("Int64").astype(str)
    )
    st.caption(
        f"{len(games)} partidos. Haz clic en uno para ver sus estadísticas y lo que predijo el modelo."
    )
    hist_event = st.dataframe(
        games[["match_date", "round", "Partido", "Tiros", "Córners", "Amarillas"]],
        hide_index=True,
        use_container_width=True,
        height=420,
        on_select="rerun",
        selection_mode="single-row",
        key=f"history_{h_league}_{h_season}_{h_team}",
        column_config={
            "match_date": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
            "round": st.column_config.NumberColumn("Jornada", format="%d"),
        },
    )
    if not games.empty:
        pick = hist_event.selection.rows[0] if hist_event.selection.rows else 0
        if not hist_event.selection.rows:
            st.caption(f"Mostrando el partido más reciente: {games.iloc[0].Partido}.")
        one_x_two, counts = pre_match_predictions(h_league, h_season)
        with st.container(border=True):
            match_card.render_past(games.iloc[pick], one_x_two, counts)

# ---------------------------------------------------------------- simulator
with tab_sim:
    st.caption(
        "Elige cualquier enfrentamiento de la temporada actual. El modelo se ajusta con los partidos de las "
        "últimas tres temporadas hasta hoy."
    )
    c1, c2, c3 = st.columns(3)
    sim_league = c1.selectbox(
        "Liga", list(LEAGUE_NAMES), format_func=LEAGUE_NAMES.get, key="sim_league"
    )
    teams = season_teams(sim_league, current_season(sim_league))
    home = c2.selectbox("Local", teams, index=0, key=f"sim_home_{sim_league}")
    away = c3.selectbox(
        "Visitante", [t for t in teams if t != home], index=0, key=f"sim_away_{sim_league}"
    )
    fc_sim = forecaster(sim_league)
    referee = None
    if fc_sim.referee_factor:
        refs = sorted(fc_sim.referee_factor)
        choice = st.selectbox(
            "Árbitro (opcional, solo Premier League)", ["(sin especificar)", *refs]
        )
        referee = None if choice == "(sin especificar)" else choice
    with st.container(border=True):
        match_card.render(fc_sim, home, away, market_score_all, referee=referee)

# ---------------------------------------------------------------- per-market reliability
with tab_rel:
    st.markdown(
        "Cada mercado se validó con el mismo backtest semana a semana que el 1X2 (2015/16 – hoy, cinco ligas). "
        "La referencia es la **media de la liga**: la frecuencia con que se supera esa línea en las últimas "
        "temporadas, igual para todos los partidos. Si el modelo no le gana, sus diferencias entre partidos son "
        "ruido. En goles también se compara con las **cuotas de cierre**."
    )
    rel = market_score_all.copy()
    rel["mae_total"] = pd.to_numeric(
        rel.mae_total, errors="coerce"
    )  # None -> NaN so na_rep applies
    rel["Mercado"] = rel.market.map(MARKET_LABELS)
    rel["Modelo"] = rel.model.map(MARKET_MODEL_NAMES)
    rel_view = (
        rel[["Mercado", "Modelo", "matches", "log_loss", "brier", "mae_total", "ll_vs_base", "ll_vs_base_lo",
             "ll_vs_base_hi"]]
        .style.apply(diverging_inverse, subset=["ll_vs_base"], axis=0)
        .format({"log_loss": "{:.4f}", "brier": "{:.4f}", "mae_total": "{:.2f}", "ll_vs_base": "{:+.4f}",
                 "ll_vs_base_lo": "{:+.4f}", "ll_vs_base_hi": "{:+.4f}", "matches": "{:,.0f}"}, na_rep="–")
    )  # fmt: skip
    st.dataframe(
        rel_view,
        hide_index=True,
        use_container_width=True,
        column_config={
            "matches": "Partidos", "log_loss": "Log loss", "brier": "Brier",
            "mae_total": st.column_config.Column(
                "Error medio del total", help="Diferencia media absoluta entre el total esperado y el real"
            ),
            "ll_vs_base": "Δ vs media de la liga", "ll_vs_base_lo": "IC95 inf.", "ll_vs_base_hi": "IC95 sup.",
        },
    )  # fmt: skip
    st.caption(
        "Δ < 0 (azul) = mejor que la media de la liga. Si el intervalo IC95 cruza el 0, la diferencia no se "
        "distingue del azar. El modelo sin árbitro solo existe para la Premier League (única liga con árbitro "
        "en los datos)."
    )
    for market in ["shots", "goals", "yellow", "corners"]:
        icon, text = market_verdict(market_score_all, market)
        st.markdown(f"{icon} **{MARKET_LABELS[market]}**: {text}.")

    st.subheader("Calibración del modelo por mercado")
    cal_market = st.radio(
        "Mercado",
        ["goals", "corners", "yellow", "shots"],
        format_func=MARKET_LABELS.get,
        horizontal=True,
    )
    cal = market_calibration(cal_market)
    diag = (
        alt.Chart(pd.DataFrame({"x": [0.1, 0.9]}))
        .mark_line(color=INK_MUTED, strokeDash=[4, 4])
        .encode(x="x:Q", y="x:Q")
    )
    cal_line = (
        alt.Chart(cal)
        .mark_line(color=BLUE, strokeWidth=2, point=alt.OverlayMarkDef(size=60, color=BLUE))
        .encode(
            x=alt.X("predicted:Q", title="P(más de la línea) predicha", axis=alt.Axis(format="%")),
            y=alt.Y("observed:Q", title="Frecuencia observada", axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip("predicted:Q", format=".1%", title="Predicha"),
                alt.Tooltip("observed:Q", format=".1%", title="Observada"),
            ],
        )  # fmt: skip
    )
    st.altair_chart((diag + cal_line).properties(height=340), use_container_width=True)


def score_view(experiment: str, models: list[str]) -> pd.DataFrame:
    s = scores(experiment)
    s = s[s.model.isin(models)].copy()
    s["Modelo"] = s.model.map(MODEL_NAMES)
    return s


SCORE_CONFIG = {
    "matches": st.column_config.NumberColumn("Partidos", format="%d"),
    "brier": st.column_config.NumberColumn("Brier", format="%.4f"),
    "log_loss": st.column_config.NumberColumn("Log loss", format="%.4f"),
    "rps": st.column_config.NumberColumn("RPS", format="%.4f"),
    "accuracy": st.column_config.NumberColumn("Acierto (favorito)", format="percent"),
    "ll_vs_ref": st.column_config.NumberColumn("Δ log loss vs mercado", format="%+.4f"),
    "ll_vs_ref_lo": st.column_config.NumberColumn("IC95 inf.", format="%+.4f"),
    "ll_vs_ref_hi": st.column_config.NumberColumn("IC95 sup.", format="%+.4f"),
}


def calibration_chart(experiment: str, model: str) -> alt.LayerChart:
    cal = calibration(experiment, model)
    cal["Resultado"] = cal.outcome.map({"home": "Local", "draw": "Empate", "away": "Visitante"})
    series = ["Local", "Empate", "Visitante"]
    diag = (
        alt.Chart(pd.DataFrame({"x": [0, 0.8]}))
        .mark_line(color=INK_MUTED, strokeDash=[4, 4])
        .encode(x="x:Q", y="x:Q")
    )
    lines = alt.Chart(cal).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60)).encode(
        x=alt.X("predicted:Q", title="Probabilidad predicha", axis=alt.Axis(format="%")),
        y=alt.Y("observed:Q", title="Frecuencia observada", axis=alt.Axis(format="%")),
        color=alt.Color("Resultado:N", scale=color_scale(series), legend=alt.Legend(orient="top", title=None)),
        tooltip=["Resultado:N", alt.Tooltip("predicted:Q", format=".1%", title="Predicha"),
                 alt.Tooltip("observed:Q", format=".1%", title="Observada")],
    )  # fmt: skip
    return (diag + lines).properties(height=380)


# ---------------------------------------------------------------- multi-season validation
with tab_val:
    st.markdown(
        "Backtest **walk-forward**: cada semana el modelo se reajusta solo con partidos jugados *antes* de ese lunes "
        "y predice los partidos de la semana. Hiperparámetros elegidos en 2013/14–2014/15, que no se puntúan. "
        "Todas las métricas: menor es mejor."
    )
    models = ["market", "dixon_coles_goals", "poisson_goals", "base_rates"]
    s = score_view("multi_season", models)
    if s.empty:
        st.info("Ejecuta `scripts/run_backtest.py`.")
    else:
        st.dataframe(
            s[["Modelo", *SCORE_CONFIG]],
            hide_index=True,
            use_container_width=True,
            column_config=SCORE_CONFIG,
        )
        st.caption(
            "Referencias: un pronóstico uniforme (1/3 cada resultado) tiene Brier 0.667 y log loss 1.099. "
            "Δ vs mercado < 0 significaría superar a las casas de apuestas; el intervalo indica si la diferencia es "
            "distinguible del azar."
        )

        st.subheader("Log loss por temporada")
        by_season = scores("multi_season", by_season=True)
        by_season = by_season[by_season.model.isin(models[:3])]
        by_season["Modelo"] = by_season.model.map(MODEL_NAMES)
        names = [MODEL_NAMES[m] for m in models[:3]]
        chart = alt.Chart(by_season).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=50)).encode(
            x=alt.X("season:N", title="Temporada"),
            y=alt.Y("log_loss:Q", title="Log loss", scale=alt.Scale(zero=False)),
            color=alt.Color("Modelo:N", scale=color_scale(names), legend=alt.Legend(orient="top", title=None)),
            tooltip=["season:N", "Modelo:N", alt.Tooltip("log_loss:Q", format=".4f"),
                     alt.Tooltip("matches:Q", title="Partidos")],
        )  # fmt: skip
        st.altair_chart(chart.properties(height=320), use_container_width=True)

        st.subheader("Calibración")
        st.caption(
            "Si el modelo dice 40%, ¿ocurre el 40% de las veces? Puntos sobre la diagonal = bien calibrado. "
            "Cada punto agrupa el mismo número de partidos."
        )
        model = st.radio(
            "Modelo", ["dixon_coles_goals", "market"], format_func=MODEL_NAMES.get, horizontal=True
        )
        st.altair_chart(calibration_chart("multi_season", model), use_container_width=True)

# ---------------------------------------------------------------- xG vs goals
with tab_xg:
    st.markdown(
        "Solo existe xG de StatsBomb para la temporada 2015/16 completa de Premier League, LaLiga, Serie A y Ligue 1. "
        "Para comparar en igualdad, los tres primeros modelos usan **solo partidos de esa misma temporada** "
        "(sin historial previo) y empiezan a predecir tras ~5 jornadas."
    )
    models = [
        "market",
        "dixon_coles_goals_3y_history",
        "dixon_coles_mix",
        "dixon_coles_xg",
        "dixon_coles_goals",
    ]
    s = score_view("xg_vs_goals", models)
    if s.empty:
        st.info("Ejecuta `scripts/build_metrics.py` y `scripts/run_backtest.py`.")
    else:
        st.dataframe(
            s[["Modelo", *SCORE_CONFIG]],
            hide_index=True,
            use_container_width=True,
            column_config=SCORE_CONFIG,
        )
