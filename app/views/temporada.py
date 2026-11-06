"""Season in progress: standings, shots, form, market expected points, model predictions."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from common import (
    LEAGUE_NAMES,
    NEGATIVE,
    POSITIVE,
    current_season,
    season_matches,
    season_predictions,
    short_season,
    upcoming_predictions,
)
from styles import (
    FORM_COLUMNS,
    apply_columns,
    diverging,
    form_cell,
    sequential,
    split_form,
    zone_cells,
    zone_legend_html,
)

from futbol.leagues import LEAGUES, tournament_of
from futbol.models import odds_to_probs
from futbol.season import regular_phase, standings

st.title("Temporada en curso")

league = (
    st.segmented_control(
        "Liga", list(LEAGUE_NAMES), format_func=LEAGUE_NAMES.get, default="E0", key="league"
    )
    or "E0"
)
info = LEAGUES[league]
season = current_season(league)
matches = season_matches(league, season)

if info.tournaments:  # Liga MX: Apertura / Clausura, each with its own table and liguilla
    matches = matches.assign(
        tournament=pd.to_datetime(matches.match_date).dt.date.map(tournament_of)
    )
    options = list(dict.fromkeys(matches.sort_values("match_date", ascending=False).tournament))
    tournament = st.selectbox("Torneo", options, key=f"tournament_{season}")
    matches = matches[matches.tournament == tournament]
    table_matches = matches[regular_phase(matches)]
    playoffs = len(matches.dropna(subset=["home_goals"])) - len(table_matches)
else:
    table_matches, playoffs = matches, 0

played = matches.dropna(subset=["home_goals"])
if played.empty:
    st.info("Todavía no hay partidos jugados esta temporada.")
    st.stop()

last = pd.Timestamp(played.match_date.max())
source = (
    "resultados, tiros y cuotas de cierre"
    if info.has_stats
    else "resultados y cuotas de cierre (sin tiros)"
)
st.subheader(
    f"{LEAGUE_NAMES[league]} · {short_season(season)}"
    + (f" · {tournament}" if info.tournaments else "")
)
st.caption(
    f"{len(played)} partidos jugados · último: {last:%d/%m/%Y} · fuente: football-data.co.uk ({source}). "
    "Se actualiza con `python scripts/update.py`."
    + (
        f" La tabla cuenta solo la fase regular; {playoffs} partidos de play-in/liguilla no suman."
        if playoffs
        else ""
    )
)
if league == "ARG":
    st.caption(
        ":material/info: La Liga Profesional se juega por zonas con eliminatorias; esta es la *tabla anual* "
        "(todos los partidos del año), que es la que decide descensos y copas internacionales."
    )

# ---------------------------------------------------------------- standings
table = standings(table_matches)
table = pd.concat([table, split_form(table.form)], axis=1)
# Form sits next to points so it is visible without scrolling sideways; one cell per match
# so every result gets its own colour (with its letter, so colour is never the only cue).
STANDINGS_COLUMNS = (
    # The essentials first: on a phone only the first ~5 columns fit without swiping the table.
    f"position team points {' '.join(FORM_COLUMNS)} played goal_diff won drawn lost goals_for goals_against "
    "xpts pts_minus_xpts shots_pm sot_pm shots_against_pm conversion"
).split()
if not info.has_stats:  # no shots in this league's data: don't show empty columns
    STANDINGS_COLUMNS = [
        c
        for c in STANDINGS_COLUMNS
        if c not in ("shots_pm", "sot_pm", "shots_against_pm", "conversion")
    ]
styled = (
    table[STANDINGS_COLUMNS]
    .style.apply(lambda s: zone_cells(s, league), subset=["position"], axis=0)
    .map(form_cell, subset=FORM_COLUMNS)
    .apply(diverging, subset=["goal_diff"], axis=0)
    .apply(diverging, subset=["pts_minus_xpts"], axis=0)
    .pipe(apply_columns, sequential, ["shots_pm", "sot_pm", "conversion"])
    .pipe(apply_columns, sequential, ["shots_against_pm"], reverse=True)
    .format(
        {
            "goal_diff": "{:+d}",
            "xpts": "{:.1f}",
            "pts_minus_xpts": "{:+.1f}",
            "shots_pm": "{:.1f}",
            "sot_pm": "{:.1f}",
            "shots_against_pm": "{:.1f}",
            "conversion": "{:.0%}",
        },
        na_rep="",
    )  # fmt: skip
)
FORM_CONFIG = {
    c: st.column_config.TextColumn(
        "Forma" if i == 0 else "",
        width=58 if i == 0 else 34,  # the first cell also carries the header text
        help="Últimos 5 partidos, el más reciente a la derecha",
    )
    for i, c in enumerate(FORM_COLUMNS)
}
st.dataframe(
    styled,
    hide_index=True,
    use_container_width=True,
    height=35 * (len(table) + 1) + 3,
    column_config={
        **FORM_CONFIG,
        "position": st.column_config.NumberColumn("#", width="small"),
        "team": "Equipo",
        "played": "PJ",
        "won": "G",
        "drawn": "E",
        "lost": "P",
        "goals_for": "GF",
        "goals_against": "GC",
        "goal_diff": st.column_config.NumberColumn("DG", format="%+d"),
        "points": "Pts",
        "xpts": st.column_config.NumberColumn(
            "Pts esp.",
            format="%.1f",
            help="Puntos que esperaba el mercado según las cuotas de cierre de cada partido",
        ),
        "pts_minus_xpts": st.column_config.NumberColumn("Pts − esp.", format="%+.1f"),
        "shots_pm": st.column_config.NumberColumn("Tiros/PJ", format="%.1f"),
        "sot_pm": st.column_config.NumberColumn("A puerta/PJ", format="%.1f"),
        "shots_against_pm": st.column_config.NumberColumn("Tiros rec./PJ", format="%.1f"),
        "conversion": st.column_config.NumberColumn(
            "Gol/tiro a puerta",
            format="percent",
            help="Goles por cada tiro a puerta. Muy ruidoso con pocos partidos.",
        ),
    },
)
st.html(zone_legend_html(league))
st.caption(
    "Azul/rojo en DG y Pts − esp.: por encima/por debajo (más intenso = mayor diferencia). Azul más oscuro en "
    "tiros y conversión = más alto; en tiros recibidos, más oscuro = recibe menos. "
    "Orden: puntos, diferencia de goles, goles a favor. Algunas ligas desempatan primero por el enfrentamiento "
    "directo, así que en empates el orden oficial puede variar."
)

# ---------------------------------------------------------------- over/under expectations
st.subheader("¿Quién rinde por encima de lo esperado?")
st.caption(
    "Antes de cada partido, las cuotas de cierre implican una probabilidad de ganar, empatar y perder; los puntos "
    "esperados suman 3 × P(ganar) + P(empatar) en cada partido. La diferencia mezcla suerte con lo que el mercado "
    "no vio: una racha positiva grande suele moderarse, pero no siempre."
)
diff = table[["team", "points", "xpts", "pts_minus_xpts"]].assign(
    signo=lambda d: d.pts_minus_xpts.map(lambda v: "Por encima" if v >= 0 else "Por debajo")
)
bars = (
    alt.Chart(diff)
    .mark_bar(cornerRadiusEnd=4, height=14)
    .encode(
        x=alt.X("pts_minus_xpts:Q", title="Puntos reales − puntos esperados"),
        y=alt.Y("team:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=160)),
        color=alt.Color(
            "signo:N",
            scale=alt.Scale(domain=["Por encima", "Por debajo"], range=[POSITIVE, NEGATIVE]),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=[
            alt.Tooltip("team:N", title="Equipo"),
            alt.Tooltip("points:Q", title="Puntos"),
            alt.Tooltip("xpts:Q", title="Esperados", format=".1f"),
            alt.Tooltip("pts_minus_xpts:Q", title="Diferencia", format="+.1f"),
        ],
    )
)
zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#b5b4ae").encode(x="x:Q")
st.altair_chart((bars + zero).properties(height=26 * len(diff)), use_container_width=True)

# ---------------------------------------------------------------- upcoming
st.subheader("Próximos partidos")
PROBS = {c: st.column_config.ProgressColumn(t, format="percent", min_value=0, max_value=1)
         for c, t in [("p_home", "Local"), ("p_draw", "Empate"), ("p_away", "Visitante")]}  # fmt: skip
up = upcoming_predictions(league)
if up.empty and not info.schedule_slug:
    st.info(
        "Todavía no tenemos el calendario de próximos partidos de esta liga. Mientras tanto, la pestaña "
        "*Simulador* de **Predicciones** predice cualquier enfrentamiento.",
        icon=":material/event_upcoming:",
    )
elif up.empty:
    st.info(
        "Aún no se han publicado los partidos de la próxima jornada de esta liga. football-data.co.uk los "
        "publica unos días antes; ejecuta `python scripts/update.py` y pulsa *Recargar datos*."
    )
else:
    market = odds_to_probs(up).add_prefix("m_")
    up = up.join(market)
    st.dataframe(
        up[["match_date", "home_team", "away_team", "p_home", "p_draw", "p_away", "m_p_home", "m_p_draw", "m_p_away"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "match_date": "Fecha", "home_team": "Local", "away_team": "Visitante", **PROBS,
            "m_p_home": st.column_config.NumberColumn("Mercado L", format="percent"),
            "m_p_draw": st.column_config.NumberColumn("Mercado E", format="percent"),
            "m_p_away": st.column_config.NumberColumn("Mercado V", format="percent"),
        },
    )  # fmt: skip

# ---------------------------------------------------------------- recent results vs model
st.subheader("Últimos resultados y lo que predijo el modelo")
st.caption(
    "Probabilidades de Dixon-Coles calculadas *antes* de cada partido, solo con datos anteriores. "
    "La última columna es la probabilidad que el modelo le dio al resultado que ocurrió."
)
preds = season_predictions(league, season).head(20)
if not preds.empty:
    hg, ag = preds.home_goals.astype(int), preds.away_goals.astype(int)
    result = np.select([hg > ag, hg == ag], ["home", "draw"], "away")
    preds = preds.assign(
        Resultado=hg.astype(str) + "–" + ag.astype(str),
        p_result=np.choose(np.select([result == "home", result == "draw"], [0, 1], 2),
                           [preds.p_home, preds.p_draw, preds.p_away]),
    )  # fmt: skip

    def team_cells(_: pd.Series, side: str) -> list[str]:
        """Winner green, loser red, draw grey: same status colours as the form column."""
        letters = {"home": {"home": "G", "away": "P", "draw": "E"},
                   "away": {"home": "P", "away": "G", "draw": "E"}}[side]  # fmt: skip
        return [form_cell(letters[r]).replace(" text-align: center;", "") for r in result]

    styled_preds = (
        preds[
            [
                "match_date",
                "home_team",
                "Resultado",
                "away_team",
                "p_home",
                "p_draw",
                "p_away",
                "p_result",
            ]
        ]
        .style.apply(lambda s: team_cells(s, "home"), subset=["home_team"], axis=0)
        .apply(lambda s: team_cells(s, "away"), subset=["away_team"], axis=0)
        .apply(sequential, subset=["p_result"], axis=0)
        .format({"p_result": "{:.0%}"})
    )
    st.dataframe(
        styled_preds,
        hide_index=True,
        use_container_width=True,
        column_config={
            "match_date": "Fecha", "home_team": "Local", "away_team": "Visitante", **PROBS,
            "p_result": st.column_config.NumberColumn(
                "P(lo que pasó)", help="Azul claro = el modelo no lo esperaba (sorpresa)"
            ),
        },
    )  # fmt: skip
