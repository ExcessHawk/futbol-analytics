"""Teams, current season (API-Football): xG, shots, corners, cards for and against."""

import altair as alt
import pandas as pd
import streamlit as st
from common import (
    BLUE,
    LEAGUE_NAMES,
    api_freshness_note,
    api_season,
    api_season_label,
    api_team_rows,
    color_scale,
)
from styles import apply_columns, diverging, sequential

from futbol.api_stats import team_match_view, team_table

st.title("Equipos")

league = (
    st.segmented_control(
        "Liga", list(LEAGUE_NAMES), format_func=LEAGUE_NAMES.get, default="E0", key="teams_league"
    )
    or "E0"
)
season = api_season(league)
if season is None:
    st.info(
        "Todavía no hay datos de API-Football para esta liga. Ejecuta `python scripts/download_api_football.py`."
    )
    st.stop()

rows = api_team_rows(league, season)
view = team_match_view(rows)
table = team_table(view)
has_xg = table.xg_matches.sum() > 0
st.subheader(f"{LEAGUE_NAMES[league]} · {api_season_label(league, season)}")
st.caption(api_freshness_note(rows) + " Valores por partido salvo puntos y goles.")

# Essentials first: a phone shows the first ~5 columns without swiping the table.
COLUMNS = ["position", "team", "points", "xg_pm", "xga_pm", "goals_minus_xg", "played", "goals_for", "goals_against",
           "shots_total_pm", "shots_on_pm", "shots_on_against_pm", "corners_pm", "corners_against_pm", "yellow_pm",
           "possession_pm"]  # fmt: skip
if not has_xg:
    COLUMNS = [c for c in COLUMNS if c not in ("xg_pm", "xga_pm", "goals_minus_xg")]
styled = (
    table[COLUMNS]
    .style.pipe(apply_columns, diverging, ["goals_minus_xg"])
    .pipe(
        apply_columns,
        sequential,
        ["xg_pm", "shots_total_pm", "shots_on_pm", "corners_pm", "possession_pm"],
    )
    .pipe(
        apply_columns,
        sequential,
        ["xga_pm", "shots_on_against_pm", "corners_against_pm"],
        reverse=True,
    )
    .format(
        {c: "{:.1f}" for c in COLUMNS if c.endswith("_pm")}
        | {"goals_minus_xg": "{:+.1f}", "possession_pm": "{:.0f}%"},
        na_rep="—",
    )  # fmt: skip
)
st.dataframe(
    styled,
    hide_index=True,
    use_container_width=True,
    height=35 * (len(table) + 1) + 3,
    column_config={
        "position": st.column_config.NumberColumn("#", width="small"),
        "team": "Equipo", "played": "PJ", "points": "Pts", "goals_for": "GF", "goals_against": "GC",
        "xg_pm": st.column_config.Column("xG", help="Goles esperados a favor por partido"),
        "xga_pm": st.column_config.Column("xGA", help="Goles esperados en contra por partido"),
        "goals_minus_xg": st.column_config.Column("G − xG", help="Goles marcados menos los esperados en la temporada"),
        "shots_total_pm": "Tiros", "shots_on_pm": "A puerta",
        "shots_on_against_pm": st.column_config.Column("A puerta rec.", help="Tiros a puerta recibidos"),
        "corners_pm": "Córners", "corners_against_pm": "Córners rec.", "yellow_pm": "Amarillas",
        "possession_pm": "Posesión",
    },
)  # fmt: skip
st.caption(
    "Azul más oscuro = mejor (en columnas *recibidos*, más oscuro = recibe menos). G − xG azul: marca más de lo "
    "que sus ocasiones justifican; rojo: menos. Orden por puntos; los desempates oficiales pueden variar."
)

# ---------------------------------------------------------------- xG for vs against
if has_xg:
    st.subheader("Calidad de ocasiones: generadas vs concedidas")
    ranked = table.sort_values("xg_pm", ascending=False).team.tolist()
    labelled = set(ranked[:4] + ranked[-2:])
    base = alt.Chart(table).encode(
        x=alt.X("xg_pm:Q", title="xG a favor por partido", scale=alt.Scale(zero=False)),
        y=alt.Y("xga_pm:Q", title="xG en contra por partido (mejor abajo)", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("team:N", title="Equipo"), alt.Tooltip("points:Q", title="Puntos"),
                 alt.Tooltip("xg_pm:Q", title="xG/PJ", format=".2f"),
                 alt.Tooltip("xga_pm:Q", title="xGA/PJ", format=".2f")],
    )  # fmt: skip
    points = base.mark_circle(size=110, color=BLUE, opacity=0.85, stroke="white", strokeWidth=2)
    labels = (
        alt.Chart(table[table.team.isin(labelled)])
        .mark_text(align="left", dx=9, fontSize=11, color="#52514e")
        .encode(x="xg_pm:Q", y="xga_pm:Q", text="team:N")
    )
    st.altair_chart((points + labels).properties(height=420), use_container_width=True)

# ---------------------------------------------------------------- one team
st.divider()
team = st.selectbox("Equipo", table.team.tolist(), key=f"team_{league}")
tm = view[view.team == team].reset_index(drop=True)
c1, c2, c3, c4 = st.columns(4)
row = table[table.team == team].iloc[0]
c1.metric("Puntos", int(row.points), help=f"{int(row.played)} partidos")
c2.metric("Goles", f"{int(row.goals_for)} – {int(row.goals_against)}")
if has_xg:
    c3.metric("xG – xGA por partido", f"{row.xg_pm:.2f} – {row.xga_pm:.2f}")
c4.metric("Córners por partido", f"{row.corners_pm:.1f} – {row.corners_against_pm:.1f}")

if has_xg and tm.xg.notna().sum() >= 3:
    window = 5
    trend = pd.DataFrame(
        {
            "Partido": range(1, len(tm) + 1),
            "Rival": tm.opponent,
            "xG a favor": tm.xg.rolling(window, min_periods=1).mean(),
            "xG en contra": tm.xg_against.rolling(window, min_periods=1).mean(),
        }
    ).melt(["Partido", "Rival"], var_name="Serie", value_name="xG")
    series = ["xG a favor", "xG en contra"]
    st.markdown(f"**Tendencia de xG** (media móvil de {window} partidos)")
    chart = (
        alt.Chart(trend)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=40))
        .encode(
            x=alt.X("Partido:Q", title="Partido (orden cronológico)", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("xG:Q", title="xG por partido"),
            color=alt.Color(
                "Serie:N", scale=color_scale(series), legend=alt.Legend(orient="top", title=None)
            ),
            tooltip=["Partido:Q", "Rival:N", "Serie:N", alt.Tooltip("xG:Q", format=".2f")],
        )
    )
    st.altair_chart(chart.properties(height=280), use_container_width=True)

st.markdown("**Partidos**")
log = tm.sort_values("kickoff_utc", ascending=False).assign(
    Resultado=lambda d: (
        d.goals_for.astype("Int64").astype(str) + "–" + d.goals_against.astype("Int64").astype(str)
    ),
    Campo=lambda d: d.is_home.map({True: "Local", False: "Visitante"}),
)
st.dataframe(
    log[["kickoff_utc", "opponent", "Campo", "Resultado", "xg", "xg_against", "shots_on", "shots_on_against",
         "corners", "corners_against", "yellow"]],
    hide_index=True,
    use_container_width=True,
    column_config={
        "kickoff_utc": st.column_config.DatetimeColumn("Fecha", format="DD/MM/YYYY"), "opponent": "Rival",
        "xg": st.column_config.NumberColumn("xG", format="%.2f"),
        "xg_against": st.column_config.NumberColumn("xGA", format="%.2f"),
        "shots_on": "A puerta", "shots_on_against": "A puerta rec.", "corners": "Córners",
        "corners_against": "Córners rec.", "yellow": "Amarillas",
    },
)  # fmt: skip
