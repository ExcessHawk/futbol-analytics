"""Players, current season (API-Football): per-90 leaderboard and a player's match log with
how often he cleared each betting line - the numbers behind a player prop."""

import altair as alt
import pandas as pd
import streamlit as st
from common import (
    BLUE,
    LEAGUE_NAMES,
    api_freshness_note,
    api_player_rows,
    api_season,
    api_season_label,
)
from styles import apply_columns, sequential

from futbol.api_stats import hit_rates, player_log, player_table

st.title("Jugadores")

POSITIONS = {"F": "Delanteros", "M": "Centrocampistas", "D": "Defensas", "G": "Porteros"}
STATS = {
    "shots_on": ("Tiros a puerta", [1, 2, 3, 4]),
    "shots_total": ("Tiros", [1, 2, 3, 4, 5]),
    "goals": ("Goles", [1, 2, 3]),
    "assists": ("Asistencias", [1, 2]),
    "yellow": ("Tarjetas amarillas", [1]),
    "fouls_committed": ("Faltas cometidas", [1, 2, 3]),
}

league = (
    st.segmented_control(
        "Liga", list(LEAGUE_NAMES), format_func=LEAGUE_NAMES.get, default="E0", key="players_league"
    )
    or "E0"
)
season = api_season(league)
if season is None:
    st.info(
        "Todavía no hay datos de API-Football para esta liga. Ejecuta `python scripts/download_api_football.py`."
    )
    st.stop()

rows = api_player_rows(league, season)
st.subheader(f"{LEAGUE_NAMES[league]} · {api_season_label(league, season)}")
st.caption(api_freshness_note(rows))

c1, c2 = st.columns([1, 2])
position = c1.selectbox(
    "Posición", list(POSITIONS), format_func=POSITIONS.get, key="players_position"
)
min_minutes = c2.slider("Minutos mínimos", 0, 1500, 270, step=90,
                        help="Con pocos minutos, las tasas por 90 son sobre todo ruido.")  # fmt: skip
table = player_table(rows, min_minutes)
pool = table[table.position == position]

PER90 = {"goals_p90": "Goles", "assists_p90": "Asist.", "shots_total_p90": "Tiros", "shots_on_p90": "A puerta",
         "key_passes_p90": "Pases clave", "yellow_p90": "Amarillas"}  # fmt: skip
view = pool[
    [
        "player",
        "sot_per_match",
        *PER90,
        "rating",
        "matches",
        "starts",
        "minutes",
    ]  # essentials first (phones)
].sort_values("shots_on_p90" if position in ("F", "M") else "rating", ascending=False)
st.dataframe(
    view.style.pipe(apply_columns, sequential, [*PER90, "sot_per_match", "rating"]).format(
        {c: "{:.2f}" for c in PER90} | {"sot_per_match": "{:.2f}", "rating": "{:.2f}", "minutes": "{:.0f}"},
        na_rep="—",
    ),
    hide_index=True,
    use_container_width=True,
    column_config={
        "player": "Jugador", "matches": "PJ", "starts": "Titular", "minutes": "Min",
        **{c: st.column_config.Column(label, help="Por 90 minutos") for c, label in PER90.items()},
        "sot_per_match": st.column_config.Column("A puerta / partido", help="Lo que usan las apuestas de jugador"),
        "rating": st.column_config.Column("Rating", help="Valoración media de API-Football, ponderada por minutos"),
    },
)  # fmt: skip
st.caption(
    f"{len(pool)} {POSITIONS[position].lower()} con al menos {min_minutes} minutos. Azul más oscuro = más alto."
)

# ---------------------------------------------------------------- one player
st.divider()
if pool.empty:
    st.stop()
names = pool.loc[view.index]  # same order as the leaderboard above: its leader comes first
pick = st.selectbox(
    "Jugador", names.player_id.tolist(), format_func=dict(zip(names.player_id, names.player, strict=True)).get,
    key=f"player_{league}_{position}",
)  # fmt: skip
p = pool[pool.player_id == pick].iloc[0]
log = player_log(rows, pick)
played = log[log.minutes.fillna(0) > 0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Partidos", f"{int(p.matches)} ({int(p.starts)} de titular)")
c2.metric("Goles", int(p.goals))
c3.metric("Tiros a puerta / partido", f"{p.sot_per_match:.2f}")
c4.metric("Rating", f"{p.rating:.2f}" if pd.notna(p.rating) else "—")

c1, c2 = st.columns([1, 2])
stat = c1.selectbox(
    "Estadística", list(STATS), format_func=lambda k: STATS[k][0], key="player_stat"
)
only_starts = c1.toggle("Solo partidos de titular", value=False, key="player_only_starts",
                        help="Una apuesta de jugador suele perderse si sale desde el banquillo.")  # fmt: skip
label, thresholds = STATS[stat]
rates = hit_rates(log, stat, thresholds, only_starts)
with c1:
    st.markdown(f"**¿Cuántas veces lo consiguió?** ({label.lower()})")
    st.dataframe(
        rates.assign(Linea=rates.threshold.map(lambda k: f"{k} o más"),
                     Veces=rates.hits.astype(str) + " de " + rates.matches.astype(str))[["Linea", "Veces", "rate"]],
        hide_index=True,
        use_container_width=True,
        column_config={"Linea": "Línea", "rate": st.column_config.ProgressColumn("Frecuencia", format="percent",
                                                                                  min_value=0, max_value=1)},
    )  # fmt: skip
    st.caption(
        "Es la frecuencia histórica de esta temporada, no una predicción: no ajusta por rival ni por campo."
    )

with c2:
    st.markdown(f"**{label} por partido**")
    series = played.sort_values("kickoff_utc").assign(
        n=lambda d: range(1, len(d) + 1),
        titular=lambda d: d.started.map({True: "Titular", False: "Suplente"}),
    )
    line = st.select_slider("Marca una línea", options=thresholds, value=thresholds[min(2, len(thresholds) - 1)],
                            key=f"line_{stat}")  # fmt: skip
    bars = (
        alt.Chart(series)
        .mark_bar(cornerRadiusEnd=3, color=BLUE)
        .encode(
            x=alt.X("n:O", title="Partido (orden cronológico)"),
            y=alt.Y(f"{stat}:Q", title=label, axis=alt.Axis(tickMinStep=1)),
            opacity=alt.condition(alt.datum.titular == "Titular", alt.value(1), alt.value(0.45)),
            tooltip=[alt.Tooltip("kickoff_utc:T", title="Fecha", format="%d/%m/%Y"),
                     alt.Tooltip("opponent:N", title="Rival"), alt.Tooltip("titular:N", title=" "),
                     alt.Tooltip("minutes:Q", title="Minutos"), alt.Tooltip(f"{stat}:Q", title=label)],
        )
    )  # fmt: skip
    rule = (
        alt.Chart(pd.DataFrame({"y": [line]}))
        .mark_rule(color="#e34948", strokeDash=[5, 4], strokeWidth=2)
        .encode(y="y:Q")
    )
    st.altair_chart((bars + rule).properties(height=260), use_container_width=True)
    st.caption("Barras claras = salió de suplente. Línea roja = la línea elegida.")

st.markdown("**Partido a partido**")
st.dataframe(
    log.assign(Titular=log.started.map({True: "Sí", False: "No"}))[
        ["kickoff_utc", "opponent", "venue", "Titular", "minutes", "shots_total", "shots_on", "goals", "assists",
         "yellow", "rating"]
    ],
    hide_index=True,
    use_container_width=True,
    column_config={
        "kickoff_utc": st.column_config.DatetimeColumn("Fecha", format="DD/MM/YYYY"), "opponent": "Rival",
        "venue": "Campo", "minutes": st.column_config.NumberColumn("Min", format="%d"),
        "shots_total": st.column_config.NumberColumn("Tiros", format="%d"),
        "shots_on": st.column_config.NumberColumn("A puerta", format="%d"),
        "goals": st.column_config.NumberColumn("Goles", format="%d"),
        "assists": st.column_config.NumberColumn("Asist.", format="%d"),
        "yellow": st.column_config.NumberColumn("Amarillas", format="%d"),
        "rating": st.column_config.NumberColumn("Rating", format="%.1f"),
    },
)  # fmt: skip
