"""Lineups on a pitch: the probable XI of an upcoming match (recent lineups minus the injury
report) or the real XI of a played one. Clicking a player opens his season numbers."""

import html

import pandas as pd
import streamlit as st
from common import (
    INJURY_REASONS_ES,
    LEAGUE_NAMES,
    api_played_with_lineups,
    api_player_rows,
    api_season,
    api_upcoming,
    fecha_es,
    fixture_lineups,
    local_time,
    player_photo_uri,
    team_injuries,
    team_recent_lineups,
)
from pitch import TEAM_COLORS, pitch_chart, place

from futbol.api_stats import player_log, player_table
from futbol.lineups import RECENT_MATCHES, probable_xi

st.title("Alineaciones")

POS_ES = {"G": "Portero", "D": "Defensa", "M": "Centrocampista", "F": "Delantero"}

league = (
    st.segmented_control(
        "Liga", list(LEAGUE_NAMES), format_func=LEAGUE_NAMES.get, default="E0", key="xi_league"
    )
    or "E0"
)
season = api_season(league)
if season is None:
    st.info("Todavía no hay datos de API-Football para esta liga.")
    st.stop()

mode = (
    st.segmented_control(
        "Partido",
        ["next", "played"],
        default="next",
        key="xi_mode",
        format_func={
            "next": "Próximo (alineación probable)",
            "played": "Jugado (alineación real)",
        }.get,
    )  # fmt: skip
    or "next"
)


def short(name: str) -> str:
    """'Kylian Mbappé' -> 'Mbappé' for the label under the photo."""
    parts = name.split()
    return parts[-1] if len(parts) > 1 else name


def fixture_label(r) -> str:
    return f"{fecha_es(local_time(r.kickoff_utc))} · {r.home_name} vs {r.away_name}"


# ---------------------------------------------------------------- fixture and lineups
sides = {}  # side -> dict(team_id, name, formation, players DataFrame, injuries, exact)
if mode == "next":
    fixtures = api_upcoming(league)
    if fixtures.empty:
        st.info("No hay próximos partidos guardados. Se descargan con `python scripts/update.py`.")
        st.stop()
    pick = st.selectbox(
        "Partido", fixtures.fixture_id.tolist(), key=f"xi_next_{league}",
        format_func=lambda f: fixture_label(fixtures[fixtures.fixture_id == f].iloc[0]),
    )  # fmt: skip
    fx = fixtures[fixtures.fixture_id == pick].iloc[0]
    for side in ("home", "away"):
        team_id, name = int(fx[f"{side}_id"]), fx[f"{side}_name"]
        history = team_recent_lineups(team_id, fx.kickoff_utc, RECENT_MATCHES)
        injuries, exact = team_injuries(team_id, int(fx.fixture_id), fx.kickoff_utc)
        out = set(injuries[injuries.kind == "Missing Fixture"].player_id)
        doubt = set(injuries[injuries.kind == "Questionable"].player_id)
        guess = probable_xi(history, out, doubt) if not history.empty else None
        players = guess.players if guess else pd.DataFrame()
        sides[side] = {"team_id": team_id, "name": name, "players": players, "injuries": injuries,
                       "exact": exact, "formation": guess.formation if guess else None,
                       "based_on": guess.based_on if guess else 0}  # fmt: skip
else:
    fixtures = api_played_with_lineups(league, season)
    if fixtures.empty:
        st.info("No hay alineaciones de partidos jugados esta temporada.")
        st.stop()
    pick = st.selectbox(
        "Partido", fixtures.fixture_id.tolist(), key=f"xi_played_{league}",
        format_func=lambda f: (lambda r: f"{fixture_label(r)} ({int(r.home_goals)}-{int(r.away_goals)})")(
            fixtures[fixtures.fixture_id == f].iloc[0]),
    )  # fmt: skip
    fx = fixtures[fixtures.fixture_id == pick].iloc[0]
    lineup = fixture_lineups(int(fx.fixture_id))
    for side in ("home", "away"):
        team_id = int(fx[f"{side}_id"])
        team = lineup[lineup.team_id == team_id]
        sides[side] = {"team_id": team_id, "name": fx[f"{side}_name"],
                       "players": team[team.started & team.grid.notna()].assign(doubtful=False),
                       "bench": team[~team.started], "formation": team.formation.dropna().iloc[0]
                       if team.formation.notna().any() else None,
                       "coach": team.coach.dropna().iloc[0] if team.coach.notna().any() else None}  # fmt: skip

# ---------------------------------------------------------------- pitch
# Desktop: pitch on the left, the selected player on the right. Phone: stacked.
left, right = st.columns([3, 2], gap="large")


def team_header(side: str) -> None:
    s = sides[side]
    arrow = "▼" if side == "away" else "▲"  # away attacks down, home attacks up
    left.markdown(
        f"<span style='color:{TEAM_COLORS[side]};font-size:1.15rem;font-weight:700'>{arrow} {s['name']}"
        f"</span> · {s['formation'] or 'sin datos'}"
        + (f" · {s['coach']}" if s.get("coach") else ""),
        unsafe_allow_html=True,
    )


frames = []
for side, s in sides.items():
    p = s["players"]
    if p.empty or "grid" not in p or p.grid.isna().all():
        continue
    p = p[p.grid.notna()].assign(side=side, formation=s["formation"])
    if mode == "next":
        p["tt_Titular en sus últimos partidos"] = (p.confidence * 100).round().astype(int).astype(
            str
        ) + "%"
        p["tt_Estado"] = p.doubtful.map({True: "Duda", False: "Disponible"})
    else:
        p["tt_Rating"] = p.rating.map(lambda r: f"{r:.1f}" if pd.notna(r) else "—")
        p["tt_Minutos"] = p.minutes.fillna(0).astype(int)
        p["tt_Goles"] = p.goals.fillna(0).astype(int)
    p["tt_Jugador"] = p.player_name
    p["tt_Posición"] = p.pos.map(POS_ES)
    frames.append(p)
if not frames:
    st.info("No hay alineaciones recientes de estos equipos para estimar el once.")
    st.stop()
on_pitch = place(pd.concat(frames, ignore_index=True))
on_pitch["player_name"] = on_pitch.player_name.map(html.unescape)  # the API sends "O&apos;Brien"
on_pitch["label"] = on_pitch.player_name.map(short)
on_pitch["photo"] = [
    player_photo_uri(int(pid), name, TEAM_COLORS[side])
    for pid, name, side in zip(on_pitch.player_id, on_pitch.player_name, on_pitch.side, strict=True)
]
on_pitch["player_id"] = on_pitch.player_id.astype(int)
on_pitch["doubtful"] = on_pitch.doubtful.astype(bool)
chart_cols = ["x", "y", "side", "player_id", "label", "photo", "doubtful",
              *[c for c in on_pitch.columns if c.startswith("tt_")]]  # fmt: skip
team_header("away")
event = left.altair_chart(
    pitch_chart(on_pitch[chart_cols]), use_container_width=True, on_select="rerun",
    selection_mode="player", key=f"pitch_{pick}",
)  # fmt: skip
team_header("home")

if mode == "next":
    left.caption(
        f"Alineación **probable**: la formación del último partido y, en cada puesto, quien más lo ha "
        f"ocupado en los últimos {RECENT_MATCHES} partidos de liga, sin los lesionados ni sancionados. "
        "Comprobado con los partidos de 2026: acierta de media **8.3 de los 11 titulares** (repetir el "
        "once del último partido acierta 8.2). La oficial se publica ~30 minutos antes del partido. "
        "Borde amarillo = duda."
    )
else:
    left.caption(
        "Alineación real. Pasa el cursor (o toca) sobre un jugador para ver su rating en el partido."
    )

# ---------------------------------------------------------------- injuries / bench
c1, c2 = st.columns(2)
for col, side in ((c1, "home"), (c2, "away")):
    s = sides[side]
    with col:
        if mode == "next":
            inj = s["injuries"]
            st.markdown(f"**Bajas · {s['name']}**")
            if inj.empty:
                st.caption(
                    "Sin bajas reportadas."
                    if league not in ("MEX", "ARG")
                    else "API-Football no publica bajas de esta liga."
                )
                continue
            if not s["exact"]:
                when = fecha_es(local_time(inj.kickoff_utc.iloc[0]), with_time=False)
                st.caption(f"Aún no hay parte para este partido: es el del partido del {when}.")
            table = inj.assign(
                Estado=inj.kind.map({"Missing Fixture": "Baja", "Questionable": "Duda"}).fillna(
                    inj.kind
                ),
                Motivo=inj.reason.map(lambda r: INJURY_REASONS_ES.get(r, r)),
            )[["player_name", "Estado", "Motivo"]].rename(columns={"player_name": "Jugador"})
            table["Jugador"] = table.Jugador.map(html.unescape)
            st.dataframe(table, hide_index=True, use_container_width=True)
        else:
            bench = s["bench"]
            used = bench[bench.minutes.fillna(0) > 0]
            st.markdown(f"**Suplentes que jugaron · {s['name']}**")
            if used.empty:
                st.caption("Ninguno.")
            else:
                st.dataframe(
                    used[["player_name", "minutes", "rating"]].sort_values("minutes", ascending=False),
                    hide_index=True, use_container_width=True,
                    column_config={"player_name": "Jugador", "minutes": "Min",
                                   "rating": st.column_config.NumberColumn("Rating", format="%.1f")},
                )  # fmt: skip

# ---------------------------------------------------------------- one player
with right:
    options = (
        on_pitch.assign(order=on_pitch.side.map({"home": 0, "away": 1}))
        .sort_values(["order", "row", "col"])
        .player_id.tolist()
    )
    names = dict(zip(on_pitch.player_id, on_pitch.player_name, strict=True))
    select_key = f"xi_player_{pick}"
    clicked = (event.selection.get("player") or [{}])[0].get("player_id") if event else None
    # A click on the pitch moves the selector, but only when it is a new click: the chart keeps
    # its selection across reruns and would otherwise pin the selector forever.
    if clicked is not None and st.session_state.get(f"{select_key}_last_click") != clicked:
        st.session_state[f"{select_key}_last_click"] = clicked
        if clicked in options:
            st.session_state[select_key] = clicked
    player_id = st.selectbox(
        "Jugador (toca uno en el campo o elígelo aquí)",
        options,
        format_func=names.get,
        key=select_key,
    )
    row = on_pitch[on_pitch.player_id == player_id].iloc[0]
    team_id = sides[row.side]["team_id"]

    rows = api_player_rows(league, season)
    mine = rows[(rows.player_id == player_id) & (rows.team_id == team_id)]
    c_photo, c_info = st.columns([1, 4])
    c_photo.image(row.photo, width=96)
    with c_info:
        st.markdown(f"### {row.player_name}")
        st.caption(f"{sides[row.side]['name']} · {POS_ES.get(row.get('pos'), '')}")
        if mode == "next":
            inj = sides[row.side]["injuries"]
            note = inj[inj.player_id == player_id]
            if not note.empty:
                st.warning(
                    f"En el parte: {INJURY_REASONS_ES.get(note.reason.iloc[0], note.reason.iloc[0])}"
                )
            st.caption(
                f"Titular en el {row.confidence:.0%} de sus últimos partidos (ponderado por cercanía)."
            )

    if mine.empty:
        st.info("Sin partidos con este equipo en la temporada.")
        st.stop()
    season_row = player_table(mine).iloc[0]
    # Two rows: the right-hand column is narrow on desktop, five metrics would be cut off.
    m = st.columns(3)
    m[0].metric("Partidos", int(season_row.matches), help=f"{int(season_row.starts)} de titular")
    m[1].metric("Goles", int(season_row.goals))
    m[2].metric("Asist.", int(season_row.assists))
    m = st.columns(3)
    m[0].metric("A puerta/PJ", f"{season_row.sot_per_match:.2f}")
    m[1].metric("Rating", f"{season_row.rating:.2f}" if pd.notna(season_row.rating) else "—")
    m[2].metric("Minutos", f"{int(season_row.minutes)}")

    log = player_log(mine, player_id).head(8)
    log["Partido"] = [
        f"{fecha_es(local_time(k), with_time=False)} · {'vs' if v == 'Local' else 'en'} {o}"
        for k, v, o in zip(log.kickoff_utc, log.venue, log.opponent, strict=True)
    ]
    log["Titular"] = log.started.map({True: "Sí", False: "No"})
    st.markdown("**Últimos partidos**")
    st.dataframe(
        log[["Partido", "Titular", "minutes", "goals", "assists", "shots_on", "shots_total", "yellow", "rating"]],
        hide_index=True,
        use_container_width=True,
        column_config={"minutes": "Min", "goals": "Goles", "assists": "Asist.", "shots_on": "A puerta",
                       "shots_total": "Tiros", "yellow": "Amarillas",
                       "rating": st.column_config.NumberColumn("Rating", format="%.1f")},
    )  # fmt: skip
