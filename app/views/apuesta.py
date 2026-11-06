"""Bet-slip analyser: upload a photo of a slip, confirm what OCR read, get the model's view."""

import numpy as np
import pandas as pd
import streamlit as st
from common import (
    LEAGUE_NAMES,
    api_player_rows,
    api_season,
    forecaster,
    market_scores,
    market_verdict,
    player_lookup,
    read_slip,
    slip_fixtures,
    team_index,
)
from styles import diverging, sequential

from futbol.api_stats import find_player, prop_record
from futbol.betslip import (
    MARKET_NAMES,
    PICK_COLUMNS,
    PLAYER_MARKETS,
    SELECTION_NAMES,
    evaluate,
    odds_consistency,
    parse_meta,
    parse_slip_variants,
    risk_label,
    summarize,
    suspect_odds,
)
from futbol.ingest.api_football import has_key, per_match_average
from futbol.ingest.football_data import current_season_start
from futbol.sample_ticket import as_phone_photo, render_ticket, to_png_bytes

st.title("Analizar un boleto")
st.markdown(
    "Sube una foto o captura de tu boleto. **OpenCV** limpia la imagen, **Tesseract** lee el texto y la app "
    "reconoce los partidos, la selección y la cuota de cada apuesta. Luego compara cada cuota con la probabilidad "
    "del modelo."
)

# ---------------------------------------------------------------- 1. image
source = st.radio(
    "Imagen",
    ["Subir foto o captura", "Usar el boleto de ejemplo"],
    horizontal=True,
    key="slip_source",
)
images: list[bytes] = []
if source == "Subir foto o captura":
    uploads = st.file_uploader(
        "Boleto", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True,
        label_visibility="collapsed",
        help="Si el boleto no cabe en una captura, sube todas en orden (de arriba abajo).",
    )  # fmt: skip
    images = [u.getvalue() for u in uploads or []]
else:
    style = st.segmented_control(
        "Variante", ["Captura", "Captura en modo oscuro", "Foto con el celular"], default="Captura",
        key="slip_variant",
    ) or "Captura"  # fmt: skip
    ticket = render_ticket(dark=style == "Captura en modo oscuro")
    if style == "Foto con el celular":
        ticket = as_phone_photo(ticket)
    images = [to_png_bytes(ticket)]

if not images:
    st.info(
        "Sube una o varias capturas, o prueba con el boleto de ejemplo.",
        icon=":material/photo_camera:",
    )
    st.stop()

variant_lines: dict[str, list[str]] = {}
previews, confidences = [], []
try:
    for data in images:
        preview, lines_by_variant, conf = read_slip(data)
        previews.append(preview)
        confidences.append(conf)
        for name, lines in lines_by_variant.items():
            variant_lines.setdefault(name, []).extend(lines)
except Exception as exc:  # Tesseract missing or unreadable image: explain, don't crash
    st.error(f"No se pudo leer la imagen: {exc}")
    st.stop()
lines = variant_lines["clahe_otsu"]  # the reading shown to the user and used for the totals

cols = st.columns(min(len(images), 3))
for k, data in enumerate(images):
    cols[k % len(cols)].image(data, caption=f"Captura {k + 1}", use_container_width=True)
with st.expander("Ver lo que ve Tesseract tras OpenCV"):
    pcols = st.columns(min(len(previews), 3))
    for k, preview in enumerate(previews):
        pcols[k % len(pcols)].image(preview, use_container_width=True, clamp=True)
with st.expander(
    f"Texto leído ({len(lines)} líneas, confianza media {np.mean(confidences):.0f}%, "
    f"{len(variant_lines)} lecturas por imagen)"
):
    st.code("\n".join(lines) or "(nada)", language=None)

# ---------------------------------------------------------------- 2. confirm picks
idx = team_index()
league_of = {team: league for team, league in idx.choices.values()}
all_teams = sorted(league_of)
market_label = MARKET_NAMES
label_to_market = {v: k for k, v in market_label.items()}
label_to_selection = {v: k for k, v in SELECTION_NAMES.items()}

parsed = parse_slip_variants(list(variant_lines.values()), idx, slip_fixtures())
meta = parse_meta(lines)
if parsed.empty:
    st.warning(
        "No reconocí ningún partido. Puedes escribir las apuestas a mano en la tabla.",
        icon=":material/search_off:",
    )
    parsed = pd.DataFrame([{c: None for c in PICK_COLUMNS}])

# ---------------------------------------------------------------- player stats (stored API-Football data)
STAT_LABEL = {"player_sot": "tiros a puerta", "player_shots": "tiros", "player_goals": "goles",
              "player_cards": "amarillas"}  # fmt: skip
api_notes, api_error = [], None
player_rows = parsed.index[
    (parsed.bet_type == "Jugador") & parsed.player.notna() & parsed.league_code.notna()
]
for i in player_rows:
    r = parsed.loc[i]
    league_season = api_season(r.league_code)
    found = None
    if league_season is not None:
        rows_db = api_player_rows(r.league_code, league_season)
        found = find_player(rows_db, r.player, [r.home_team, r.away_team])
    if found:  # this season's match-by-match data is already in the database: no API call
        pid, full_name, side = found
        rec = prop_record(rows_db, pid, r.market, r.line if pd.notna(r.line) else 0.5)
        parsed.at[i, "player"] = full_name
        if side:
            parsed.at[i, "player_team"] = side
        parsed.at[i, "player_avg"] = rec["avg"]
        api_notes.append(
            f"**{full_name}**: {rec['avg']:.2f} {STAT_LABEL[r.market]} por partido; "
            f"{rec['need']} o más en **{rec['hits']} de {rec['matches']}** partidos esta temporada."
        )
    elif has_key():  # not stored (e.g. a league we don't download): ask the API
        candidates, api_error = player_lookup(r.player, r.league_code, current_season_start())
        if api_error:
            break
        if candidates.empty:
            api_notes.append(
                f"**{r.player}**: no aparece en API-Football para esta liga y temporada."
            )
            continue
        candidates = candidates.assign(
            fd_team=[(idx.match(t) or (None,))[0] for t in candidates.team]
        )
        in_match = candidates[candidates.fd_team.isin([r.home_team, r.away_team])]
        best = (in_match if not in_match.empty else candidates).sort_values("appearances").iloc[-1]
        avg = per_match_average(best, r.market)
        if best.fd_team in (r.home_team, r.away_team):
            parsed.at[i, "player_team"] = best.fd_team
        parsed.at[i, "player_avg"] = avg
        api_notes.append(
            f"**{best.player}** ({best.team}): {avg:.2f} {STAT_LABEL[r.market]} por partido."
        )

st.subheader("1. Revisa lo que leímos")
st.caption(
    "El OCR propone y tú confirmas: corrige cualquier celda o añade filas. Las cuotas americanas (+150, −120) ya "
    "están convertidas a decimales."
)
editable = pd.DataFrame(
    {
        "Tipo": parsed.bet_type.fillna("Partido"),
        "Local": parsed.home_team,
        "Visitante": parsed.away_team,
        "Mercado": parsed.market.map(market_label),
        "Selección": parsed.selection.map(SELECTION_NAMES),
        "Línea": parsed.line,
        "Cuota": parsed.odds,
        "Jugador": parsed.player,
        "Equipo del jugador": parsed.player_team,
        "Promedio por partido": pd.to_numeric(parsed.player_avg, errors="coerce"),
    }
)
edited = st.data_editor(
    editable,
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    key=f"slip_editor_{hash(tuple(images))}",  # a new set of images resets the edits
    column_config={
        "Tipo": st.column_config.SelectboxColumn(options=["Partido", "Jugador"], required=True, width="small"),
        "Local": st.column_config.SelectboxColumn(options=all_teams, required=True),
        "Visitante": st.column_config.SelectboxColumn(options=all_teams, required=True),
        "Mercado": st.column_config.SelectboxColumn(options=list(market_label.values()), required=True),
        "Selección": st.column_config.SelectboxColumn(options=list(SELECTION_NAMES.values()), required=True),
        "Línea": st.column_config.NumberColumn(help="Más/menos: 2.5, 9.5… · '3 o más' = 2.5", step=0.5, format="%.1f"),
        "Cuota": st.column_config.NumberColumn(help="Cuota decimal", min_value=1.01, step=0.01, format="%.2f"),
        "Jugador": st.column_config.TextColumn(help="Solo en apuestas de jugador"),
        "Equipo del jugador": st.column_config.SelectboxColumn(options=all_teams, help="Local o visitante"),
        "Promedio por partido": st.column_config.NumberColumn(
            help="Tiros a puerta / tiros / goles / amarillas del jugador por partido esta temporada",
            min_value=0.0, step=0.05, format="%.2f",
        ),
    },
)  # fmt: skip

if len(player_rows) or (edited["Tipo"] == "Jugador").any():
    if api_error:
        st.warning(f"API-Football respondió: *{api_error}*. Puedes escribir el promedio del jugador a mano.",
                   icon=":material/cloud_off:")  # fmt: skip
    elif not has_key():
        st.info(
            "Las apuestas de jugador necesitan su promedio de esta temporada. Para rellenarlo solo, crea una clave "
            "gratuita en [API-Football](https://dashboard.api-football.com) y ponla en `APIFOOTBALL_KEY` del `.env` "
            "(el plan gratuito limita las temporadas disponibles). Mientras tanto, escríbelo a mano.",
            icon=":material/key:",
        )
    for note in api_notes:
        st.caption(f":material/person: {note} Fuente: API-Football, partido a partido.")

picks = pd.DataFrame(
    {
        "bet_type": edited["Tipo"],
        "home_team": edited["Local"],
        "away_team": edited["Visitante"],
        "market": edited["Mercado"].map(label_to_market),
        "selection": edited["Selección"].map(label_to_selection),
        "line": pd.to_numeric(edited["Línea"], errors="coerce"),
        "odds": pd.to_numeric(edited["Cuota"], errors="coerce"),
        "player": edited["Jugador"],
        "player_team": edited["Equipo del jugador"],
        "player_avg": pd.to_numeric(edited["Promedio por partido"], errors="coerce"),
    }
).dropna(how="all", subset=["home_team", "away_team", "market", "odds"])
home_league, away_league = picks.home_team.map(league_of), picks.away_team.map(league_of)
picks["league_code"] = home_league.where(home_league == away_league)

base_total = meta.base_total
warning = odds_consistency(picks, base_total)
if warning and base_total:
    product = float(picks.odds.prod())
    missing = base_total / product
    suspect = suspect_odds(picks.reset_index(drop=True), base_total)
    if suspect:
        pos, fixed = suspect
        row = picks.reset_index(drop=True).iloc[pos]
        warning += (
            f" La sospechosa es **{row.home_team} vs {row.away_team}**: se leyó {row.odds:.2f} y, para que el "
            f"total cuadre, debería ser **{fixed:.2f}**. Corrígela en la tabla."
        )
    elif 1.05 < missing < 50:
        warning += (
            f" Parece que **falta una apuesta de cuota ≈ {missing:.2f}** (o hay una captura sin subir): "
            "añádela como fila nueva."
        )
    st.warning(warning, icon=":material/rule:")
elif base_total:
    boost_note = f" (antes del boost de +{meta.boost_pct:.0f}%)" if meta.boost_pct else ""
    st.caption(
        f":material/check_circle: Comprobado: las {len(picks)} cuotas multiplicadas dan {picks.odds.prod():.2f}, "
        f"igual que la cuota total del boleto{boost_note}."
    )

if picks.empty:
    st.stop()

# ---------------------------------------------------------------- 3. evaluation
st.subheader("2. Qué dice el modelo")
evaluated = evaluate(picks, forecaster)
scores = market_scores()


def pct(p: float) -> str:
    """Two significant figures, so a long parlay shows 0.044%, not 0.0%."""
    return f"{p:.1%}" if p >= 0.01 else f"{p * 100:.2g}%"


def money(x: float) -> str:
    """'+$157' / '−$12' / '$0' (no '-0' from rounding)."""
    r = round(x)
    return "$0" if r == 0 else f"{'+' if r > 0 else '−'}${abs(r):,}"


pending = evaluated[evaluated.problem.notna()]
if not pending.empty:
    items = "".join(
        f"\n- **{r.player if isinstance(r.player, str) and r.player else f'{r.home_team} vs {r.away_team}'}**: {r.problem}"
        for r in pending.itertuples()
    )
    st.warning(f"{len(pending)} apuesta(s) sin evaluar todavía:{items}", icon=":material/pending:")


def describe(row: pd.Series) -> str:
    sel = SELECTION_NAMES.get(row.selection, "?")
    if row.market == "1x2":
        return {
            "home": f"Gana {row.home_team}",
            "away": f"Gana {row.away_team}",
            "draw": "Empate",
        }.get(row.selection, sel)
    if row.market in ("goals", "corners", "yellow", "shots", "sot") and pd.notna(row.line):
        unit = {
            "goals": "goles",
            "corners": "córners",
            "yellow": "amarillas",
            "shots": "tiros",
            "sot": "tiros a puerta",
        }[row.market]
        return f"{sel} {row.line:g} {unit}"
    if row.market in PLAYER_MARKETS:
        who = row.player if isinstance(row.player, str) and row.player else "Jugador"
        if pd.isna(row.line):
            return f"{who}: {MARKET_NAMES[row.market]}"
        if row.market == "player_goals" and row.line == 0.5:
            return f"{who} marca" if row.selection == "over" else f"{who} no marca"
        if row.market == "player_cards" and row.line == 0.5:
            return f"{who} ve amarilla" if row.selection == "over" else f"{who} sin amarilla"
        unit = {"player_sot": "tiros a puerta", "player_shots": "tiros", "player_goals": "goles",
                "player_cards": "amarillas"}[row.market]  # fmt: skip
        if row.selection == "over" and float(row.line).is_integer() is False:
            return f"{who}: {int(row.line + 0.5)} o más {unit}"
        return f"{who}: {sel} {row.line:g} {unit}"
    if row.market == "btts":
        return f"Ambos marcan: {sel}"
    return f"{MARKET_NAMES.get(row.market, '?')}: {sel}"


def reliability(row: pd.Series) -> str:
    if pd.notna(row.problem):
        return "—"
    if row.market in ("1x2", "double"):
        return "El mercado acierta más que el modelo (1X2 validado)"
    if row.market in PLAYER_MARKETS:
        return f"Sin validar: promedio del jugador × rival ({row.opponent_factor:.2f})"
    key = {"btts": "goals"}.get(row.market, row.market)
    icon, text = market_verdict(scores, key)
    return (
        "No mejor que la media de la liga"
        if "no se distingue" in text
        else "Mejor que la media de la liga"
    )


def season_record(row: pd.Series) -> str:
    """'6 de 11 (55%)' for player props: how often he cleared this exact line this season."""
    if row.market not in PLAYER_MARKETS or pd.isna(row.line) or pd.isna(row.league_code):
        return ""
    league_season = api_season(row.league_code)
    if league_season is None or not isinstance(row.player, str):
        return ""
    rows_db = api_player_rows(row.league_code, league_season)
    found = find_player(rows_db, row.player, [row.home_team, row.away_team])
    if not found:
        return ""
    rec = prop_record(rows_db, found[0], row.market, row.line)
    if not rec["matches"]:
        return ""
    return f"{rec['hits']} de {rec['matches']} ({rec['hits'] / rec['matches']:.0%})"


view = pd.DataFrame(
    {  # decision columns first: on a phone only the first few fit without swiping
        "Apuesta": evaluated.apply(describe, axis=1),
        "Cuota": evaluated.odds,
        "p_model": evaluated.p_model,
        "Esta temporada": evaluated.apply(season_record, axis=1),
        "ev": evaluated.ev,
        "p_implied": evaluated.p_implied,
        "fair_odds": evaluated.fair_odds,
        "Partido": evaluated.home_team.fillna("?") + " vs " + evaluated.away_team.fillna("?"),
        "Liga": evaluated.league_code.map(LEAGUE_NAMES),
        "Fiabilidad del modelo aquí": evaluated.apply(reliability, axis=1),
        "Problema": evaluated.problem.fillna(""),
    }
)
styled = (
    view.style.apply(diverging, subset=["ev"], axis=0)
    .apply(sequential, subset=["p_model"], axis=0)
    .format(
        {
            "Cuota": "{:.2f}",
            "p_model": "{:.0%}",
            "p_implied": "{:.0%}",
            "fair_odds": "{:.2f}",
            "ev": "{:+.0%}",
        },
        na_rep="—",
    )  # fmt: skip
)
st.dataframe(
    styled,
    hide_index=True,
    use_container_width=True,
    column_config={
        "p_model": st.column_config.Column("Prob. modelo", help="Probabilidad de que la apuesta gane según el modelo"),
        "p_implied": st.column_config.Column("Prob. según la cuota", help="1 / cuota: lo que la casa de apuestas implica"),
        "fair_odds": st.column_config.Column("Cuota justa (modelo)", help="La cuota a la que el modelo no ganaría ni perdería"),
        "ev": st.column_config.Column(
            "Valor esperado", help="Ganancia media por cada peso apostado según el modelo. Azul = positivo, rojo = negativo"
        ),
    },
)  # fmt: skip

summary = summarize(evaluated)
if summary:
    st.subheader("3. Tu boleto en conjunto")
    as_parlay = st.toggle(
        "Es una combinada (hay que acertar todas)", value=summary.n_picks > 1, key="slip_parlay"
    )
    c_stake, c_boost = st.columns(2)
    stake = c_stake.number_input(
        "Apuesta ($)", min_value=1.0, value=float(meta.stake or 100.0), step=10.0, key="slip_stake"
    )
    boost = c_boost.number_input(
        "Boost de la casa (%)", min_value=0.0, value=float(meta.boost_pct or 0.0), step=5.0, key="slip_boost",
        help="Algunas casas suben la cuota de la combinada (p. ej. +80%). Cambia el pago y el valor esperado.",
    )  # fmt: skip
    boosted_odds = summary.odds * (1 + boost / 100)
    boosted_ev = summary.p_win * boosted_odds - 1
    ok = evaluated.dropna(subset=["p_model"])
    if not pending.empty:
        st.caption(
            f":material/warning: Este resumen cubre **{len(ok)} de {len(evaluated)}** apuestas. Con la(s) que faltan, "
            "la probabilidad real de acertar la combinada es **menor** que la que se muestra."
        )
    if as_parlay:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Probabilidad de acertar todo", pct(summary.p_win))
        c2.metric(
            "Cuota combinada", f"{boosted_odds:,.2f}",
            help=f"Sin boost: {summary.odds:,.2f}" if boost else None,
        )  # fmt: skip
        c3.metric("Cobrarías si aciertas", f"${stake * boosted_odds:,.0f}")
        c4.metric(
            "Resultado esperado",
            money(stake * boosted_ev),
            help="Promedio a largo plazo según el modelo",
        )
        if warning:
            st.caption(
                ":material/warning: Estos números usan cuotas que no cuadran con el total del boleto: corrige la "
                "tabla de arriba antes de fiarte de ellos."
            )
        st.markdown(
            f"**{risk_label(summary.p_win)}.** Según el modelo, este boleto se gana aproximadamente **1 de cada "
            f"{max(1, round(1 / summary.p_win)):,} veces**."
        )
        if summary.same_match_picks:
            st.warning(
                "Hay varias apuestas del mismo partido: no son independientes (si hay muchos goles, también es más "
                "probable que marquen ambos), así que la probabilidad combinada es solo aproximada.",
                icon=":material/link:",
            )
    else:
        expected_hits = float(ok.p_model.sum())
        profit = float((stake * ok.ev).sum())
        c1, c2 = st.columns(2)
        c1.metric("Aciertos esperados", f"{expected_hits:.1f} de {len(ok)}")
        c2.metric("Resultado esperado (apostando lo mismo en cada una)", money(profit))

    st.info(
        "**Cómo leer esto.** El modelo está bien calibrado, pero **el mercado de apuestas es más preciso** (lo medimos "
        "en 20,000 partidos: ver *Predicciones → Validación 1X2*). Si el modelo ve valor esperado positivo, lo más "
        "probable es que el modelo se equivoque, no la casa de apuestas. Los córners no son más predecibles que la "
        "media de la liga. Las casas cobran un margen en cada cuota, así que a largo plazo el resultado esperado de "
        "apostar suele ser negativo.",
        icon=":material/info:",
    )
