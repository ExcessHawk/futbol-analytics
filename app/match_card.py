"""Full forecast for one match: 1X2, goals, scorelines, corners, cards, shots.

Used by the upcoming-fixtures detail and by the simulator, so both look the same.
Every market block shows its backtest verdict next to the numbers.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from common import BLUE, ORANGE, market_verdict

from futbol.markets import MARKETS, MatchForecaster, p_total_over

DRAW_GREY = "#a3a29c"
SHOWN_GOALS = 5  # scoreline grid 0..5 (everything above is <2% combined)


def probability_bar(p: dict[str, float], home: str, away: str) -> alt.LayerChart:
    """One stacked bar: home / draw / away, each segment labelled with its percentage."""
    df = pd.DataFrame(
        {
            "Resultado": [f"Gana {home}", "Empate", f"Gana {away}"],
            "p": [p["p_home"], p["p_draw"], p["p_away"]],
            "orden": [0, 1, 2],
        }
    )
    df["start"] = df.p.cumsum() - df.p
    df["mid"] = df.start + df.p / 2
    color = alt.Color(
        "Resultado:N",
        scale=alt.Scale(domain=df.Resultado.tolist(), range=[BLUE, DRAW_GREY, ORANGE]),
        legend=alt.Legend(orient="top", title=None),
    )
    bars = (
        alt.Chart(df)
        .mark_bar(height=34, stroke="white", strokeWidth=2)
        .encode(
            x=alt.X("start:Q", axis=None, scale=alt.Scale(domain=[0, 1])),
            x2="end:Q",
            color=color,
            tooltip=["Resultado:N", alt.Tooltip("p:Q", format=".1%", title="Probabilidad")],
        )
        .transform_calculate(end="datum.start + datum.p")
    )
    labels = (
        alt.Chart(df[df.p >= 0.08])
        .mark_text(color="white", fontWeight="bold", fontSize=13)
        .encode(x="mid:Q", text=alt.Text("p:Q", format=".0%"))
    )
    return (bars + labels).properties(height=70)


def score_heatmap(matrix: np.ndarray, home: str, away: str) -> alt.LayerChart:
    """P(exact score) grid; darker = more likely, text = probability."""
    n = SHOWN_GOALS + 1
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    df = pd.DataFrame({"home": i.ravel(), "away": j.ravel(), "p": matrix[:n, :n].ravel()})
    df["score"] = df.home.astype(str) + "-" + df.away.astype(str)
    base = alt.Chart(df).encode(
        x=alt.X("away:O", title=f"Goles de {away}"),
        y=alt.Y("home:O", title=f"Goles de {home}", sort="descending"),
    )
    rect = base.mark_rect(stroke="white", strokeWidth=2).encode(
        # interpolate in RGB: Altair's default (HCL) drifts through teal between these two blues.
        color=alt.Color(
            "p:Q", scale=alt.Scale(range=["#fcfcfb", "#1c5cab"], interpolate="rgb"), legend=None
        ),
        tooltip=[alt.Tooltip("score:N", title="Marcador"), alt.Tooltip("p:Q", format=".1%")],
    )
    text = base.mark_text(fontSize=11).encode(
        text=alt.Text("p:Q", format=".0%"),
        color=alt.condition("datum.p > 0.09", alt.value("white"), alt.value("#0b0b0b")),
    )
    return (rect + text).properties(height=300)


def over_table(lam: float, mu: float, center: float, spread: int) -> pd.DataFrame:
    """P(over) for a few lines around the league's usual line (under = 1 - over)."""
    lines = [center + k for k in range(-spread, spread + 1)]
    p = [float(p_total_over(np.array([lam]), np.array([mu]), line)[0]) for line in lines]
    return pd.DataFrame({"Línea": [f"Más de {line:g}" for line in lines], "Probabilidad": p})


PCT = st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)


def verdict(scores: pd.DataFrame, market: str) -> None:
    icon, text = market_verdict(scores, market)
    st.caption(f"{icon} Fiabilidad en el backtest: {text}")


def render(
    fc: MatchForecaster,
    home: str,
    away: str,
    scores: pd.DataFrame,
    referee: str | None = None,
    odds_over25: float | None = None,
    odds_under25: float | None = None,
) -> None:
    fixture = pd.DataFrame({"home_team": [home], "away_team": [away], "referee": [referee]})
    f = fc.forecast(fixture).iloc[0]

    st.markdown(f"#### {home} vs {away}")
    st.altair_chart(probability_bar(f.to_dict(), home, away), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Goles esperados", f"{f.exp_home_goals:.2f} – {f.exp_away_goals:.2f}")
    c2.metric("Marcador más probable", f"{f.top_score} ({f.p_top_score:.0%})")
    c3.metric("Más de 2.5 goles", f"{f.p_over_2_5:.0%}")
    c4.metric("Ambos marcan", f"{f.p_btts:.0%}")
    if odds_over25 and odds_under25:
        inv_o, inv_u = 1 / odds_over25, 1 / odds_under25
        c3.caption(f"Mercado: {inv_o / (inv_o + inv_u):.0%}")
    verdict(scores, "goals")

    m = fc.score_matrix(home, away)
    left, right = st.columns([3, 2])
    with left:
        st.altair_chart(score_heatmap(m, home, away), use_container_width=True)
    with right:
        st.markdown("**Marcadores más probables**")
        flat = [(f"{i}-{j}", m[i, j]) for i in range(m.shape[0]) for j in range(m.shape[1])]
        top = pd.DataFrame(
            sorted(flat, key=lambda t: -t[1])[:6], columns=["Marcador", "Probabilidad"]
        )
        st.dataframe(
            top, hide_index=True, use_container_width=True, column_config={"Probabilidad": PCT}
        )

    st.divider()
    cc, cy, cs = st.columns(3)
    with cc:
        count_block(f, "corners", "Córners", scores)
    with cy:
        extra = None
        if referee and fc.referee_factor.get(referee):
            extra = (
                f"Árbitro {referee}: ×{fc.referee_factor[referee]:.2f} sobre la media de tarjetas."
            )
        count_block(f, "yellow", "Tarjetas amarillas", scores, extra)
    with cs:
        count_block(f, "shots", "Tiros", scores)
    st.caption("Probabilidad de *menos de* una línea = 100% − probabilidad de *más de* esa línea.")


def count_block(
    f: pd.Series, key: str, title: str, scores: pd.DataFrame, extra: str | None = None
) -> None:
    """Expected home/away/total for one count market, P(over) around its line, verdict."""
    st.markdown(f"**{title}**")
    if f"exp_home_{key}" not in f or pd.isna(f[f"exp_home_{key}"]):
        st.caption(
            ":material/block: Sin datos de este mercado para esta liga todavía (la fuente gratuita solo trae "
            "resultados y cuotas). Llegará con las estadísticas de API-Football."
        )
        return
    home, away = f[f"exp_home_{key}"], f[f"exp_away_{key}"]
    st.metric("Esperados (local – visitante)", f"{home:.1f} – {away:.1f}")
    st.caption(f"Total esperado: **{home + away:.1f}**")
    st.dataframe(
        over_table(home, away, f[f"line_{key}"], 2),
        hide_index=True,
        use_container_width=True,
        column_config={"Probabilidad": PCT},
    )
    if extra:
        st.caption(extra)
    verdict(scores, key)


# --------------------------------------------------------------------------- played matches

STATS = [
    ("Goles", "home_goals", "away_goals"),
    ("Tiros", "home_shots", "away_shots"),
    ("Tiros a puerta", "home_sot", "away_sot"),
    ("Córners", "home_corners", "away_corners"),
    ("Faltas", "home_fouls", "away_fouls"),
    ("Amarillas", "home_yellow", "away_yellow"),
    ("Rojas", "home_red", "away_red"),
]
MARKET_ROWS = [
    ("goals", "Goles"),
    ("shots", "Tiros"),
    ("corners", "Córners"),
    ("yellow", "Amarillas"),
]


def stat_bars(match: pd.Series, home: str, away: str) -> alt.LayerChart:
    """Head-to-head bars: each stat split by share, home (blue) left, away (orange) right."""
    rows = []
    for order, (label, h_col, a_col) in enumerate(STATS):
        h, a = match.get(h_col), match.get(a_col)
        if pd.isna(h) or pd.isna(a):
            continue
        h, a = int(h), int(a)
        share = 0.5 if h + a == 0 else h / (h + a)
        rows.append({"stat": label, "order": order, "home": h, "away": a, "share": share})
    df = pd.DataFrame(rows)
    y = alt.Y(
        "stat:N",
        sort=alt.SortField("order"),
        title=None,
        axis=alt.Axis(labelFontSize=12, ticks=False, domain=False),
    )
    home_bar = alt.Chart(df).mark_bar(color=BLUE, height=16).encode(
        x=alt.X("zero:Q", scale=alt.Scale(domain=[-0.12, 1.12]), axis=None), x2="share:Q", y=y,
        tooltip=["stat:N", alt.Tooltip("home:Q", title=home)],
    ).transform_calculate(zero="0")  # fmt: skip
    away_bar = alt.Chart(df).mark_bar(color=ORANGE, height=16).encode(
        x="share:Q", x2="one:Q", y=y, tooltip=["stat:N", alt.Tooltip("away:Q", title=away)],
    ).transform_calculate(one="1")  # fmt: skip
    home_txt = alt.Chart(df).mark_text(align="right", dx=-6, fontWeight="bold", fontSize=13).encode(
        x="zero:Q", y=y, text="home:Q"
    ).transform_calculate(zero="0")  # fmt: skip
    away_txt = alt.Chart(df).mark_text(align="left", dx=6, fontWeight="bold", fontSize=13).encode(
        x="one:Q", y=y, text="away:Q"
    ).transform_calculate(one="1")  # fmt: skip
    return (home_bar + away_bar + home_txt + away_txt).properties(height=32 * len(df))


def outcome_label(h: int, a: int) -> str:
    return "p_home" if h > a else "p_draw" if h == a else "p_away"


def render_past(match: pd.Series, one_x_two: pd.DataFrame, counts: pd.DataFrame) -> None:
    home, away = match.home_team, match.away_team
    hg, ag = int(match.home_goals), int(match.away_goals)
    meta = [f"{pd.Timestamp(match.match_date):%d/%m/%Y}"]
    if pd.notna(match.get("round")):
        meta.append(f"jornada {int(match['round'])}")
    if isinstance(match.get("referee"), str):
        meta.append(f"árbitro {match.referee}")
    st.markdown(f"#### {home} {hg} – {ag} {away}")
    st.caption(" · ".join(meta))

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Estadísticas del partido**")
        st.html(
            f'<div style="display:flex;justify-content:space-between;font-size:0.9rem;margin:0 8% -4px 22%">'
            f'<span style="color:{BLUE};font-weight:700">{home}</span>'
            f'<span style="color:{ORANGE};font-weight:700">{away}</span></div>'
        )
        st.altair_chart(stat_bars(match, home, away), use_container_width=True)

    sel = (one_x_two.home_team == home) & (one_x_two.away_team == away)
    pre = one_x_two[sel].set_index("model")
    with right:
        st.markdown("**Antes del partido**")
        if "dixon_coles_goals" not in pre.index:
            st.info(
                "Sin predicción para este partido (temporadas anteriores a 2015/16 o primeras jornadas)."
            )
        else:
            happened = outcome_label(hg, ag)
            model = pre.loc["dixon_coles_goals"]
            st.altair_chart(probability_bar(model.to_dict(), home, away), use_container_width=True)
            text = f"El modelo le daba **{model[happened]:.0%}** a lo que pasó"
            if "market" in pre.index:
                text += f"; el mercado, **{pre.loc['market', happened]:.0%}**"
            st.markdown(text + ".")
            if pd.notna(model.exp_home):
                st.caption(
                    f"Goles esperados por el modelo: {model.exp_home:.2f} – {model.exp_away:.2f}"
                )

    sel = (counts.home_team == home) & (counts.away_team == away)
    pre_counts = counts[sel].set_index("market")
    if pre_counts.empty:
        return
    rows = []
    for key, label in MARKET_ROWS:
        if key not in pre_counts.index:
            continue
        m = MARKETS[key]
        p = pre_counts.loc[key]
        actual_h, actual_a = match.get(m.home_col), match.get(m.away_col)
        if pd.isna(actual_h) or pd.isna(actual_a):
            continue
        total = int(actual_h) + int(actual_a)
        rows.append(
            {
                "Mercado": label,
                "Esperado (L – V)": f"{p.exp_home:.1f} – {p.exp_away:.1f}",
                "Real (L – V)": f"{int(actual_h)} – {int(actual_a)}",
                "Línea": f"Más de {p.line:g}",
                "P(modelo)": p.p_over,
                "¿Pasó?": "Sí" if total > p.line else "No",
            }
        )
    st.markdown("**Lo que predijo el modelo frente a lo que pasó**")
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
        column_config={"P(modelo)": PCT},
    )
