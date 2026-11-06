"""Home view: what the dashboard covers and where each number comes from."""

import pandas as pd
import streamlit as st
from common import LEAGUE_NAMES, freshness, scores

from futbol.db import read_sql

f = freshness()

st.title("⚽ Fútbol Analytics")
st.markdown(
    f"{len(LEAGUE_NAMES)} ligas ({', '.join(LEAGUE_NAMES.values())}), su temporada en curso y un modelo de "
    "predicción **Dixon-Coles** validado partido a partido contra los resultados reales y contra las cuotas de "
    "cierre de las casas de apuestas."
)

counts = read_sql(
    "SELECT (SELECT count(*) FROM fd_matches WHERE home_goals IS NOT NULL) AS results, "
    "(SELECT count(*) FROM api_fixture) AS api_fixtures, "
    "(SELECT count(*) FROM api_player_match) AS player_rows"
).iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("Partidos con resultado y cuotas", f"{counts.results:,}",
          help=f"Desde 2012. Último partido: {pd.Timestamp(f.last_played):%d/%m/%Y}")  # fmt: skip
c2.metric(
    "Partidos con estadísticas detalladas",
    f"{counts.api_fixtures:,}",
    help="API-Football, desde 2015",
)
c3.metric(
    "Actuaciones de jugadores",
    f"{counts.player_rows:,}",
    help="Un jugador en un partido = una fila",
)

st.subheader("Temporada en curso")
st.markdown(
    """
- **Clasificación y forma**: tabla, forma reciente y puntos frente a los *puntos esperados* según las cuotas;
  en Liga MX, por torneo (Apertura / Clausura) sin contar la liguilla.
- **Equipos**: xG y xG en contra, tiros, tiros a puerta, córners y tarjetas a favor y en contra, por partido.
- **Jugadores**: ranking por 90 minutos y la ficha de cada jugador partido a partido, con **cuántas veces superó
  cada línea** (p. ej. "3 o más tiros a puerta en 6 de 11 partidos").
"""
)
st.subheader("Apuestas")
st.markdown(
    """
- **Predicciones**: próximos partidos con 1X2, goles, marcadores, córners y tarjetas; simulador de cualquier
  enfrentamiento; historial con lo que predijo el modelo; y la fiabilidad medida de cada mercado.
- **Analizar boleto**: sube capturas de tu boleto y la app lee las apuestas (OCR) y las compara con el modelo.
"""
)
st.caption(
    "Fuentes: football-data.co.uk (resultados y cuotas), API-Football (estadísticas de partidos y jugadores, xG), "
    "fixturedownload.com (calendario de las ligas europeas)."
)

try:
    s = scores("multi_season").set_index("model")
    dc, mk = s.loc["dixon_coles_goals"], s.loc["market"]
    st.info(
        f"**Resultado principal:** en {int(dc.matches):,} partidos desde 2015, Dixon-Coles obtiene un log loss de "
        f"{dc.log_loss:.4f} frente a {mk.log_loss:.4f} del mercado: está bien calibrado, pero no supera a las casas "
        "de apuestas. Detalle en *Predicciones → Validación 1X2*."
    )
except Exception:
    st.caption("Ejecuta `scripts/run_backtest.py` para ver los resultados del modelo.")
