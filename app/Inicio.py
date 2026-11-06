"""Streamlit entry point and navigation.  Run:  streamlit run app/Inicio.py

Two groups: the season in progress (tables, teams, players) and the betting tools built on
the models (predictions, bet-slip analyser).
"""

import streamlit as st
from common import freshness, sidebar_status

st.set_page_config(page_title="Fútbol Analytics", page_icon="⚽", layout="wide")

try:
    freshness()
except Exception as exc:  # DB down or not initialised: say so instead of a traceback
    st.error(f"No se pudo leer la base de datos: {exc}")
    st.stop()

nav = st.navigation(
    {
        "": [st.Page("views/inicio.py", title="Inicio", icon=":material/home:", default=True)],
        "Temporada en curso": [
            st.Page(
                "views/temporada.py", title="Clasificación y forma", icon=":material/leaderboard:"
            ),
            st.Page("views/equipos.py", title="Equipos", icon=":material/groups:"),
            st.Page("views/jugadores.py", title="Jugadores", icon=":material/person:"),
            st.Page("views/alineaciones.py", title="Alineaciones", icon=":material/stadium:"),
        ],
        "Apuestas": [
            st.Page("views/predicciones.py", title="Predicciones", icon=":material/insights:"),
            st.Page(
                "views/apuesta.py", title="Analizar boleto", icon=":material/document_scanner:"
            ),
        ],
    }
)
sidebar_status()
nav.run()
