# Desarrollo

## Entorno

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e .
docker compose up -d
python scripts/check_env.py
```

Las versiones de `requirements.txt` están fijadas. `requirements-dev.txt` añade pytest, ruff y JupyterLab
(`notebooks/`).

## Pruebas y estilo

```bash
pytest                          # todo
pytest tests/test_betslip.py    # un módulo
pytest -k "simulator"           # por nombre
ruff check . && ruff format --check .
```

| Archivo | Qué cubre | Necesita base de datos |
|---|---|---|
| `test_models.py` | Ajuste, recuperación de parámetros conocidos, recién ascendidos, ρ | No |
| `test_markets.py` | Mercados de conteo, líneas, factor rival, apuestas de jugador | No |
| `test_validation.py` | Backtest sin fuga de datos, métricas, calibración | No |
| `test_metrics.py` | PPDA, pases progresivos, xG | No |
| `test_season.py` | Clasificación, forma, fase regular de Liga MX | No |
| `test_betslip.py` | OCR con boletos sintéticos, parser, cuotas, comprobaciones | No (sí Tesseract) |
| `test_real_slip.py` | Parser con el texto de un boleto real | No |
| `test_lineups.py` | Parser de alineaciones y bajas, alineación probable, su backtest, orientación del campo | No |
| `test_leagues_api.py` | Registro de ligas, transformaciones de API-Football | No |
| `test_config.py` | Variables de entorno | No |
| `test_app.py` | Cada vista con `streamlit.testing` (AppTest), en las 8 ligas; Alineaciones en sus dos modos | Sí |

Las pruebas que necesitan base de datos o Tesseract se saltan si no están disponibles.

## Convenciones

- **Idiomas:** código, comentarios y registros en inglés; todo lo que ve el usuario del dashboard, en español.
- **Funciones puras separadas de la E/S:** parsers, modelos y métricas reciben y devuelven DataFrames; la red y la base
  de datos se quedan en `ingest/`, `db.py`, `experiments.py` y `app/common.py`.
- **Configuración** solo desde `config.py` (`get_settings()`), nunca `os.getenv` disperso.
- **SQL** con parámetros con nombre (`read_sql("… WHERE league_code = :l", l=code)`).
- **Ligas** desde `futbol.leagues`, nunca listas de códigos escritas en las vistas.
- Líneas de 100 caracteres (ruff).

## Dashboard

- `app/Inicio.py` define la navegación (`st.navigation`). Una página nueva es un archivo en `app/views/` y una entrada
  en esa lista.
- **Toda consulta va en `app/common.py`** con `@st.cache_data(ttl=3600)` (o `cache_resource` para objetos como los
  modelos ajustados). Streamlit vuelve a ejecutar la vista completa en cada clic; sin caché, cada clic repetiría las
  consultas.
- Colores de tablas: `app/styles.py` (resultado y forma, divergentes centrados en 0, secuenciales, zonas de la
  clasificación) aplicados con `pandas.Styler`.
- Fichas de partido: `app/match_card.py` (`render` para próximos, `render_past` para jugados).
- Tablas en móvil: las columnas importantes primero, porque en pantallas estrechas el resto queda fuera y hay que
  desplazar la tabla.
- **Clics en gráficos:** `st.altair_chart(..., on_select="rerun")` con una `selection_point`. Vega no detecta
  bien los clics sobre imágenes, por eso el campo pone encima de cada foto un círculo casi transparente
  (`opacity=0.001`) que recibe el clic y el tooltip. AppTest no puede hacer clic en un gráfico: eso se comprobó con
  Chrome real.
- Para probar una vista nueva, añadirla a `test_app.py` siguiendo el patrón existente (`AppTest.from_file` +
  `switch_page`) y comprobar que no lanza excepciones con cada liga.

## Añadir un mercado de conteo

1. La columna por equipo en `fd_matches` (o en `api_team_match`) y en la ingesta correspondiente.
2. Una entrada en `MARKETS` (`src/futbol/markets.py`); `MatchForecaster` la ajusta automáticamente y omite las ligas
   que no tienen la columna.
3. Palabras clave en `MARKET_KEYWORDS`, nombres en `MARKET_NAMES` y selecciones en `VALID_SELECTIONS`
   (`betslip.py`), si debe reconocerse en boletos.
4. `python scripts/run_backtest.py --markets-only` y revisar su veredicto en **Predicciones → Fiabilidad** antes de
   mostrarlo como fiable.

## Añadir una liga

Ver [Datos → Añadir una liga](datos.md#añadir-una-liga).
