# Operación

## Scripts

Todos se ejecutan desde la raíz del proyecto con el entorno virtual activado. Son idempotentes: volver a ejecutarlos
actualiza datos, no los duplica.

| Script | Qué hace | Opciones |
|---|---|---|
| `check_env.py` | Comprueba la versión de Python, las dependencias y la conexión a la base de datos | — |
| `init_db.py` | Crea o actualiza el esquema (también lo hacen los demás scripts al arrancar) | — |
| `get_tessdata.py` | Descarga los modelos de idioma de Tesseract (spa, eng) a `data/tessdata/` | — |
| `ingest_football_data.py` | Resultados, estadísticas y cuotas de football-data | `--leagues E0 SP1 …` (por defecto las cinco grandes), `--from 2012`, `--to <temporada actual>` |
| `update.py` | Actualización diaria completa (ver abajo) | — |
| `download_api_football.py` | Estadísticas de equipo y jugador por partido | `--leagues`, `--seasons 2015 … 2026`, `--budget 3000` (máximo de peticiones) |
| `ingest_statsbomb.py` | Eventos StatsBomb 2015/16 | `--workers 8`, `--limit N` (pruebas), `--force` |
| `build_metrics.py` | Tablas derivadas de StatsBomb y cruce de equipos | — |
| `run_backtest.py` | Ajuste de hiperparámetros, backtest de todas las ligas, mercados, próximos partidos y fuerza de los equipos | `--skip-tuning`, `--leagues`, `--current-only`, `--markets-only` |

## Actualización diaria (`update.py`)

1. football-data: temporada en curso de las cinco grandes, partidos de la próxima semana y calendario de
   fixturedownload.
2. Liga MX, Argentina y Brasil (resultados y cuotas).
3. Elimina partidos aplazados que siguen como "sin jugar" con fecha pasada.
4. Predicciones walk-forward de la temporada en curso y de los próximos partidos, con los hiperparámetros guardados.
5. API-Football: partidos terminados nuevos de la temporada actual (máximo 400 peticiones). Si falta la clave o el plan
   ha caducado, lo registra y sigue: el resto de la actualización no se pierde.
6. Muestra la fecha del último partido jugado de cada liga.

Tarda ~1 minuto. Ejemplo de cron (la cuota de API-Football se reinicia a las 00:00 UTC):

```cron
0 6 * * * cd /ruta/futbol-analytics && .venv/bin/python scripts/update.py >> data/raw/update.log 2>&1
```

El backtest completo (`run_backtest.py` sin opciones) solo hace falta cuando cambia el modelo o se añade una liga.

## API-Football

| Plan | Peticiones | Temporadas |
|---|---|---|
| Free | 100/día, 10/min | 2022–2024 |
| Pro (19 USD/mes, sin renovación automática) | 7,500/día, 300/min | Todas |

- El cliente (`ingest/api_football.py`) usa como máximo el 40% del límite por minuto, deja una reserva de 50
  peticiones diarias y espera 10 s si la cabecera `X-RateLimit-Remaining` baja a 2.
- `/status` no consume cuota (`account_status()` en `ingest/api_football.py`); el registro de cada descarga muestra
  las peticiones usadas.
- Una petición `/fixtures?ids=` trae hasta 20 partidos con estadísticas, jugadores, alineaciones y eventos. La carga
  inicial de 8 ligas × 12 temporadas (~33,000 partidos) costó ~1,800 peticiones.
- Las respuestas originales se guardan en `data/raw/api_football/{liga}/{temporada}/{id}.json.gz`. Si el plan caduca,
  todo lo descargado sigue disponible; solo dejan de llegar partidos nuevos de la temporada actual.
- La búsqueda de jugadores del analizador de boletos se guarda 24 h en `api_player_cache`.

## Solución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| `ImportError` o `AttributeError` en el dashboard tras actualizar el código | Streamlit conserva en memoria la versión anterior de `common.py` o de `src/futbol` | Reiniciar el proceso de Streamlit |
| La app se queda cargando o muestra un error de conexión | PostgreSQL no está en marcha (la conexión se rinde a los 5 s) | `docker compose up -d`; comprobar `DB_PORT` en `.env` |
| Datos de ayer en la app después de `update.py` | Caché de una hora | Botón **Recargar datos** de la barra lateral |
| "No se pudo leer la imagen: … tesseract is not installed or it's not in your PATH" | Tesseract no instalado o no está en el `PATH` | `apt install tesseract-ocr`, o `TESSERACT_CMD` con la ruta completa |
| El OCR no reconoce texto en español | Faltan los modelos de idioma | `python scripts/get_tessdata.py` (o `TESSDATA_DIR`) |
| `Free plans do not have access to this season` | El plan Pro caducó | Renovarlo o limitarse a lo descargado; `update.py` sigue funcionando sin esa parte |
| `download_api_football.py` se detiene antes de terminar | Se alcanzó `--budget` o la reserva diaria | Volver a ejecutarlo: salta lo ya descargado |
| Liga sin próximos partidos | football-data publica `fixtures.csv` solo para la próxima semana; Liga MX, Argentina y Brasil no tienen calendario | Esperar a la siguiente actualización; en esas ligas el simulador sí funciona |
| Un partido aparece dos veces | Aplazamiento con fecha nueva | `update.py` lo limpia (`drop_stale_fixtures`) |
