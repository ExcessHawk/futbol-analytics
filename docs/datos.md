# Datos

## Fuentes

| Fuente | Cobertura | Contenido | Módulo |
|---|---|---|---|
| football-data.co.uk, ligas principales (`mmz4281/{temporada}/{liga}.csv`) | Premier, LaLiga, Bundesliga, Serie A, Ligue 1 desde 2012/13 | Resultado, tiros, tiros a puerta, córners, faltas, amarillas, rojas, árbitro (Premier), cuotas 1X2 y más/menos 2.5 | `ingest/football_data.py` |
| football-data.co.uk, ligas extra (`new/{liga}.csv`) | Liga MX, Argentina, Brasil desde 2012 | Resultado y cuotas 1X2 de cierre (sin estadísticas) | `ingest/football_data.py` |
| football-data.co.uk, `fixtures.csv` | Próxima semana | Partidos sin jugar con cuotas previas | `ingest/football_data.py` |
| API-Football (`/fixtures?ids=`) | 8 ligas desde 2015 (Liga MX desde 2016) | Estadísticas de equipo por partido (con xG), de cada jugador (minutos, titularidad, tiros, a puerta, goles, tarjetas, rating), alineaciones y eventos | `ingest/api_fixtures.py` |
| fixturedownload.com | 5 ligas europeas, temporada en curso | Calendario completo: jornada, hora UTC, estadio | `ingest/schedule.py` |
| StatsBomb Open Data | Premier, LaLiga, Serie A, Ligue 1 2015/16 | Eventos con coordenadas | `ingest/statsbomb.py` |

### Disponibilidad del xG de API-Football

| Ligas | xG por partido |
|---|---|
| Cinco grandes de Europa | desde 2022 |
| Brasil, Argentina | desde 2024 |
| Liga MX | temporada 2026/27 |

### Cuotas

De cada partido se toma el primer conjunto completo según esta preferencia: cierre de Pinnacle → media de cierre del
mercado → apertura de Pinnacle → media del mercado → otros. La columna `odds_source` registra cuál se usó. Las cuotas
de cierre son la referencia contra la que se mide el modelo, porque son la estimación pública más afinada de la
probabilidad real.

## Esquema

El esquema completo está en [`sql/schema.sql`](../sql/schema.sql). Es idempotente (se puede ejecutar en cada arranque)
e incluye sus migraciones. Las tablas principales:

### Partidos y resultados

**`fd_matches`**: un partido de football-data. Clave: `(league_code, season, home_team, away_team, match_date)`.

| Columna | Notas |
|---|---|
| `league_code` | `E0`, `SP1`, `D1`, `I1`, `F1`, `MEX`, `ARG`, `BRA` |
| `season` | `'2026/2027'` o `'2026'` (año natural en Brasil y Argentina) |
| `home_goals`, `away_goals` | `NULL` = todavía sin jugar |
| `home_shots`, `home_sot`, `home_corners`, `home_yellow`, `home_red`, `home_fouls` (y `away_*`) | Solo en las ligas principales |
| `odds_home/draw/away`, `odds_source` | Cuotas 1X2 elegidas y su origen |
| `odds_over25`, `odds_under25` | Más/menos 2.5 goles |
| `referee` | Solo Premier League |

**`schedule`**: calendario de fixturedownload con los nombres ya convertidos a los de football-data
(`round`, `kickoff_utc`, `venue`).

### API-Football

| Tabla | Una fila por | Columnas destacadas |
|---|---|---|
| `api_fixture` | partido | `fixture_id`, `league_code`, `season` (año de inicio), `kickoff_utc`, `round`, `status`, equipos, goles, `referee` |
| `api_team_match` | equipo y partido | `shots_on`, `shots_total`, `corners`, `fouls`, `yellow`, `red`, `possession`, `xg` |
| `api_player_match` | jugador y partido | `minutes`, `started`, `substitute`, `position`, `rating`, `shots_total`, `shots_on`, `goals`, `assists`, `key_passes`, `yellow` |
| `api_player_cache` | búsqueda | Respuestas de `/players?search=` guardadas 24 h |
| `api_lineup` | equipo y partido | `formation`, `coach` |
| `api_lineup_player` | jugador convocado y partido | `number`, `pos`, `started`, `grid` (`fila:columna` de los titulares: fila 1 = portero, columna 1 = banda izquierda del equipo) |
| `api_upcoming` | próximo partido | Los siguientes 20 partidos de cada liga, aún sin jugar |
| `api_injury` | jugador y partido | Parte de bajas: `kind` (`Missing Fixture` = baja, `Questionable` = duda) y `reason`. Solo las cinco grandes y Brasil |

Las fotos de los jugadores (`media.api-sports.io`, no gastan cuota) se descargan una vez a
`data/raw/api_football_media/players/` y la app las sirve desde ahí: API-Football pide no enlazarlas directamente.
Los nombres llegan a veces con entidades HTML ("O&apos;Brien") y se decodifican al guardarlos.

Cuando un jugador no hace ningún tiro, la API devuelve `null`, no 0; `api_stats.py` lo trata como 0.

### Modelos

| Tabla | Contenido |
|---|---|
| `predictions` | Probabilidades 1X2 de cada experimento y modelo (`multi_season`, `xg_vs_goals`, `upcoming`) |
| `market_predictions` | Más/menos de goles, córners, tarjetas, tiros y tiros a puerta: esperado, línea y P(más de) |
| `model_params` | Hiperparámetros elegidos en el último ajuste (`xi`, `ridge`) |
| `team_ratings` | Fuerza actual de cada equipo (ataque, defensa, ventaja de local, ρ) |

### StatsBomb (experimento xG)

`competitions`, `teams`, `players`, `matches`, `lineups`, `events` (subconjunto tipado de los eventos; los de tipo
*Ball Receipt* se descartan), y las derivadas `team_match_stats`, `player_season_stats` y la vista `match_xg`.

## Cruce de equipos entre fuentes

Cada fuente escribe los nombres a su manera ("Man United", "Manchester United", "Man Utd"). En lugar de tablas de
equivalencias escritas a mano, los nombres se cruzan **por calendario**: dos partidos de la misma liga con la misma
fecha y el mismo marcador son el mismo partido, así que sus equipos locales son el mismo equipo. Cada coincidencia es
un voto, y cada nombre se asigna al equipo con más votos. Un equipo real acumula decenas de votos por temporada; una
coincidencia casual, uno o dos.

- StatsBomb ↔ football-data: `ingest/xref.py` → `team_xref` (mínimo 10 votos).
- fixturedownload ↔ football-data: `ingest/schedule.py` (mínimo 3 votos) → `schedule`, y los nombres alternativos se
  guardan en `team_alias` para el lector de boletos.

## Identidad de un partido

La fecha forma parte de la clave porque Liga MX y Argentina repiten el mismo local–visitante dentro de una misma
temporada: Apertura y Clausura, play-in y liguilla. Sin la fecha, el segundo partido sobrescribiría al primero (más de
800 casos por liga). Las comparaciones del backtest también incluyen la fecha.

Un partido aplazado puede llegar dos veces: como "sin jugar" con la fecha antigua y como jugado con la nueva.
`drop_stale_fixtures()` elimina los "sin jugar" con fecha pasada.

## Temporadas

| Contexto | Formato | Ejemplo |
|---|---|---|
| football-data, Europa y Liga MX | `inicio/fin` | `'2026/2027'` |
| football-data, Brasil y Argentina | año | `'2026'` (Argentina usaba `'2013/2014'` hasta 2014) |
| API-Football | año de inicio | `2026` |

El código nunca compara temporadas como texto para decidir qué puntuar: usa fechas (`League.first_test_date()`) y
"la temporada del partido más reciente" de cada liga.

## Añadir una liga

1. Entrada nueva en `LEAGUES` (`src/futbol/leagues.py`): código de football-data, nombre, fuente (`main` o `extra`),
   `calendar_season`, `has_stats`, `api_football_id` y, si existe, `schedule_slug` de fixturedownload.
2. Añadir el código a `BIG5` o `EXTRA` (determina qué función de ingesta la descarga).
3. Si tiene zonas propias en la tabla (clasificación a torneos, descenso), añadirlas en `ZONES` de `app/styles.py`.
4. Alias de equipos para los boletos, si hace falta: `MANUAL_ALIASES` en `src/futbol/betslip.py`.
5. Ejecutar `python scripts/update.py`,
   `python scripts/download_api_football.py --leagues CODIGO` y
   `python scripts/run_backtest.py --skip-tuning --leagues CODIGO`.

El dashboard toma las ligas del registro; no hay que tocar las vistas.
