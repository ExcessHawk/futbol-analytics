# Fútbol Analytics

Dashboard de análisis de fútbol para **8 ligas** (Premier League, LaLiga, Bundesliga, Serie A, Ligue 1, Liga MX,
Liga Profesional Argentina y Brasileirão) con un modelo de predicción de partidos **Dixon-Coles**, validado partido a
partido contra los resultados reales y contra las cuotas de cierre de las casas de apuestas, y un **analizador de
boletos de apuestas por foto** (OpenCV + Tesseract).

> **Conclusión principal, sin adornos:** el modelo está bien calibrado, pero **no supera al mercado de apuestas** en
> ninguna liga. En ~34,000 partidos fuera de muestra pierde contra las cuotas de cierre por ~0.02 de log loss, con
> intervalos de confianza que no incluyen el cero. El dashboard lo dice en cada predicción.

📚 Documentación técnica en [`docs/`](docs/README.md): [arquitectura](docs/arquitectura.md), [datos](docs/datos.md),
[modelos y validación](docs/modelos.md), [analizador de boletos](docs/boletos.md), [operación](docs/operacion.md) y
[desarrollo](docs/desarrollo.md).

---

## Qué hace

| Vista | Contenido |
|---|---|
| **Clasificación y forma** | Tabla, forma reciente (G/E/P), zonas de la liga y **puntos frente a los puntos esperados** según las cuotas. Liga MX por torneo (Apertura/Clausura) sin contar la liguilla. |
| **Equipos** | xG y xG en contra, goles − xG, tiros, tiros a puerta, córners y tarjetas a favor y en contra por partido; tendencia de xG y registro de cada partido. |
| **Jugadores** | Ranking por 90 minutos y ficha partido a partido de cada jugador, con **cuántas veces superó cada línea** (p. ej. "3 o más tiros a puerta en 6 de 11 partidos"). |
| **Alineaciones** | Campo con los dos onces y la foto de cada jugador. Próximo partido: **alineación probable** (la formación del último partido, quien más ha jugado en cada puesto, sin los lesionados ni sancionados) y el **parte de bajas** con su motivo. Partido jugado: el once real con el rating de cada jugador. Un clic en un jugador abre su temporada y sus últimos partidos. |
| **Predicciones** | Próximos partidos (clic para ver el detalle): 1X2, goles esperados, marcadores, más/menos, ambos marcan, córners, tarjetas y tiros. Simulador de cualquier enfrentamiento, historial con lo que predijo el modelo y la **fiabilidad medida de cada mercado**. |
| **Analizar boleto** | Sube una o varias capturas del boleto: OCR, apuestas reconocidas (partido, mercado, cuota, apuestas de jugador), comprobación de la cuota total, probabilidad del modelo, valor esperado y probabilidad de acertar la combinada (con boost). |

## Resultados de la validación

Backtest **walk-forward**: cada semana el modelo se reajusta solo con partidos jugados antes de ese lunes y predice
los de la semana (el código lo verifica en cada bloque). Hiperparámetros elegidos en 2013/14–2014/15, temporadas que
nunca se puntúan. Métricas: menor es mejor; un pronóstico uniforme tiene log loss 1.099 y Brier 0.667.

### Resultado 1X2

| Liga(s) | Partidos | Modelo (log loss) | Mercado (log loss) | Diferencia vs mercado (IC 95%) |
|---|---|---|---|---|
| Cinco grandes de Europa (2015/16 – hoy) | 20,215 | 0.9847 | 0.9640 | +0.021 [+0.018, +0.023] |
| Liga MX, Argentina y Brasil (2015 – hoy) | 13,803 | 1.0380 | 1.0174 | +0.021 [+0.018, +0.024] |

- Frecuencias históricas (sin modelo): 1.0709 en Europa. El modelo aprende mucho, pero no tanto como el mercado.
- Dixon-Coles apenas mejora a Poisson simple (Δ ≈ 0.0004): la corrección de marcadores bajos tiene poco efecto en el 1X2.
- La calibración es buena: cuando el modelo dice 40%, ocurre cerca del 40% de las veces (gráfico en la app).

### Otros mercados (Europa, frente a usar la media de la liga)

| Mercado | Δ log loss (IC 95%) | ¿Sirve? |
|---|---|---|
| Tiros (más/menos) | −0.021 [−0.025, −0.016] | Sí, la mejora más clara |
| Tiros a puerta | −0.015 [−0.019, −0.012] | Sí |
| Tarjetas amarillas | −0.009 [−0.012, −0.006] | Sí, modesta (el factor árbitro aporta muy poco) |
| Goles más/menos 2.5 | −0.009 [−0.012, −0.006] | Sí, pero el mercado es mejor (−0.020) |
| **Córners** | −0.0015 [−0.004, +0.0015] | **No** se distingue de la media de la liga |

En Liga MX, Argentina y Brasil, el modelo de goles más/menos 2.5 tampoco se distingue de la media de la liga
(−0.0005 [−0.003, +0.002]).

### ¿Mejora el xG frente a los goles?

Con el xG de StatsBomb (2015/16, solo partidos de la misma temporada), el modelo entrenado con xG mejora al de goles
en −0.004 de log loss, **no significativo** con 1,337 partidos; mezclar xG y goles sí mejora (−0.006, significativo).
Pesa más tener historial: el modelo de goles con 3 temporadas previas supera a todos los anteriores.

## Fuentes de datos

| Fuente | Qué aporta | Acceso |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk) | Resultados, tiros, córners, tarjetas, árbitro (Premier) y cuotas de cierre desde 2012; Liga MX, Argentina y Brasil solo resultados y cuotas | CSV gratuitos |
| [API-Football](https://www.api-football.com) | Estadísticas por equipo y **por jugador** en cada partido (tiros, tiros a puerta, minutos, titularidad, tarjetas, rating) y **xG**, 8 ligas desde 2015 | API con clave; el plan gratuito solo da 2022–2024, la temporada actual requiere plan de pago |
| [fixturedownload.com](https://fixturedownload.com) | Calendario completo de las ligas europeas (jornada, hora, estadio) | JSON público |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | Eventos con coordenadas de 2015/16 (experimento xG frente a goles) | Abierto, con atribución |

Los nombres de equipos difieren entre fuentes ("Man United" / "Manchester United"); se cruzan **por calendario**:
dos partidos con la misma fecha y el mismo marcador son el mismo partido, y cada coincidencia es un voto. No hay
tablas de equivalencias escritas a mano para las fuentes, salvo los alias que usan los boletos.

**Antes de publicar los datos de API-Football en abierto, revisa sus términos de uso.**

## Metodología

**Modelo de partido.** Ataque y defensa por equipo con ventaja de local:
`goles_local ~ Poisson(exp(μ + local + ataque_L + defensa_V))`, y lo análogo para el visitante. Se ajusta por máxima
verosimilitud con pesos que decaen con la antigüedad del partido (vida media ~2 años), regularización L2 hacia el
promedio y un *prior* de equipo débil para los recién ascendidos. La corrección de Dixon-Coles (ρ) se ajusta después,
sobre los marcadores reales. El mismo modelo, entrenado con otros conteos, predice córners, tarjetas, tiros y tiros a
puerta.

**Supuestos y límites.** Los goles se tratan como Poisson (ritmo constante en el partido; ignora el estado del
marcador y las expulsiones), los dos equipos son independientes salvo la corrección de marcadores bajos, y la fuerza
de cada equipo es fija dentro de la ventana de entrenamiento: no ve fichajes, lesiones ni cambios de entrenador. Los
córners tienen más varianza de la que asume Poisson.

**Validación.** Walk-forward semanal, log loss, Brier y RPS, curvas de calibración y diferencias emparejadas frente al
mercado con intervalos de confianza. Los partidos se identifican con su fecha, porque Liga MX y Argentina repiten el
mismo enfrentamiento dentro de una temporada (torneos cortos y liguilla).

**Apuestas de jugador.** Promedio del jugador en la temporada × factor del rival (cuánto produce su equipo en ese
partido frente a su media, según el modelo de equipos) → Poisson. **No está validado** con un backtest propio; la app
lo indica y muestra al lado la frecuencia real de esa línea en la temporada.

**Lector de boletos.**
1. OpenCV prepara tres versiones de cada imagen (gris, CLAHE + Otsu, umbral adaptativo; inversión si es modo oscuro,
   enderezado si está inclinada).
2. Tesseract (español + inglés) lee cada una, y las cuotas se deciden **por votación** entre lecturas.
3. Las apuestas se separan por la línea de fecha y hora de cada una, o por la línea "local vs visitante", según el
   formato del boleto. Los nombres se reconocen con búsqueda aproximada sobre los alias, y las abreviaturas ("VAL vs RMA")
   se resuelven con el calendario.
4. Comprobación final: el producto de las cuotas leídas tiene que coincidir con la cuota total impresa (quitando el
   boost). Si no coincide, la app señala la cuota sospechosa o la apuesta que falta.

## Puesta en marcha

> **El repositorio no trae datos.** Son ~200 MB de respuestas originales de las APIs y una base de 1.1 GB, y
> ninguno de los dos cabe aquí. Al clonar, la base está vacía y el dashboard arranca con todo en cero: eso no
> es un fallo, es el estado inicial. La sección siguiente lo llena desde las fuentes, y las dos primeras
> —resultados, cuotas y eventos de StatsBomb— son públicas y no piden ninguna clave.

Requisitos: Python 3.14, Docker (para PostgreSQL) y Tesseract OCR (solo para el analizador de boletos).

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt    # en un servidor basta requirements.txt
pip install -e .                       # instala src/futbol en modo editable

cp .env.example .env                   # y rellena APIFOOTBALL_KEY si tienes clave
docker compose up -d                   # PostgreSQL en localhost:5450
python scripts/get_tessdata.py         # modelos de idioma de Tesseract (spa, eng)
python scripts/check_env.py            # verifica dependencias y conexión a la base de datos
```

Tesseract: en Debian/Ubuntu `sudo apt install tesseract-ocr`; en Windows, el instalador de UB Mannheim y su ruta en
`TESSERACT_CMD`.

### Carga inicial de datos (en este orden)

| Paso | Comando | Tiempo aprox. |
|---|---|---|
| 1. Resultados y cuotas, 8 ligas desde 2012 | `python scripts/ingest_football_data.py` | 1 min |
| 2. Liga MX, Argentina, Brasil y calendario | `python scripts/update.py` | 1 min |
| 3. Estadísticas de API-Football (necesita clave) | `python scripts/download_api_football.py` | 15–20 min, ~1,800 peticiones |
| 4. Eventos StatsBomb 2015/16 (opcional, experimento xG) | `python scripts/ingest_statsbomb.py` y `python scripts/build_metrics.py` | 10 min |
| 5. Modelos y backtest | `python scripts/run_backtest.py` | 15–20 min |
| 6. Dashboard | `streamlit run app/Inicio.py` | — |

Las alineaciones se guardan con cada descarga. En una base de datos creada antes de que existieran, se cargan una
vez desde los JSON del disco, sin gastar peticiones: `python scripts/load_lineups.py`.

`download_api_football.py` es reanudable (salta lo ya descargado) y va al 40% del límite por minuto de la API para no
arriesgar un bloqueo. Guarda el JSON original comprimido en `data/raw/api_football/`, así que los datos siguen
disponibles aunque el plan de pago caduque.

### Mantenerlo al día

```bash
python scripts/update.py      # ~1 min: resultados, cuotas, calendario, API-Football y predicciones
```

Pensado para un cron diario, por ejemplo a las 06:00 (la cuota diaria de API-Football se reinicia a las 00:00 UTC):

```cron
0 6 * * * cd /ruta/futbol-analytics && .venv/bin/python scripts/update.py >> data/raw/update.log 2>&1
```

Después de cada actualización, el botón **Recargar datos** del dashboard limpia la caché. Tras cambios **de código**
hay que reiniciar Streamlit: no siempre recarga los módulos compartidos (`app/common.py`, `src/futbol`).

## Variables de entorno

| Variable | Por defecto | Para qué |
|---|---|---|
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | `localhost`, `5450`, `futbol`… | PostgreSQL |
| `APIFOOTBALL_KEY` | vacía | API-Football; sin ella se omiten sus pasos y la app funciona con el resto de fuentes |
| `TESSERACT_CMD` | `tesseract` | Binario de Tesseract (ruta completa en Windows) |
| `TESSDATA_DIR` | `data/tessdata` | Modelos de idioma del OCR |
| `LOG_LEVEL` | `INFO` | Nivel de log de los scripts |

## Notas para el despliegue

- **La clave de API-Football solo en el servidor** (`.env`): nunca llega al navegador.
- **Streamlit no trae inicio de sesión.** Para compartirlo con un grupo cerrado, protégelo en el proxy inverso
  (autenticación básica o un proveedor OAuth) y sirve por HTTPS.
- **El servidor necesita** Tesseract (`apt install tesseract-ocr`) y los modelos (`python scripts/get_tessdata.py`)
  para el analizador de boletos. El resto de la app funciona sin ellos.
- **Tamaño:** la tabla más grande son los eventos de StatsBomb (~3.6 millones de filas); si no te interesa el
  experimento xG, puedes omitir el paso 4 de la carga inicial. El JSON de API-Football ocupa unos cientos de MB
  comprimido en `data/raw/`.
- **Datos de API-Football:** el plan de pago no se renueva solo. Al volver al plan gratuito, la temporada actual deja
  de actualizarse (`update.py` lo registra y sigue con lo demás).

## Estructura

```
src/futbol/
  leagues.py          registro de ligas: fuente, formato de temporada, ids de API-Football
  models.py           Poisson / Dixon-Coles, matriz de marcadores
  markets.py          córners, tarjetas, tiros, más/menos, apuestas de jugador
  validation.py       backtest walk-forward, log loss, Brier, RPS, calibración
  experiments.py      experimentos y predicciones que se guardan en la base de datos
  season.py           clasificación, forma, puntos esperados, fase regular de Liga MX
  api_stats.py        tablas de equipos y jugadores a partir de API-Football
  betslip.py          OCR de boletos: OpenCV, Tesseract, parser y evaluación
  lineups.py          alineación probable (últimas alineaciones + parte de bajas) y su backtest
  ingest/             football-data, API-Football, calendario, StatsBomb, cruce de equipos
scripts/              comandos (ingesta, backtest, actualización diaria)
app/                  dashboard de Streamlit (Inicio.py + views/)
sql/schema.sql        esquema idempotente, con migraciones
tests/                pytest (los de la app se saltan si no hay base de datos)
docs/                 documentación técnica
```

## Calidad

```bash
pytest                 # 127 pruebas: modelos, métricas, alineaciones, parser de boletos (también con un boleto real), vistas
ruff check . && ruff format --check .
```

Las pruebas de las vistas recorren cada página con `streamlit.testing` contra la base de datos real, incluidas las 8
ligas. Las del OCR generan boletos sintéticos (captura, modo oscuro, foto inclinada con sombra) y comprueban que se
leen las apuestas y las cuotas.

## Limitaciones

- El modelo **no gana al mercado**. Úsalo para entender partidos, no como sistema de apuestas: con el margen de las
  casas, el valor esperado de apostar suele ser negativo.
- Las predicciones de córners no son mejores que la media de la liga.
- Las apuestas de jugador no están validadas con backtest; no saben si el jugador será titular.
- El lector de boletos se probó con un formato de casa de apuestas y con boletos sintéticos; otros formatos pueden
  necesitar ajustes (la tabla de revisión permite corregir cualquier lectura).
- La alineación probable acierta de media 8.3 de 11 titulares; no ve las rotaciones por copa o competición europea
  (solo conoce los partidos de liga). Liga MX y Argentina no tienen parte de bajas en API-Football.
- Liga MX Femenil no está incluida: API-Football no tiene estadísticas de jugadoras ni de partido para esa liga.

---

## Para qué sirve y para qué no

Esto es una herramienta de **análisis**, no un servicio de apuestas y no un sistema para ganar dinero. El propio
proyecto mide que su modelo pierde contra las cuotas de cierre en las ocho ligas, y esa medición está arriba, no
escondida. El analizador de boletos existe para leer lo que ya apostaste y decirte qué probabilidad le da el
modelo — que, repito, es peor que la del mercado.

Si buscas un sistema rentable, el resultado honesto de este repositorio es que aquí no está.

## Licencia

[MIT](LICENSE). Haz lo que quieras con el código, con atribución y sin garantía.

Los **datos** son otra cosa y no los cubre esta licencia: football-data.co.uk, StatsBomb Open Data,
fixturedownload.com y API-Football tienen cada uno sus propias condiciones, y quien los descargue las acepta por su
cuenta. Aquí no se redistribuye ninguno: el repositorio solo trae el código que los va a buscar.
