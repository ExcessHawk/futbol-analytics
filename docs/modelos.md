# Modelos y validación

## Modelo de partido (`models.py`)

Cada equipo tiene un parámetro de **ataque** y uno de **defensa** (cuántos goles concede: más alto = peor), con una
ventaja de local común a la liga:

```
goles local     ~ Poisson(λ),  log λ = μ + local + ataque[local]     + defensa[visitante]
goles visitante ~ Poisson(μ'), log μ' = μ +         ataque[visitante] + defensa[local]
```

### Ajuste

- **Máxima verosimilitud ponderada** con L-BFGS y gradiente analítico (cada ajuste tarda ~12 ms, y el backtest hace
  miles).
- **Decaimiento temporal:** peso = `exp(-xi · días)`. Con el `xi` elegido (0.001) un partido de hace ~2 años pesa la
  mitad que uno de hoy.
- **Regularización L2** (`ridge`) hacia 0 (equipo promedio), para que los equipos con pocos partidos no tengan
  parámetros extremos.
- **Recién ascendidos:** los equipos con menos de 10 partidos en la ventana (o ninguno) se acercan a la media de los 3
  equipos más débiles, no al equipo promedio.
- **Dixon-Coles:** después se ajusta ρ (entre −0.25 y 0.25) sobre los marcadores reales, que corrige la probabilidad
  de 0-0, 1-0, 0-1 y 1-1.

### Del modelo a las probabilidades

La matriz de marcadores (0 a 10 goles por equipo, renormalizada) da todo: 1X2, doble oportunidad, más/menos de
cualquier línea, ambos marcan y el marcador más probable.

### Supuestos y dónde fallan

1. Goles Poisson: ritmo constante; ignora el estado del marcador, las expulsiones y los cambios tácticos.
2. Independencia entre equipos, salvo los cuatro marcadores bajos que corrige ρ.
3. Fuerza constante dentro de la ventana: no ve fichajes, lesiones ni cambios de entrenador.
4. Una única ventaja de local por liga (la altura de Toluca o Pachuca no se modela aparte).

### Entrenar con xG

La verosimilitud de Poisson `k·log λ − λ` sigue siendo válida con `k` no entero, así que ataque y defensa pueden
ajustarse con xG en lugar de goles. ρ se ajusta siempre sobre marcadores enteros.

## Mercados (`markets.py`)

El mismo modelo, entrenado con otros conteos por partido, da córners, amarillas, tiros y tiros a puerta. El total de
dos conteos Poisson independientes es Poisson(λ + μ'), así que `P(total > línea)` tiene forma cerrada. La línea
principal de cada liga es el `.5` más cercano a su media.

- **Árbitro** (solo Premier, que es donde lo tienen los datos): factor multiplicativo sobre las amarillas, con
  contracción hacia la media de la liga (equivale a añadir 10 partidos "de media" al historial de cada árbitro).
- **Apuestas de jugador:** `promedio del jugador × factor del rival`. El factor es cuánto produce su equipo en ese
  partido, según el modelo, frente a lo que produce de media contra todos los rivales (mitad en casa, mitad fuera). El
  promedio del jugador ya refleja la fuerza de su propio equipo; solo hay que ajustar rival y campo. **No está
  validado** con un backtest propio.

## Validación (`validation.py`, `experiments.py`)

### Backtest walk-forward

- Los partidos a predecir se agrupan por semana (lunes a domingo).
- Antes de cada semana, el modelo se reajusta con los partidos jugados **antes de ese lunes** dentro de una ventana de
  3 años. El código lo comprueba en cada bloque (`assert train.max_date < semana.min_date`).
- Los partidos de mitad de semana usan datos del lunes: es una elección conservadora, nunca optimista.
- Los hiperparámetros (`xi` en {0.001, 0.0019, 0.003}, `ridge` en {2, 5, 10}) se eligen con las temporadas
  2013/14 y 2014/15 de las cinco grandes, que nunca se puntúan. Elegirlos con los datos de prueba sería otra forma de
  hacer trampa con el futuro.

### Métricas (menor es mejor)

| Métrica | Definición | Pronóstico uniforme |
|---|---|---|
| Log loss | `−log p(resultado ocurrido)`; castiga mucho los errores con confianza | 1.099 |
| Brier | `Σ (p − o)²` sobre los 3 resultados | 0.667 |
| RPS | Brier sobre probabilidades acumuladas; un empate está "más cerca" de una victoria local que una visitante | — |

Las comparaciones entre modelos usan **solo los partidos que todos predijeron**, y la diferencia frente a la
referencia se da con su intervalo de confianza del 95% (media ± 1.96 errores estándar de la diferencia partido a
partido).

### Experimentos guardados

| Experimento | Qué compara |
|---|---|
| `multi_season` | Frecuencias históricas, Poisson, Dixon-Coles y mercado (cuotas de cierre), en las 8 ligas desde 2015 |
| `xg_vs_goals` | Dixon-Coles con goles, con xG y con su media, en StatsBomb 2015/16 usando solo la misma temporada |
| `upcoming` | Predicciones de los próximos partidos |
| mercados | Modelo frente a la media de la liga (y frente al mercado en más/menos 2.5) en goles, córners, amarillas, tiros y tiros a puerta |

## Resultados

### 1X2

| Liga(s) | Partidos | Mercado | Dixon-Coles | Poisson | Frecuencias | DC − mercado (IC 95%) |
|---|---|---|---|---|---|---|
| Cinco grandes | 20,215 | 0.9640 | 0.9847 | 0.9850 | 1.0709 | +0.021 [+0.018, +0.023] |
| Liga MX, Argentina, Brasil | 13,803 | 1.0174 | 1.0380 | 1.0381 | 1.0680 | +0.021 [+0.018, +0.024] |
| Argentina | 5,507 | 1.0381 | 1.0582 | — | — | +0.020 [+0.015, +0.025] |
| Brasil | 4,509 | 0.9950 | 1.0180 | — | — | +0.023 [+0.017, +0.029] |
| Liga MX | 3,787 | 1.0142 | 1.0325 | — | — | +0.018 [+0.013, +0.024] |

Cifras de log loss. En las cinco grandes: Brier 0.5866 frente a 0.5725 del mercado; el favorito del modelo acierta el
52.6% frente al 54.0% del mercado.

**Lectura:** el modelo aprende mucho más que las frecuencias históricas, está bien calibrado, y aun así el mercado es
más preciso en todas las ligas, también en Latinoamérica. Un modelo que solo ve resultados no compite con un precio que
incorpora alineaciones, lesiones y dinero informado.

### Mercados (Europa, Δ log loss frente a la media de la liga)

| Mercado | Δ (IC 95%) | Veredicto |
|---|---|---|
| Tiros | −0.0206 [−0.0252, −0.0160] | Mejor |
| Tiros a puerta | −0.0152 [−0.0188, −0.0117] | Mejor |
| Amarillas | −0.0089 [−0.0119, −0.0060] | Mejor |
| Goles más/menos 2.5 | −0.0088 [−0.0115, −0.0062] | Mejor, pero el mercado lo es más (−0.0199) |
| Córners | −0.0015 [−0.0044, +0.0015] | No se distingue |

En la Premier, añadir el factor árbitro a las amarillas mejora apenas 0.0014, una diferencia no concluyente. En
Latinoamérica, más/menos 2.5 goles no se distingue de la media de la liga (−0.0005 [−0.003, +0.002]).

La calibración de más/menos 2.5 en Europa se desvía en la zona baja: cuando el modelo da un 34%, ocurre ~41% de las
veces.

### xG frente a goles (StatsBomb 2015/16, 1,337 partidos, sin historial previo)

| Modelo | Log loss |
|---|---|
| Mercado | 0.9749 |
| Dixon-Coles goles, con 3 temporadas de historial | 0.9919 |
| Dixon-Coles (xG + goles)/2 | 1.0087 |
| Dixon-Coles xG | 1.0114 |
| Dixon-Coles goles | 1.0150 |

El xG mejora a los goles en −0.004 (no significativo); la mezcla, en −0.006 (significativo). El historial pesa más que
el xG. API-Football tiene ya xG en temporadas recientes, lo que permite repetir el experimento con historial.
