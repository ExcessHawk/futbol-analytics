# Analizador de boletos

`src/futbol/betslip.py` convierte la foto o captura de un boleto en una tabla de apuestas y evalúa cada una con los
modelos. La vista es `app/views/apuesta.py`.

```
imagen ─► OpenCV (3 variantes) ─► Tesseract ─► parser ─► votación ─► revisión del usuario ─► evaluación
```

El OCR nunca es fiable al 100% (cada casa de apuestas tiene su diseño), así que el parser **propone** las apuestas y la
app las muestra en una tabla editable para confirmarlas antes de evaluar.

## 1. Preprocesado (OpenCV)

`preprocess_variants()` produce tres imágenes binarias de la misma foto:

| Variante | Técnica | Funciona mejor con |
|---|---|---|
| `gray` | Escala de grises | Capturas de pantalla limpias |
| `clahe_otsu` | Contraste local (CLAHE) + umbral de Otsu | Fotos con poco contraste |
| `adaptive` | Umbral adaptativo | Fotos con sombras o iluminación desigual |

Antes de eso, a todas:

- **Modo oscuro:** si el brillo medio es menor que 110, se invierte la imagen (Tesseract espera texto oscuro sobre
  fondo claro).
- **Escalado** a 1600 px de ancho: Tesseract lee mejor con caracteres de 25–35 px de alto.
- **Enderezado:** el ángulo del rectángulo mínimo que contiene la tinta (`cv2.minAreaRect`); solo se corrige entre
  0.3° y 15°, para no girar boletos que ya están rectos ni confundir el texto con otra cosa.

## 2. OCR (Tesseract)

Modo de página `--psm 6` (un bloque de texto uniforme) con español e inglés. El modo 4 (columnas) perdía las cuotas
alineadas a la derecha. Los modelos de idioma se descargan con `scripts/get_tessdata.py` en `data/tessdata/`.

## 3. Parser

### Dividir el boleto en apuestas

Cada casa organiza el boleto de forma distinta: el partido arriba o abajo de la apuesta, la hora al principio o al
final. `parse_slip()` prueba varias formas de cortarlo en bloques, una por apuesta:

- por la línea de la hora (`21:00`), con el bloque empezando o terminando en ella;
- por la línea del partido ("Valencia vs Real Madrid"), con los detalles debajo o encima.

De cada lectura cuenta `3 × apuestas + mercados reconocidos + cuotas`, y se queda con la de más puntos.

### Reconocer el partido

`TeamIndex` contiene los equipos que jugaron en los últimos 400 días y sus alias (los de `team_alias` y
`MANUAL_ALIASES`). Para cada línea busca:

1. coincidencia exacta del nombre (`fuzz.ratio ≥ 88`) o parcial (`WRatio ≥ 85`);
2. códigos de tres letras ("VAL vs RMA"): si hay varias lecturas posibles (RMA = Real Madrid; VAL = Valencia o
   Vallecano), gana el partido programado más próximo entre ellas;
3. nombres que existen en dos ligas (Santos en Brasil y Santos Laguna en Liga MX): se elige el equipo de la misma liga
   que el rival.

### Mercado, selección, línea y cuota

| Mercado | Ejemplo en el boleto |
|---|---|
| 1X2, doble oportunidad | "Real Madrid", "Empate", "1X" |
| Goles más/menos, ambos marcan | "Más de 2.5", "Ambos equipos anotan: Sí" |
| Córners, amarillas, tiros, tiros a puerta | "Total de córners más de 9.5" |
| Jugador: tiros a puerta, tiros, goles, tarjeta | "Kylian Mbappé · Tiros a puerta: 3 o más" |

- "3 o más" equivale a "más de 2.5". El OCR lee a menudo "3 0 más" o "30 más", y la expresión regular lo contempla.
- Cuotas decimales (`1.90`), americanas (`+150`, `-200`) y fraccionarias (`9/10`), con o sin sufijo `x`.
- La cuota se extrae antes de buscar la línea, para que "2.40 Más de 1.5" no tome 2.40 como línea.
- "Ambos equipos anotan" no es una apuesta de jugador aunque contenga "anotan".
- El pie del boleto (importe, ganancia potencial, código de reserva) se descarta con la expresión `FOOTER`.

### Votación entre variantes

`parse_slip_variants()` analiza las tres variantes y, para cada partido, se queda con la apuesta que más variantes
leyeron y con la cuota más repetida. Si una variante lee "1.67x" como "167x", las otras dos la corrigen.

### Datos del boleto

`parse_meta()` lee el importe apostado, el *boost* (p. ej. +80%) y la cuota o ganancia total impresas.

## 4. Comprobaciones

- **Producto de cuotas:** si el producto de las cuotas leídas no coincide (±2%) con la cuota total impresa, alguna se
  leyó mal (`odds_consistency()`).
- **Cuota sospechosa:** para cada apuesta se calcula qué cuota haría cuadrar el total; si difiere de la leída en un
  solo carácter (1.90 frente a 4.90), se señala y se propone la corrección (`suspect_odds()`).
- **Apuesta perdida:** si la diferencia es mayor (entre ×1.05 y ×50), lo más probable es que falte una apuesta y la
  app lo indica.
- `check_pick()` explica por qué una apuesta no se puede evaluar: faltan equipos, línea, cuota o el promedio del
  jugador.

## 5. Evaluación

`evaluate()` añade a cada apuesta:

| Columna | Significado |
|---|---|
| `p_model` | Probabilidad según el modelo |
| `p_implied` | `1 / cuota`: lo que cree la casa (con su margen) |
| `fair_odds` | `1 / p_model` |
| `ev` | `p_model × cuota − 1`: ganancia esperada por unidad apostada |

Las apuestas de jugador usan su promedio de la temporada en la base de datos (API-Football) ajustado por el rival. La
combinada multiplica las probabilidades, y la app avisa cuando hay dos apuestas del mismo partido, porque entonces no
son independientes.

**Cómo leerla:** el mercado es más preciso que el modelo (log loss 0.964 frente a 0.985 en 1X2). Cuando el modelo "ve
valor" en una cuota, lo más probable es que el que se equivoque sea el modelo. Lo que sí es fiable es la
probabilidad aproximada de que la combinada salga.

## Adaptarlo a otra casa de apuestas

1. Guardar una captura del boleto y pasarla por la app; abrir el expander con la imagen procesada y el texto leído.
2. Si no reconoce equipos: añadir los alias en `MANUAL_ALIASES`.
3. Si no reconoce el mercado: palabras clave en `MARKET_KEYWORDS` o `PLAYER_MARKETS`.
4. Si se cuela texto del pie o de la cabecera: ampliar `FOOTER` o `NOT_A_NAME`.
5. Si el diseño no encaja con ninguna forma de dividirlo: añadir una función `_blocks_by_...` y sumarla a la lista
   de `parse_slip()`.
6. Copiar el texto leído a una prueba como `tests/test_real_slip.py`, para que no se rompa con cambios futuros.

## Pruebas

- `tests/test_betslip.py` genera boletos sintéticos (captura, modo oscuro, foto inclinada con sombra) y comprueba
  equipos, mercados y cuotas, además de las funciones de cuotas, alias y comprobaciones.
- `tests/test_real_slip.py` usa las líneas de OCR de un boleto real de 11 apuestas y exige leer las 11, el importe, el
  boost y la ganancia total.
