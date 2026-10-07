# Decisiones de la Competencia 1

Resumen de las cinco decisiones que llevaron al archivo entregado, del 5 al 7-oct-2026. El detalle completo, con
los análisis, las pruebas y las verificaciones, está en mi carpeta de trabajo y no se publica.

## Procedimiento

Para cada decisión:

1. escribí la pregunta, con las opciones y los criterios;
2. dos asistentes de IA la analizaron por separado, y cada análisis se selló con su SHA-256 antes de que el otro
   lo leyera;
3. cada uno criticó el análisis del otro;
4. decidí yo, con los dos análisis y las críticas a la vista;
5. se implementó el código, que se revisó antes de cada corrida y de cada envío. La excepción fue la búsqueda de
   D04: a mi pedido arrancó con la revisión en paralelo, y no se frenó porque la revisión no encontró problemas
   graves;
6. los resultados se verificaron por separado, recalculándolos desde las predicciones guardadas.

Las reglas para adoptar un cambio se fijaron antes de ver los resultados.

## Cómo se midió

**Ganancia:** G(k) = 1.100.000 · TP(k) − 27.500 · k, donde TP cuenta sólo los BAJA+2 entre los k clientes de
mayor probabilidad. A_f es la media de G(k), en millones, sobre todos los enteros k de 8.000 a 15.000 del corte f
(desde D03; en D02 se promediaba la grilla de 8.000 a 15.000 con paso 500):

| Corte | Entrenamiento | Validación |
|---|---|---|
| F1 | 202103 | 202105 |
| F2 | 202103–202104 | 202106 |
| F3, control | 202104 | 202106 |

La métrica principal es A = (A_F1 + A_F2) / 2, sobre el promedio de las probabilidades de las semillas.

**Regla de reemplazo** (D03 a D05): una alternativa reemplaza a la vigente sólo si
- A mejora al menos 10 M;
- A_F1 y A_F2 mejoran los dos;
- A_F3 no pierde más de 5 M;
- en F1 y en F2, la diferencia media por semilla es positiva, con al menos 4 de 5 semillas a favor en la
  pantalla y 16 de 20 en la confirmación.

## D01 · Datos (5-oct)

- **Clase:** `clase_ternaria` recalculada con la lógica de la cátedra. Si el cliente falta el mes siguiente es
  BAJA+1, aunque después vuelva; si está el siguiente y falta el subsiguiente, BAJA+2.
- **Reparaciones:**
  - `ccajas_depositos` y `ccajas_otras` se reemplazan por su suma, porque entre abril y junio se reclasificaron
    transacciones entre las dos;
  - se sacan `Visa_mconsumototal` y `Master_mconsumototal`, idénticas a `*_mconsumospesos`;
  - NA en `*_mfinanciacion_limite` cuando el límite de compra es positivo y la financiación lo supera 10 veces,
    errores de escala;
  - NA en `*_Fvencimiento` menores que −1.000.000, un valor centinela;
  - se sacan `cprestamos_personales` y `mprestamos_personales`: en junio cambia su relación con la baja.
- **Meses:** el modelo final se entrena con 202103–202106 y predice 202108; 202107 se usa sólo como historia.

## D02 · Variables (5 y 6-oct)

- **Validación local:** ningún bloque cumplió la regla fijada antes: cierres centrados, indicadores de
  mantenimiento, rankings, historia selectiva, historia amplia y variables nuevas. El resultado local fue C0, los
  datos de D01 sin variables nuevas.
- **Tablero público,** media de 20 archivos de una semilla con N = 11.000:

  | Envío | Ganancia pública |
  |---|---:|
  | C0 | 81,00 |
  | C0 + historia selectiva (C7) | **84,77** |
  | C0 + *lags* y deltas 1 y 2 de todas las variables | 82,86 |

- **Decisión: C7,** es decir C0 más *lag* 1 y delta 1 de seis conteos y del ranking mensual de cuatro montos.
  Me aparté del resultado local por la evidencia de agosto en el tablero, sabiendo que el tablero mide alrededor
  del 25% de agosto.

## D03 · Clase de entrenamiento (6-oct)

- **Variantes:** K1, BAJA+2 contra el resto; K2, BAJA+1 y BAJA+2 contra CONTINUA; K3, BAJA+2 contra CONTINUA sin
  las filas BAJA+1.
- **K2 pasó la regla** y se confirmó con 20 semillas: A +28,54 M (F1 +39,40, F2 +17,68 y F3 +35,93 M), con las 20
  semillas a favor en los tres cortes. K3 no pasó.
- **Tablero,** un archivo de ensamble de 20 semillas por clase, N = 11.000: K1 92,81 y K2 86,74. La diferencia,
  −6,08, no llegó al veto fijado antes de mandar, −9,94.
- **Decisión: K2,** con K1 como alternativa.

## D04 · Hiperparámetros (6 y 7-oct)

- **Búsqueda:** 40 pruebas con Optuna (TPE, semilla 147347) sobre la receta de Denicolay que se compartió en el
  Zulip de la materia: `learning_rate` 0,005, `max_bin` 31, `min_data_in_leaf` 0, `feature_fraction` 0,5 y
  `feature_fraction_bynode` 0,2, sin *bagging*. Quedaron libres las hojas, el hessiano mínimo por hoja y las
  rondas.
- **Resultado: ninguna configuración cumplió la regla.** Las cinco mejores mejoraban A entre 3,89 y 6,19 M, y todas
  empeoraban el ensamble en F1. La receta nueva da modelos individuales mejores, pero gana mucho menos al promediar
  semillas. Tampoco hubo candidatas al ampliar la grilla a 250, 500 y 750 rondas.
- **Control de clase** con la configuración vigente, N = 11.500: en el tablero, K1 91,82 y K2 86,74. La
  diferencia, −5,09, no llegó al veto, −9,58.
- **Corte:** N = 11.500, el que minimiza la peor pérdida entre F1 y F2 en la grilla de 8.000 a 15.000 con paso
  500, con la pérdida medida contra el máximo de cada corte.
- **Decisión:** se conservan los hiperparámetros vigentes, K2 y N = 11.500.

## D05 · Archivo final (7-oct)

- **Decisión: K2 sola,** con 20 semillas y N = 11.500: `d04k2en11500.csv`, el último archivo que mandé al bot.
- **Alternativas medidas y no adoptadas,** porque ninguna cumplió la regla de reemplazo:

  | Alternativa | Contra K2, en validación |
  |---|---|
  | Promedio de probabilidades K1 + K2 | A −0,03 M; en N = 11.500, F1 −2,20 y F2 −4,40 M |
  | Promedio de rangos K1 + K2 | A −9,09 M; F3 −7,81 M |
  | K2 + la receta de D04 (prueba 38, la mejor finalista de la selección) | entre +5,8 y +6,3 M en la media de F1 y F2, por debajo de los 10 M; sin F3 |
  | K1 sola | A −28,54 M; mejor en el tablero: 92,81 y 91,82 contra 86,74 |

## Uso de IA

Usé dos asistentes de IA como analistas independientes, con el procedimiento de arriba:

- **Claude** (Anthropic), en Claude Code: análisis de cada decisión, código, pruebas, corridas y documentación.
- **Codex** (OpenAI): análisis independientes, críticas cruzadas, revisión del código antes de cada corrida y de
  cada envío (salvo la búsqueda de D04, revisada en paralelo) y verificación de los resultados.

Las decisiones las tomé yo y quedaron registradas con fecha en cada paso. En D02 me aparté del resultado local
por la evidencia del tablero; en D03 y D04 mantuve K2 por las reglas fijadas antes, aunque el tablero favorecía a
K1.
