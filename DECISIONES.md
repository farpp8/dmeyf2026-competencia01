# Decisiones de la Competencia 1

Resumen de las once decisiones que llevaron al archivo entregado, del 5 al 9-oct-2026. El detalle completo, con
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

## D01 · Datos

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

## D02 · Variables

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

## D03 · Clase de entrenamiento

- **Variantes:** K1, BAJA+2 contra el resto; K2, BAJA+1 y BAJA+2 contra CONTINUA; K3, BAJA+2 contra CONTINUA sin
  las filas BAJA+1.
- **K2 pasó la regla** y se confirmó con 20 semillas: A +28,54 M (F1 +39,40, F2 +17,68 y F3 +35,93 M), con las 20
  semillas a favor en los tres cortes. K3 no pasó.
- **Tablero,** un archivo de ensamble de 20 semillas por clase, N = 11.000: K1 92,81 y K2 86,74. La diferencia,
  −6,08, no llegó al veto fijado antes de mandar, −9,94.
- **Decisión: K2,** con K1 como alternativa.

## D04 · Hiperparámetros

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

## D05 · Archivo final

- **Decisión: K2 sola,** con 20 semillas y N = 11.500: `d04k2en11500.csv`, el último envío hasta ese momento.
- **Alternativas medidas y no adoptadas,** porque ninguna cumplió la regla de reemplazo:

  | Alternativa | Contra K2, en validación |
  |---|---|
  | Promedio de probabilidades K1 + K2 | A −0,03 M; en N = 11.500, F1 −2,20 y F2 −4,40 M |
  | Promedio de rangos K1 + K2 | A −9,09 M; F3 −7,81 M |
  | K2 + la receta de D04 (prueba 38, la mejor finalista de la selección) | entre +5,8 y +6,3 M en la media de F1 y F2, por debajo de los 10 M; sin F3 |
  | K1 sola | A −28,54 M; mejor en el tablero: 92,81 y 91,82 contra 86,74 |

## D06 · Plan para los últimos días

- **Un punto ciego de la validación local:**
  - F1 no usa ninguna variable de historia, porque marzo no tiene mes anterior;
  - la historia de orden 2 no existe en ningún corte: 0 columnas útiles en F1, F2 y F3, y 294 en el modelo final
    (147 *lags* 2 y 147 deltas 2).
- **Grilla cerrada contra la base:** ninguna variante pasó la regla. Fueron un ranking con «cero fijo», el signo de
  la comisión de mantenimiento y sacar la historia.
- **Tablero, con candidatos y regla fijados antes de ver resultados:**
  - un envío por candidato, todos con 20 semillas y N = 11.500;
  - para adoptar un candidato sin evidencia local hay que superar a la base por un margen corregido por la cantidad
    de candidatos mirados.

  | Candidato | Público | Contra la base |
  |---|---:|---:|
  | Base (K2 sobre C7) | 86,74 | — |
  | K2 + prueba 38, mezcla 50/50 | 89,95 | +3,2 |
  | K1 | 91,82 | +5,1 |
  | L: *lags* y deltas 1 y 2 de todo, configuración vigente | 98,59 | +11,9 |
  | **D: las variables de L con la receta de la prueba 38** | **103,18** | **+16,4** |

- **La regla elige D**, lo que verificó Codex por su cuenta.

## D07 a D09 · Rondas alrededor de D

Tres decisiones más intentaron mejorar a D, con reglas fijadas antes y el público sólo como control. Ninguna lo
reemplazó:

| Decisión | Qué se probó | Resultado |
|---|---|---|
| D07 | Un corte F4 (202105 → 202106), el único que ve la historia de orden 2; D con K1 y otras tres rondas | D-K1 perdió 10,6 en el público (92,57). Las rondas 2 y 3 no dejaron candidata y la 4 se omitió |
| D08 | 24 configuraciones de hiperparámetros, medidas en F4 y F2 | La mejor pasó la pantalla (+4,56) pero no la confirmación con 20 semillas nuevas (+3,52, con una barra de 5,4). En el público, informativo, dio 99,19 |
| D09 | Julio con etiquetas parciales, pesos para BAJA+1, reparar los cierres de tarjeta y un modelo auxiliar | La reparación se descartó antes de la pantalla, porque los cierres pesaban el 0,23% de la ganancia de los árboles. Las demás líneas no pasaron la pantalla: julio parcial ganaba en mayo y perdía 21 en junio |

## D10 · Investigación profunda y el corte

- **Análisis:** Con un análisis de Claude (un flujo de agentes con verificación adversarial) y otro de Codex, cruzados     como en las decisiones anteriores.
- **Hallazgo:** una cohorte de clientes con el préstamo personal congelado (el mismo monto positivo tres meses
  seguidos) tiene 1,23% de BAJA+2 en mayo y 12,32% en junio. Explica entre el 21 y el 23% de la ganancia de D en
  junio. Se mantuvo la métrica de D09, que pondera mayo y junio por igual, y se agregó un control que saca a esa
  cohorte.
- **Pantalla:** cinco cambios sobre D, cada uno por separado:
  - un bloque de 21 variables de comportamiento por canal;
  - *bagging* 0,8;
  - `extra_trees`;
  - pesos por recencia;
  - BAJA+3 como positivo con peso 0,3.

  **Ninguno llegó siquiera a empatar a D:**

  | Cambio | Contra D (media de mayo y junio, M) |
  |---|---:|
  | Bloque de comportamiento | −3,70 |
  | *Bagging* 0,8 | −0,24 |
  | `extra_trees` | −24,30 |
  | Pesos por recencia | −0,58 |
  | BAJA+3 con peso 0,3 | −0,51 |

- **El corte.** Esta regla se fijó antes de mirar los resultados:
  - siete escenarios, con las 20 semillas de D en cuatro cortes de validación, junio sin la cohorte y junio con el
    92% y el 85% de los aciertos;
  - N\* es el de menor arrepentimiento máximo sobre la grilla 9.000–12.000;
  - se cambia sólo si la media mejora.

  **Dio N\* = 11.000:** el arrepentimiento máximo baja de 14,30 a 6,05 y la media sube 2,99 M. Sólo el corte F4
  pierde al pasar de 11.500 a 11.000.
- **El público como veto:** se adoptaba salvo que perdiera más de 2σ contra D (4,39). `d10dn11000.csv` dio
  **102,60**, contra 103,18 de D: −0,58, dentro del ruido.
- **Decisión: D con N = 11.000,** `d10dn11000.csv`. Es el último envío y lo reproduce este repositorio.

## D11 · `max_bin` 255 (9-oct, informativa)

- **Qué:** D con 255 cortes por variable en lugar de 31, con la misma pantalla de la D10, sellada antes de correr.
- **Resultado:** ΔT +2,17, debajo de la barra de +3:
  - mayo +3,03 y junio +1,30;
  - los diagnósticos, −1,14 y +1,95;
  - sin la cohorte de préstamos, junio queda en −0,27.
- **No pasó, así que no cambió nada.** No se bajó la barra después de ver el resultado.

## De dónde salen los hiperparámetros y el corte

Las 83 hojas y el hessiano de D salen de la búsqueda y la selección de la D04; N = 11.000, de la regla de la D10. El
código que los produjo, con las instrucciones para correrlo, está en [origen/](origen/README.md).

## Uso de IA

Usé dos asistentes de IA como analistas independientes, con el procedimiento de arriba:

- **Claude** (Anthropic), en Claude Code: análisis de cada decisión, código, pruebas, corridas y documentación.
- **Codex** (OpenAI): análisis independientes, críticas cruzadas, revisión del código antes de cada corrida y de
  cada envío (salvo la búsqueda de D04, revisada en paralelo) y verificación de los resultados.

Las decisiones las tomé yo y quedaron registradas con fecha en cada paso. En D02 me aparté del resultado local
por la evidencia del tablero; en D03 y D04 mantuve K2 por las reglas fijadas antes, aunque el tablero favorecía a
K1.
