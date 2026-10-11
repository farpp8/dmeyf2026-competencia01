# De dónde salen los hiperparámetros y el corte de D

El archivo entregado, `d10dn11000.csv`, se reproduce con `python reproducir.py` desde la raíz. Usa valores fijos:
83 hojas, un hessiano de referencia h = 12,791817301585011 y N = 11.000 clientes. Esta carpeta tiene el código que
produjo esos valores, como pidió la cátedra: si los hiperparámetros salen de una optimización anterior, hay que subir
también esa optimización y explicar cómo correrla.

No hace falta correr nada de esto para reproducir el archivo entregado. Sirve para reproducir el camino.

## La cadena

| Paso | Qué se decidió | Script | Qué compara | Tiempo |
|---|---|---|---|---:|
| 1. Búsqueda bayesiana (D04, 6-oct) | 40 pruebas de hojas y h con Optuna, sobre la base C7 con la clase K2 | `d04_busqueda.py` | cada prueba contra `esperado/pruebas_d04.csv` | 3 h (las 4 primeras, 17 min) |
| 2. Selección (D04, 7-oct) | las 5 mejores pruebas, con 5 semillas, contra la configuración vigente; la de mayor mejora fue la prueba 38: **83 hojas y h = 12,79** | `d04_seleccion.py` | la tabla contra `esperado/seleccion_d04.csv` | 10 min |
| 3. D (D06, 7-oct) | la receta de la prueba 38 sobre las variables con historia (C0 con *lags* y deltas 1 y 2 de todo) | `../06_receta_denicolay.py` | — | 40 min |
| 4. El corte (D10, 9-oct) | la regla de N, fijada antes de mirar, sobre D en cuatro cortes de validación: **N = 11.000** | `regla_n.py` | G, los arrepentimientos y N contra `esperado/regla_n_d10.json` | 25 min |
| 5. El archivo | los 11.000 primeros de D | `../07_archivo_d10.py` | `../reproducir.py` lo compara byte por byte | segundos |

**El hessiano:** `min_sum_hessian_in_leaf` = h · filas de entrenamiento / 326.184. 326.184 son las filas de
entrenamiento de F2 (202103–202104), así que en F2 h es exactamente el mínimo por hoja, y en los demás cortes se
escala por sus filas.

**Los fijos de la receta,** que no se buscaron: `learning_rate` 0,005, `max_bin` 31, `min_data_in_leaf` 0,
`feature_fraction` 0,5, `feature_fraction_bynode` 0,2 y sin *bagging*. Son la receta que Gustavo Denicolay compartió en
el Zulip de la materia. Las 1.000 rondas salen de la búsqueda: las cinco finalistas eligieron 1.000 entre 1.000 y
8.000.

**Lo que no es código:**
- **La D06 eligió D en el tablero público,** con un umbral corregido por la cantidad de candidatos y fijado antes de
  mirar. Fue después de que la regla local de la D04 (ΔA ≥ 10 M) no dejara ninguna elegible. El tablero no se puede
  reproducir desde acá. Los números están en `../DECISIONES.md`.
- **Lo que se probó después de la D06 y no cambió D** (D07 a D10, y la prueba de `max_bin` 255) está resumido en
  `../DECISIONES.md`.

## Cómo correrlo

Desde la raíz del repositorio, con el entorno de la reproducción principal más Optuna:

```bash
pip install -r origen/requirements.txt
python 01_preparar_datos.py           # el dataset de trabajo, desde el crudo en datos/ o en COMPETENCIA_01_CRUDO
python origen/d04_busqueda.py         # paso 1; con --hasta 4 corre sólo las 4 primeras pruebas
python origen/d04_seleccion.py        # paso 2
python origen/regla_n.py              # paso 4; necesita también el crudo, por la cohorte de préstamos
```

- Cada script termina diciendo si **reproduce** o **no reproduce** los resultados originales, y sale con código 1 si
  no reproduce.
- Las salidas van a `origen/salidas/`, que no se versiona.
- **La búsqueda se reproduce sólo de corrido y en una carpeta vacía.** El muestreador de Optuna no guarda el estado
  de su generador: si la búsqueda se corta y se reanuda, vuelve a sortear desde la semilla y toma otro camino. Por eso
  el script se niega a reanudar. Si se corta, hay que borrar `origen/salidas/d04_busqueda/` y empezar de nuevo.
  `--hasta K` escribe en otra carpeta, `d04_busqueda_hasta<K>/`.
- `d04_seleccion.py` y `regla_n.py` sí reutilizan las predicciones que ya guardaron, sin volver a controlar con qué
  datos, código o entorno se hicieron. Si cambia cualquiera de esas tres cosas, o se vuelve a correr la búsqueda, hay
  que borrar antes `origen/salidas/d04_seleccion/` y `origen/salidas/regla_n/`.
- `d04_seleccion.py` toma las finalistas de la búsqueda corrida acá si tiene las 40 pruebas; si no, de la original.
  - Con las finalistas propias, para las semillas 147347 y 322607 usa las predicciones de la búsqueda, como la
    corrida original.
  - Con las de la original, entrena las cinco semillas.

**Lo que fija los resultados:**
- las semillas: 147347 para el muestreador TPE de Optuna, y las de cada script para LightGBM;
- `deterministic=True`, `force_col_wise=True` y 12 hilos;
- las versiones de `requirements.txt`.

**Qué puede hacer que no coincida:**
- en otro sistema operativo o con otra compilación de LightGBM, las probabilidades pueden cambiar en los últimos
  decimales. Entonces A puede diferir un poco y, en la búsqueda, el muestreador puede tomar otro camino desde la
  prueba en que difiera;
- reanudar la búsqueda, por lo explicado arriba.

## Qué cambia respecto de los scripts que se corrieron

Los originales están en la carpeta de trabajo del autor, que no se publica: `experimentos/d04_busqueda/` y
`experimentos/d10/`.
- Las rutas son las de este repositorio.
- `d04_busqueda.py` no tiene los límites de tiempo del original: 8 horas, o el 8-oct a las 06:00. La corrida original
  terminó las 40 pruebas antes de llegar a ellos. El de fecha, hoy, cortaría la búsqueda en el acto.
- `d04_seleccion.py` reentrena la configuración vigente de la D03 en lugar de leer sus predicciones guardadas.
- `regla_n.py` junta en un archivo el entrenamiento de D en los cuatro cortes (con la receta de
  `06_receta_denicolay.py`) y la regla de `experimentos/d10/analizar.py`.

**Lo que se comprobó antes de publicar,** en una copia limpia de este repositorio y con este mismo código (10-oct):
- **`d04_busqueda.py --hasta 4`:** las 4 primeras pruebas (las 3 anclas y la primera al azar) dieron exactamente las de
  la corrida original. Las 40 no se volvieron a correr. Un segundo intento sobre la misma carpeta se negó a reanudar;
- **`d04_seleccion.py`:** la tabla de la selección dio igual a la original, con la configuración vigente de la D03
  reentrenada;
- **`regla_n.py`:** los siete escenarios dieron exactamente los originales, y N* = 11.000. Las 80 predicciones de D
  fueron idénticas, bit a bit, a las de la corrida de la D10.
