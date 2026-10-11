# DMEyF 2026 · Competencia 1

**Federico Ramírez** · Data Mining en Economía y Finanzas, Maestría en Data Mining (UBA), 2026 · comisión del lunes.

Este repositorio reproduce, desde el dataset crudo de la cátedra, el último archivo que envié al bot de la
competencia: **`d10dn11000.csv`**, con 11.000 clientes de 202108 y SHA-256
`8a03bf3d5fd792569187e8093a1663f7c7f97eeab7b785af412b06fc595a70b5`. Lo mandé el 9-oct-2026 a las 18:27, hora de
Buenos Aires.

**El modelo, D:** el promedio de las probabilidades de 20 LightGBM, uno por semilla.
- **Entrenamiento:** 202103–202106, para separar BAJA+1 y BAJA+2 de CONTINUA.
- **Variables:** los datos reparados más *lag* 1, delta 1, *lag* 2 y delta 2 de todas las variables, 735 columnas
  útiles.
- **Receta de hiperparámetros:** la de Denicolay, ajustada en mi búsqueda de D04 (prueba 38).
- **Envío:** los 11.000 clientes con mayor promedio. Es el mismo modelo que mandé el 7-oct con 11.500 clientes
  (`d06dn11500.csv`); el 9-oct una regla fijada antes de mirar bajó el corte a 11.000 (D10, en
  [DECISIONES.md](DECISIONES.md)). El archivo de 11.000 es exactamente las primeras 11.000 líneas del de 11.500.

También reproduce los envíos anteriores: **`d06dn11500.csv`** (D con 11.500), con `python reproducir.py d06d`, y
**`d04k2en11500.csv`** (la «base»), con `python reproducir.py base`.

## Cómo reproducirlo

Hace falta Python 3.13 de 64 bits y el crudo de la cátedra, `competencia_01_crudo.csv`, que no está en el
repositorio: se baja de <https://storage.googleapis.com/open-courses/dmeyf2026-9c6f/competencia_01_crudo.csv> (ver
`datos/LEEME.md`). En macOS, LightGBM necesita además OpenMP (`brew install libomp`).

```bash
git clone https://github.com/farpp8/dmeyf2026-competencia01.git
cd dmeyf2026-competencia01
python -m venv .venv
.venv\Scripts\activate            # en Linux o macOS: source .venv/bin/activate
pip install -r requirements.txt
# copiar competencia_01_crudo.csv a datos/, o definir COMPETENCIA_01_CRUDO con su ruta
python reproducir.py
```

**Comprobado en Windows 11,** en una carpeta limpia y con las versiones de `requirements.txt`: termina con
`OK: d10dn11000.csv coincide byte por byte con el archivo enviado`. Tardó 43 minutos, casi todos en entrenar los 20
modelos. `python reproducir.py d06d` tarda lo mismo, y la base (`python reproducir.py base`), unos 4 minutos. La
memoria máxima de un proceso que arma las variables y entrena D es de unos 12 GB.

En otro sistema operativo o con otra compilación de LightGBM, `deterministic=True` no garantiza las mismas
probabilidades. Si cambian, `reproducir.py` lo avisa, y el CSV puede diferir en clientes cercanos al corte.

## Qué hace cada script

| Script | Qué hace |
|---|---|
| `01_preparar_datos.py` | Lee el crudo, calcula `clase_ternaria` con la lógica de la cátedra y aplica las reparaciones de D01. Escribe `datos/competencia_01_d01.parquet`. |
| `02_variables.py` | Módulo de variables. Para D se usa `construir(df, hist_1y2_todas=True)`: *lags* y deltas 1 y 2 de todos los predictores. La base usa `BASE`, la historia selectiva de C7. |
| `03_envios_tablero.py` | Módulo con los meses, las 20 semillas y la escritura del CSV. Como script arma los envíos de prueba de D02; `reproducir.py` no lo corre así. |
| `06_receta_denicolay.py` | **D:** los 20 modelos con la receta de la prueba 38, sus probabilidades para 202108 y el CSV con los 11.500 clientes de mayor promedio. |
| `07_archivo_d10.py` | **El último envío:** los 11.000 clientes de mayor promedio de esas mismas 20 probabilidades, sin reentrenar. Controla que los 11.500 primeros sean los de `d06dn11500.csv`. |
| `05_envios_ensamble.py` | **La base:** `entrenar d03k2` y `archivo d03k2 d04k2e 11500`. |
| `test_01_preparar_datos.py`, `test_02_variables.py` | Pruebas con paneles chicos: clase, reparaciones, identificadores, rutas con apóstrofo e historia por mes exacto. |
| `reproducir.py` | Corre las pruebas y los pasos en orden, y compara cada etapa con `huellas_esperadas.json`. |
| `origen/` | **De dónde salen los hiperparámetros y el corte:** la búsqueda bayesiana y la selección de la D04, que dieron las 83 hojas y el hessiano, y la regla de N de la D10, que dio 11.000. Ver [origen/README.md](origen/README.md). No hace falta para reproducir el archivo. |

## Qué fija el resultado de D

- **Semillas:** 147347, 322607, 462467, 782149 y 983809, y los tres primos siguientes a cada una
  (`semillas()` en `03_envios_tablero.py`).
- **LightGBM** (`06_receta_denicolay.py`):
  - `objective` binary, `learning_rate` 0,005, `max_bin` 31, `min_data_in_leaf` 0, `feature_fraction` 0,5,
    `feature_fraction_bynode` 0,2 y sin *bagging*;
  - 83 hojas, 1.000 rondas y `min_sum_hessian_in_leaf` = 12,791817 × filas / 326.184;
  - `deterministic=True`, `force_col_wise=True` y `num_threads=12`, fijos en el código. No hace falta tener 12
    núcleos, pero con menos tarda más;
  - el `Dataset` se construye una vez y se reutiliza entre semillas.
- **Clase:** BAJA+1 y BAJA+2 contra CONTINUA. La ganancia cuenta sólo BAJA+2.
- **Empates:** los clientes de 202108 están ordenados por `numero_de_cliente` y el orden se toma con `argsort`
  estable.
- **CSV:** un `numero_de_cliente` por línea, sin encabezado, en ASCII y con saltos de línea CRLF.
- **Entorno de la corrida original:** Windows 11 Pro, AMD Ryzen 7 9800X3D (8 núcleos, 16 hilos), Python 3.13.3 y
  las versiones de `requirements.txt`.

`reproducir.py` se detiene si el crudo no es el de la cátedra o si el CSV final no coincide. Si difieren el parquet
o las probabilidades, por ejemplo con otro procesador, avisa y sigue: lo que se envió es la lista de clientes.

## Huellas

| Archivo | SHA-256 |
|---|---|
| `competencia_01_crudo.csv`, la entrada (492.769.580 bytes) | `c72c8f5c1149f1d972e996c2963c5e3dbb31eb38cfe41c2bb1bbaf847d7a7ca1` |
| `d10dn11000.csv`, D con N = 11.000, el último envío | `8a03bf3d5fd792569187e8093a1663f7c7f97eeab7b785af412b06fc595a70b5` |
| `d06dn11500.csv`, D | `de534992228e5f26ebda4dc867f86dc78282cc194030e1884ecdd2ffd1a3d294` |
| `d04k2en11500.csv`, la base | `91c18043de05f63a85108a0463599171b7e8ede3d1fa73f0e65aa3a0208684ba` |

Las intermedias, del parquet, las claves, las columnas y las 20 probabilidades de cada entrega, están en
`huellas_esperadas.json`.

## Diferencias con los scripts que generaron los envíos

Son copias de los que usé, con estos cambios, que no alteran el resultado:

1. `01_preparar_datos.py` lee el crudo de `datos/` o de `COMPETENCIA_01_CRUDO`. Antes lo buscaba en la carpeta
   de mi proyecto.
2. `01_preparar_datos.py` duplica las comillas simples de las rutas que pasa a SQL. Sin eso, una carpeta como
   `O'Connor` hacía fallar la lectura del crudo.
3. `03_envios_tablero.py` escribe el CSV con `newline="\r\n"`. En Windows el modo texto ya convertía `\n` en
   CRLF, así que el archivo es el mismo; en Linux o macOS, sin este cambio, cambiaría la huella.
4. El docstring de `01_preparar_datos.py` y su prueba usan identificadores sintéticos en lugar de un cliente real
   del crudo.
5. `06_receta_denicolay.py` es el camino de entrenamiento de D de mi armador de candidatos, en un solo archivo y sin
   depender de mis carpetas de trabajo. Antes de armar D, ese armador reprodujo bit a bit los dos caminos de
   entrenamiento que usa: el de la configuración vigente y el de la prueba 38.
6. `02_variables.py` no incluye dos opciones que agregué después en mi copia, `cero_fijo` y `mantenimiento_signo`.
   Las probé en D06, no las adopté y ni D ni la base las usan.
7. `07_archivo_d10.py` es el camino de D de mi armador del archivo de D10, en un solo archivo: toma las 20
   probabilidades de D y escribe los 11.000 primeros.

Los docstrings mencionan archivos de mi carpeta de trabajo que no se publican: `decisiones/`, `experimentos/`,
`envios/` y el script que manda los archivos al bot. El resumen de las decisiones está en
[DECISIONES.md](DECISIONES.md).

## Qué no incluye

- el crudo ni nada derivado de él: parquet, modelos o probabilidades;
- los CSV enviados ni otras listas de clientes;
- el script de envío al bot ni credenciales.
