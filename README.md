# DMEyF 2026 · Competencia 1

**Federico Ramírez** · Data Mining en Economía y Finanzas, Maestría en Data Mining (UBA), 2026 · comisión del lunes.

Este repositorio reproduce, desde el dataset crudo de la cátedra, el archivo que entregué al bot de la
competencia: **`d04k2en11500.csv`**, con 11.500 clientes de 202108 y SHA-256
`91c18043de05f63a85108a0463599171b7e8ede3d1fa73f0e65aa3a0208684ba`.

**El modelo:** el promedio de las probabilidades de 20 LightGBM, uno por semilla, entrenados con 202103–202106
para separar BAJA+1 y BAJA+2 de CONTINUA. Usan los datos reparados más 20 variables de historia. Se envían los
11.500 clientes con mayor promedio.

## Cómo reproducirlo

Hace falta Python 3.13 de 64 bits y el crudo de la cátedra, `competencia_01_crudo.csv`, que no está en el
repositorio. En macOS, LightGBM necesita además OpenMP (`brew install libomp`).

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
`OK: d04k2en11500.csv coincide byte por byte con el archivo entregado`. Tardó 3,3 minutos, casi todos en entrenar
los 20 modelos, de unos 9 segundos cada uno.

En otro sistema operativo o con otra compilación de LightGBM, `deterministic=True` no garantiza las mismas
probabilidades. Si cambian, `reproducir.py` lo avisa, y el CSV puede diferir en clientes cercanos al corte.

## Qué hace cada script

| Script | Qué hace |
|---|---|
| `01_preparar_datos.py` | Lee el crudo, calcula `clase_ternaria` con la lógica de la cátedra y aplica las reparaciones de D01. Escribe `datos/competencia_01_d01.parquet`. |
| `02_variables.py` | Módulo de variables. La base que se usa, `BASE`, es C7: los datos de D01 más *lag* 1 y delta 1 de seis conteos y del ranking mensual de cuatro montos. |
| `03_envios_tablero.py` | Módulo con los meses, las 20 semillas, los hiperparámetros y la escritura del CSV. Como script arma los envíos de prueba de D02; `reproducir.py` no lo corre así. |
| `05_envios_ensamble.py` | `entrenar d03k2`: los 20 modelos de clase K2 y sus probabilidades para 202108. `archivo d03k2 d04k2e 11500`: el CSV con los 11.500 clientes de mayor promedio. |
| `test_01_preparar_datos.py`, `test_02_variables.py` | Pruebas con paneles chicos: clase, reparaciones, identificadores, rutas con apóstrofo e historia por mes exacto. |
| `reproducir.py` | Corre las pruebas y los pasos en orden, y compara cada etapa con `huellas_esperadas.json`. |

## Qué fija el resultado

- **Semillas:** 147347, 322607, 462467, 782149 y 983809, y los tres primos siguientes a cada una
  (`semillas()` en `03_envios_tablero.py`).
- **LightGBM:** `objective` binary, `learning_rate` 0,05, `num_leaves` 63, `min_data_in_leaf` 200,
  `feature_fraction` 0,5, `bagging_fraction` 0,8 con `bagging_freq` 1 y 500 rondas. `deterministic=True`,
  `force_col_wise=True` y `num_threads=12` están fijos en el código. No hace falta tener 12 núcleos, pero con
  menos tarda más.
- **Empates:** los clientes de 202108 están ordenados por `numero_de_cliente` y el orden se toma con `argsort`
  estable.
- **CSV:** un `numero_de_cliente` por línea, sin encabezado, en ASCII y con saltos de línea CRLF.
- **Entorno de la corrida original:** Windows 11 Pro, AMD Ryzen 7 9800X3D (8 núcleos, 16 hilos), Python 3.13.3 y
  las versiones de `requirements.txt`.

`reproducir.py` se detiene si el crudo no es el de la cátedra o si el CSV final no coincide. Si difieren el
parquet o las probabilidades, por ejemplo con otro procesador, avisa y sigue: lo que se entregó es la lista de
clientes.

## Huellas

| Archivo | SHA-256 |
|---|---|
| `competencia_01_crudo.csv`, la entrada (492.769.580 bytes) | `c72c8f5c1149f1d972e996c2963c5e3dbb31eb38cfe41c2bb1bbaf847d7a7ca1` |
| `d04k2en11500.csv`, la entrega | `91c18043de05f63a85108a0463599171b7e8ede3d1fa73f0e65aa3a0208684ba` |

Las intermedias, del parquet, las claves, las columnas y las 20 probabilidades, están en `huellas_esperadas.json`.

## Diferencias con los scripts que generaron el envío

Son copias de los que usé, con estos cambios, que no alteran el resultado:

1. `01_preparar_datos.py` lee el crudo de `datos/` o de `COMPETENCIA_01_CRUDO`. Antes lo buscaba en la carpeta
   de mi proyecto.
2. `01_preparar_datos.py` duplica las comillas simples de las rutas que pasa a SQL. Sin eso, una carpeta como
   `O'Connor` hacía fallar la lectura del crudo.
3. `03_envios_tablero.py` escribe el CSV con `newline="\r\n"`. En Windows el modo texto ya convertía `\n` en
   CRLF, así que el archivo es el mismo; en Linux o macOS, sin este cambio, cambiaría la huella.
4. El docstring de `01_preparar_datos.py` y su prueba usan identificadores sintéticos en lugar de un cliente real
   del crudo.

Los docstrings mencionan archivos de mi carpeta de trabajo que no se publican: `decisiones/`, `experimentos/`,
`envios/` y el script que manda los archivos al bot. El resumen de las decisiones está en
[DECISIONES.md](DECISIONES.md).

## Qué no incluye

- el crudo ni nada derivado de él: parquet, modelos o probabilidades;
- el CSV entregado ni otras listas de clientes;
- el script de envío al bot ni credenciales.
