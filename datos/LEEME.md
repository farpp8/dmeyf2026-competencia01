# datos/

Acá va el dataset crudo de la cátedra, `competencia_01_crudo.csv` (492.769.580 bytes, SHA-256
`c72c8f5c1149f1d972e996c2963c5e3dbb31eb38cfe41c2bb1bbaf847d7a7ca1`). No se incluye en el repositorio.

Se baja del almacenamiento de la cátedra, la dirección que usan los notebooks del curso:
<https://storage.googleapis.com/open-courses/dmeyf2026-9c6f/competencia_01_crudo.csv>. `reproducir.py` controla su
huella antes de empezar.

También se puede dejar en otro lugar y definir la variable de entorno `COMPETENCIA_01_CRUDO` con su ruta.

`01_preparar_datos.py` escribe en esta carpeta `competencia_01_d01.parquet`, `reparaciones_d01.parquet` y
`resumen_d01.json`.
