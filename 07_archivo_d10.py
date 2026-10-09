# -*- coding: utf-8 -*-
"""Competencia 1: el archivo de D con N = 11.000, la regla de N de la decisión D10 (9-oct-2026).

Es el modelo D de `06_receta_denicolay.py`, sin reentrenar ni cambiar nada: sólo cambia cuántos clientes se mandan.
- **Por qué 11.000:** la decisión D10 fijó una regla antes de mirar, que se aplicó sobre las 20 semillas de producción
  de D en cuatro cortes de validación (202103 → 202105, 202104 → 202105, 202103–04 → 202106 y 202105 → 202106) y en
  tres escenarios más (junio sin la cohorte de préstamos congelados, y junio con el 92% y el 85% de los aciertos).
  - N* es el que minimiza el arrepentimiento máximo sobre la grilla 9.000–12.000.
  - Se cambia de 11.500 a N* sólo si la media de la ganancia mejora.
  - Dio N* = 11.000, con una mejora media de +2,99 M. Ver DECISIONES.md.
- **Entrada:** las 20 probabilidades y las claves de agosto de `envios/d06d/` (de `06_receta_denicolay.py`).
- **Control:** los 11.500 primeros del mismo promedio tienen que ser, en el mismo orden, los de `d06dn11500.csv`.
- **Archivo:** los 11.000 clientes con mayor promedio de las 20 probabilidades, con desempate estable y el formato del
  envío. Es el prefijo de 11.000 líneas de `d06dn11500.csv`.

Uso: `python 07_archivo_d10.py`, después de `06_receta_denicolay.py`. Tarda unos segundos.

Salidas en `envios/d10d/`: `d10dn11000.csv` y `manifiesto.json`.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tablero", AQUI / "03_envios_tablero.py")
tb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tb)

ORIGEN, NOMBRE, N, N_D = "d06d", "d10d", 11_000, 11_500


def main():
    origen, carpeta = tb.SALIDA / ORIGEN, tb.SALIDA / NOMBRE
    if not (origen / "manifiesto.json").exists():
        raise SystemExit(f"falta envios/{ORIGEN}: primero python 06_receta_denicolay.py")
    if (carpeta / "manifiesto.json").exists():
        raise SystemExit(f"{NOMBRE} ya existe; no se sobrescribe")
    claves = np.load(origen / "claves_202108.npy", allow_pickle=False)
    probs = [np.load(origen / f"probabilidades_{s}.npy", allow_pickle=False) for s in tb.semillas()]
    if any(p.shape != claves.shape or not np.isfinite(p).all() for p in probs):
        raise RuntimeError("probabilidades inválidas en envios/d06d")
    orden = np.argsort(-np.mean(probs, axis=0), kind="stable")
    ids_d = [int(x) for x in (origen / f"{ORIGEN}n{N_D}.csv").read_text(encoding="ascii").split()]
    if ids_d != claves[orden[:N_D]].tolist():
        raise RuntimeError(f"el promedio de las 20 probabilidades no reproduce {ORIGEN}n{N_D}.csv")
    carpeta.mkdir(parents=True, exist_ok=True)
    csv = carpeta / f"{NOMBRE}n{N}.csv"
    tb.escribir_csv(csv, claves[orden[:N]])
    ids = [int(x) for x in csv.read_text(encoding="ascii").split()]
    if len(ids) != N or len(set(ids)) != N or ids != ids_d[:N]:
        raise RuntimeError(f"{csv.name}: control fallido")
    manifiesto = {"envio": NOMBRE, "descripcion": "D con N = 11.000 (regla de N de D10)", "origen": f"envios/{ORIGEN}",
                  "semillas": tb.semillas(), "n_por_archivo": N,
                  "huellas": {f"{ORIGEN}/manifiesto.json": tb.huella(origen / "manifiesto.json"),
                              "07_archivo_d10.py": tb.huella(__file__)},
                  "archivos": [{"archivo": csv.name, "filas": N, "sha256": tb.huella(csv)}]}
    (carpeta / "manifiesto.json").write_text(json.dumps(manifiesto, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{NOMBRE}: {csv.name}, {N} clientes, el prefijo de {ORIGEN}n{N_D}.csv; "
          f"SHA-256 {manifiesto['archivos'][0]['sha256']}")


if __name__ == "__main__":
    main()
