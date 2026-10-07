# -*- coding: utf-8 -*-
"""Competencia 1: el archivo D, la receta de Denicolay sobre la historia completa (decisión D06, 7-oct-2026).

- **Variables:** C0, los datos de D01 sin transformaciones, más *lag* 1, delta 1, *lag* 2 y delta 2 de todos los
  predictores (`construir(df, hist_1y2_todas=True)` de `02_variables.py`). La historia se arma sobre el panel completo
  antes de filtrar meses, así que julio aporta los *lags* de agosto. Quedan 735 columnas útiles.
- **Clase K2:** BAJA+1 y BAJA+2 contra CONTINUA. La ganancia cuenta sólo BAJA+2.
- **Receta,** la de la prueba 38 de la búsqueda de D04, sobre la receta de Denicolay:
  - `learning_rate` 0,005, `max_bin` 31, `min_data_in_leaf` 0, `feature_fraction` 0,5, `feature_fraction_bynode`
    0,2, sin *bagging*;
  - 83 hojas, 1.000 rondas y `min_sum_hessian_in_leaf` = 12,791817… × filas de entrenamiento / 326.184;
  - `deterministic`, `force_col_wise` y 12 hilos;
  - el `Dataset` se construye una vez, con `max_bin` 31 y `feature_pre_filter` False, y se reutiliza entre semillas.
- **Entrenamiento:** 202103–202106, con las 20 semillas de `03_envios_tablero.py`; predice 202108.
- **Archivo:** los 11.500 clientes con mayor promedio de las 20 probabilidades, con desempate estable y el formato del
  envío (`escribir_csv`: un entero por línea, sin encabezado, ASCII con CRLF).

Uso: `python 06_receta_denicolay.py`. Tarda unos 40 minutos, casi todos en entrenar.

Salidas en `envios/d06d/`: las 20 probabilidades, `claves_202108.npy`, `columnas.json`, `d06dn11500.csv` y
`manifiesto.json`.
"""
import importlib.util
import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tablero", AQUI / "03_envios_tablero.py")
tb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tb)
var = tb.var

NOMBRE = "d06d"
N = 11_500
POSITIVAS = ["BAJA+1", "BAJA+2"]
FIJOS = dict(objective="binary", learning_rate=0.005, max_bin=31, min_data_in_leaf=0, feature_fraction=0.5,
             feature_fraction_bynode=0.2, bagging_fraction=1.0, bagging_freq=0, feature_pre_filter=False,
             verbosity=-1, num_threads=12, deterministic=True, force_col_wise=True)
HOJAS, H_REF, FILAS_REF, RONDAS = 83, 12.791817301585011, 326_184, 1000


def main():
    carpeta = tb.SALIDA / NOMBRE
    if (carpeta / "manifiesto.json").exists():
        raise SystemExit(f"{NOMBRE} ya existe; no se sobrescribe")
    carpeta.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    panel = var.construir(pd.read_parquet(tb.DATOS), hist_1y2_todas=True)
    panel = panel[panel.foto_mes.isin(tb.MESES_TRAIN + [tb.MES_PREDICCION])].reset_index(drop=True)
    predictores = var.predictores_de(panel)
    en_train = panel.foto_mes.isin(tb.MESES_TRAIN).to_numpy()
    en_pred = (panel.foto_mes == tb.MES_PREDICCION).to_numpy()
    clase = panel[var.CLASE].to_numpy(dtype=object)
    if set(clase[en_train]) != {"BAJA+1", "BAJA+2", "CONTINUA"} or pd.notna(clase[en_pred]).any():
        raise RuntimeError("clases inesperadas: 202103–202106 deben tener las tres y 202108 ninguna")
    X = panel[predictores].astype("float32")
    cols = [c for c in predictores if X.loc[en_train, c].nunique(dropna=False) >= 2]
    X_train = np.ascontiguousarray(X.loc[en_train, cols].to_numpy())
    X_pred = np.ascontiguousarray(X.loc[en_pred, cols].to_numpy())
    y = np.isin(clase[en_train], POSITIVAS).astype("int8")
    claves = panel.loc[en_pred, "numero_de_cliente"].to_numpy(dtype="int64")
    if len(np.unique(claves)) != len(claves) or not (np.diff(claves) > 0).all():
        raise RuntimeError("las claves de 202108 deben ser únicas y estar ordenadas")
    np.save(carpeta / "claves_202108.npy", claves)
    (carpeta / "columnas.json").write_text(json.dumps(cols), encoding="utf-8")
    print(f"{NOMBRE}: {len(cols)} columnas, {len(y)} filas, {int(y.sum())} positivos, {len(claves)} clientes de agosto "
          f"({time.time() - t0:.0f} s)", flush=True)
    ds = lgb.Dataset(X_train, label=y, params={"max_bin": FIJOS["max_bin"], "feature_pre_filter": False, "verbosity": -1},
                     free_raw_data=False)
    ds.construct()
    params = {**FIJOS, "num_leaves": HOJAS, "min_sum_hessian_in_leaf": H_REF * len(y) / FILAS_REF}
    probs = []
    for s in tb.semillas():
        t1 = time.time()
        p = lgb.train({**params, "seed": s}, ds, num_boost_round=RONDAS).predict(X_pred)
        if p.shape != (len(claves),) or not np.isfinite(p).all():
            raise RuntimeError(f"semilla {s}: predicción inválida")
        np.save(carpeta / f"probabilidades_{s}.npy", p.astype("float64"))
        probs.append(p)
        print(f"  semilla {s}: {time.time() - t1:.0f} s", flush=True)
    prom = np.mean(probs, axis=0)
    elegidos = claves[np.argsort(-prom, kind="stable")[:N]]
    csv = carpeta / f"{NOMBRE}n{N}.csv"
    tb.escribir_csv(csv, elegidos)
    ids = [int(x) for x in csv.read_text(encoding="ascii").split()]
    if len(ids) != N or len(set(ids)) != N or not set(ids) <= set(claves.tolist()):
        raise RuntimeError(f"{csv.name}: control fallido")
    manifiesto = {"envio": NOMBRE, "descripcion": "D: C0 + lags y deltas 1 y 2 de todo, receta de la prueba 38, K2",
                  "positivas": POSITIVAS, "meses_train": tb.MESES_TRAIN, "mes_prediccion": tb.MES_PREDICCION,
                  "params": params, "rondas": RONDAS, "semillas": tb.semillas(), "columnas": len(cols),
                  "filas_train": int(len(y)), "n_por_archivo": N,
                  "huellas": {"datos": tb.huella(tb.DATOS), "02_variables.py": tb.huella(AQUI / "02_variables.py"),
                              "03_envios_tablero.py": tb.huella(AQUI / "03_envios_tablero.py"),
                              "06_receta_denicolay.py": tb.huella(__file__)},
                  "versiones": {"python": platform.python_version(), "lightgbm": lgb.__version__},
                  "archivos": [{"archivo": csv.name, "filas": N, "sha256": tb.huella(csv)}]}
    (carpeta / "manifiesto.json").write_text(json.dumps(manifiesto, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{NOMBRE}: {csv.name}, {N} clientes; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
