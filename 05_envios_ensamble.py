# -*- coding: utf-8 -*-
"""Competencia 1: envíos de un solo archivo con el ensamble, el promedio de las probabilidades de 20 semillas.

Sirve para la verificación en el tablero que pidió el usuario el 6-oct-2026 (`decisiones/D03_clase_corte/
decision.md`): K1 contra K2, cada uno en un envío de un archivo, con N = 11.000. Sólo prepara los archivos; los
manda `04_enviar_al_bot.py`.

Uso:
    python 05_envios_ensamble.py entrenar d03k2              # K2 con 202103–202106 y las 20 semillas, predice 202108
    python 05_envios_ensamble.py archivo d02c7 d03k1e 11000  # top N del promedio de las probabilidades de un envío
    python 05_envios_ensamble.py archivo d03k2 d03k2e 11000

- **Modelos:** la base C7 (`BASE`), los hiperparámetros, las rondas, los meses y las 20 semillas son los de
  `03_envios_tablero.py`. Lo único que cambia entre K1 y K2 es la clase positiva del entrenamiento. K1 ya existe:
  es el envío `d02c7`.
- **Archivo del ensamble:** se promedian las probabilidades de las 20 semillas y se toman los N clientes de mayor
  promedio. Los empates se resuelven en el orden de `claves_202108.npy`, que está ordenado por cliente. Un
  `numero_de_cliente` entero por línea, sin encabezado.
- **Salidas:** cada carpeta de `envios/` tiene su `manifiesto.json`, con el formato que lee `04_enviar_al_bot.py`.
"""
import importlib.util
import json
import re
import sys
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

MODELOS = {"d03k2": {"positivas": ["BAJA+1", "BAJA+2"], "descripcion": "K2: BAJA+1 y BAJA+2 contra CONTINUA, base C7"}}


def entrenar(nombre):
    cfg = MODELOS[nombre]
    carpeta = tb.SALIDA / nombre
    if (carpeta / "manifiesto.json").exists():
        raise SystemExit(f"{nombre} ya existe; no se sobrescribe")
    carpeta.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    panel = var.construir(pd.read_parquet(tb.DATOS), **var.BASE)
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
    y = np.isin(clase[en_train], cfg["positivas"]).astype("int8")
    claves = panel.loc[en_pred, "numero_de_cliente"].to_numpy(dtype="int64")
    if len(np.unique(claves)) != len(claves) or not (np.diff(claves) > 0).all():
        raise RuntimeError("las claves de 202108 deben ser únicas y estar ordenadas")
    np.save(carpeta / "claves_202108.npy", claves)
    print(f"{nombre}: {len(cols)} predictores útiles, {en_train.sum()} filas de entrenamiento, {y.sum()} positivos, "
          f"{len(claves)} clientes de agosto ({time.time() - t0:.0f} s)", flush=True)
    for s in tb.semillas():
        t1 = time.time()
        modelo = lgb.train({**tb.PARAMS, "seed": s}, lgb.Dataset(X_train, label=y), num_boost_round=tb.RONDAS)
        p = modelo.predict(X_pred)
        if p.shape != (len(claves),) or not np.isfinite(p).all():
            raise RuntimeError(f"semilla {s}: predicción inválida")
        np.save(carpeta / f"probabilidades_{s}.npy", p.astype("float64"))
        print(f"  semilla {s}: {time.time() - t1:.0f} s", flush=True)
    manifiesto = {"modelo": nombre, "descripcion": cfg["descripcion"], "opciones_construir": var.BASE,
                  "positivas": cfg["positivas"], "meses_train": tb.MESES_TRAIN, "mes_prediccion": tb.MES_PREDICCION,
                  "params": tb.PARAMS, "rondas": tb.RONDAS, "predictores_utiles": len(cols), "positivos_train": int(y.sum()),
                  "semillas": tb.semillas(),
                  "huellas": {"datos": tb.huella(tb.DATOS), "02_variables.py": tb.huella(AQUI / "02_variables.py"),
                              "03_envios_tablero.py": tb.huella(AQUI / "03_envios_tablero.py"),
                              "05_envios_ensamble.py": tb.huella(__file__)},
                  "versiones": {"lightgbm": lgb.__version__}, "archivos": []}
    (carpeta / "manifiesto.json").write_text(json.dumps(manifiesto, indent=1), encoding="utf-8")
    (carpeta / "columnas.json").write_text(json.dumps(cols), encoding="utf-8")
    print(f"{nombre}: listo en {time.time() - t0:.0f} s", flush=True)


def huellas_entrada(origen):
    """SHA-256 de las claves y de las 20 probabilidades que consume el ensamble de `origen`."""
    o = tb.SALIDA / origen
    m = json.loads((o / "manifiesto.json").read_text(encoding="utf-8"))
    nombres = ["claves_202108.npy"] + [f"probabilidades_{s}.npy" for s in m["semillas"]]
    return {nombre: tb.huella(o / nombre) for nombre in nombres}


def archivo(origen, destino, n):
    """Un envío de un solo archivo: los n clientes con mayor promedio de probabilidades de las 20 semillas."""
    if not 8000 <= n <= 15000:
        raise SystemExit("N debe estar entre 8.000 y 15.000 (regla de la entrega, Zulip #191492)")
    if not re.fullmatch(r"[A-Za-z0-9]+", f"{destino}n{n}"):
        raise SystemExit("el nombre del envío tiene que ser alfanumérico, como exige el bot")
    o = tb.SALIDA / origen
    m = json.loads((o / "manifiesto.json").read_text(encoding="utf-8"))
    semillas = m["semillas"]
    if len(semillas) != 20 or semillas != tb.semillas():
        raise SystemExit(f"{origen}: debe tener las 20 semillas de los envíos")
    claves = np.load(o / "claves_202108.npy")
    if len(np.unique(claves)) != len(claves) or not (np.diff(claves) > 0).all():
        raise SystemExit(f"{origen}: las claves de 202108 deben ser únicas y estar ordenadas")
    P = np.vstack([np.load(o / f"probabilidades_{s}.npy") for s in semillas])
    if P.shape != (20, len(claves)) or not np.isfinite(P).all():
        raise SystemExit(f"{origen}: probabilidades inválidas")
    prom = P.mean(axis=0)
    orden = np.argsort(-prom, kind="stable")
    elegidos = claves[orden[:n]]
    d = tb.SALIDA / destino
    if (d / "manifiesto.json").exists():
        raise SystemExit(f"{destino} ya existe; no se sobrescribe")
    d.mkdir(parents=True, exist_ok=True)
    csv = d / f"{destino}n{n}.csv"
    tb.escribir_csv(csv, elegidos)
    ids = [int(x) for x in csv.read_text(encoding="ascii").split()]
    if len(ids) != n or len(set(ids)) != n or not set(ids) <= set(claves.tolist()):
        raise RuntimeError(f"{csv.name}: control fallido")
    manifiesto = {"envio": destino, "tipo": "ensamble: promedio de las probabilidades de 20 semillas, top N",
                  "origen": origen, "origen_manifiesto_sha256": tb.huella(o / "manifiesto.json"),
                  "n_por_archivo": n, "semillas": semillas, "prob_corte": float(prom[orden[n - 1]]),
                  "huellas": {"05_envios_ensamble.py": tb.huella(__file__)},
                  "huellas_entrada": huellas_entrada(origen),
                  "archivos": [{"archivo": csv.name, "filas": n, "sha256": tb.huella(csv)}]}
    (d / "manifiesto.json").write_text(json.dumps(manifiesto, indent=1), encoding="utf-8")
    print(f"{destino}: {csv.name}, {n} clientes; probabilidad promedio en el corte {manifiesto['prob_corte']:.5f}")


if __name__ == "__main__":
    accion = sys.argv[1] if len(sys.argv) > 1 else ""
    if accion == "entrenar" and len(sys.argv) == 3 and sys.argv[2] in MODELOS:
        entrenar(sys.argv[2])
    elif accion == "archivo" and len(sys.argv) == 5:
        archivo(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        raise SystemExit(__doc__)
