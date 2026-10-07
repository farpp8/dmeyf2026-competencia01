# -*- coding: utf-8 -*-
"""Competencia 1: arma los tres envíos al tablero público. Sólo los prepara; no envía nada.

Los aprobó el usuario el 5-oct-2026. Cada envío tiene 20 modelos LightGBM, uno por semilla, entrenados con
202103–202106 (BAJA+2 contra el resto), que predicen 202108. Cada archivo tiene los N clientes con mayor
probabilidad, un `numero_de_cliente` entero por línea y sin encabezado, como pide el bot
(github.com/alejandro-bolanos/zulip-competition-bot).

| Envío | Variables |
|---|---|
| `d02c0` | C0, la decisión de D02 |
| `d02c7` | C0 + historia selectiva |
| `d02l12` | C0 + *lags* y deltas de orden 1 y 2 de todas las variables, la receta de Denicolay |

- **Semillas:** las cinco del usuario (CheckIn #182829) y, de cada una, los tres primos siguientes: 20 primos.
- **Hiperparámetros:** los mismos fijos de las pruebas de D02. Todavía no están optimizados; eso es D04.
- **Cantidad de clientes:** N = 11.000 en todos los archivos, de forma provisoria hasta D03. El bot devuelve
  una sola media por envío, así que la cantidad tiene que ser la misma en todos.

Uso: `python 03_envios_tablero.py [envio ...]`

Salidas en `envios/<envio>/`:
- los 20 CSV;
- `probabilidades_<semilla>.npy` y `claves_202108.npy`, para el ensamble final;
- `manifiesto.json`, con huellas, parámetros, semillas y controles de cada archivo.
"""
import hashlib
import importlib.util
import json
import platform
import re
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("variables", AQUI / "02_variables.py")
var = importlib.util.module_from_spec(spec)
spec.loader.exec_module(var)

DATOS = AQUI / "datos" / "competencia_01_d01.parquet"
SALIDA = AQUI / "envios"
MESES_TRAIN = [202103, 202104, 202105, 202106]
MES_PREDICCION = 202108
N_ENVIO = 11_000
SEMILLAS_USUARIO = [147347, 322607, 462467, 782149, 983809]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.5,
              bagging_fraction=0.8, bagging_freq=1, verbosity=-1, num_threads=12, deterministic=True,
              force_col_wise=True)
RONDAS = 500
ENVIOS = {"d02c0": dict(), "d02c7": dict(hist_selectiva=True), "d02l12": dict(hist_1y2_todas=True)}


def es_primo(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def semillas():
    """Las cinco del usuario y los tres primos siguientes a cada una."""
    out = []
    for p in SEMILLAS_USUARIO:
        out.append(p)
        q = p
        for _ in range(3):
            q += 1
            while not es_primo(q):
                q += 1
            out.append(q)
    assert len(set(out)) == 20 and all(es_primo(s) for s in out)
    return out


def huella(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def escribir_csv(path, ids):
    """Un entero por línea, sin encabezado; se verifica el texto escrito."""
    ids = np.asarray(ids, dtype="int64")
    # Saltos de línea CRLF explícitos: así se escribió el archivo enviado (en Windows) y la huella no depende del sistema.
    path.write_text("\n".join(str(int(i)) for i in ids) + "\n", encoding="ascii", newline="\r\n")
    lineas = path.read_text(encoding="ascii").splitlines()
    if len(lineas) != len(ids) or not all(re.fullmatch(r"[0-9]+", l) for l in lineas):
        raise RuntimeError(f"{path.name}: formato inválido")


def armar(nombre, base):
    carpeta = SALIDA / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    panel = var.construir(base, **ENVIOS[nombre])
    panel = panel[panel.foto_mes.isin(MESES_TRAIN + [MES_PREDICCION])].reset_index(drop=True)
    predictores = var.predictores_de(panel)
    en_train = panel.foto_mes.isin(MESES_TRAIN).to_numpy()
    en_pred = (panel.foto_mes == MES_PREDICCION).to_numpy()
    X = panel[predictores].astype("float32")
    cols = [c for c in predictores if X.loc[en_train, c].nunique(dropna=False) >= 2]
    X_train = np.ascontiguousarray(X.loc[en_train, cols].to_numpy())
    X_pred = np.ascontiguousarray(X.loc[en_pred, cols].to_numpy())
    y = (panel.loc[en_train, var.CLASE] == "BAJA+2").to_numpy(dtype="int8")
    claves = panel.loc[en_pred, "numero_de_cliente"].to_numpy(dtype="int64")
    if len(np.unique(claves)) != len(claves) or (panel.loc[en_pred, var.CLASE].notna().any()):
        raise RuntimeError("202108 debe tener clientes únicos y sin clase")
    np.save(carpeta / "claves_202108.npy", claves)
    print(f"{nombre}: {len(cols)} predictores útiles de {len(predictores)}, {en_train.sum()} filas de entrenamiento, "
          f"{len(claves)} clientes de agosto ({time.time() - t0:.0f} s)", flush=True)
    archivos = []
    for s in semillas():
        t1 = time.time()
        modelo = lgb.train({**PARAMS, "seed": s}, lgb.Dataset(X_train, label=y), num_boost_round=RONDAS)
        p = modelo.predict(X_pred)
        np.save(carpeta / f"probabilidades_{s}.npy", p.astype("float64"))
        elegidos = claves[np.argsort(-p, kind="stable")[:N_ENVIO]]
        csv = carpeta / f"{nombre}s{s}.csv"
        escribir_csv(csv, elegidos)
        archivos.append({"archivo": csv.name, "semilla": s, "filas": int(len(elegidos)), "sha256": huella(csv),
                         "prob_corte": float(np.sort(p)[::-1][N_ENVIO - 1])})
        print(f"  semilla {s}: {time.time() - t1:.0f} s, probabilidad en el corte {archivos[-1]['prob_corte']:.4f}", flush=True)
    # Controles del conjunto
    agosto = set(claves.tolist())
    for a in archivos:
        ids = [int(l) for l in (carpeta / a["archivo"]).read_text(encoding="ascii").split()]
        if len(ids) != N_ENVIO or len(set(ids)) != N_ENVIO or not set(ids) <= agosto:
            raise RuntimeError(f"{a['archivo']}: control fallido")
    manifiesto = {"envio": nombre, "opciones_construir": ENVIOS[nombre], "meses_train": MESES_TRAIN,
                  "mes_prediccion": MES_PREDICCION, "n_por_archivo": N_ENVIO, "params": PARAMS, "rondas": RONDAS,
                  "predictores_utiles": len(cols), "semillas": semillas(),
                  "huellas": {"datos": huella(DATOS), "02_variables.py": huella(AQUI / "02_variables.py"),
                              "03_envios_tablero.py": huella(__file__)},
                  "versiones": {"python": platform.python_version(), "lightgbm": lgb.__version__},
                  "archivos": archivos}
    (carpeta / "manifiesto.json").write_text(json.dumps(manifiesto, indent=1), encoding="utf-8")
    (carpeta / "columnas.json").write_text(json.dumps(cols), encoding="utf-8")
    print(f"{nombre}: listo en {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    elegidos = sys.argv[1:] or list(ENVIOS)
    base = pd.read_parquet(DATOS)
    for nombre in elegidos:
        armar(nombre, base)
