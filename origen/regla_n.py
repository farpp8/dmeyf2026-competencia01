# -*- coding: utf-8 -*-
"""Origen de N = 11.000: la regla de N de la decisión D10 (9-oct-2026), aplicada a D.

Es el cálculo de `experimentos/d10/` de la carpeta de trabajo (el entrenamiento de D en los cuatro cortes y
`analizar.py n`), en un solo archivo. La regla se fijó antes de mirar los resultados.

**Qué hace:**
1. **Entrena D en cuatro cortes de validación,** con las 20 semillas de `03_envios_tablero.semillas()` y la receta de
   `06_receta_denicolay.py`: las mismas variables (C0 con *lags* y deltas 1 y 2 de todo), la clase K2 y los
   hiperparámetros. Las columnas útiles se calculan con las filas de entrenamiento de cada corte.

   | Corte | Entrenamiento | Validación |
   |---|---|---|
   | F1 | 202103 | 202105 |
   | F5 | 202104 | 202105 |
   | F2 | 202103–202104 | 202106 |
   | F4 | 202105 | 202106 |
2. **Arma siete escenarios** con el promedio de las 20 probabilidades. En cada uno, la ganancia
   G(N) = 1,1·TP(N) − 0,0275·N (M), con TP sólo de BAJA+2:
   - F1, F5, F2 y F4;
   - F2_L: junio sin la cohorte de préstamos personales congelados, en el orden de los demás. La cohorte son los
     clientes con `mprestamos_personales` > 0 e igual en 202104, 202105 y 202106. Se lee del crudo, porque esa columna
     no está en el dataset de `01_preparar_datos.py`;
   - F2 con el TP × 0,92 y × 0,85: un agosto más difícil.
3. **Aplica la regla** sobre la grilla N ∈ {9.000, 9.500, …, 12.000}:
   - el arrepentimiento de cada escenario es el máximo de G en la grilla menos G(N);
   - N* es el de menor arrepentimiento máximo; en un empate, el más cercano a 11.500, y si sigue, el menor;
   - se cambia de 11.500 a N* sólo si la media entre escenarios de G(N*) − G(11.500) es > 0.
4. **El control:** compara G, los arrepentimientos y N con la corrida original (`origen/esperado/regla_n_d10.json`).

Resultado original: N* = 11.000, con un arrepentimiento máximo de 6,05 contra 14,30 de 11.500 y una mejora media de
+2,99 M. Por eso el archivo entregado tiene 11.000 clientes (`07_archivo_d10.py`).

Uso, desde la raíz y después de `python 01_preparar_datos.py`, con el crudo en `datos/` o en
`COMPETENCIA_01_CRUDO`: `python origen/regla_n.py`. Entrena 80 modelos, unos 25 minutos.

Salidas en `origen/salidas/regla_n/`: las predicciones y `regla_n.json`.
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import comun

var, tb, rd = comun.var, comun.tb, comun.receta_d
SALIDA = comun.SALIDAS / "regla_n"
ESPERADO = comun.ESPERADO / "regla_n_d10.json"
CORTES = {"F1": ([202103], 202105), "F2": ([202103, 202104], 202106), "F5": ([202104], 202105),
          "F4": ([202105], 202106)}                     # el orden de la D10, que fija el de la media entre escenarios
GRILLA, ACTUAL, FACTORES_TP = [9000, 9500, 10000, 10500, 11000, 11500, 12000], 11_500, [0.92, 0.85]


def periodo(foto_mes):
    f = np.asarray(foto_mes, dtype="int64")
    return (f // 100) * 12 + f % 100


def cohorte(crudo, mes):
    """Clientes con `mprestamos_personales` > 0 e igual en mes, mes−1 y mes−2 (meses calendario). No mira la clase."""
    per, p = periodo(crudo["foto_mes"]), periodo([mes])[0]
    t = [crudo[per == p - j].set_index("numero_de_cliente")["mprestamos_personales"] for j in range(3)]
    ids = t[0].index.intersection(t[1].index).intersection(t[2].index)
    a, b, z = (x.loc[ids].to_numpy(dtype="float64") for x in t)
    return set(ids[(a > 0) & (a == b) & (a == z)].astype("int64"))


def regla_n(G):
    """`G`: {escenario: {N: ganancia}}. Devuelve N*, la mejora media, el N final y los arrepentimientos."""
    reg = {s: {n: max(G[s][m] for m in GRILLA) - G[s][n] for n in GRILLA} for s in G}
    peor = {n: max(reg[s][n] for s in G) for n in GRILLA}
    n_star = min(GRILLA, key=lambda n: (peor[n], abs(n - ACTUAL), n))
    mejora = float(np.mean([G[s][n_star] - G[s][ACTUAL] for s in G]))
    return {"N_estrella": n_star, "mejora_media": mejora, "N_final": n_star if mejora > 0 else ACTUAL,
            "arrepentimiento_maximo": peor}


def predicciones():
    """El promedio de las 20 semillas de D en cada corte, más la y y la cohorte del mes validado."""
    comun.exigir_datos()
    if not comun.CRUDO.exists():
        raise SystemExit(f"falta el crudo en {comun.CRUDO}: copiarlo a datos/ o definir COMPETENCIA_01_CRUDO")
    SALIDA.mkdir(parents=True, exist_ok=True)
    panel = var.construir(pd.read_parquet(comun.DATOS), hist_1y2_todas=True)
    panel = panel[panel.foto_mes.between(202103, 202106)].reset_index(drop=True)
    crudo = pd.read_csv(comun.CRUDO, usecols=["numero_de_cliente", "foto_mes", "mprestamos_personales"])
    predictores = var.predictores_de(panel)
    clase = panel[var.CLASE].to_numpy(dtype="U8")
    salida = {}
    for corte, (meses, mes_val) in CORTES.items():
        en_train = panel.foto_mes.isin(meses).to_numpy()
        en_val = (panel.foto_mes == mes_val).to_numpy()
        X = panel.loc[en_train | en_val, predictores].astype("float32")
        tr = en_train[en_train | en_val]
        cols = [c for c in predictores if X.loc[tr, c].nunique(dropna=False) >= 2]
        y_train = np.isin(clase[en_train], rd.POSITIVAS).astype("int8")
        X_val = np.ascontiguousarray(X.loc[~tr, cols].to_numpy())
        ds = lgb.Dataset(np.ascontiguousarray(X.loc[tr, cols].to_numpy()), label=y_train,
                         params={"max_bin": rd.FIJOS["max_bin"], "feature_pre_filter": False, "verbosity": -1},
                         free_raw_data=False)
        ds.construct()
        params = {**rd.FIJOS, "num_leaves": rd.HOJAS, "min_sum_hessian_in_leaf": rd.H_REF * len(y_train) / rd.FILAS_REF}
        probs = []
        for s in tb.semillas():
            path = SALIDA / f"D_{corte}_{s}.npy"
            if path.exists():
                probs.append(np.load(path, allow_pickle=False))
                continue
            t0 = time.time()
            p = lgb.train({**params, "seed": s}, ds, num_boost_round=rd.RONDAS).predict(X_val)
            np.save(path, p.astype("float64"), allow_pickle=False)
            probs.append(p)
            print(f"  D {corte} {s}: {time.time() - t0:.0f} s", flush=True)
        claves = panel.loc[en_val, "numero_de_cliente"].to_numpy(dtype="int64")
        salida[corte] = {"p": np.mean(probs, axis=0), "y": (clase[en_val] == "BAJA+2").astype("int8"),
                         "cohorte": np.isin(claves, np.fromiter(cohorte(crudo, mes_val), dtype="int64"))}
        print(f"{corte}: {len(cols)} columnas, {len(y_train)} filas de entrenamiento; la cohorte del mes validado "
              f"tiene {int(salida[corte]['cohorte'].sum())} clientes", flush=True)
    return salida


def escenarios(d):
    G = {c: {n: float(comun.curva(d[c]["y"], d[c]["p"])[n - 1]) for n in GRILLA} for c in CORTES}
    y, coh, p = d["F2"]["y"], d["F2"]["cohorte"], d["F2"]["p"]
    gl = comun.curva(y[~coh], p[~coh])
    G["F2_L"] = {n: float(gl[n - 1]) for n in GRILLA}
    tp = np.cumsum(y[np.argsort(-p, kind="stable")])
    for f in FACTORES_TP:
        G[f"F2_TPx{f}"] = {n: float(1.1 * f * tp[n - 1] - 0.0275 * n) for n in GRILLA}
    return G


def comparar_con_original(G, r):
    o = json.loads(ESPERADO.read_text(encoding="utf-8"))
    print("\nComparación con la corrida original:")
    ok = list(G) == list(o["G"])
    for s in G:
        ok &= comun.comparar(f"G de {s}", [G[s][n] for n in GRILLA], [o["G"][s][str(n)] for n in GRILLA], 1e-9)
    ok &= comun.comparar("arrepentimiento máximo", [r["arrepentimiento_maximo"][n] for n in GRILLA],
                         [o["arrepentimiento_maximo"][str(n)] for n in GRILLA], 1e-9)
    ok &= r["N_estrella"] == o["N_estrella"] and r["N_final"] == o["N_final"]
    print(f"  N* y N final: {r['N_estrella']} y {r['N_final']} (original {o['N_estrella']} y {o['N_final']})")
    print("→ " + ("reproduce la regla de N original" if ok else "NO reproduce la regla de N original"))
    return ok


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    G = escenarios(predicciones())
    r = regla_n(G)
    print(pd.DataFrame(G).T.round(2).to_string())
    print("arrepentimiento máximo: " + ", ".join(f"{n}: {x:.2f}" for n, x in r["arrepentimiento_maximo"].items()))
    print(f"N* = {r['N_estrella']}; media de G(N*) − G(11.500) = {r['mejora_media']:+.3f} → N = {r['N_final']}")
    (SALIDA / "regla_n.json").write_text(json.dumps({"G": G, **r}, indent=1), encoding="utf-8")
    if not comparar_con_original(G, r):
        raise SystemExit(1)
